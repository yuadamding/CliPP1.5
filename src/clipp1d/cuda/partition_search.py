"""Bounded streaming partition selection, separate from raw continuation."""
from dataclasses import dataclass
from copy import deepcopy
from time import perf_counter
import torch
from .partition import LastPartitionCache, _Snapshot, _SCALAR_FIELDS, grouping, refit_labels
from .policy import QualificationError
from .refinement import refine_memberships


@dataclass
class PartitionCandidate:
    family: str
    refit: object
    raw_reference: object
    reference_refit: object
    reference_lambda: torch.Tensor
    origin: str
    separate_exact_one: bool
    proposal: dict

    def __post_init__(self):
        self.reference_lambda = self.reference_lambda.clone()
        self.proposal = deepcopy(self.proposal)
        self._objects = tuple(id(x) for x in (self.refit, self.raw_reference, self.reference_refit))
        self._metadata = self._metadata_values()
        self._snapshot = _Snapshot(self._tensors())

    def _metadata_values(self):
        return (self.family, self.origin, self.separate_exact_one, deepcopy(self.proposal),
                tuple((r.clonal, r.scalar.subdivisions) for r in (self.refit, self.reference_refit)))

    def _tensors(self):
        tensors = [self.reference_lambda, self.raw_reference.x, self.raw_reference.objective]
        for r in (self.refit, self.reference_refit):
            tensors += [r.labels, r.centers, r.phi, r.sizes, r.loss, r.gap, r.score]
            tensors += [getattr(r.scalar, name) for name in _SCALAR_FIELDS]
        return tuple(tensors)

    def validate_identity(self):
        if (tuple(id(x) for x in (self.refit, self.raw_reference, self.reference_refit)) != self._objects or
                self._metadata_values() != self._metadata):
            raise QualificationError("Partition candidate identity/provenance was modified")
        self._snapshot.validate(self._tensors(), "Partition candidate")


@dataclass
class PartitionSearchResult:
    candidate: PartitionCandidate
    records: list
    status: str
    policy: object
    seconds: float

    def __post_init__(self):
        self.records = deepcopy(self.records)
        self._metadata = (id(self.candidate), deepcopy(self.records), self.status, self.policy, self.seconds)

    def validate_identity(self):
        if (id(self.candidate), self.records, self.status, self.policy, self.seconds) != self._metadata:
            raise QualificationError("Partition search evidence was modified")
        self.candidate.validate_identity()


class PartitionSearch:
    """Retain one winning reference, one refit cache, and <=4 seen partitions.

    A callback does not retain every start's dense dual matrix. Raw-start
    failures remain in raw-path diagnostics; proposal/refit failures have their
    own coverage records and cannot falsely complete the raw search.
    """

    def __init__(self, model, graph, pilots, policy, search_policy):
        self.model, self.graph, self.pilots = model, graph, pilots
        self.policy, self.search_policy = policy, search_policy
        self.cache = LastPartitionCache(model, policy)
        self.best = None
        self.records = []
        self.seconds = 0.
        self.seen = []
        self.penalty_index = -1

    def begin_lambda(self, index, lam):
        self.penalty_index = index
        self.lam = lam
        self.seen = []

    @staticmethod
    def key(candidate):
        return (float(candidate.refit.score), candidate.refit.centers.numel(),
                float(candidate.reference_lambda), candidate.origin != "baseline")

    def _accept(self, candidate):
        if self.best is None or self.key(candidate) < self.key(self.best):
            self.best = candidate

    def observe(self, raw, start):
        begin = perf_counter()
        origin = f"penalty_{self.penalty_index}:{start}"
        record = dict(origin=origin, candidate_family="raw_fusion_path",
                      reference_lambda=float(self.lam), raw_objective=float(raw.objective))
        try:
            if not raw.qualified or not bool(torch.isfinite(raw.objective)):
                raise QualificationError("Unqualified start cannot supply a partition candidate")
            labels = grouping(raw.x, self.policy.fusion_tol,
                              separate_clonal=self.search_policy.separate_exact_one)[2]
            if any(torch.equal(labels, seen) for seen in self.seen):
                record.update(status="duplicate_memberships")
                return
            self.seen.append(labels.clone())
            fitted = refit_labels(self.model, labels, self.policy,
                                  pilot_reuse=self.pilots, cache=self.cache)
            record.update(status="qualified", score=float(fitted.score),
                          clusters=fitted.centers.numel(), refit_gap=float(fitted.gap))
            candidate = PartitionCandidate("raw_fusion_path", fitted, raw, fitted, self.lam,
                                           origin, self.search_policy.separate_exact_one, {})
            self._accept(candidate)
        except QualificationError as error:
            record.update(status="unresolved", error=str(error), failure_diagnostics=error.diagnostics)
        finally:
            self.records.append(record)
            self.seconds += perf_counter() - begin

    def finish(self, raw, refit, lam):
        begin = perf_counter()
        baseline = PartitionCandidate("raw_fusion_path", refit, raw, refit, lam,
                                      "baseline", True, {})
        self.records.append(dict(origin="baseline", candidate_family="raw_fusion_path",
                                 reference_lambda=float(lam), raw_objective=float(raw.objective),
                                 status="qualified", score=float(refit.score),
                                 clusters=refit.centers.numel(), refit_gap=float(refit.gap)))
        self._accept(baseline)
        seed = self.best
        try:
            fitted, rounds, status = refine_memberships(self.model, seed.refit,
                                                       self.policy, self.search_policy)
        except QualificationError as error:
            fitted, status = seed.refit, "unresolved"
            rounds = [dict(status="unresolved", error=str(error), failure_diagnostics=error.diagnostics)]
        if bool(fitted.score < seed.refit.score):
            origin = "reassignment:" + seed.origin
            proposal = dict(algorithm="sequential_exact_score_moves_and_qualified_refits_v1",
                            seed_origin=seed.origin, seed_score=float(seed.refit.score),
                            status=status, rounds=rounds,
                            truth_used=False, raw_certificate_inherited=False)
            direct = PartitionCandidate("direct_partition", fitted, seed.raw_reference,
                                        seed.reference_refit, seed.reference_lambda,
                                        origin, seed.separate_exact_one, proposal)
            self._accept(direct)
            self.records.append(dict(origin=origin, candidate_family="direct_partition",
                                     reference_lambda=float(seed.reference_lambda), status="qualified",
                                     score=float(fitted.score), clusters=fitted.centers.numel(),
                                     refit_gap=float(fitted.gap), proposal=proposal))
        elif rounds:
            self.records.append(dict(origin="reassignment:" + seed.origin,
                                     candidate_family="direct_partition", status="no_improvement",
                                     proposal_status=status, rounds=rounds))
        self.best.validate_identity()
        if not bool(self.best.refit.score <= refit.score):
            raise QualificationError("Partition search discarded the baseline score")
        unresolved = any(r["status"] == "unresolved" for r in self.records) or status == "unresolved"
        coverage = "incomplete" if unresolved or status != "fixed_point" else "complete"
        self.seconds += perf_counter() - begin
        return PartitionSearchResult(self.best, self.records, coverage, self.search_policy, self.seconds)
