"""Bounded streaming partition selection, separate from raw continuation."""
from dataclasses import dataclass, replace
from copy import deepcopy
from time import perf_counter
import torch
from .partition import LastPartitionCache, _Snapshot, _SCALAR_FIELDS, grouping, refit_labels
from .policy import QualificationError
from .refinement import refine_memberships
from .birth import birth_candidates
from .ancestry import ancestry_step


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
    ancestors: tuple = ()

    def __post_init__(self):
        self.reference_lambda = self.reference_lambda.clone()
        self.proposal = deepcopy(self.proposal)
        self._objects = tuple(id(x) for x in (self.refit, self.raw_reference, self.reference_refit, *self.ancestors))
        self._metadata = self._metadata_values()
        self._snapshot = _Snapshot(self._tensors())

    def _metadata_values(self):
        return (self.family, self.origin, self.separate_exact_one, deepcopy(self.proposal),
                tuple((r.clonal, r.scalar.subdivisions) for r in (self.refit, self.reference_refit, *self.ancestors)))

    def _tensors(self):
        tensors = [self.reference_lambda, self.raw_reference.x, self.raw_reference.objective]
        for r in (self.refit, self.reference_refit, *self.ancestors):
            tensors += [r.labels, r.centers, r.phi, r.sizes, r.loss, r.gap, r.score]
            tensors += [getattr(r.scalar, name) for name in _SCALAR_FIELDS]
        return tuple(tensors)

    def validate_identity(self):
        if (tuple(id(x) for x in (self.refit, self.raw_reference, self.reference_refit, *self.ancestors)) != self._objects or
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
    preserved: object

    def __post_init__(self):
        self.records = deepcopy(self.records)
        self._metadata = (id(self.candidate), deepcopy(self.records), self.status, self.policy, self.seconds, id(self.preserved))

    def validate_identity(self):
        if (id(self.candidate), self.records, self.status, self.policy, self.seconds, id(self.preserved)) != self._metadata:
            raise QualificationError("Partition search evidence was modified")
        self.candidate.validate_identity()
        self.preserved.validate_identity()


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
        self.bank = []

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

    def _retain_seed(self, candidate):
        """Bounded bank: best score, then distinct occupied K, then memberships."""
        pool = sorted(self.bank + [candidate], key=self.key)
        unique = []
        for c in pool:
            if not any(torch.equal(c.refit.labels, x.refit.labels) for x in unique):
                unique.append(c)
        chosen = unique[:1]
        for c in unique[1:]:
            if all(c.refit.centers.numel() != x.refit.centers.numel() for x in chosen):
                chosen.append(c)
        chosen.extend(c for c in unique if all(c is not x for x in chosen))
        self.bank = chosen[:self.search_policy.seed_bank_size]

    def _direct(self, parent, fitted, operation, details, status):
        origin = f"{operation}_{len(self.records)}:{parent.origin}"
        step = ancestry_step(operation, parent.refit, fitted, details, self.search_policy.minimum_decrease)
        proposal = dict(algorithm="partition_ancestry_v2", parent_origin=parent.origin,
                        root_origin=parent.proposal.get('root_origin', parent.origin),
                        ancestry=deepcopy(parent.proposal.get('ancestry', [])) + [step],
                        status=status, truth_used=False, raw_certificate_inherited=False)
        child = PartitionCandidate("direct_partition", fitted, parent.raw_reference,
                                   parent.reference_refit, parent.reference_lambda, origin,
                                   parent.separate_exact_one, proposal, parent.ancestors + (parent.refit,))
        self.records.append(dict(origin=origin, candidate_family="direct_partition",
                                 reference_lambda=float(parent.reference_lambda), status="qualified",
                                 score=float(fitted.score), clusters=fitted.centers.numel(),
                                 refit_gap=float(fitted.gap), proposal=proposal,
                                 coverage_status=status))
        self._accept(child)
        return child

    def _refine(self, seed, search_policy):
        try:
            fitted, rounds, status = refine_memberships(self.model, seed.refit, self.policy, search_policy)
        except QualificationError as error:
            fitted, status = seed.refit, "unresolved"
            rounds = [dict(status="unresolved", error=str(error), failure_diagnostics=error.diagnostics)]
        if bool(fitted.score < seed.refit.score):
            return self._direct(seed, fitted, "reassignment", dict(rounds=rounds), status)
        self.records.append(dict(origin=f"refinement_{len(self.records)}:{seed.origin}",
                                 candidate_family="direct_partition", status="no_improvement",
                                 proposal_status=status, coverage_status=status, rounds=rounds))
        return seed

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
            self._retain_seed(candidate)
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
        self._retain_seed(baseline)
        seed = self.best
        self._refine(seed, self.search_policy)
        # Preserve the exact old estimator AFTER its refinement, not only its
        # earlier raw seed. New searches cannot discard this qualified result.
        preserved = self.best
        seeds = [preserved]
        if self.search_policy.seed_bank_size > 1:
            additional = [baseline] + self.bank
            seen = [seed.refit.labels]
            for candidate in additional:
                if any(torch.equal(candidate.refit.labels, labels) for labels in seen):
                    continue
                seen.append(candidate.refit.labels)
                seeds.append(self._refine(candidate, self.search_policy))
        # Birth has an independent generation/refinement budget. Provisional
        # worse candidates may be refined or explored without replacing best.
        birth_policy = replace(self.search_policy, max_rounds=self.search_policy.birth_max_rounds)
        for generation in range(self.search_policy.birth_generations):
            next_seeds = []
            for parent in seeds:
                bank = []
                try:
                    for born, failure in birth_candidates(self.model, parent.refit, self.policy, self.search_policy):
                        if failure is not None:
                            status = failure['status']
                            self.records.append(dict(origin=f"birth_probe_{len(self.records)}:{parent.origin}",
                                status=status, generation=generation, details=failure,
                                coverage_status="fixed_point" if status == "infeasible_center_pair" else status))
                            continue
                        candidate = self._direct(parent, born.refit, "birth", born.proposal, "fixed_point")
                        bank.append(candidate)
                        bank = sorted(bank, key=self.key)[:self.search_policy.birth_refine_seeds]
                except QualificationError as error:
                    self.records.append(dict(origin=f"birth_{len(self.records)}:{parent.origin}", status="unresolved",
                                             error=str(error), coverage_status="unresolved"))
                next_seeds.extend(self._refine(c, birth_policy) for c in bank)
            seeds = sorted(next_seeds, key=self.key)[:self.search_policy.birth_refine_seeds]
            if not seeds:
                break
        self.best.validate_identity()
        if not bool(self.best.refit.score <= preserved.refit.score):
            raise QualificationError("Partition search discarded the preserved current result")
        coverage = "incomplete" if any(r['status'] == 'unresolved' or
            r.get('coverage_status', 'fixed_point') != 'fixed_point' for r in self.records) else "complete"
        self.seconds += perf_counter() - begin
        return PartitionSearchResult(self.best, self.records, coverage, self.search_policy, self.seconds, preserved)
