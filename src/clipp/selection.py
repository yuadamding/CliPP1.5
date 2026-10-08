"""Refit distance-to-set chain partitions and select observed-likelihood BIC.

Multiplicity is marginalized over 1..major_cn. Each occupied partition block
has one fitted cellular prevalence (CP), with CCF = CP / purity. The scalar
refit searches a dense grid and every component's binomial mode, then refines
all bracketed maxima; this is a deterministic multimode numerical search, not
a certificate of global optimality. BIC marginalizes both unknown cluster and
multiplicity, fitting cluster weights at the candidate's fixed CP centers. It
counts q center parameters and q-1 independent weights for q occupied blocks.
Adjacent coarsenings and conditional boundary polishing improve the proposal
bank on the same frozen chain. Centers use conditional block refitting;
observed-mixture weights are fitted only for BIC. No fusion/allocation penalty enters BIC.
"""

from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from ._candidate_store import CandidateStore
from ._io import partition_labels
from ._proposals import iter_chain_proposals, _integers as _integer_vector
from ._progress import NumericalProgress
from .config import MAX_CLUSTERS
from .refinement import polish_chain_partition
from .model import MultiplicityModel
from .refitting import refit_center
from .scoring import fit_cluster_weights
from .output import _write_result
from .versions import IDENTITIES, FIT_FIELDS, FIT_SCORE_FIELDS


_CONDITIONAL_FIELDS = ("num_clusters",) + FIT_SCORE_FIELDS
_PARENT_FIELDS = ("parent_requested_k", "parent_replicate", "parent_partition_sha256")


def _cuts(labels):
    return np.flatnonzero(np.diff(labels)) + 1


def _fingerprint(chain_digest, cuts):
    digest = chain_digest.copy()
    digest.update(np.asarray(cuts, dtype="<i8").tobytes())
    return digest.hexdigest()


def _parameters(result, order):
    """Lossless O(blocks + q) evidence; visits need not follow center order."""
    ordered = result["labels"][order]
    cuts = _cuts(ordered)
    return json.dumps(
        {
            "cuts": cuts.tolist(),
            "block_labels": ordered[np.r_[0, cuts]].tolist(),
            "centers": result["centers"].tolist(),
            "weights": result["cluster_weights"].tolist(),
        },
        separators=(",", ":"),
        allow_nan=False,
    )


class _ChainRefitCache(OrderedDict):
    """One immutable model/order's bounded numerical caches, without provenance."""

    def __init__(
        self,
        model,
        chain_order,
        max_column_bytes=32 * 1024 * 1024,
        *,
        max_partition_bytes=8 * 1024 * 1024,
        progress=None,
    ):
        super().__init__()
        self.model = model
        self.order = np.asarray(chain_order, dtype=np.int64).copy()
        self.order.setflags(write=False)
        self.ranks = np.empty(len(model), dtype=np.int64)
        self.ranks[self.order] = np.arange(len(model))
        self.ranks.setflags(write=False)
        self.progress = progress
        self.columns = OrderedDict()
        self.column_bytes = 0
        self.max_column_bytes = max_column_bytes
        self.max_intervals = 16384
        self.weight_results = OrderedDict()
        self.weight_bytes = 0
        self.max_weight_bytes = 8 * 1024 * 1024
        self.partition_results = OrderedDict()
        self.partition_bytes = 0
        self.max_partition_bytes = max_partition_bytes
        self.telemetry = dict(
            interval_cache_hits=0,
            interval_refits=0,
            conditional_refit_seconds=0.0,
            likelihood_cache_hits=0,
            likelihood_evaluations=0,
            weight_cache_hits=0,
            weight_solver_calls=0,
            weight_solver_failures=0,
            weight_solver_seconds=0.0,
            partition_cache_hits=0,
            partition_cache_misses=0,
        )

    def partition_key(self, labels):
        ordered = validate_chain_labels(labels[self.order], len(self.model), len(self.model))
        cuts = tuple(int(value) for value in _cuts(ordered))
        # Input cuts are essential: fitting blocks that later merge is not
        # necessarily numerically identical to directly refitting their union.
        return refit_center, fit_cluster_weights, cuts

    def get_partition(self, key):
        cached = self.partition_results.pop(key, None)
        if cached is None:
            self.telemetry["partition_cache_misses"] += 1
            return None
        self.partition_results[key] = cached
        self.telemetry["partition_cache_hits"] += 1
        if self.progress is not None:
            self.progress.add(partition_cache_hits=1)
            self.progress.update("partition_cache_hit")
        cuts, block_labels, value, _ = cached
        return {
            **value,
            "labels": partition_labels(self.order, cuts, block_labels),
            "centers": value["centers"].copy(),
            "cluster_weights": value["cluster_weights"].copy(),
        }

    def put_partition(self, key, result):
        ordered = result["labels"][self.order]
        cuts = _cuts(ordered)
        block_labels = ordered[np.r_[0, cuts]].copy()
        value = {name: item for name, item in result.items() if name != "labels"}
        for name in ("centers", "cluster_weights"):
            value[name] = value[name].copy()
            value[name].setflags(write=False)
        cuts.setflags(write=False)
        block_labels.setflags(write=False)
        # Conservative accounting includes Python containers, key cuts and
        # owned ndarray buffers; no N-length labels are retained per result.
        size = (
            sys.getsizeof(key)
            + sys.getsizeof(key[2])
            + sum(sys.getsizeof(cut) for cut in key[2])
            + sys.getsizeof(cuts)
            + sys.getsizeof(block_labels)
            + sys.getsizeof(value)
            + sum(sys.getsizeof(name) + sys.getsizeof(item) for name, item in value.items())
            + 256
        )
        if size <= self.max_partition_bytes:
            while self.partition_bytes + size > self.max_partition_bytes:
                _, removed = self.partition_results.popitem(last=False)
                self.partition_bytes -= removed[-1]
            self.partition_results[key] = (cuts, block_labels, value, size)
            self.partition_bytes += size

    def get(self, key, default=None):
        value = super().get(key, default)
        if key in self:
            self.move_to_end(key)
            self.telemetry["interval_cache_hits"] += 1
        return value

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.max_intervals:
            self.popitem(last=False)
        super().__setitem__(key, value)

    def fit_weights(self, centers, counts):
        # Model/order and numerical implementation are fixed by this cache.
        # Ordered float64 centers AND original count initialization are required;
        # duplicate columns can otherwise change exact-zero support.
        key = (fit_cluster_weights, centers.dtype.str, centers.tobytes(), counts.dtype.str, counts.tobytes())
        cached = self.weight_results.pop(key, None)
        if cached is not None:
            self.weight_results[key] = cached
            self.telemetry["weight_cache_hits"] += 1
            value = cached[0]
            return {**value, "cluster_weights": value["cluster_weights"].copy()}
        kernel = self.log_kernel(centers)
        start = time.perf_counter()
        self.telemetry["weight_solver_calls"] += 1
        try:
            with self.progress.timer("weight_optimization") if self.progress is not None else nullcontext():
                value = fit_cluster_weights(
                    self.model, centers, counts, log_kernel=kernel, telemetry=self.telemetry
                )
        except RuntimeError:
            self.telemetry["weight_solver_failures"] += 1
            raise
        finally:
            self.telemetry["weight_solver_seconds"] += time.perf_counter() - start
        size = (
            sys.getsizeof(key)
            + sum(sys.getsizeof(part) for part in key)
            + sys.getsizeof(value)
            + sys.getsizeof(value["cluster_weights"])
            + 256
        )
        if size <= self.max_weight_bytes:
            while self.weight_bytes + size > self.max_weight_bytes:
                _, (_, removed_size) = self.weight_results.popitem(last=False)
                self.weight_bytes -= removed_size
            self.weight_results[key] = ({**value, "cluster_weights": value["cluster_weights"].copy()}, size)
            self.weight_bytes += size
        return value

    def block_key(self, rows):
        ranks = self.ranks[rows]
        left, right = int(ranks.min()), int(ranks.max()) + 1
        if right - left != len(rows):
            raise ValueError("Cached refit requires a contiguous chain interval")
        return left, right

    def log_kernel(self, centers):
        if self.progress is not None:
            self.progress.add(likelihood_matrix_requests=1)
        with (
            self.progress.timer("likelihood_column_assembly") if self.progress is not None else nullcontext()
        ):
            return self._log_kernel(centers)

    def _log_kernel(self, centers):
        columns = []
        for cp in centers:
            key = float(cp)
            column = self.columns.pop(key, None)
            if column is None:
                self.telemetry["likelihood_evaluations"] += 1
                column = self.model.log_likelihood(cp)
                if self.progress is not None:
                    self.progress.add(
                        likelihood_columns_computed=1,
                        valid_state_evaluations=self.model.valid_states,
                    )
                    self.progress.update("likelihood_column_completed")
                if column.nbytes <= self.max_column_bytes:
                    while self.column_bytes + column.nbytes > self.max_column_bytes:
                        _, removed = self.columns.popitem(last=False)
                        self.column_bytes -= removed.nbytes
                    self.columns[key] = column
                    self.column_bytes += column.nbytes
            else:
                self.telemetry["likelihood_cache_hits"] += 1
                if self.progress is not None:
                    self.progress.add(likelihood_column_cache_hits=1)
                self.columns[key] = column
            columns.append(column)
        return np.column_stack(columns)


def refit_partition(model, labels, cache=None, chain_order=None):
    """Refit blocks; merge exact adjacent equal centers on a supplied chain.

    Without an explicit chain permutation no adjacency is inferred from labels.
    Equal centers separated by another center always remain separate blocks.
    """
    labels = np.atleast_1d(np.asarray(labels))
    if labels.shape != (len(model),) or not np.all(np.isfinite(labels)) or np.any(labels != np.rint(labels)):
        raise ValueError("Partition labels must be one integer per mutation")
    if isinstance(cache, _ChainRefitCache) and (
        cache.model is not model or chain_order is None or not np.array_equal(cache.order, chain_order)
    ):
        raise ValueError("Chain refit cache belongs to a different model or order")
    # Canonicalize before validating/cache lookup: arbitrary int64 labels can
    # be distinct above float64's exact-integer range, but cuts depend only on
    # membership. This is the same inverse used by the original refit path.
    clusters, inverse = np.unique(labels, return_inverse=True)
    partition_key = cache.partition_key(inverse) if isinstance(cache, _ChainRefitCache) else None
    if partition_key is not None:
        cached = cache.get_partition(partition_key)
        if cached is not None:
            return cached
    centers, log_likelihoods = [], []
    for index in range(len(clusters)):
        rows = np.flatnonzero(inverse == index)
        key = cache.block_key(rows) if isinstance(cache, _ChainRefitCache) else rows.tobytes()
        fitted = cache.get(key) if cache is not None else None
        if fitted is None:
            start = time.perf_counter()
            reporter = cache.progress if isinstance(cache, _ChainRefitCache) else None
            with (
                reporter.context(interval_left=key[0], interval_right=key[1])
                if reporter is not None
                else nullcontext()
            ):
                fitted = (
                    refit_center(model.subset(rows), progress=reporter)
                    if reporter is not None
                    else refit_center(model.subset(rows))
                )
            if isinstance(cache, _ChainRefitCache):
                cache.telemetry["interval_refits"] += 1
                cache.telemetry["conditional_refit_seconds"] += time.perf_counter() - start
            if cache is not None:
                cache[key] = fitted
        centers.append(fitted[0])
        log_likelihoods.append(fitted[1])
    centers = np.asarray(centers)
    if chain_order is not None:
        chain_order = _integer_vector(chain_order, "Frozen chain order", len(model))
        if not np.array_equal(np.sort(chain_order), np.arange(len(model))):
            raise ValueError("Frozen chain order must be a permutation of original row indices")
        ordered = validate_chain_labels(inverse[chain_order], len(model), len(centers))
        ordered_cp = centers[ordered]
        starts = np.r_[True, ordered_cp[1:] != ordered_cp[:-1]]
        inverse[chain_order] = np.cumsum(starts) - 1
        centers = ordered_cp[starts]
    order = np.argsort(-centers, kind="stable")
    relabel = np.empty(len(order), dtype=int)
    relabel[order] = np.arange(len(order))
    labels = relabel[inverse]
    centers = centers[order]
    conditional_log_likelihood = math.fsum(log_likelihoods)
    k = len(centers)
    counts = np.bincount(labels, minlength=k)
    mixture = (
        cache.fit_weights(centers, counts)
        if isinstance(cache, _ChainRefitCache)
        else fit_cluster_weights(model, centers, counts)
    )
    parameters = 2 * k - 1
    bic = -2 * mixture["log_likelihood"] + parameters * math.log(len(model))
    result = {
        "labels": labels,
        "centers": centers,
        **mixture,
        "conditional_log_likelihood": conditional_log_likelihood,
        "num_clusters": k,
        "num_parameters": parameters,
        "num_mutations": len(model),
        "bic": bic,
        "bic_definition": IDENTITIES["scoring_version"],
    }
    if partition_key is not None:
        cache.put_partition(partition_key, result)
    return result


def validate_chain_labels(labels, length, requested_k):
    """Allow label permutations, but require each occupied label to be one run."""
    labels = _integer_vector(labels, "Chain labels", length)
    starts = np.r_[True, labels[1:] != labels[:-1]]
    block_labels = labels[starts]
    if len(np.unique(block_labels)) != len(block_labels):
        raise ValueError("Each cluster must be contiguous along the frozen chain")
    if len(block_labels) > requested_k:
        raise ValueError("Partition exceeds its requested K block budget")
    return labels


def expand_subsample_partition(labels, sample_indices, chain_order, requested_k):
    """Expand sample blocks at adjacent sampled-rank midpoints, ties to the left.

    The returned labels are in full chain order. This interpolation uses only
    frozen chain positions and cannot create noncontiguous full-data clusters.
    """
    n = len(chain_order)
    indices = _integer_vector(sample_indices, "Subsample indices")
    if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
        raise ValueError("Invalid or duplicate subsample indices")
    ranks = np.empty(n, dtype=int)
    ranks[chain_order] = np.arange(n)
    sample_ranks = ranks[indices]
    if np.any(np.diff(sample_ranks) <= 0):
        raise ValueError("Subsample indices must follow the frozen full-chain order")
    labels = validate_chain_labels(labels, len(indices), requested_k)
    edges = np.flatnonzero(labels[1:] != labels[:-1])
    boundaries = (sample_ranks[edges] + sample_ranks[edges + 1]) // 2 + 1
    return np.searchsorted(boundaries, np.arange(n), side="right")


def _raw_diagnostics(path):
    if not os.path.isfile(path):
        raise ValueError("Missing native diagnostics: " + str(path))
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 1:
        raise ValueError("Native fit diagnostics must contain exactly one row: %s" % path)
    return {"raw_" + str(key): value for key, value in frame.iloc[0].items()}


def search_chain_coarsenings(
    model, seeds, chain_order, budgets, cache=None, *, store=None, replicate=1, progress=None
):
    """Compare native proposals and all their adjacent-block coarsenings.

    K remains a requested hyperparameter. A derived q-block partition is routed
    only to requested K=q, not used to introduce unrequested lower-K searches.
    Native proposals may themselves collapse below their capacity. No new cuts,
    ordering, likelihood, penalty, or unconstrained label reassignment is used.
    This searches a finite proposal bank, not all feasible chain partitions.
    """
    n = len(model)
    chain_order = _integer_vector(chain_order, "Frozen chain order", n)
    if not np.array_equal(np.sort(chain_order), np.arange(n)):
        raise ValueError("Frozen chain order must be a permutation of original row indices")
    budgets = _integer_vector(budgets, "Cluster budgets")
    if (
        len(np.unique(budgets)) != len(budgets)
        or np.any(budgets < 1)
        or np.any(budgets > min(MAX_CLUSTERS, n))
    ):
        raise ValueError(f"Expected unique K in 1..min({MAX_CLUSTERS},N)")
    cache = (
        _ChainRefitCache(
            model, chain_order, progress=NumericalProgress(progress) if progress is not None else None
        )
        if cache is None
        else cache
    )
    reporter = getattr(cache, "progress", None)
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())

    records = CandidateStore() if store is None else store
    winners, winner_keys = {}, {}
    unique_partitions = 0
    started = time.perf_counter()

    def append(record):
        record["replicate"] = replicate
        record["publication_eligible"] = (
            record["status"] == "scored" and record["active_mixture_components"] == record["num_clusters"]
        )
        return records.append(record)

    for cuts, by_budget in iter_chain_proposals(seeds, chain_order, budgets):
        unique_partitions += 1
        if progress is not None and (unique_partitions == 1 or unique_partitions % 1024 == 0):
            progress(
                dict(
                    stage="candidate_search",
                    replicate=replicate,
                    unique_proposals_started=unique_partitions,
                    candidate_records=records.next_id(replicate),
                    elapsed_seconds=time.perf_counter() - started,
                    scratch_bytes=records.scratch_bytes,
                    **cache.telemetry,
                )
            )
        labels = partition_labels(chain_order, cuts)
        proposal_digest = _fingerprint(chain_digest, cuts)
        try:
            with (
                reporter.context(
                    phase="candidate_search",
                    replicate=replicate,
                    requested_k=next(iter(by_budget)) if len(by_budget) == 1 else None,
                    requested_ks=sorted(by_budget),
                )
                if reporter is not None
                else nullcontext()
            ):
                result = refit_partition(model, labels, cache, chain_order=chain_order)
        except RuntimeError as error:
            if any(p["candidate_kind"] == "native" for p in by_budget.values()):
                raise
            for provenance in by_budget.values():
                append(
                    {
                        **provenance,
                        "status": "refit_failed",
                        "error": str(error),
                        "proposal_partition_sha256": proposal_digest,
                        "chain_cuts": ",".join(map(str, cuts)),
                        "candidate_search_version": IDENTITIES["candidate_search_version"],
                        "selected_for_k": False,
                    }
                )
            continue
        ordered_result = result["labels"][chain_order]
        final_cuts = tuple(int(i) for i in _cuts(ordered_result))
        digest = proposal_digest if final_cuts == cuts else _fingerprint(chain_digest, final_cuts)
        parameters = _parameters(result, chain_order)
        for budget, provenance in sorted(by_budget.items()):
            validate_chain_labels(result["labels"][chain_order], n, budget)
            record = {
                **provenance,
                "status": "scored",
                "partition_sha256": digest,
                "proposal_partition_sha256": proposal_digest,
                "chain_cuts": ",".join(map(str, cuts)),
                "num_input_blocks": len(cuts) + 1,
                "num_clusters": result["num_clusters"],
                "partition_parameters": parameters,
                "bic": result["bic"],
                "log_likelihood": result["log_likelihood"],
                "conditional_log_likelihood": result["conditional_log_likelihood"],
                "weight_optimality_gap": result["weight_optimality_gap"],
                "weight_active_score_gap": result["weight_active_score_gap"],
                "active_mixture_components": int(np.count_nonzero(result["cluster_weights"] > 0)),
                "candidate_search_version": IDENTITIES["candidate_search_version"],
                "selected_for_k": False,
            }
            record = append(record)
            rank = (
                result["bic"],
                result["num_clusters"],
                provenance["candidate_kind"] != "native",
                provenance["parent_requested_k"],
                provenance["parent_replicate"],
                cuts,
            )
            if budget not in winners or rank < winner_keys[budget]:
                winners[budget] = {**record, "cuts": cuts, "result": result}
                winner_keys[budget] = rank
    for winner in winners.values():
        records.update(replicate, winner["candidate_id"], {"selected_for_k": True})
    return {
        "replicate": replicate,
        "winners": winners,
        "candidates": records,
        "stats": {
            "unique_partitions": unique_partitions,
            "scored_budget_candidates": records.count(replicate=replicate),
            "cached_refit_blocks": len(cache),
            "likelihood_cache_bytes": getattr(cache, "column_bytes", 0),
        },
    }


def refine_chain_search(model, search, seeds, chain_order, cache):
    """Improve the same constrained objective and reconcile boundary weights.

    The native and coarsening bank remains evidence, including degenerate
    scores. Each budget's bank winner and native partition seed an alternating
    exact-boundary/conditional-center polish. This never optimizes unrestricted
    mixture memberships or changes the frozen order. An exactly zero mixture
    weight cannot describe a published occupied component: repair it by actual
    adjacent merges and refits, not by relabeling or a positive weight floor.
    """
    n = len(model)
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())
    records = search["candidates"]
    winners = {}
    replicate = search["replicate"]
    visited_scope = f"refine:{replicate}"
    reporter = getattr(cache, "progress", None)

    def fit(labels, budget):
        with (
            reporter.context(phase="repair_and_polish", replicate=replicate, requested_k=int(budget))
            if reporter is not None
            else nullcontext()
        ):
            return refit_partition(model, labels, cache, chain_order)

    def cuts_of(result):
        ordered = result["labels"][chain_order]
        return tuple(int(i) for i in _cuts(ordered))

    def admissible(result):
        return bool(np.all(result["cluster_weights"] > 0))

    for candidate in search["winners"].values():
        records.update(replicate, candidate["candidate_id"], {"selected_for_k": False})

    def consider(result, parent, kind, diagnostics=None, existing=None):
        budget = int(parent["requested_k"])
        cuts = cuts_of(result)
        validate_chain_labels(result["labels"][chain_order], n, budget)
        if not records.visited_add(visited_scope, (budget, cuts)):
            return
        if existing is not None:
            record = existing
        else:
            digest = _fingerprint(chain_digest, cuts)
            record = {key: parent[key] for key in ("requested_k",) + _PARENT_FIELDS}
            record.update(
                {
                    "candidate_id": records.next_id(replicate),
                    "candidate_kind": kind,
                    "status": "scored",
                    "proposal_partition_sha256": digest,
                    "partition_sha256": digest,
                    "chain_cuts": ",".join(map(str, cuts)),
                    "num_input_blocks": len(cuts) + 1,
                    "selected_for_k": False,
                    "candidate_search_version": IDENTITIES["candidate_search_version"],
                    "refinement_parent_partition_sha256": parent["partition_sha256"],
                }
            )
            record.update({key: result[key] for key in _CONDITIONAL_FIELDS})
        record.update(
            {
                "partition_parameters": _parameters(result, chain_order),
                "publication_eligible": admissible(result),
                "active_mixture_components": int(np.count_nonzero(result["cluster_weights"] > 0)),
                "minimum_mixture_weight": float(np.min(result["cluster_weights"])),
                "distinct_centers": int(len(np.unique(result["centers"]))),
                "joint_mixture_center_mle": False,
            }
        )
        if diagnostics:
            record.update({"boundary_" + key: value for key, value in diagnostics.items()})
        record["replicate"] = replicate
        if existing is None:
            record = records.append(record)
        else:
            record = records.update(
                replicate,
                record["candidate_id"],
                {k: v for k, v in record.items() if k not in {"replicate", "candidate_id"}},
            )
        candidate = {**record, "result": result, "cuts": cuts}
        if admissible(result):
            rank = (result["bic"], result["num_clusters"], record["candidate_id"])
            old = winners.get(budget)
            if old is None or rank < (old["bic"], old["num_clusters"], old["candidate_id"]):
                winners[budget] = candidate
            return
        # Explore only merges touching unsupported blocks. Every accepted
        # reduction is a newly fitted chain partition under the same K budget.
        ordered = result["labels"][chain_order]
        starts = np.r_[0, np.asarray(cuts)]
        unsupported = np.flatnonzero(result["cluster_weights"][ordered[starts]] == 0)
        remove = set()
        for block in unsupported:
            if block > 0:
                remove.add(block - 1)
            if block < len(cuts):
                remove.add(block)
        for edge in sorted(remove):
            reduced_cuts = cuts[:edge] + cuts[edge + 1 :]
            reduced_key = (budget, reduced_cuts)
            if records.visited_contains(visited_scope, reduced_key):
                continue
            labels = partition_labels(chain_order, reduced_cuts)
            try:
                repaired = fit(labels, budget)
            except RuntimeError as error:
                records.visited_add(visited_scope, reduced_key)
                records.append(
                    {
                        "replicate": replicate,
                        "candidate_id": records.next_id(replicate),
                        "requested_k": budget,
                        "candidate_kind": "unsupported_component_coarsening",
                        "status": "refit_failed",
                        "publication_eligible": False,
                        "selected_for_k": False,
                        "error": str(error),
                        **{key: parent[key] for key in _PARENT_FIELDS},
                        "proposal_partition_sha256": _fingerprint(chain_digest, reduced_cuts),
                        "chain_cuts": ",".join(map(str, reduced_cuts)),
                        "candidate_search_version": IDENTITIES["candidate_search_version"],
                    }
                )
                continue
            consider(repaired, candidate, "unsupported_component_coarsening")

    # Preserve every admissible old winner before adding improvements, so a
    # conditional-likelihood improvement cannot silently worsen the BIC winner.
    # Duplicate starts were already ignored by polishing; avoid refitting them.
    starts, start_partitions = [], set()
    for budget, candidate in sorted(search["winners"].items()):
        original = records.find_first(
            replicate=replicate,
            requested_k=budget,
            proposal_partition_sha256=candidate["proposal_partition_sha256"],
        )
        consider(candidate["result"], candidate, candidate["candidate_kind"], existing=original)
        starts.append(candidate)
        start_partitions.add((budget, candidate["partition_sha256"]))
        baseline = records.best(replicate, budget, eligible=True)
        if baseline is not None:
            if (budget, baseline["partition_sha256"]) in start_partitions:
                continue
            cuts = tuple(int(x) for x in baseline["chain_cuts"].split(",") if x)
            labels = partition_labels(chain_order, cuts)
            fitted = fit(labels, budget)
            consider(fitted, baseline, baseline["candidate_kind"], existing=baseline)
            starts.append({**baseline, "result": fitted})
            start_partitions.add((budget, baseline["partition_sha256"]))
    for seed in seeds:
        budget = seed["requested_k"]
        record = records.find_first(replicate=replicate, requested_k=budget, candidate_kind="native")
        if (budget, record["partition_sha256"]) in start_partitions:
            continue
        fitted = fit(seed["labels"], budget)
        consider(fitted, record, "native", existing=record)
        starts.append({**record, "result": fitted})
        start_partitions.add((budget, record["partition_sha256"]))
    polished = set()
    for start_index, parent in enumerate(starts, 1):
        key = (parent["requested_k"], cuts_of(parent["result"]))
        if key in polished:
            continue
        polished.add(key)
        kwargs = {"likelihood_provider": cache.log_kernel} if isinstance(cache, _ChainRefitCache) else {}
        if reporter is not None:
            kwargs["progress"] = reporter
        with (
            reporter.context(
                phase="repair_and_polish",
                replicate=replicate,
                requested_k=int(parent["requested_k"]),
                refinement_start=start_index,
            )
            if reporter is not None
            else nullcontext()
        ):
            refinement = polish_chain_partition(
                model,
                parent["result"],
                chain_order,
                lambda labels: fit(labels, parent["requested_k"]),
                **kwargs,
            )
        consider(refinement["result"], parent, "chain_boundary_polish", refinement["diagnostics"])
        # Preserve convergence evidence even when polish leaves the same cuts.
        original = records.find_first(
            replicate=replicate,
            requested_k=parent["requested_k"],
            proposal_partition_sha256=parent["proposal_partition_sha256"],
        )
        records.update(
            replicate,
            original["candidate_id"],
            {"polish_" + key: value for key, value in refinement["diagnostics"].items()},
        )
    for budget, parent in search["winners"].items():
        if budget not in winners:
            fallback = fit(np.zeros(n, dtype=int), budget)
            # A failed optional route must not hide a finite single-block
            # candidate. This is an explicit refit within the at-most-K set.
            records.visited_discard(visited_scope, (int(budget), ()))
            consider(fallback, parent, "supported_single_block_fallback")
    # Visited final partitions suppress repeated exploration, not candidate
    # ranking. A repair can visit an older eligible record's partition first,
    # leaving its smaller ID unconsidered. Reconcile only after the search so
    # its starts, refits and proposal order stay unchanged. Restore the recorded
    # fit exactly; refitting here could change a tied score or zero-weight support.
    for budget, winner in winners.items():
        record = records.best(replicate, budget, eligible=True)
        if record["candidate_id"] == winner["candidate_id"]:
            continue
        parameters = json.loads(record["partition_parameters"])
        labels = partition_labels(chain_order, parameters["cuts"], parameters["block_labels"])
        centers = np.asarray(parameters["centers"], dtype=float)
        weights = np.asarray(parameters["weights"], dtype=float)
        result = {key: record[key] for key in _CONDITIONAL_FIELDS}
        result.update(
            labels=labels,
            centers=centers,
            cluster_weights=weights,
            num_parameters=2 * record["num_clusters"] - 1,
            num_mutations=n,
            bic_definition=IDENTITIES["scoring_version"],
        )
        record = records.update(
            replicate,
            record["candidate_id"],
            {
                "minimum_mixture_weight": float(weights.min()),
                "distinct_centers": int(len(np.unique(centers))),
                "joint_mixture_center_mle": False,
            },
        )
        winners[budget] = {**record, "result": result, "cuts": tuple(parameters["cuts"])}
    for winner in winners.values():
        records.update(
            replicate, winner["candidate_id"], {"selected_for_k": True, "selected_for_replicate_k": True}
        )
    return {
        "winners": winners,
        "candidates": records,
        "original_bank_stats": search["stats"],
        "stats": {
            "unique_partitions": records.distinct_proposals(replicate),
            "scored_budget_candidates": records.count(replicate=replicate, status="scored"),
            "cached_refit_blocks": len(cache),
            "likelihood_cache_bytes": getattr(cache, "column_bytes", 0),
        },
    }


def run_model_selection(
    canonical,
    chain_order,
    preliminary_result,
    final_result,
    cluster_list,
    reps=None,
    *,
    candidate_store=None,
    progress=None,
):
    """Write the selected result and return compact fits plus transient audit tables.

    K is a block budget: q occupied blocks contribute 2q-1 parameters, including
    mixture weights. Unsupported components require actual adjacent refits.
    Native proposals and adjacent coarsenings are not global-optimality proofs.
    Subsample boundaries expand in chain-rank space before full-data refits;
    all candidate BICs use exactly the same N mutations and likelihood.
    """
    model = MultiplicityModel(*canonical.arrays)
    coordinates = canonical.retained[["chromosome_index", "position", "mutation_id", "original_row"]].copy()
    chain_order = _integer_vector(chain_order, "Frozen chain order", len(model))
    if not np.array_equal(np.sort(chain_order), np.arange(len(model))):
        raise ValueError("Frozen chain order must be a permutation of original row indices")
    budgets = _integer_vector(cluster_list, "Cluster budgets")
    if (
        len(np.unique(budgets)) != len(budgets)
        or np.any(budgets < 1)
        or np.any(budgets > min(MAX_CLUSTERS, len(model)))
        or (reps is not None and (isinstance(reps, bool) or int(reps) != reps or reps < 1))
    ):
        raise ValueError(f"Expected unique K in 1..min({MAX_CLUSTERS},N) and a positive replicate count")
    os.makedirs(final_result, exist_ok=True)
    replicate_count = 1 if reps is None else int(reps)
    records = [
        {
            "requested_k": k,
            "replicate": rep,
            "status": "missing_partition",
            "num_mutations": len(model),
            "selected_for_k": False,
            "selected": False,
            "raw_status": np.nan,
        }
        for k in budgets
        for rep in range(1, replicate_count + 1)
    ]
    by_attempt = {(int(row["requested_k"]), row["replicate"]): row for row in records}
    reporter = NumericalProgress(progress) if progress is not None else None
    cache = _ChainRefitCache(model, chain_order, progress=reporter)
    best_by_k = {}
    search_records = CandidateStore() if candidate_store is None else candidate_store
    for rep in range(1, replicate_count + 1):
        seeds, raw_diagnostics = [], {}
        suffix = "_rep%d" % rep if reps is not None else ""
        sample_indices = (
            _integer_vector(
                np.loadtxt(os.path.join(preliminary_result, "sample_indices_rep%d.txt" % rep), ndmin=1),
                "Subsample indices",
            )
            if reps is not None
            else None
        )
        for requested_k in budgets:
            record = by_attempt[(int(requested_k), rep)]
            labels_path = os.path.join(preliminary_result, "K%d_label%s.txt" % (requested_k, suffix))
            if sample_indices is not None and requested_k > len(sample_indices):
                record["status"] = "k_exceeds_subsample_size"
                continue
            if not os.path.isfile(labels_path):
                continue
            ordered_labels = np.atleast_1d(np.loadtxt(labels_path))
            if sample_indices is not None:
                ordered_labels = expand_subsample_partition(
                    ordered_labels, sample_indices, chain_order, requested_k
                )
            else:
                ordered_labels = validate_chain_labels(ordered_labels, len(model), requested_k)
            labels = np.empty(len(model), dtype=int)
            labels[chain_order] = ordered_labels
            seeds.append({"requested_k": int(requested_k), "replicate": rep, "labels": labels})
            raw_diagnostics[int(requested_k)] = _raw_diagnostics(
                os.path.join(preliminary_result, "K%d_fit%s.tsv" % (requested_k, suffix))
            )
        if not seeds:
            continue
        # Missing/invalid native attempts remain failures, even if another
        # capacity could supply a feasible coarsening for them.
        available = [seed["requested_k"] for seed in seeds]
        search = search_chain_coarsenings(
            model,
            seeds,
            chain_order,
            available,
            cache,
            store=search_records,
            replicate=rep,
            progress=progress,
        )
        if progress is not None:
            progress(dict(stage="repair_and_polish", replicate=rep, **search["stats"]))
        search = refine_chain_search(model, search, seeds, chain_order, cache)
        if progress is not None:
            progress(
                dict(
                    stage="selection_complete",
                    replicate=rep,
                    scratch_bytes=search_records.scratch_bytes,
                    **search["stats"],
                    **cache.telemetry,
                )
            )
        for candidate in search["winners"].values():
            search_records.update(rep, candidate["candidate_id"], {"selected_for_k": False})
        native_records = {
            row["requested_k"]: row
            for row in search_records.iter_rows(replicate=rep, candidate_kind="native")
        }
        for requested_k, candidate in search["winners"].items():
            record = by_attempt[(requested_k, rep)]
            result = candidate["result"]
            record["center_refit"] = "conditional"
            record.update(
                {
                    key: result[key]
                    for key in (
                        "num_clusters",
                        "num_parameters",
                        "log_likelihood",
                        "conditional_log_likelihood",
                        "weight_optimality_gap",
                        "weight_active_score_gap",
                        "bic",
                        "bic_definition",
                    )
                }
            )
            record.update(
                {
                    key: candidate[key]
                    for key in (
                        "candidate_kind",
                        *_PARENT_FIELDS,
                        "proposal_partition_sha256",
                        "partition_sha256",
                        "candidate_search_version",
                        "active_mixture_components",
                        "candidate_id",
                        "publication_eligible",
                        "minimum_mixture_weight",
                        "distinct_centers",
                        "joint_mixture_center_mle",
                    )
                }
            )
            record["native_bic"] = native_records[requested_k]["bic"]
            record["native_num_clusters"] = native_records[requested_k]["num_clusters"]
            record["candidate_count"] = search_records.count(replicate=rep, requested_k=requested_k)
            parent_raw = raw_diagnostics[candidate["parent_requested_k"]]
            record.update({"parent_" + key: value for key, value in parent_raw.items()})
            # A derived partition has a raw parent, not its own raw certificate.
            if candidate["candidate_kind"] == "native":
                record.update(parent_raw)
            record["status"] = "numerical_multimode_refit"
            current = (result["bic"], result["num_clusters"], rep, result, record)
            if requested_k not in best_by_k or current[:3] < best_by_k[requested_k][:3]:
                best_by_k[requested_k] = current
    winners = []
    for requested_k, best in sorted(best_by_k.items()):
        best[4]["selected_for_k"] = True
        winners.append((best[0], best[1], requested_k, best[2], best[3], best[4]))
    if not winners:
        raise RuntimeError("No requested K produced a partition for BIC selection")
    winner = min(winners, key=lambda item: item[:4])
    winner[5]["selected"] = True
    for requested_k, best in best_by_k.items():
        row = best[4]
        search_records.update(
            row["replicate"],
            row["candidate_id"],
            {"selected_for_k": True, "selected": requested_k == winner[2]},
        )
    _write_result(model, coordinates, winner[4], final_result)
    fits = []
    for record in records:
        candidate = search_records.get(record["replicate"], record["candidate_id"])
        fits.append(
            {
                **{
                    name: record[name].item() if isinstance(record[name], np.generic) else record[name]
                    for name in FIT_FIELDS
                },
                "proposal_cuts": [int(cut) for cut in candidate["chain_cuts"].split(",") if cut],
                "partition_parameters": json.loads(candidate["partition_parameters"]),
            }
        )
    return {
        "selected_k": int(winner[2]),
        "fits": fits,
        "selection": pd.DataFrame(records),
        "candidates": search_records,
        "telemetry": {**cache.telemetry, **(reporter.snapshot() if reporter is not None else {})},
    }
