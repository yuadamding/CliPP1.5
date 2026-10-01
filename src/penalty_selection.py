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
import hashlib
import itertools
import math
import os
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import gammaln, logsumexp, xlog1py, xlogy

from chain_refinement import polish_chain_partition


MODEL_VERSION = "uniform_1_to_major_v1"
BIC_DEFINITION = "observed_cluster_multiplicity_2K_minus_1_v1"
CANDIDATE_SEARCH_VERSION = "native_chain_boundary_polish_supported_weights_v2"


class MultiplicityModel:
    """Single-sample binomial mixture with fixed, uniform multiplicity prior."""

    def __init__(self, r, n, major, total, purity):
        arrays = [np.atleast_1d(np.asarray(x, dtype=float)) for x in (r, n, major, total)]
        if (any(x.ndim != 1 for x in arrays) or not arrays[0].size or
                any(x.shape != arrays[0].shape for x in arrays) or
                any(not np.all(np.isfinite(x)) or np.any(x != np.rint(x)) for x in arrays)):
            raise ValueError("Counts and copy numbers must be matching finite integer vectors")
        self.r, self.n, self.major, self.total = arrays
        self.purity = float(purity)
        if (not math.isfinite(self.purity) or not 0 < self.purity <= 1 or
                np.any(self.r < 0) or np.any(self.n <= 0) or np.any(self.r > self.n) or
                np.any(self.major < 1) or np.any(self.total < self.major)):
            raise ValueError("Invalid counts, major/total copy numbers, or purity")
        self.major = self.major.astype(int)
        self.m = np.arange(1, int(self.major.max()) + 1)
        self.valid = self.m[None, :] <= self.major[:, None]
        self.scale = self.m[None, :] / (2 * (1 - self.purity) + self.purity * self.total[:, None])
        self.log_constant = (gammaln(self.n + 1) - gammaln(self.r + 1)
                             - gammaln(self.n - self.r + 1) - np.log(self.major))

    def __len__(self):
        return len(self.r)

    def subset(self, indices):
        return MultiplicityModel(self.r[indices], self.n[indices], self.major[indices],
                                 self.total[indices], self.purity)

    def _log_kernel(self, cp):
        cp = np.broadcast_to(np.asarray(cp, dtype=float), (len(self),))
        if np.any(~np.isfinite(cp)) or np.any(cp < 0) or np.any(cp > self.purity):
            raise ValueError("Cellular prevalence must be finite and in [0, purity]")
        # Padded states are masked; clip their p before evaluating logarithms.
        p = np.minimum(1.0, cp[:, None] * self.scale)
        kernel = (xlogy(self.r[:, None], p) + xlog1py((self.n - self.r)[:, None], -p)
                  + self.log_constant[:, None])
        return np.where(self.valid, kernel, -np.inf)

    def log_likelihood(self, cp):
        """Per-mutation marginal log likelihood, including binomial constants."""
        return logsumexp(self._log_kernel(cp), axis=1)

    def posterior(self, cp):
        """Conditional probabilities for m=1,...,max(major), with padded zeros."""
        kernel = self._log_kernel(cp)
        normalizer = logsumexp(kernel, axis=1)
        if not np.all(np.isfinite(normalizer)):
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.exp(kernel - normalizer[:, None])

    def grid_log_likelihood(self, grid):
        """Evaluate center proposals in bounded-memory batches."""
        result = np.empty(len(grid))
        batch = max(1, min(32, 1_000_000 // max(1, self.valid.size)))
        for start in range(0, len(grid), batch):
            cp = grid[start:start + batch]
            p = np.minimum(1.0, cp[:, None, None] * self.scale[None, :, :])
            kernel = (xlogy(self.r[None, :, None], p)
                      + xlog1py((self.n - self.r)[None, :, None], -p)
                      + self.log_constant[None, :, None])
            kernel = np.where(self.valid[None, :, :], kernel, -np.inf)
            result[start:start + batch] = logsumexp(kernel, axis=2).sum(axis=1)
        return result


def refit_center(model):
    """Refine bracketed modes and edge intervals, retaining physical endpoints."""
    component_modes = (model.r / model.n)[:, None] / model.scale
    grid = np.unique(np.concatenate((np.linspace(0, model.purity, 513),
                                    np.clip(component_modes[model.valid], 0, model.purity))))
    values = model.grid_log_likelihood(grid)
    best_index = int(np.argmax(values))
    best = (float(values[best_index]), float(grid[best_index]))
    maxima = np.flatnonzero((values[1:-1] >= values[:-2]) &
                            (values[1:-1] >= values[2:]) &
                            ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))) + 1
    # A maximum just inside a finite boundary can beat that endpoint while
    # every interior grid point is worse. It then has no grid-local maximum
    # to bracket. Search both edge intervals even when neither is bracketed;
    # the explicit grid candidates above still preserve exact endpoints.
    intervals = {(float(grid[0]), float(grid[1])),
                 (float(grid[-2]), float(grid[-1]))}
    intervals.update((float(grid[index - 1]), float(grid[index + 1]))
                     for index in maxima)
    for lower, upper in sorted(intervals):
        fit = minimize_scalar(lambda cp: -float(model.log_likelihood(cp).sum()),
                              bounds=(lower, upper), method="bounded",
                              options={"xatol": 1e-12, "maxiter": 500})
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError("Marginal-likelihood center refit failed: %s" % fit.message)
        candidate = (-float(fit.fun), float(fit.x))
        if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if not np.isfinite(best[0]):
        raise RuntimeError("No finite marginal likelihood in cluster refit")
    return best[1], best[0]


def fit_cluster_weights(model, centers, initial_weights=None, *, log_kernel=None):
    """Maximize the observed likelihood over the simplex at fixed CP centers.

    This is a concave weight problem. The reported Frank-Wolfe gap bounds the
    remaining log-likelihood improvement at these centers, even with duplicate
    centers or boundary weights. Zero weights are returned for explicit repair
    by the caller; this optimizer does not change partition memberships.
    Center fitting itself remains conditional on the frozen chain partition;
    this function does not claim a joint mixture-center maximum.
    """
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    if (centers.ndim != 1 or not centers.size or np.any(~np.isfinite(centers)) or
            np.any(centers < 0) or np.any(centers > model.purity)):
        raise ValueError("Expected finite cluster centers in [0, purity]")
    if log_kernel is None:
        log_kernel = np.column_stack([model.log_likelihood(cp) for cp in centers])
    else:
        log_kernel = np.asarray(log_kernel, dtype=float)
        if (log_kernel.shape != (len(model), len(centers)) or
                np.any(np.isnan(log_kernel)) or np.any(np.isposinf(log_kernel))):
            raise ValueError("Invalid cached center likelihoods")
    offset = np.max(log_kernel, axis=1)
    if np.any(~np.isfinite(offset)):
        raise ValueError("Every mutation needs finite likelihood in some cluster")
    k = len(centers)
    if k == 1:
        return {"cluster_weights": np.ones(1), "log_likelihood": float(offset.sum()),
                "weight_optimality_gap": 0.0, "weight_active_score_gap": 0.0}
    kernel = np.exp(log_kernel - offset[:, None])
    weights = (np.ones(k) / k if initial_weights is None
               else np.asarray(initial_weights, dtype=float))
    if (weights.shape != (k,) or np.any(~np.isfinite(weights)) or
            np.any(weights <= 0)):
        raise ValueError("Initial cluster weights must be positive and finite")
    weights = weights / weights.sum()

    def objective(w):
        mass = kernel @ w
        return -float(np.log(mass).mean()) if np.all(mass > 0) else np.inf

    def gradient(w):
        mass = np.maximum(kernel @ w, np.finfo(float).tiny)
        return -(kernel / mass[:, None] / len(model)).sum(axis=0)

    tolerance = 1e-8
    fit = minimize(objective, weights, jac=gradient, method="SLSQP",
                   bounds=[(0.0, 1.0)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                "jac": lambda w: np.ones(k)},
                   options={"ftol": 1e-13, "maxiter": 2000})
    if np.all(np.isfinite(fit.x)):
        candidate = np.maximum(fit.x, 0.0)
        if candidate.sum() > 0:
            candidate /= candidate.sum()
            if objective(candidate) <= objective(weights):
                weights = candidate
    # SLSQP's success flag is not an optimality check. EM can stall indefinitely
    # between nearly identical columns; instead transfer mass between the most
    # and least favorable coordinates. The 1-D log likelihood is concave, so
    # its derivative either brackets the maximum or chooses the simplex edge.
    # Previously zero weights may enter; no likelihood or support is discarded.
    for _ in range(10000):
        score = -gradient(weights)
        positive = np.flatnonzero(weights > 0)
        active_gap = float(score.max() - score[positive].min())
        # A tiny but positive unsupported weight barely changes the global
        # gap. Check active-coordinate stationarity as well, so exact line
        # searches reach true boundary zeros without an arbitrary weight floor.
        if max(0.0, float(score.max() - 1.0)) <= tolerance and active_gap <= tolerance:
            break
        grow = int(np.argmax(score))
        shrink = int(positive[np.argmin(score[positive])])
        if grow == shrink:
            break
        mass = kernel @ weights
        direction = weights[shrink] * (kernel[:, grow] - kernel[:, shrink])

        def derivative(fraction):
            normalizer = np.maximum(mass + fraction * direction, np.finfo(float).tiny)
            return float(np.sum(direction / normalizer / len(model)))

        fraction = (1.0 if derivative(1.0) >= 0 else
                    brentq(derivative, 0.0, 1.0, xtol=1e-14))
        transfer = fraction * weights[shrink]
        weights[grow] += transfer
        weights[shrink] = 0.0 if fraction == 1.0 else weights[shrink] - transfer
        weights /= weights.sum()
    score = -gradient(weights)
    optimality_gap = len(model) * max(0.0, float(score.max() - 1.0))
    active_gap = float(score.max() - score[weights > 0].min())
    if (not np.isfinite(objective(weights)) or optimality_gap > len(model) * tolerance or
            active_gap > tolerance):
        raise RuntimeError("Cluster-weight likelihood optimization did not converge")
    likelihood = float(np.sum(offset + np.log(kernel @ weights)))
    return {"cluster_weights": weights, "log_likelihood": likelihood,
            "weight_optimality_gap": optimality_gap, "weight_active_score_gap": active_gap}


class _ChainRefitCache(dict):
    """One model/order's interval refits and a bounded LRU of likelihood columns."""

    def __init__(self, model, chain_order, max_column_bytes=32 * 1024 * 1024):
        super().__init__()
        self.model = model
        self.order = np.asarray(chain_order, dtype=np.int64).copy()
        self.ranks = np.empty(len(model), dtype=np.int64)
        self.ranks[self.order] = np.arange(len(model))
        self.columns = OrderedDict()
        self.column_bytes = 0
        self.max_column_bytes = max_column_bytes

    def block_key(self, rows):
        ranks = self.ranks[rows]
        left, right = int(ranks.min()), int(ranks.max()) + 1
        if right - left != len(rows):
            raise ValueError("Cached refit requires a contiguous chain interval")
        return left, right

    def log_kernel(self, centers):
        columns = []
        for cp in centers:
            key = float(cp)
            column = self.columns.pop(key, None)
            if column is None:
                column = self.model.log_likelihood(cp)
                if column.nbytes <= self.max_column_bytes:
                    while self.column_bytes + column.nbytes > self.max_column_bytes:
                        _, removed = self.columns.popitem(last=False)
                        self.column_bytes -= removed.nbytes
                    self.columns[key] = column
                    self.column_bytes += column.nbytes
            else:
                self.columns[key] = column
            columns.append(column)
        return np.column_stack(columns)


def refit_partition(model, labels, cache=None, chain_order=None):
    """Refit blocks; merge exact adjacent equal centers on a supplied chain.

    Without an explicit chain permutation no adjacency is inferred from labels.
    Equal centers separated by another center always remain separate blocks.
    """
    labels = np.atleast_1d(np.asarray(labels))
    if (labels.shape != (len(model),) or not np.all(np.isfinite(labels)) or
            np.any(labels != np.rint(labels))):
        raise ValueError("Partition labels must be one integer per mutation")
    if isinstance(cache, _ChainRefitCache) and (
            cache.model is not model or chain_order is None or
            not np.array_equal(cache.order, chain_order)):
        raise ValueError("Chain refit cache belongs to a different model or order")
    clusters, inverse = np.unique(labels, return_inverse=True)
    centers, log_likelihoods = [], []
    for index in range(len(clusters)):
        rows = np.flatnonzero(inverse == index)
        key = cache.block_key(rows) if isinstance(cache, _ChainRefitCache) else rows.tobytes()
        fitted = cache.get(key) if cache is not None else None
        if fitted is None:
            fitted = refit_center(model.subset(rows))
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
    log_kernel = cache.log_kernel(centers) if isinstance(cache, _ChainRefitCache) else None
    mixture = fit_cluster_weights(model, centers, np.bincount(labels, minlength=k),
                                  log_kernel=log_kernel)
    parameters = 2 * k - 1
    bic = -2 * mixture["log_likelihood"] + parameters * math.log(len(model))
    return {"labels": labels, "centers": centers, **mixture,
            "conditional_log_likelihood": conditional_log_likelihood,
            "num_clusters": k, "num_parameters": parameters,
            "num_mutations": len(model), "bic": bic, "bic_definition": BIC_DEFINITION}


def _load_vector(directory, filename):
    return np.atleast_1d(np.loadtxt(os.path.join(directory, filename)))


def _write_table(data, path):
    temporary = path + ".tmp"
    data.to_csv(temporary, sep="\t", index=False, float_format="%.17g")
    os.replace(temporary, path)


def _write_result(model, coordinates, result, directory, requested_k):
    if np.any(result["cluster_weights"] <= 0):
        raise ValueError("Published occupied clusters require positive fitted mixture weights")
    assignments = coordinates.copy()
    assignments["cluster_index"] = result["labels"]
    structures = pd.DataFrame({"cluster_index": np.arange(result["num_clusters"]),
                               "num_SNV": np.bincount(result["labels"]),
                               "cellular_prevalence": result["centers"],
                               "mixture_weight": result["cluster_weights"]})
    posterior = model.posterior(result["centers"][result["labels"]])
    calls = np.argmax(posterior, axis=1)
    multiplicity = assignments.copy()
    multiplicity["major_cn"] = model.major
    multiplicity["multiplicity"] = calls + 1
    multiplicity["multiplicity_probability"] = posterior[np.arange(len(model)), calls]
    multiplicity["expected_multiplicity"] = posterior @ model.m
    files = []
    for stem, frame in (("mutation_assignments", assignments), ("subclonal_structure", structures),
                        ("posterior_multiplicity", multiplicity)):
        filename = "%s_K%d.txt" % (stem, requested_k)
        _write_table(frame, os.path.join(directory, filename))
        files.append(filename)
    return files


def _integer_vector(values, name, length=None):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if (values.ndim != 1 or not values.size or
            (length is not None and values.size != length) or
            np.any(~np.isfinite(values)) or np.any(values != np.rint(values))):
        raise ValueError("%s must be a nonempty integer vector of the expected length" % name)
    return values.astype(np.int64)


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
        return {}
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 1:
        raise ValueError("Native fit diagnostics must contain exactly one row: %s" % path)
    return {"raw_" + str(key): value for key, value in frame.iloc[0].items()}


def search_chain_coarsenings(model, seeds, chain_order, budgets, cache=None):
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
    if (len(np.unique(budgets)) != len(budgets) or np.any(budgets < 1) or
            np.any(budgets > min(10, n))):
        raise ValueError("Expected unique K in 1..min(10,N)")
    allowed = set(int(k) for k in budgets)
    cache = _ChainRefitCache(model, chain_order) if cache is None else cache
    chain_bytes = np.asarray(chain_order, dtype="<i8").tobytes()

    def fingerprint(cuts):
        return hashlib.sha256(chain_bytes + np.asarray(cuts, dtype="<i8").tobytes()).hexdigest()

    proposals = {}

    def add(cuts, target, source_k, rep, parent_cuts, kind):
        provenance = {"requested_k": target, "parent_requested_k": source_k,
                      "parent_replicate": rep, "candidate_kind": kind,
                      "parent_partition_sha256": fingerprint(parent_cuts)}
        by_budget = proposals.setdefault(cuts, {})
        rank = lambda p: (p["candidate_kind"] != "native", p["parent_requested_k"],
                          p["parent_replicate"], p["candidate_kind"])
        if target not in by_budget or rank(provenance) < rank(by_budget[target]):
            by_budget[target] = provenance

    for seed in seeds:
        source_k = int(seed["requested_k"])
        rep = int(seed.get("replicate", 1))
        if (source_k != seed["requested_k"] or not 1 <= source_k <= min(10, n) or
                rep != seed.get("replicate", 1) or rep < 1):
            raise ValueError("Invalid native candidate K or replicate")
        labels = _integer_vector(seed["labels"], "Native chain labels", n)
        ordered = validate_chain_labels(labels[chain_order], n, source_k)
        cuts = tuple(int(i) for i in np.flatnonzero(ordered[1:] != ordered[:-1]) + 1)
        if source_k in allowed:
            add(cuts, source_k, source_k, rep, cuts, "native")
        for target in sorted(allowed):
            if target > len(cuts) + 1:
                continue
            for subset in itertools.combinations(cuts, target - 1):
                kind = "adjacent_coarsening" if subset != cuts else "capacity_reuse"
                add(subset, target, source_k, rep, cuts, kind)

    winners, winner_keys, records = {}, {}, []
    # Group by source where possible to reuse its at-most-55 interval centers.
    def proposal_order(item):
        cuts, sources = item
        parent = min((p["parent_requested_k"], p["parent_replicate"]) for p in sources.values())
        return parent, len(cuts), cuts

    for cuts, by_budget in sorted(proposals.items(), key=proposal_order):
        labels = np.empty(n, dtype=int)
        labels[chain_order] = np.searchsorted(cuts, np.arange(n), side="right")
        proposal_digest = fingerprint(cuts)
        try:
            result = refit_partition(model, labels, cache, chain_order=chain_order)
        except RuntimeError as error:
            if any(p["candidate_kind"] == "native" for p in by_budget.values()):
                raise
            for provenance in by_budget.values():
                records.append({**provenance, "status": "refit_failed", "error": str(error),
                                "proposal_partition_sha256": proposal_digest,
                                "chain_cuts": ",".join(map(str, cuts)),
                                "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                                "selected_for_k": False})
            continue
        ordered_result = result["labels"][chain_order]
        final_cuts = tuple(int(i) for i in np.flatnonzero(ordered_result[1:] != ordered_result[:-1]) + 1)
        digest = fingerprint(final_cuts)
        for budget, provenance in sorted(by_budget.items()):
            validate_chain_labels(result["labels"][chain_order], n, budget)
            record = {**provenance, "status": "scored", "partition_sha256": digest,
                      "proposal_partition_sha256": proposal_digest,
                      "chain_cuts": ",".join(map(str, cuts)),
                      "num_input_blocks": len(cuts) + 1,
                      "num_clusters": result["num_clusters"], "bic": result["bic"],
                      "log_likelihood": result["log_likelihood"],
                      "conditional_log_likelihood": result["conditional_log_likelihood"],
                      "weight_optimality_gap": result["weight_optimality_gap"],
                      "weight_active_score_gap": result["weight_active_score_gap"],
                      "active_mixture_components": int(np.count_nonzero(result["cluster_weights"] > 0)),
                      "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                      "selected_for_k": False}
            records.append(record)
            rank = (result["bic"], result["num_clusters"],
                    provenance["candidate_kind"] != "native", provenance["parent_requested_k"],
                    provenance["parent_replicate"], cuts)
            if budget not in winners or rank < winner_keys[budget]:
                winners[budget] = {**record, "cuts": cuts, "result": result}
                winner_keys[budget] = rank
    for record in records:
        if record["status"] == "scored":
            winner = winners[record["requested_k"]]
            record["selected_for_k"] = record["proposal_partition_sha256"] == winner["proposal_partition_sha256"]
    return {"winners": winners, "candidates": records,
            "stats": {"unique_partitions": len(proposals), "scored_budget_candidates": len(records),
                      "cached_refit_blocks": len(cache),
                      "likelihood_cache_bytes": getattr(cache, "column_bytes", 0)}}


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
    chain_bytes = np.asarray(chain_order, dtype="<i8").tobytes()
    records = search["candidates"]
    winners = {}
    seen = set()

    def cuts_of(result):
        ordered = result["labels"][chain_order]
        return tuple(int(i) for i in np.flatnonzero(np.diff(ordered)) + 1)

    def fingerprint(cuts):
        return hashlib.sha256(chain_bytes + np.asarray(cuts, dtype="<i8").tobytes()).hexdigest()

    def admissible(result):
        return bool(np.all(result["cluster_weights"] > 0))

    for index, record in enumerate(records):
        record["candidate_id"] = index
        record["selected_for_k"] = False
        record["publication_eligible"] = (record["status"] == "scored" and
            record["active_mixture_components"] == record["num_clusters"])

    def consider(result, parent, kind, diagnostics=None, existing=None):
        budget = int(parent["requested_k"])
        cuts = cuts_of(result)
        validate_chain_labels(result["labels"][chain_order], n, budget)
        key = (budget, cuts)
        if key in seen:
            return
        seen.add(key)
        if existing is not None:
            record = existing
        else:
            record = {key: parent[key] for key in (
                "requested_k", "parent_requested_k", "parent_replicate", "parent_partition_sha256")}
            record.update({"candidate_id": len(records), "candidate_kind": kind,
                "status": "scored", "proposal_partition_sha256": fingerprint(cuts),
                "partition_sha256": fingerprint(cuts), "chain_cuts": ",".join(map(str, cuts)),
                "num_input_blocks": len(cuts) + 1, "selected_for_k": False,
                "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                "refinement_parent_partition_sha256": parent["partition_sha256"]})
            record.update({key: result[key] for key in (
                "num_clusters", "bic", "log_likelihood", "conditional_log_likelihood",
                "weight_optimality_gap", "weight_active_score_gap")})
            records.append(record)
        record.update({"publication_eligible": admissible(result),
            "active_mixture_components": int(np.count_nonzero(result["cluster_weights"] > 0)),
            "minimum_mixture_weight": float(np.min(result["cluster_weights"])),
            "distinct_centers": int(len(np.unique(result["centers"]))),
            "joint_mixture_center_mle": False})
        if diagnostics:
            record.update({"boundary_" + key: value for key, value in diagnostics.items()})
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
            reduced_cuts = cuts[:edge] + cuts[edge + 1:]
            if (budget, reduced_cuts) in seen:
                continue
            labels = np.empty(n, dtype=int)
            labels[chain_order] = np.searchsorted(reduced_cuts, np.arange(n), side="right")
            try:
                repaired = refit_partition(model, labels, cache, chain_order)
            except RuntimeError as error:
                seen.add((budget, reduced_cuts))
                records.append({"candidate_id": len(records), "requested_k": budget,
                    "candidate_kind": "unsupported_component_coarsening", "status": "refit_failed",
                    "publication_eligible": False, "selected_for_k": False, "error": str(error),
                    "parent_requested_k": parent["parent_requested_k"],
                    "parent_replicate": parent["parent_replicate"],
                    "parent_partition_sha256": parent["parent_partition_sha256"],
                    "proposal_partition_sha256": fingerprint(reduced_cuts),
                    "chain_cuts": ",".join(map(str, reduced_cuts)),
                    "candidate_search_version": CANDIDATE_SEARCH_VERSION})
                continue
            consider(repaired, candidate, "unsupported_component_coarsening")

    # Preserve every admissible old winner before adding improvements, so a
    # conditional-likelihood improvement cannot silently worsen the BIC winner.
    starts = []
    for budget, candidate in sorted(search["winners"].items()):
        original = next(row for row in records if row["requested_k"] == budget and
                        row["proposal_partition_sha256"] == candidate["proposal_partition_sha256"])
        consider(candidate["result"], candidate, candidate["candidate_kind"], existing=original)
        starts.append(candidate)
        eligible = [row for row in records if row["requested_k"] == budget and
                    row["publication_eligible"]]
        if eligible:
            baseline = min(eligible, key=lambda row: (row["bic"], row["num_clusters"], row["candidate_id"]))
            cuts = tuple(int(x) for x in baseline["chain_cuts"].split(",") if x)
            labels = np.empty(n, dtype=int)
            labels[chain_order] = np.searchsorted(cuts, np.arange(n), side="right")
            fitted = refit_partition(model, labels, cache, chain_order)
            consider(fitted, baseline, baseline["candidate_kind"], existing=baseline)
            starts.append({**baseline, "result": fitted})
    for seed in seeds:
        budget = seed["requested_k"]
        record = next(row for row in records if row["requested_k"] == budget and
                      row["candidate_kind"] == "native")
        fitted = refit_partition(model, seed["labels"], cache, chain_order)
        consider(fitted, record, "native", existing=record)
        starts.append({**record, "result": fitted})
    polished = set()
    for parent in starts:
        key = (parent["requested_k"], cuts_of(parent["result"]))
        if key in polished:
            continue
        polished.add(key)
        refinement = polish_chain_partition(model, parent["result"], chain_order,
            lambda labels: refit_partition(model, labels, cache, chain_order))
        consider(refinement["result"], parent, "chain_boundary_polish",
                 refinement["diagnostics"])
        # Preserve convergence evidence even when polish leaves the same cuts.
        parent_id = next(row["candidate_id"] for row in records
            if row["requested_k"] == parent["requested_k"] and
            row["proposal_partition_sha256"] == parent["proposal_partition_sha256"])
        records[parent_id].update({"polish_" + key: value
                                   for key, value in refinement["diagnostics"].items()})
    for budget, parent in search["winners"].items():
        if budget not in winners:
            fallback = refit_partition(model, np.zeros(n, dtype=int), cache, chain_order)
            # A failed optional route must not hide a finite single-block
            # candidate. This is an explicit refit within the at-most-K set.
            seen.discard((budget, ()))
            consider(fallback, parent, "supported_single_block_fallback")
    for record in records:
        winner = winners.get(record["requested_k"])
        record["selected_for_k"] = winner is not None and record["candidate_id"] == winner["candidate_id"]
    return {"winners": winners, "candidates": records,
            "original_bank_stats": search["stats"],
            "stats": {"unique_partitions": len({row["proposal_partition_sha256"] for row in records}),
                      "scored_budget_candidates": sum(row["status"] == "scored" for row in records),
                      "cached_refit_blocks": len(cache),
                      "likelihood_cache_bytes": getattr(cache, "column_bytes", 0)}}


def run_model_selection(preprocess_dir, preliminary_result, final_result, cluster_list, reps=None,
                        center_refit="conditional"):
    """Improve fixed-K chain proposals and publish the minimum BIC-form score.

    K is a block budget: q occupied blocks contribute 2q-1 parameters, including
    mixture weights. Unsupported components require actual adjacent refits.
    Native proposals and adjacent coarsenings are not global-optimality proofs.
    Subsample boundaries expand in chain-rank space before full-data refits;
    all candidate BICs use exactly the same N mutations and likelihood.
    """
    if center_refit != "conditional":
        raise ValueError("Only conditional center refitting is supported; got %r" % center_refit)
    with open(os.path.join(preprocess_dir, "multiplicity_model.txt")) as handle:
        if handle.read().strip() != MODEL_VERSION:
            raise ValueError("Preprocessing does not declare the uniform multiplicity model")
    model = MultiplicityModel(*[_load_vector(preprocess_dir, f) for f in
                                ("r.txt", "n.txt", "major.txt", "total.txt")],
                              purity=float(_load_vector(preprocess_dir, "purity_ploidy.txt")[0]))
    index = pd.read_csv(os.path.join(preprocess_dir, "multiplicity.txt"), sep=r"\s+", header=None,
                        dtype=str)
    if index.shape != (len(model), 4):
        raise ValueError("Multiplicity coordinates must have four columns and one row per mutation")
    if (not np.array_equal(index[2].astype(float), model.total) or
            not np.array_equal(index[3].astype(float), model.major)):
        raise ValueError("Multiplicity coordinate CN does not match the model inputs")
    coordinates = index[[0, 1]].copy()
    coordinates.columns = ["chromosome_index", "position"]
    if coordinates.duplicated().any():
        raise ValueError("Mutation coordinates must be unique")
    chain_order = _integer_vector(_load_vector(preliminary_result, "chain_order.txt"),
                                   "Frozen chain order", len(model))
    if not np.array_equal(np.sort(chain_order), np.arange(len(model))):
        raise ValueError("Frozen chain order must be a permutation of original row indices")
    budgets = _integer_vector(cluster_list, "Cluster budgets")
    if (len(np.unique(budgets)) != len(budgets) or np.any(budgets < 1) or
            np.any(budgets > min(10, len(model))) or
            (reps is not None and (isinstance(reps, bool) or int(reps) != reps or reps < 1))):
        raise ValueError("Expected unique K in 1..min(10,N) and a positive replicate count")
    os.makedirs(final_result, exist_ok=True)
    replicate_count = 1 if reps is None else int(reps)
    records = [{"requested_k": k, "replicate": rep, "status": "missing_partition",
                "num_mutations": len(model), "selected_for_k": False, "selected": False,
                "raw_status": np.nan}
               for k in budgets for rep in range(1, replicate_count + 1)]
    by_attempt = {(int(row["requested_k"]), row["replicate"]): row for row in records}
    cache = _ChainRefitCache(model, chain_order)
    best_by_k, search_records = {}, []
    for rep in range(1, replicate_count + 1):
        seeds, raw_diagnostics = [], {}
        for requested_k in budgets:
            record = by_attempt[(int(requested_k), rep)]
            suffix = "_rep%d" % rep if reps is not None else ""
            labels_path = os.path.join(preliminary_result, "K%d_label%s.txt" % (requested_k, suffix))
            sample_indices = None
            if reps is not None:
                sample_indices = _integer_vector(_load_vector(preliminary_result,
                    "sample_indices_rep%d.txt" % rep), "Subsample indices")
                if requested_k > len(sample_indices):
                    record["status"] = "k_exceeds_subsample_size"
                    continue
            if not os.path.isfile(labels_path):
                continue
            ordered_labels = np.atleast_1d(np.loadtxt(labels_path))
            if sample_indices is not None:
                ordered_labels = expand_subsample_partition(ordered_labels, sample_indices,
                                                           chain_order, requested_k)
            else:
                ordered_labels = validate_chain_labels(ordered_labels, len(model), requested_k)
            labels = np.empty(len(model), dtype=int)
            labels[chain_order] = ordered_labels
            seeds.append({"requested_k": int(requested_k), "replicate": rep, "labels": labels})
            raw_diagnostics[int(requested_k)] = _raw_diagnostics(os.path.join(preliminary_result,
                "K%d_fit%s.tsv" % (requested_k, suffix)))
        if not seeds:
            continue
        # Missing/invalid native attempts remain failures, even if another
        # capacity could supply a feasible coarsening for them.
        available = [seed["requested_k"] for seed in seeds]
        search = search_chain_coarsenings(model, seeds, chain_order, available, cache)
        search = refine_chain_search(model, search, seeds, chain_order, cache)
        for row in search["candidates"]:
            copied = {**row, "replicate": rep,
                      "selected_for_replicate_k": row["selected_for_k"],
                      "selected_for_k": False, "selected": False}
            search_records.append(copied)
        native_records = {row["requested_k"]: row for row in search["candidates"]
                          if row["candidate_kind"] == "native"}
        for requested_k, candidate in search["winners"].items():
            record = by_attempt[(requested_k, rep)]
            result = candidate["result"]
            record["center_refit"] = center_refit
            record.update({key: result[key] for key in (
                "num_clusters", "num_parameters", "log_likelihood",
                "conditional_log_likelihood", "weight_optimality_gap", "weight_active_score_gap", "bic",
                "bic_definition")})
            record.update({key: candidate[key] for key in (
                "candidate_kind", "parent_requested_k", "parent_replicate",
                "parent_partition_sha256", "proposal_partition_sha256", "partition_sha256", "candidate_search_version",
                "active_mixture_components", "candidate_id", "publication_eligible",
                "minimum_mixture_weight", "distinct_centers", "joint_mixture_center_mle")})
            record["native_bic"] = native_records[requested_k]["bic"]
            record["native_num_clusters"] = native_records[requested_k]["num_clusters"]
            record["candidate_count"] = sum(row["requested_k"] == requested_k
                                            for row in search["candidates"])
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
        files = _write_result(model, coordinates, best[3], final_result, requested_k)
        winners.append((best[0], best[1], requested_k, best[2], files, best[4]))
    if not winners:
        _write_table(pd.DataFrame(search_records), os.path.join(final_result, "chain_candidates.tsv"))
        _write_table(pd.DataFrame(records), os.path.join(final_result, "bic_selection.tsv"))
        raise RuntimeError("No requested K produced a partition for BIC selection")
    winner = min(winners, key=lambda item: item[:4])
    winner[5]["selected"] = True
    for row in search_records:
        best = best_by_k[row["requested_k"]][4]
        row["selected_for_k"] = (row["status"] == "scored" and row["replicate"] == best["replicate"] and
                                  row["candidate_id"] == best["candidate_id"])
        row["selected"] = row["selected_for_k"] and row["requested_k"] == winner[2]
    _write_table(pd.DataFrame(search_records), os.path.join(final_result, "chain_candidates.tsv"))
    _write_table(pd.DataFrame(records), os.path.join(final_result, "bic_selection.tsv"))
    destination = os.path.join(final_result, "Best_K")
    os.makedirs(destination, exist_ok=True)
    for filename in os.listdir(destination):
        if filename.endswith(".txt") and filename.startswith(("mutation_assignments_K", "subclonal_structure_K", "posterior_multiplicity_K")):
            os.remove(os.path.join(destination, filename))
    for filename in winner[4]:
        shutil.copyfile(os.path.join(final_result, filename), os.path.join(destination, filename))
    return int(winner[2])
