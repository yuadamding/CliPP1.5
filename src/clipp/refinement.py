"""Conditional likelihood refinement on one frozen chain.

Only boundaries move. Component visits follow the supplied partition, including
nonmonotone CP sequences. No mixture weights or unconstrained MAP assignments
enter this conditional objective; the caller retains responsibility for BIC.
"""

import math
from contextlib import nullcontext

import numpy as np


def fixed_center_chain_partition(log_likelihood):
    """Exactly optimize nonempty, ordered blocks for fixed component centers.

    ``log_likelihood[i, j]`` is observation i's conditional marginal log
    likelihood at the jth center *in chain visit order*. Every column must be
    used by one nonempty contiguous interval. The dynamic program takes O(Nq)
    time and memory, permits -inf entries, and deterministically chooses the
    earliest preceding cut when scores tie. Centers need not increase with j.
    """
    likelihood = np.asarray(log_likelihood, dtype=float)
    if (
        likelihood.ndim != 2
        or min(likelihood.shape) == 0
        or likelihood.shape[1] > likelihood.shape[0]
        or np.any(np.isnan(likelihood))
        or np.any(np.isposinf(likelihood))
    ):
        raise ValueError("Expected an N-by-q finite-or-negative-infinity array, 1 <= q <= N")
    n, q = likelihood.shape
    previous = np.full(n + 1, -np.inf)
    previous[0] = 0.0
    parents = np.full((q, n + 1), -1, dtype=np.int64)
    for component in range(q):
        column = likelihood[:, component]
        current = np.full(n + 1, -np.inf)
        # A block cannot cross a -inf observation. Independent finite runs
        # avoid subtracting infinite prefix sums or producing NaNs.
        finite = np.isfinite(column)
        starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
        stops = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
        for start, stop in zip(starts, stops):
            prefix = np.r_[0.0, np.cumsum(column[start:stop])]
            if not np.isfinite(prefix).all():
                raise ValueError("Conditional log-likelihood accumulation overflowed")
            proposals = previous[start:stop] - prefix[:-1]
            best = np.maximum.accumulate(proposals)
            improves = proposals > np.r_[-np.inf, best[:-1]]
            cuts = np.maximum.accumulate(np.where(improves, np.arange(start, stop), -1))
            current[start + 1 : stop + 1] = prefix[1:] + best
            parents[component, start + 1 : stop + 1] = cuts
        previous = current
    if not np.isfinite(previous[n]):
        raise RuntimeError("No finite partition uses every ordered component in a nonempty block")
    labels = np.empty(n, dtype=np.int64)
    end = n
    for component in range(q - 1, -1, -1):
        start = int(parents[component, end])
        if not 0 <= start < end:
            raise RuntimeError("Invalid chain dynamic-program backtrack")
        labels[start:end] = component
        end = start
    if end != 0:
        raise RuntimeError("Chain dynamic-program backtrack did not cover every observation")
    # Sum the actual selected entries, independently of prefix-sum subtraction.
    score = math.fsum(likelihood[np.arange(n), labels])
    return {
        "labels": labels,
        "cuts": np.flatnonzero(np.diff(labels)) + 1,
        "conditional_log_likelihood": score,
        "num_clusters": q,
        "status": "exact_fixed_center_chain_partition",
    }


def _partition_state(model, result, chain_order, likelihood_provider=None):
    labels = np.asarray(result["labels"])
    centers = np.asarray(result["centers"], dtype=float)
    n = len(model)
    if (
        labels.shape != (n,)
        or not np.isfinite(labels).all()
        or np.any(labels != np.rint(labels))
        or centers.ndim != 1
        or not len(centers)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("Invalid refitted chain labels or centers")
    labels = labels.astype(np.int64)
    if not np.array_equal(np.unique(labels), np.arange(len(centers))):
        raise ValueError("Every refitted center must have an occupied label")
    ordered = labels[chain_order]
    cuts = np.flatnonzero(np.diff(ordered)) + 1
    visits = ordered[np.r_[0, cuts]]
    if len(visits) != len(centers) or len(np.unique(visits)) != len(visits):
        raise ValueError("Each center must visit exactly one contiguous chain block")
    # The provider returns original-row columns in the requested visit order.
    # Numerically sorting centers here would change the constrained DP problem.
    visited_centers = centers[visits]
    kernel = (
        np.column_stack([model.log_likelihood(cp) for cp in visited_centers])
        if likelihood_provider is None
        else np.asarray(likelihood_provider(visited_centers), dtype=float)
    )
    if kernel.shape != (n, len(visits)) or np.any(np.isnan(kernel)) or np.any(np.isposinf(kernel)):
        raise ValueError("Invalid refinement likelihood columns")
    kernel = kernel[chain_order]
    states = np.searchsorted(cuts, np.arange(n), side="right")
    score = math.fsum(kernel[np.arange(n), states])
    if not np.isfinite(score):
        raise ValueError("Supplied chain partition has nonfinite conditional likelihood")
    return labels, cuts, kernel, score


def polish_chain_partition(
    model,
    result,
    chain_order,
    refit,
    max_iterations=100,
    mean_loglik_tolerance=1e-10,
    *,
    likelihood_provider=None,
    progress=None,
):
    """Alternate exact conditional boundaries and caller-supplied block refits.

    ``refit(labels)`` receives labels in original mutation-row order and returns
    a result dictionary containing ``labels`` and ``centers``. It may merge
    adjacent blocks, but cannot add or move boundaries itself. The returned
    result is the best conditional-likelihood candidate, with scalar-only
    iteration history. A finite iteration budget or a local fixed point is
    explicitly distinguished from global constrained optimality.
    """
    if (
        not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
        or not np.isfinite(mean_loglik_tolerance)
        or mean_loglik_tolerance < 0
    ):
        raise ValueError("Require a positive iteration budget and nonnegative finite tolerance")
    order = np.asarray(chain_order)
    n = len(model)
    if (
        order.shape != (n,)
        or not np.isfinite(order).all()
        or np.any(order != np.rint(order))
        or not np.array_equal(np.sort(order), np.arange(n))
    ):
        raise ValueError("Frozen chain order must be a permutation of original rows")
    order = order.astype(np.int64)
    current = result
    _, cuts, kernel, score = _partition_state(model, current, order, likelihood_provider)
    initial_score, initial_q = score, kernel.shape[1]
    best, best_score = current, score
    history = []

    def completed(entry, iteration_status):
        entry["status"] = iteration_status
        history.append(entry)
        if progress is not None:
            progress.update(
                "polish_iteration_completed", refinement_iteration=entry["iteration"], status=iteration_status
            )

    status = "iteration_budget"
    for iteration in range(1, max_iterations + 1):
        if progress is not None:
            progress.update("polish_iteration_started", refinement_iteration=iteration)
        with progress.timer("boundary_dynamic_programming") if progress is not None else nullcontext():
            proposal = fixed_center_chain_partition(kernel)
        fixed_score = proposal["conditional_log_likelihood"]
        roundoff = 64 * np.finfo(float).eps * max(1.0, abs(score), abs(fixed_score))
        if fixed_score < score - roundoff:
            raise RuntimeError("Exact chain boundary update decreased conditional likelihood")
        entry = {
            "iteration": iteration,
            "num_clusters": kernel.shape[1],
            "previous_conditional_log_likelihood": score,
            "fixed_center_conditional_log_likelihood": fixed_score,
            "boundaries_changed": not np.array_equal(proposal["cuts"], cuts),
        }
        if not entry["boundaries_changed"]:
            status = "fixed_center_boundaries_stable"
            completed(entry, status)
            break
        labels = np.empty(n, dtype=np.int64)
        labels[order] = proposal["labels"]
        try:
            with progress.context(refinement_iteration=iteration) if progress is not None else nullcontext():
                candidate = refit(labels)
                _, new_cuts, new_kernel, new_score = _partition_state(
                    model, candidate, order, likelihood_provider
                )
        except RuntimeError as error:
            entry["error"] = str(error)
            status = "refit_failed"
            completed(entry, status)
            break
        if not set(new_cuts) <= set(proposal["cuts"]):
            raise ValueError("Refit may only preserve boundaries or merge adjacent blocks")
        roundoff = 64 * np.finfo(float).eps * max(1.0, abs(fixed_score), abs(new_score))
        entry.update(refit_conditional_log_likelihood=new_score, refit_num_clusters=new_kernel.shape[1])
        if new_score < fixed_score - roundoff or new_score < score:
            # A numerical scalar refit need not recover its supplied-center
            # witness. Keep the accepted incumbent instead of hiding a loss.
            status = "refit_conditional_decrease"
            completed(entry, status)
            break
        improvement = new_score - score
        entry["mean_loglik_improvement"] = improvement / n
        completed(entry, "accepted")
        current, cuts, kernel, score = candidate, new_cuts, new_kernel, new_score
        if score > best_score or (score == best_score and len(current["centers"]) < len(best["centers"])):
            best, best_score = current, score
        if improvement / n <= mean_loglik_tolerance:
            status = "conditional_improvement_tolerance"
            break
    return {
        "result": best,
        "history": history,
        "diagnostics": {
            "method": "ordered_conditional_chain_boundary_refinement_v1",
            "status": status,
            "iterations": len(history),
            "max_iterations": int(max_iterations),
            "mean_loglik_tolerance": float(mean_loglik_tolerance),
            "initial_num_clusters": initial_q,
            "final_num_clusters": len(best["centers"]),
            "initial_conditional_log_likelihood": initial_score,
            "conditional_log_likelihood": best_score,
            "conditional_log_likelihood_improvement": best_score - initial_score,
            "fixed_chain": True,
            "unconstrained_reassignment": False,
            "constrained_optimum_certified": False,
        },
    }
