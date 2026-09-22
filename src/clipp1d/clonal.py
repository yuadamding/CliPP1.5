"""Historical constrained reference, retained for numerical regression tests.

Production inference uses the unconstrained search in ``fitting.py``.
"""

import numpy as np
from time import perf_counter

from .model import evaluate
from .policy import Policy
from .solver import _kink_check, objective, prepare_audit, solve_profiled, stationarity
from .types import ClonalConstraintInfeasibleError, NumericalQualificationError, RawFit, WarmState, PrimalWarmState


def fit_fixed_lambda(model, chain, pilot, lambda_value, warm_start=None, policy=Policy()):
    started = perf_counter()
    if warm_start is not None:
        if not isinstance(warm_start, (WarmState, PrimalWarmState)):
            raise ValueError("Continuation requires a chain-bound WarmState")
        warm_start.validate(chain)
    ordered = model.subset(chain.order)
    eligible = np.flatnonzero(ordered.upper == 1.0)
    if eligible.size == 0:
        raise ClonalConstraintInfeasibleError("No retained mutation can reach CCF one under its original bound")
    p = pilot.phi[chain.order]
    lb = pilot.lower_bounds[chain.order]
    at_one = evaluate(ordered, np.minimum(1, ordered.upper)).loss
    bounds = at_one[eligible] + float(np.sum(lb)) - lb[eligible]
    costs = at_one[eligible] - pilot.losses[chain.order][eligible]
    visit = eligible[np.lexsort((eligible, costs))]
    if lambda_value == 0:
        witness = int(visit[0])
        x = p.copy()
        x[witness] = 1
        value = objective(ordered, x, chain.weights * 0)
        gap = max(0, value - float(np.min(bounds)))
        lower, upper = ordered.lower.copy(), ordered.upper.copy()
        lower[witness] = upper[witness] = 1.0
        context = prepare_audit(ordered, x)
        residual, feasible = stationarity(ordered, x, np.zeros(len(model) - 1), lower, upper,
                                          chain.weights * 0, context=context)
        kink_ok, restart = _kink_check(ordered, x, chain.weights * 0, lower, upper, policy, context=context)
        stationary = feasible and residual <= policy.stationarity_tol and kink_ok and restart is None
        # Separable branch optima use global scalar gaps, not a fictitious smooth KKT.
        return RawFit(x, np.zeros(len(model) - 1), value, witness, True,
                      {"status": "qualified_separable", "inner_gap_qualified": True, "inner_gap": 0.0,
                       "inner_kkt_qualified": True, "inner_kkt_residual": 0.0,
                       "raw_branch_stationarity_qualified": stationary,
                       "stationarity_residual": residual, "separable_scalar_gap_qualified": True,
                       "clonal_feasible": True, "witness_search_complete": True, "search_complete": True,
                       "search_policy": "separable_exact_witness_profile",
                       "global_optimality_proven": False, "separable_global_gap": gap,
                       "witnesses_eligible": int(eligible.size),
                       "witnesses_solved": int(eligible.size), "witnesses_screened": 0,
                       "witnesses_unresolved": 0, "starts_attempted": 0,
                       "inner_iterations": 0, "outer_iterations": 0,
                       "search_inner_iterations": 0, "search_inner_seconds": 0.0,
                       "starts_seconds": 0.0, "witness_search_seconds": perf_counter() - started})
    weights = np.maximum(pilot.curvature[chain.order], 1.0)
    pooled = np.clip(np.full(len(model), np.average(p, weights=weights)), ordered.lower, ordered.upper)
    starts = []
    for candidate in (warm_start, p, pooled, pilot.alternative_phi[chain.order]):
        if candidate is None:
            continue
        values = candidate.x if isinstance(candidate, (WarmState, PrimalWarmState)) else candidate
        x = np.clip(values, ordered.lower, ordered.upper).copy()
        if not np.any(x == 1):
            losses = evaluate(ordered, x).loss
            # A feasible start, not a permanent anchor: every surrogate can switch.
            anchor = int(eligible[np.argmin(at_one[eligible] - losses[eligible])])
            x[anchor] = 1.
        if not any(np.array_equal(x, old) for old in starts):
            starts.append(x)
    best = None
    starts_seconds = inner_seconds = 0.0
    total_inner = profiles = profiled = qualified = 0
    failures, statuses = {}, {}
    start_progress = []
    for start in starts:
        begin = perf_counter()
        result = solve_profiled(ordered, chain, lambda_value, start, policy)
        progress_keys = ("status", "outer_iterations", "interval_restart_count", "interval_restart_length_sum",
                         "interval_restart_length_min", "interval_restart_length_max",
                         "interval_restart_objective_decrease", "surrogate_objective_decrease",
                         "objective_decrease", "witness_switches", "restart_witness_switches",
                         "largest_curvature_scale", "backtracks", "progress_tail", "audit_context_count",
                         "audit_anchor_count", "audit_scan_count", "audit_context_seconds",
                         "audit_scan_seconds", "audit_finite_search_seconds")
        start_progress.append({"start_index": len(start_progress), "objective": result.objective,
                               **{key: result.diagnostics[key] for key in progress_keys if key in result.diagnostics}})
        starts_seconds += perf_counter() - begin
        total_inner += result.diagnostics["inner_iterations"]
        inner_seconds += result.diagnostics["inner_solve_seconds"]
        profiles += result.diagnostics["profile_calls"]
        profiled += result.diagnostics["surrogate_witnesses_profiled"]
        if result.qualified:
            qualified += 1
            if best is None or result.objective < best.objective:
                best = result
        else:
            reason = result.diagnostics.get("failure_reason") or result.diagnostics["status"]
            failures[reason] = failures.get(reason, 0) + 1
            status = result.diagnostics["status"]
            statuses[status] = statuses.get(status, 0) + 1
    coverage = {"search_policy": "common_surrogate_multistart_v1",
                "nonlinear_witness_enumeration_performed": False,
                "witnesses_eligible": int(eligible.size), "starts_attempted": len(starts),
                "starts_qualified": qualified, "starts_unresolved": len(starts) - qualified,
                "search_complete": qualified == len(starts),
                "search_profile_calls": profiles, "search_surrogate_witnesses_profiled": profiled,
                "search_inner_iterations": total_inner, "start_failure_counts": failures,
                "start_progress": start_progress,
                "start_failure_status_counts": statuses, "search_inner_seconds": inner_seconds,
                "starts_seconds": starts_seconds, "witness_search_seconds": perf_counter() - started}
    if best is None:
        raise NumericalQualificationError("No qualified common-surrogate start at this penalty", **coverage)
    best.diagnostics.update(coverage)
    return best
