"""Box-constrained chain fusion with a bounded primal multistart search.

No mutation or cluster is required to reach CCF one. The historical constrained
reference in ``clonal.py`` is not part of this production path.
"""

from time import perf_counter

import numpy as np

from .policy import Policy
from .solver import _kink_check, objective, prepare_audit, solve_unconstrained, stationarity
from .types import NumericalQualificationError, PrimalWarmState, RawFit, WarmState


def fit_fixed_lambda(model, chain, pilot, lambda_value, warm_start=None, policy=Policy()):
    started = perf_counter()
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if warm_start is not None:
        if not isinstance(warm_start, (WarmState, PrimalWarmState)):
            raise ValueError("Continuation requires a chain-bound WarmState")
        warm_start.validate(chain)
    ordered = model.subset(chain.order)
    p = pilot.phi[chain.order]
    if lambda_value == 0:
        x = p.copy()
        caps = np.zeros(len(model) - 1)
        value = objective(ordered, x, caps)
        gap = float(np.sum(pilot.gaps))
        scalar_qualified = bool(np.all(np.isfinite(pilot.gaps)) and
                                np.all(pilot.gaps >= 0) and
                                np.all(pilot.gaps <= policy.scalar_atol +
                                       policy.scalar_rtol * np.abs(pilot.losses)))
        context = prepare_audit(ordered, x)
        residual, feasible = stationarity(ordered, x, caps, ordered.lower, ordered.upper,
                                          caps, context=context)
        if not feasible or not np.isfinite(value) or not scalar_qualified:
            raise NumericalQualificationError("Separable pilots did not qualify", box_feasible=feasible,
                                               separable_scalar_gap_qualified=scalar_qualified,
                                               separable_global_gap=gap)
        kink_ok, restart = _kink_check(ordered, x, caps, ordered.lower, ordered.upper,
                                       policy, context=context)
        stationary = feasible and residual <= policy.stationarity_tol and kink_ok and restart is None
        # Qualification here uses the global scalar gaps. Stationarity is a
        # separate measured claim, never inherited from the separable certificate.
        return RawFit(x, caps, value, None, True,
                      {"status": "qualified_separable", "inner_gap_qualified": True, "inner_gap": 0.0,
                       "inner_kkt_qualified": True, "inner_kkt_residual": 0.0,
                       "raw_branch_stationarity_qualified": stationary,
                       "stationarity_residual": residual, "separable_scalar_gap_qualified": True,
                       "box_feasible": feasible, "clonal_constraint": False,
                       "search_complete": True, "search_policy": "separable_unconstrained_pilots",
                       "global_optimality_proven": False, "separable_global_gap": gap,
                       "starts_attempted": 0, "inner_iterations": 0, "outer_iterations": 0,
                       "search_profile_calls": 0, "search_surrogate_witnesses_profiled": 0,
                       "search_inner_iterations": 0, "search_inner_seconds": 0.0,
                       "starts_seconds": 0.0, "fit_seconds": perf_counter() - started})
    weights = np.maximum(pilot.curvature[chain.order], 1.0)
    pooled = np.clip(np.full(len(model), np.average(p, weights=weights)), ordered.lower, ordered.upper)
    starts = []
    for candidate in (warm_start, p, pooled, pilot.alternative_phi[chain.order]):
        if candidate is None:
            continue
        values = candidate.x if isinstance(candidate, (WarmState, PrimalWarmState)) else candidate
        x = np.clip(values, ordered.lower, ordered.upper).copy()
        if not any(np.array_equal(x, old) for old in starts):
            starts.append(x)
    best = None
    starts_seconds = inner_seconds = 0.0
    total_inner = qualified = 0
    failures, statuses = {}, {}
    start_progress = []
    for start in starts:
        begin = perf_counter()
        result = solve_unconstrained(ordered, chain, lambda_value, start, policy)
        progress_keys = ("status", "outer_iterations", "interval_restart_count", "interval_restart_length_sum",
                         "interval_restart_length_min", "interval_restart_length_max",
                         "interval_restart_objective_decrease", "surrogate_objective_decrease",
                         "objective_decrease", "largest_curvature_scale", "backtracks", "progress_tail",
                         "audit_context_count", "audit_anchor_count", "audit_scan_count",
                         "audit_context_seconds", "audit_scan_seconds", "audit_finite_search_seconds")
        start_progress.append({"start_index": len(start_progress), "objective": result.objective,
                               **{key: result.diagnostics[key] for key in progress_keys if key in result.diagnostics}})
        starts_seconds += perf_counter() - begin
        total_inner += result.diagnostics["inner_iterations"]
        inner_seconds += result.diagnostics["inner_solve_seconds"]
        if result.qualified:
            qualified += 1
            if best is None or result.objective < best.objective:
                best = result
        else:
            reason = result.diagnostics.get("failure_reason") or result.diagnostics["status"]
            failures[reason] = failures.get(reason, 0) + 1
            status = result.diagnostics["status"]
            statuses[status] = statuses.get(status, 0) + 1
    coverage = {"search_policy": "unconstrained_chain_multistart_v1", "clonal_constraint": False,
                "nonlinear_witness_enumeration_performed": False, "starts_attempted": len(starts),
                "starts_qualified": qualified, "starts_unresolved": len(starts) - qualified,
                "search_complete": qualified == len(starts),
                "search_profile_calls": 0, "search_surrogate_witnesses_profiled": 0,
                "search_inner_iterations": total_inner, "start_failure_counts": failures,
                "start_progress": start_progress, "start_failure_status_counts": statuses,
                "search_inner_seconds": inner_seconds, "starts_seconds": starts_seconds,
                "fit_seconds": perf_counter() - started}
    if best is None:
        raise NumericalQualificationError("No qualified unconstrained start at this penalty", **coverage)
    best.diagnostics.update(coverage)
    return best
