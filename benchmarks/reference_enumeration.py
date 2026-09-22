"""Historical constrained nonlinear witness enumeration for policy comparisons.

This retains the df44e6a fixed-witness/multiple-start search, with direct-backend
primal start deduplication. It deliberately calls the current production
``solver.solve_branch`` at runtime so numerical repairs apply to both policies.
It is not current unconstrained production inference and is not installed in the
package. Never inject this constrained search into an unconstrained full fit.
"""

import numpy as np
from time import perf_counter

from clipp1d import solver
from clipp1d.model import evaluate
from clipp1d.policy import Policy
from clipp1d.types import (
    ClonalConstraintInfeasibleError, NumericalQualificationError, PrimalWarmState, RawFit, WarmState,
)


SEARCH_POLICY = "independent_witness_enumeration_reference"


def projected_primal_starts(candidates, lower, upper, witness):
    """Deduplicate after applying the current witness, ignoring unused duals.

    No previous witness box is inherited. The direct QP backend reconstructs
    its own dual, so different incoming duals do not define distinct starts.
    WarmState chain identity is validated once by ``fit_fixed_lambda``.
    """
    starts = []
    for candidate in candidates:
        if candidate is None:
            continue
        values = candidate.x if isinstance(candidate, (WarmState, PrimalWarmState)) else candidate
        projected = np.clip(values, lower, upper)
        projected[witness] = 1.0
        if not any(np.array_equal(projected, old) for old in starts):
            starts.append(projected)
    return starts


def fit_fixed_lambda(model, chain, pilot, lambda_value, warm_start=None, policy=Policy()):
    started = perf_counter()
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and nonnegative")
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
        value = solver.objective(ordered, x, chain.weights * 0)
        gap = max(0, value - float(np.min(bounds)))
        lower, upper = ordered.lower.copy(), ordered.upper.copy()
        lower[witness] = upper[witness] = 1.0
        residual, feasible = solver.stationarity(ordered, x, np.zeros(len(model) - 1), lower, upper, chain.weights * 0)
        kink_ok, restart = solver._kink_check(ordered, x, chain.weights * 0, lower, upper, policy)
        stationary = feasible and residual <= policy.stationarity_tol and kink_ok and restart is None
        # Separable branch optima use global scalar gaps, not a fictitious smooth KKT.
        return RawFit(x, np.zeros(len(model) - 1), value, witness, True,
                      {"search_policy": SEARCH_POLICY, "status": "qualified_separable", "inner_gap_qualified": True, "inner_gap": 0.0,
                       "inner_kkt_qualified": True, "inner_kkt_residual": 0.0,
                       "raw_branch_stationarity_qualified": stationary,
                       "stationarity_residual": residual, "separable_scalar_gap_qualified": True,
                       "clonal_feasible": True, "witness_search_complete": True,
                       "global_optimality_proven": False, "separable_global_gap": gap,
                       "witnesses_eligible": int(eligible.size),
                       "witnesses_solved": int(eligible.size), "witnesses_screened": 0,
                       "witnesses_unresolved": 0, "starts_attempted": 0,
                       "inner_iterations": 0, "outer_iterations": 0,
                       "search_inner_iterations": 0, "search_inner_seconds": 0.0,
                       "starts_seconds": 0.0, "witness_search_seconds": perf_counter() - started})
    weights = np.maximum(pilot.curvature[chain.order], 1.0)
    pooled = np.clip(np.full(len(model), np.average(p, weights=weights)), ordered.lower, ordered.upper)
    best = None
    incumbent = np.inf
    solved = screened = unresolved = starts_attempted = total_inner = 0
    starts_seconds = inner_seconds = 0.0
    unresolved_indices = []
    failures = {}
    lower_sum = float(np.sum(lb))
    for witness in visit:
        witness = int(witness)
        branch_lb = float(at_one[witness] + lower_sum - lb[witness])
        margin = 1e-10 * (1 + abs(branch_lb) + abs(incumbent)) if np.isfinite(incumbent) else 0
        if best is not None and branch_lb > incumbent + margin:
            screened += 1
            continue
        starts = projected_primal_starts(
            (warm_start, p, pooled, pilot.alternative_phi[chain.order]),
            ordered.lower, ordered.upper, witness,
        )
        branch_best = None
        for start in starts:
            start_time = perf_counter()
            result = solver.solve_branch(ordered, chain, lambda_value, witness, start, policy)
            starts_seconds += perf_counter() - start_time
            starts_attempted += 1
            total_inner += result.diagnostics["inner_iterations"]
            inner_seconds += result.diagnostics.get("inner_solve_seconds", 0.0)
            if result.qualified and (branch_best is None or result.objective < branch_best.objective):
                branch_best = result
            elif not result.qualified:
                reason = result.diagnostics["status"]
                failures[reason] = failures.get(reason, 0) + 1
        if branch_best is None:
            unresolved += 1
            unresolved_indices.append(witness)
        else:
            solved += 1
            if best is None or branch_best.objective < best.objective:
                best = branch_best
                incumbent = best.objective
    final_screened = 0
    if best is not None:
        for witness in unresolved_indices:
            branch_lb = float(at_one[witness] + lower_sum - lb[witness])
            margin = 1e-10 * (1 + abs(branch_lb) + abs(incumbent))
            if branch_lb > incumbent + margin:
                unresolved -= 1
                screened += 1
                final_screened += 1
    coverage = {"search_policy": SEARCH_POLICY, "witnesses_eligible": int(eligible.size), "witnesses_solved": solved,
                "witnesses_screened": screened, "witnesses_unresolved": unresolved,
                "witness_search_complete": unresolved == 0, "starts_attempted": starts_attempted,
                "witnesses_screened_final": final_screened,
                "witnesses_unresolved_initial": len(unresolved_indices),
                "search_inner_iterations": total_inner, "start_failure_counts": failures,
                "search_inner_seconds": inner_seconds, "starts_seconds": starts_seconds,
                "witness_search_seconds": perf_counter() - started}
    if best is None:
        raise NumericalQualificationError("No qualified raw witness branch at this penalty", **coverage)
    best.diagnostics.update(coverage)
    return best
