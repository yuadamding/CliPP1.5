"""Safeguarded likelihood majorization and a chain-only primal-dual QP solver."""

import numpy as np

from .chain import adjoint, difference
from .model import clipping_breakpoints, evaluate, one_sided_derivatives
from .policy import Policy
from .types import InnerFit, RawFit


def quadratic_gap(x, q, h, target, lower, upper, caps):
    a = adjoint(q)
    z = np.clip(target - a / h, lower, upper)
    primal = float(0.5 * np.dot(h, (x - target)**2) + np.dot(caps, np.abs(difference(x))))
    dual = float(np.sum(0.5 * h * (z - target)**2 + a * z))
    return primal - dual, primal


def solve_quadratic(h, target, lower, upper, caps, start, policy=Policy(), dual=None):
    """Solve the boxed strictly convex surrogate using exact separable proximal steps."""
    h, target, lower, upper, caps, start = [np.asarray(v, dtype=float) for v in
                                           (h, target, lower, upper, caps, start)]
    n = h.size
    if (n == 0 or any(v.shape != (n,) for v in (h, target, lower, upper, start)) or
            caps.shape != (n - 1,) or any(not np.all(np.isfinite(v)) for v in
                                        (h, target, lower, upper, caps, start)) or
            np.any(h <= 0) or np.any(lower > upper) or np.any(caps < 0)):
        raise ValueError("Invalid chain quadratic problem")
    x = np.clip(start, lower, upper)
    q = np.zeros(n - 1) if dual is None else np.clip(np.asarray(dual).copy(), -caps, caps)
    if not np.any(caps):
        x = np.clip(target, lower, upper)
        return InnerFit(x, np.zeros(n - 1), 0, True, 0)
    mu = float(np.median(h))
    tau, sigma = 0.49 / mu, 0.49 * mu
    extrapolated = x.copy()
    gap = np.inf
    for iteration in range(1, policy.inner_max_iterations + 1):
        q = np.clip(q + sigma * difference(extrapolated), -caps, caps)
        previous = x
        x = np.clip((x - tau * adjoint(q) + tau * h * target) / (1 + tau * h), lower, upper)
        extrapolated = 2 * x - previous
        if iteration % 10 == 0:
            gap, primal = quadratic_gap(x, q, h, target, lower, upper, caps)
            rounding = 128 * np.finfo(float).eps * (1 + abs(primal))
            if -rounding <= gap <= policy.inner_atol + policy.inner_rtol * abs(primal):
                return InnerFit(x, q, max(0, gap), True, iteration)
    gap, _ = quadratic_gap(x, q, h, target, lower, upper, caps)
    return InnerFit(x, q, float(gap), False, policy.inner_max_iterations)


def objective(model, x, caps):
    return float(np.sum(evaluate(model, x).loss) + np.dot(caps, np.abs(difference(x))))


def _snap_fusions(x, lower, upper, tolerance):
    """Numerical active-set polish, audited by the actual objective and raw KKT."""
    result = x.copy()
    start = 0
    low = high = x[0]
    for stop in range(1, len(x) + 1):
        if stop == len(x) or max(high, x[stop]) - min(low, x[stop]) > tolerance:
            lo, hi = np.max(lower[start:stop]), np.min(upper[start:stop])
            if lo <= hi:
                result[start:stop] = np.clip(np.mean(x[start:stop]), lo, hi)
            start = stop
            if stop < len(x):
                low = high = x[stop]
        else:
            low, high = min(low, x[stop]), max(high, x[stop])
    return result


def stationarity(model, x, q, lower, upper, caps):
    terms = evaluate(model, x, derivatives=True)
    a = adjoint(q)
    left, right = one_sided_derivatives(model, x)
    # At an upward clipping kink, use a valid subgradient from [left,right].
    # A downward kink still needs the explicit directional checks below.
    gradient = np.where(left <= right, np.clip(-a, left, right), terms.gradient)
    residual = gradient + a
    residual = np.where(x == lower, np.minimum(residual, 0), residual)
    residual = np.where(x == upper, np.maximum(residual, 0), residual)
    residual = np.where(lower == upper, 0, residual)
    node = float(np.max(np.abs(residual) / (1 + np.abs(gradient) + np.abs(a))))
    jumps = difference(x)
    edge = (float(np.max(np.where(jumps != 0,
                                 np.abs(q - caps * np.sign(jumps)) / (1 + caps), 0)))
            if jumps.size else 0.0)
    dual = float(np.max(np.maximum(np.abs(q) - caps, 0) / (1 + caps))) if caps.size else 0.0
    feasible = bool(np.all(x >= lower) and np.all(x <= upper) and np.all(np.isfinite(x)))
    return max(node, edge, dual), feasible


def _snap_crossed_breakpoints(model, previous, trial, lower, upper, caps):
    """Keep exact clipping candidates when a smooth surrogate crosses a kink.

    Only objective-improving proposals survive. The outer iteration still checks
    both descent inequalities; the QP solution retains its separate gap.
    """
    slopes = np.where(model.valid, model.slope, np.nan)
    result = trial
    value = objective(model, trial, caps)
    for threshold in (model.eps, 1 - model.eps):
        points = threshold / slopes
        for c in range(points.shape[1]):
            point = points[:, c]
            crossed = ((point >= np.minimum(previous, trial)) & (point <= np.maximum(previous, trial)) &
                       (point >= lower) & (point <= upper))
            if np.any(crossed):
                proposal = np.where(crossed, point, result)
                candidate_value = objective(model, proposal, caps)
                if candidate_value < value:
                    result, value = proposal, candidate_value
    return result


def _kink_check(model, x, caps, lower, upper, policy):
    """Check one-sided coordinate and fused-block directions at clipping plateaus.

    Return a decreasing restart if found. A kink that remains ambiguous cannot
    receive the smooth stationarity qualification.
    """
    left, right = one_sided_derivatives(model, x)
    smooth = evaluate(model, x, derivatives=True).gradient
    kink = (left != smooth) | (right != smooth)
    mass = model.slope * x[:, None]
    plateau = np.any(model.valid & (((mass <= model.eps) & (model.alt[:, None] > 0)) |
                                    ((mass >= 1 - model.eps) & (model.ref[:, None] > 0))), axis=1)
    affected = (kink | plateau) & (lower < upper)
    if not np.any(affected):
        return True, None
    cuts = np.r_[0, np.flatnonzero(difference(x) != 0) + 1, len(x)]
    current = objective(model, x, caps)
    directions = [(i, i + 1) for i in np.flatnonzero(affected)]
    directions.extend((int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if np.any(affected[a:b]))
    unresolved = False
    for start, stop in directions:
        lo, hi = float(np.max(lower[start:stop])), float(np.min(upper[start:stop]))
        if lo < hi:
            # A flat derivative below the clipping threshold does not establish
            # convergence: cross each accessible breakpoint and evaluate the
            # actual likelihood plus both incident fusion edges.
            block = model.subset(np.arange(start, stop))
            for point in clipping_breakpoints(block):
                for offset in (-1e-5, 0.0, 1e-5):
                    t = float(np.clip(point + offset, lo, hi))
                    trial = x.copy()
                    trial[start:stop] = t
                    if objective(model, trial, caps) < current - 1e-12 * (1 + current):
                        return False, trial
        for sign in (-1, 1):
            room = float(np.min(upper[start:stop] - x[start:stop]) if sign == 1 else
                         np.min(x[start:stop] - lower[start:stop]))
            if room <= 0:
                continue
            derivative = float(np.sum(right[start:stop] if sign == 1 else -left[start:stop]))
            if start:
                d = x[start] - x[start - 1]
                derivative += caps[start - 1] * (1 if d == 0 else sign * np.sign(d))
            if stop < len(x):
                d = x[stop - 1] - x[stop]
                derivative += caps[stop - 1] * (1 if d == 0 else sign * np.sign(d))
            if derivative < -policy.stationarity_tol:
                unresolved = True
                for step in (min(room, 1e-3), min(room, 1e-5), min(room, 1e-7)):
                    trial = x.copy()
                    trial[start:stop] += sign * step
                    if objective(model, trial, caps) < current - 1e-12 * (1 + current):
                        return False, trial
    return not unresolved, None


def solve_branch(model, chain, lambda_value, witness_index, start, policy=Policy()):
    """model and start are in chain order; witness index is a chain position."""
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if not 0 <= witness_index < len(model) or model.upper[witness_index] != 1:
        raise ValueError("Witness must be eligible under the original float64 box")
    lower, upper = model.lower.copy(), model.upper.copy()
    # No inherited witness bound is carried over from the preceding branch.
    lower[witness_index] = upper[witness_index] = 1.0
    x = np.clip(start, lower, upper)
    caps = lambda_value * chain.weights
    q = np.zeros(len(model) - 1)
    total_inner, accepted, backtracks = 0, 0, 0
    status, residual, inner_gap = "outer_iteration_limit", np.inf, np.inf
    accepted_surrogate_gap = np.inf
    qualified = False
    current = objective(model, x, caps)
    last_inner_qualified = False
    for outer in range(policy.outer_max_iterations):
        terms = evaluate(model, x, derivatives=True)
        h = np.maximum(terms.curvature, 1.0)
        successful = False
        for attempt in range(policy.max_backtracks):
            target = x - terms.gradient / h
            inner = solve_quadratic(h, target, lower, upper, caps, x, policy, dual=q)
            total_inner += inner.iterations
            inner_gap = inner.gap
            last_inner_qualified = inner.qualified
            if not inner.qualified:
                status = "inner_gap_unresolved"
                break
            trial = inner.x
            breakpoint_trial = _snap_crossed_breakpoints(model, x, trial, lower, upper, caps)
            # Breakpoint steps are valid inexact MM steps when they reduce Q.
            # Keep the QP's own qualified gap distinct from this accepted step.
            if not np.array_equal(breakpoint_trial, trial):
                trial = breakpoint_trial
            polished = _snap_fusions(trial, lower, upper, min(1e-8, policy.fusion_tol / 10))
            polish_gap, polish_primal = quadratic_gap(polished, inner.dual, h, target, lower, upper, caps)
            polish_rounding = 128 * np.finfo(float).eps * (1 + abs(polish_primal))
            if (-polish_rounding <= polish_gap <= policy.inner_atol + policy.inner_rtol * abs(polish_primal) and
                    objective(model, polished, caps) <= objective(model, trial, caps) + 1e-12 * (1 + current)):
                trial = polished
            step = trial - x
            new_loss = float(np.sum(evaluate(model, trial).loss))
            surrogate = float(np.sum(terms.loss) + np.dot(terms.gradient, step) + 0.5 * np.dot(h, step**2))
            new_objective = new_loss + float(np.dot(caps, np.abs(difference(trial))))
            margin = 128 * np.finfo(float).eps * (1 + abs(current))
            # Both the observed objective and surrogate descent must hold.
            old_surrogate = float(np.sum(terms.loss) + np.dot(caps, np.abs(difference(x))))
            if (new_loss <= surrogate + margin and new_objective <= current + margin and
                    surrogate + np.dot(caps, np.abs(difference(trial))) <= old_surrogate + margin):
                x, q, current = trial, inner.dual, float(new_objective)
                accepted_surrogate_gap = max(0.0, quadratic_gap(x, q, h, target, lower, upper, caps)[0])
                successful = True
                accepted += 1
                break
            h *= 2
            backtracks += 1
        if not successful:
            if last_inner_qualified:
                status = "majorization_unresolved"
            break
        residual, feasible = stationarity(model, x, q, lower, upper, caps)
        kink_ok, restart = _kink_check(model, x, caps, lower, upper, policy)
        if restart is not None:
            x = restart
            current = objective(model, x, caps)
            q.fill(0)
            continue
        if feasible and residual <= policy.stationarity_tol and kink_ok:
            status, qualified = "qualified", True
            break
        if not kink_ok:
            status = "clipping_kink_unresolved"
            break
        # Crossing a clipped plateau may require a finite scalar move. These
        # starts are normally supplied by the global pilot; do not certify stagnation.
        if not np.any(step):
            status = "stationarity_unresolved"
            break
    residual, feasible = stationarity(model, x, q, lower, upper, caps)
    return RawFit(x, q, objective(model, x, caps), witness_index, qualified,
                  {"status": status, "inner_gap_qualified": last_inner_qualified,
                   "inner_gap": float(inner_gap), "raw_branch_stationarity_qualified": qualified,
                   "accepted_surrogate_gap": accepted_surrogate_gap,
                   "stationarity_residual": residual, "clonal_feasible": feasible and x[witness_index] == 1,
                   "witness_search_complete": False, "global_optimality_proven": False,
                   "outer_iterations": outer + 1, "accepted_steps": accepted,
                   "inner_iterations": total_inner, "backtracks": backtracks})
