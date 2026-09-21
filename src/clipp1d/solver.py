"""Safeguarded likelihood majorization with a direct bounded chain-TV QP solver."""

import hashlib
import math
import numpy as np
from time import perf_counter

from .chain import adjoint, difference
from .model import clipping_breakpoints, evaluate, one_sided_derivatives
from .policy import Policy
from .tv import forward_messages, polish_blocks, reconstruct_dual, split_primal
from .types import InnerFit, QuadraticWitnessProfile, RawFit, WarmState


def quadratic_gap(x, q, h, target, lower, upper, caps):
    """Nonnegative-term gap and a scale with separable constants removed.

    Frozen coordinates are omitted *before* arithmetic involving their targets.
    The scale subtracts each free coordinate's boxed quadratic minimum, and
    excludes TV energy of edges with two frozen ends. It is invariant to an
    arbitrary constant added through a frozen target.
    """
    if (np.any(x < lower) or np.any(x > upper) or np.any(np.abs(q) > caps) or
            not np.all(np.isfinite(x)) or not np.all(np.isfinite(q))):
        return np.inf, 0.0
    a = adjoint(q)
    free = lower < upper
    hf, uf, af, xf = h[free], target[free], a[free], x[free]
    z = np.clip(uf - af / hf, lower[free], upper[free])
    delta = xf - z
    quadratic = .5 * hf * delta**2
    normal = (hf * (z - uf) + af) * delta
    jumps = difference(x)
    edge = caps * np.abs(jumps) - q * jumps
    normal_margin = 128 * np.finfo(float).eps * (1 + np.abs(hf * (z - uf)) + np.abs(af)) * np.abs(delta)
    edge_margin = 128 * np.finfo(float).eps * (1 + caps) * np.abs(jumps)
    if np.any(normal < -normal_margin) or np.any(edge < -edge_margin):
        return np.inf, 0.0
    gap = float(np.sum(quadratic) + np.sum(np.maximum(normal, 0)) + np.sum(np.maximum(edge, 0)))
    reference = np.clip(uf, lower[free], upper[free])
    displacement = xf - reference
    variable_energy = .5 * hf * displacement**2 + hf * (reference - uf) * displacement
    moving_edges = free[:-1] | free[1:]
    scale = float(np.sum(np.maximum(variable_energy, 0)) +
                  np.dot(caps[moving_edges], np.abs(jumps[moving_edges])))
    return gap, scale


def quadratic_kkt(x, q, h, target, lower, upper, caps):
    """Box-normal primal residual and a dual projection residual, both normalized."""
    free = lower < upper
    a = adjoint(q)
    g = np.zeros_like(x)
    g[free] = h[free] * (x[free] - target[free])
    residual = g + a
    residual = np.where(x == lower, np.minimum(residual, 0), residual)
    residual = np.where(x == upper, np.maximum(residual, 0), residual)
    residual[~free] = 0
    node = float(np.max(np.abs(residual) / (1 + np.abs(g) + np.abs(a))))
    dual = (float(np.max(np.abs(q - np.clip(q + difference(x), -caps, caps)) / (1 + caps)))
            if q.size else 0.0)
    return max(node, dual)


def _quadratic_inputs(h, target, lower, upper, caps, start, dual):
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
    if q.shape != (n - 1,) or not np.all(np.isfinite(q)):
        raise ValueError("Dual start must be a finite chain-edge vector")
    return h, target, lower, upper, caps, x, q


def solve_quadratic(h, target, lower, upper, caps, start, policy=Policy(), dual=None):
    """Direct bounded weighted TV; independent stable gap and O(M) KKT certificate.

    The minimizer of this strictly convex problem does not depend on a warm start.
    Starts remain accepted/validated for API parity and outer primal continuation.
    Arithmetic or certificate failure returns unresolved, without an iterative fallback.
    """
    h, target, lower, upper, caps, x, q = _quadratic_inputs(h, target, lower, upper, caps, start, dual)
    work = {}
    try:
        if not np.any(caps):
            x, q = np.clip(target, lower, upper), np.zeros(len(h) - 1)
        else:
            x, work = split_primal(h, target, lower, upper, caps)
            x = polish_blocks(x, h, target, lower, upper, caps)
            q = reconstruct_dual(x, h, target, lower, upper, caps)
        gap, scale = quadratic_gap(x, q, h, target, lower, upper, caps)
        kkt = quadratic_kkt(x, q, h, target, lower, upper, caps)
        qualified = bool(np.isfinite(gap) and gap <= policy.inner_atol + policy.inner_rtol * scale and
                         kkt <= policy.inner_kkt_tol)
    except ArithmeticError as exc:
        gap, scale, kkt, qualified = np.inf, 0., np.inf, False
        work["failure_reason"] = str(exc)
    return InnerFit(x, q, gap, qualified, 1, kkt, scale, "bounded_weighted_tv_dp", work)


def solve_quadratic_iterative(h, target, lower, upper, caps, start, policy=Policy(), dual=None):
    """First-order reference for tests/attribution only; never called by production."""
    h, target, lower, upper, caps, x, q = _quadratic_inputs(h, target, lower, upper, caps, start, dual)
    n = len(h)
    if not np.any(caps):
        x = np.clip(target, lower, upper)
        return InnerFit(x, np.zeros(n - 1), 0, True, 0, 0.0, 0.0)
    free = lower < upper
    if not np.any(free):
        q = caps * np.sign(difference(x))
        return InnerFit(x, q, 0.0, True, 0, 0.0, 0.0)
    mu = float(np.median(h[free])) if np.any(free) else 1.0
    tau, sigma = 0.49 / mu, 0.49 * mu
    extrapolated = x.copy()
    gap = np.inf
    for iteration in range(1, policy.inner_max_iterations + 1):
        q = np.clip(q + sigma * difference(extrapolated), -caps, caps)
        previous = x
        x = lower.copy()
        x[free] = np.clip((previous[free] - tau * adjoint(q)[free] + tau * h[free] * target[free]) /
                          (1 + tau * h[free]), lower[free], upper[free])
        extrapolated = 2 * x - previous
        if iteration % 10 == 0:
            gap, scale = quadratic_gap(x, q, h, target, lower, upper, caps)
            if np.isfinite(gap) and gap <= policy.inner_atol + policy.inner_rtol * scale:
                kkt = quadratic_kkt(x, q, h, target, lower, upper, caps)
                if kkt <= policy.inner_kkt_tol:
                    return InnerFit(x, q, gap, True, iteration, kkt, scale)
    gap, scale = quadratic_gap(x, q, h, target, lower, upper, caps)
    return InnerFit(x, q, float(gap), False, policy.inner_max_iterations,
                    quadratic_kkt(x, q, h, target, lower, upper, caps), scale)


def profile_quadratic_witnesses(h, target, lower, upper, caps, policy=Policy()):
    """Share two message passes across all witnesses of *one identical surrogate*.

    The nonconvex outer branch search is deliberately separate: its witnesses may
    have different h/target arrays, so these messages cannot be reused across them.
    Values omit only the common sum of original boxed unary minima; the offset is
    returned explicitly. Only the selected witness is reconstructed/certified.
    """
    h, target, lower, upper, caps, _, _ = _quadratic_inputs(
        h, target, lower, upper, caps, np.asarray(lower), None)
    if np.any(upper > 1) or not np.any((lower <= 1) & (upper == 1)):
        raise ValueError("Profiling requires CCF boxes <= 1 and an eligible witness")
    digest = hashlib.sha256()
    for value in (h, target, lower, upper, caps):
        digest.update(value.tobytes())
    _, _, _, prefix, forward_stats = forward_messages(h, target, lower, upper, caps, values_at_one=True)
    _, _, _, suffix, reverse_stats = forward_messages(h[::-1], target[::-1], lower[::-1], upper[::-1],
                                                    caps[::-1], values_at_one=True)
    reference = np.clip(target, lower, upper)
    unary_at_one = .5 * h * (1 - reference)**2 + h * (reference - target) * (1 - reference)
    values = prefix + suffix[::-1] - unary_at_one
    witness = int(np.argmin(values))
    fixed_lower, fixed_upper = lower.copy(), upper.copy()
    fixed_lower[witness] = fixed_upper[witness] = 1.
    fit = solve_quadratic(h, target, fixed_lower, fixed_upper, caps, reference, policy)
    displacement = fit.x - reference
    attained = math.fsum(.5 * h * displacement**2 + h * (reference - target) * displacement)
    attained += float(np.dot(caps, np.abs(difference(fit.x))))
    error = abs(attained - values[witness])
    tolerance = policy.inner_atol + policy.inner_rtol * abs(attained)
    eligible = upper == 1
    return QuadraticWitnessProfile(witness, fit, values, math.fsum(.5 * h * (reference - target)**2),
                                   bool(fit.qualified and np.all(np.isfinite(values[eligible])) and error <= tolerance),
                                   digest.hexdigest(), dict(scope="one_common_quadratic", reconstruction_count=1,
                                   prefix_value_error=error, forward=forward_stats, reverse=reverse_stats))


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
    # At a box endpoint, the inaccessible clipping plateau contributes no
    # derivative to feasible motion. Only the inward one-sided derivative applies.
    gradient = np.where(x == lower, right, gradient)
    gradient = np.where(x == upper, left, gradient)
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


def interval_descent(x, left, right, lower, upper, caps, tolerance=0.0):
    """Find a descending signed subinterval of an exact fused block in O(M).

    Prefix costs plus the best left boundary cover every possible right boundary.
    Frozen coordinates and infeasible signs split the scan. Negative merit means
    derivative < -tolerance * (1 + absolute node and boundary contributions).
    """
    best, best_merit = None, 0.0
    n = len(x)
    for sign in (-1, 1):
        prefix = absolute = 0.0
        start_key = np.inf
        active = False
        for i in range(n):
            feasible = x[i] > lower[i] if sign < 0 else x[i] < upper[i]
            if not feasible:
                active = False
                continue
            if not active or (i and x[i] != x[i - 1]):
                prefix = absolute = 0.0
                start_key = np.inf
            active = True
            left_cost = 0.0
            if i:
                jump = x[i] - x[i - 1]
                left_cost = caps[i - 1] * (1 if jump == 0 else sign * np.sign(jump))
            key = left_cost + tolerance * abs(left_cost) - prefix - tolerance * absolute
            if key < start_key:
                start_key = key
                start = i
                start_prefix, start_absolute, start_cost = prefix, absolute, left_cost
            value = -left[i] if sign < 0 else right[i]
            prefix += value
            absolute += abs(value)
            right_cost = 0.0
            if i + 1 < n:
                jump = x[i + 1] - x[i]
                right_cost = caps[i] * (1 if jump == 0 else -sign * np.sign(jump))
            merit = prefix + tolerance * absolute + right_cost + tolerance * abs(right_cost) + start_key + tolerance
            if merit < best_merit:
                derivative = prefix - start_prefix + start_cost + right_cost
                scale = 1 + absolute - start_absolute + abs(start_cost) + abs(right_cost)
                best = (start, i + 1, sign, float(derivative), float(scale))
                best_merit = merit
    return best


def local_interval_delta(block, x, caps, start, stop, new_values, old_losses=None):
    """Change in objective using only affected likelihoods and incident edges."""
    old = x[start:stop]
    new = np.broadcast_to(np.asarray(new_values, dtype=float), old.shape)
    if old_losses is None:
        old_losses = evaluate(block, old).loss
    new_losses = evaluate(block, new).loss
    delta = float(np.sum(new_losses - old_losses))
    delta += float(np.dot(caps[start:stop - 1], np.abs(difference(new)) - np.abs(difference(old))))
    if start:
        delta += caps[start - 1] * (abs(new[0] - x[start - 1]) - abs(old[0] - x[start - 1]))
    if stop < len(x):
        delta += caps[stop - 1] * (abs(x[stop] - new[-1]) - abs(x[stop] - old[-1]))
    return delta


def _kink_check(model, x, caps, lower, upper, policy):
    """Audit every signed fused subinterval; screen finite plateau moves locally."""
    mass = model.slope * x[:, None]
    slopes = np.where(model.valid, model.slope, np.nan)
    at_kink = np.any((x[:, None] == model.eps / slopes) |
                     (x[:, None] == (1 - model.eps) / slopes), axis=1)
    plateau = np.any(model.valid & (((mass <= model.eps) & (model.alt[:, None] > 0)) |
                                    ((mass >= 1 - model.eps) & (model.ref[:, None] > 0))), axis=1)
    if not np.any((at_kink | plateau) & (lower < upper)):
        return True, None
    left, right = one_sided_derivatives(model, x)
    direction = interval_descent(x, left, right, lower, upper, caps, policy.stationarity_tol)
    if direction is not None:
        start, stop, sign, derivative, _ = direction
        block = model.subset(np.arange(start, stop))
        old_losses = evaluate(block, x[start:stop]).loss
        margin = 128 * np.finfo(float).eps * (1 + float(np.sum(np.abs(old_losses))))
        room = float(np.min(upper[start:stop] - x[start:stop]) if sign > 0 else
                     np.min(x[start:stop] - lower[start:stop]))
        step = min(1e-3, room)
        for _ in range(48):
            values = x[start:stop] + sign * step
            delta = local_interval_delta(block, x, caps, start, stop, values, old_losses)
            if delta < -margin and delta <= 1e-4 * step * derivative:
                trial = x.copy()  # allocate the full vector only for an accepted move
                trial[start:stop] = values
                return False, trial
            step *= .5
        return False, None

    affected = plateau & (lower < upper)
    if not np.any(affected):
        return True, None
    cuts = np.r_[0, np.flatnonzero(difference(x) != 0) + 1, len(x)]
    intervals = [(i, i + 1) for i in np.flatnonzero(affected)]
    intervals.extend((int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])
                     if b - a > 1 and np.any(affected[a:b]))
    for start, stop in intervals:
        lo, hi = float(np.max(lower[start:stop])), float(np.min(upper[start:stop]))
        if lo >= hi:
            continue
        block = model.subset(np.arange(start, stop))
        old_losses = evaluate(block, x[start:stop]).loss
        margin = 128 * np.finfo(float).eps * (1 + float(np.sum(np.abs(old_losses))))
        points = clipping_breakpoints(block)
        points = points[(points >= lo) & (points <= hi)]
        # Derivative coverage above is exhaustive. Finite plateau escapes inspect
        # the closest transitions without a quadratic collection of proposals.
        pivot = np.searchsorted(points, x[start])
        points = points[max(0, pivot - 2):pivot + 3]
        for point in points:
            for offset in (-1e-5, 0.0, 1e-5):
                value = float(np.clip(point + offset, lo, hi))
                if local_interval_delta(block, x, caps, start, stop, value, old_losses) < -margin:
                    trial = x.copy()
                    trial[start:stop] = value
                    return False, trial
    return True, None


def solve_branch(model, chain, lambda_value, witness_index, start, policy=Policy()):
    """model and start are in chain order; witness index is a chain position."""
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if not 0 <= witness_index < len(model) or model.upper[witness_index] != 1:
        raise ValueError("Witness must be eligible under the original float64 box")
    lower, upper = model.lower.copy(), model.upper.copy()
    # No inherited witness bound is carried over from the preceding branch.
    lower[witness_index] = upper[witness_index] = 1.0
    caps = lambda_value * chain.weights
    if isinstance(start, WarmState):
        start.validate(chain)
        x = np.clip(start.x, lower, upper)
        q = np.clip(start.dual, -caps, caps).copy()
    else:
        x = np.clip(start, lower, upper)
        q = np.zeros(len(model) - 1)
    total_inner, accepted, backtracks = 0, 0, 0
    status, residual, inner_gap = "outer_iteration_limit", np.inf, np.inf
    accepted_surrogate_gap = np.inf
    qualified = False
    current = objective(model, x, caps)
    last_inner_qualified = False
    last_gap_qualified = last_kkt_qualified = False
    inner_kkt, inner_scale, inner_seconds = np.inf, 0.0, 0.0
    inner_algorithm, inner_work = "not_attempted", {}
    kink_seconds = 0.0
    for outer in range(policy.outer_max_iterations):
        terms = evaluate(model, x, derivatives=True)
        h = np.maximum(terms.curvature, 1.0)
        successful = False
        for attempt in range(policy.max_backtracks):
            target = x - terms.gradient / h
            inner_started = perf_counter()
            inner = solve_quadratic(h, target, lower, upper, caps, x, policy, dual=q)
            inner_seconds += perf_counter() - inner_started
            total_inner += inner.iterations
            inner_gap = inner.gap
            last_inner_qualified = inner.qualified
            inner_kkt, inner_scale = inner.kkt_residual, inner.gap_scale
            inner_algorithm, inner_work = inner.algorithm, inner.work
            last_gap_qualified = np.isfinite(inner.gap) and inner.gap <= policy.inner_atol + policy.inner_rtol * inner.gap_scale
            last_kkt_qualified = inner.kkt_residual <= policy.inner_kkt_tol
            if not inner.qualified:
                status = "inner_qualification_unresolved"
                break
            trial = inner.x
            breakpoint_trial = _snap_crossed_breakpoints(model, x, trial, lower, upper, caps)
            # Breakpoint steps are valid inexact MM steps when they reduce Q.
            # Keep the QP's own qualified gap distinct from this accepted step.
            if not np.array_equal(breakpoint_trial, trial):
                trial = breakpoint_trial
            polished = _snap_fusions(trial, lower, upper, min(1e-8, policy.fusion_tol / 10))
            polish_gap, polish_scale = quadratic_gap(polished, inner.dual, h, target, lower, upper, caps)
            if (polish_gap <= policy.inner_atol + policy.inner_rtol * polish_scale and
                    quadratic_kkt(polished, inner.dual, h, target, lower, upper, caps) <= policy.inner_kkt_tol and
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
        kink_started = perf_counter()
        kink_ok, restart = _kink_check(model, x, caps, lower, upper, policy)
        kink_seconds += perf_counter() - kink_started
        if restart is not None:
            x = restart
            current = objective(model, x, caps)
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
                  {"status": status, "inner_gap_qualified": bool(last_gap_qualified),
                   "inner_kkt_qualified": bool(last_kkt_qualified), "inner_kkt_residual": inner_kkt,
                   "inner_gap_scale": inner_scale,
                   "inner_algorithm": inner_algorithm, "last_inner_work": inner_work,
                   "inner_gap": float(inner_gap), "raw_branch_stationarity_qualified": qualified,
                   "accepted_surrogate_gap": accepted_surrogate_gap,
                   "stationarity_residual": residual, "clonal_feasible": feasible and x[witness_index] == 1,
                   "witness_search_complete": False, "global_optimality_proven": False,
                   "outer_iterations": outer + 1, "accepted_steps": accepted,
                   "inner_iterations": total_inner, "backtracks": backtracks,
                   "inner_solve_seconds": inner_seconds, "kink_check_seconds": kink_seconds})
