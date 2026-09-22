"""Safeguarded likelihood majorization with a direct bounded chain-TV QP solver."""

import hashlib
import math
from dataclasses import dataclass, field
import numpy as np
from time import perf_counter

from .chain import adjoint, difference
from .intervals import interval_descent
from .model import clipping_breakpoints, evaluate, loss, one_sided_derivatives
from .policy import Policy
from .proposals import FiniteProposal, ProposalLossMemo, iter_proposal_deltas
from .tv import forward_messages, polish_blocks, reconstruct_at_witness, reconstruct_dual, split_primal
from .types import InnerFit, QuadraticWitnessProfile, RawFit, WarmState, PrimalWarmState, readonly


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
    # Interior boxed minimizers have exactly zero normal. Computing a
    # cancelled gradient there can invent a negative normal and an infinite gap.
    unconstrained = uf - af / hf
    normal_gradient = np.where(unconstrained < lower[free], hf * (lower[free] - uf) + af,
                               np.where(unconstrained > upper[free], hf * (upper[free] - uf) + af, 0.0))
    normal = normal_gradient * delta
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


def finalize_quadratic(h, target, lower, upper, caps, x, policy=Policy(), *, work=None):
    """Polish and independently certify an already reconstructed direct primal.

    Inputs are the validated quadratic arrays from the direct solve or common
    profile. This shared finalization performs no message pass. Instrumentation
    can wrap this function to observe both reconstruction paths and their exact
    h/target/box/cap arrays before the unchanged gap and KKT admission gates.
    """
    work = {} if work is None else dict(work)
    work.update(finalization="polish_dual_gap_kkt", finalization_count=1)
    q = np.zeros(len(h) - 1)
    try:
        if np.any(caps):
            x = polish_blocks(x, h, target, lower, upper, caps)
            q = reconstruct_dual(x, h, target, lower, upper, caps)
        gap, scale = quadratic_gap(x, q, h, target, lower, upper, caps)
        kkt = quadratic_kkt(x, q, h, target, lower, upper, caps)
        qualified = bool(np.isfinite(gap) and gap <= policy.inner_atol + policy.inner_rtol * scale and
                         kkt <= policy.inner_kkt_tol)
    except ArithmeticError as exc:
        gap, scale, kkt, qualified = np.inf, 0., np.inf, False
        work["failure_reason"] = str(exc)
    if not qualified and "failure_reason" not in work:
        work["failure_reason"] = ("quadratic_gap_gate" if not np.isfinite(gap) or
                                  gap > policy.inner_atol + policy.inner_rtol * scale else "quadratic_kkt_gate")
    return InnerFit(x, q, gap, qualified, 1, kkt, scale, "bounded_weighted_tv_dp", work)


def solve_quadratic(h, target, lower, upper, caps, start, policy=Policy(), dual=None):
    """Direct bounded weighted TV; independent stable gap and O(M) KKT certificate.

    The minimizer of this strictly convex problem does not depend on a warm start.
    Starts remain accepted/validated for API parity and outer primal continuation.
    Arithmetic or certificate failure returns unresolved, without an iterative fallback.
    """
    # Validate a supplied legacy dual without copying/projecting an unused start.
    if dual is not None:
        supplied = np.asarray(dual)
        if supplied.shape != (len(h) - 1,) or not np.all(np.isfinite(supplied)):
            raise ValueError("Dual start must be a finite chain-edge vector")
    h, target, lower, upper, caps, x, q = _quadratic_inputs(h, target, lower, upper, caps, start, None)
    work = {"reconstruction_path": "direct_message_solve"}
    try:
        if not np.any(caps):
            x = np.clip(target, lower, upper)
            work["reconstruction_path"] = "separable_projection"
        else:
            x, stats = split_primal(h, target, lower, upper, caps)
            work.update(stats)
    except ArithmeticError as exc:
        work["failure_reason"] = str(exc)
        return InnerFit(x, q, np.inf, False, 1, np.inf, 0., "bounded_weighted_tv_dp", work)
    return finalize_quadratic(h, target, lower, upper, caps, x, policy, work=work)


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

    Production rebuilds these messages at each common-surrogate outer/backtrack
    step. Messages are never shared across differing h/target arrays.
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
    forward_low, forward_high, _, prefix, forward_stats = forward_messages(
        h, target, lower, upper, caps, values_at_one=True)
    reverse_low, reverse_high, _, suffix, reverse_stats = forward_messages(
        h[::-1], target[::-1], lower[::-1], upper[::-1], caps[::-1], values_at_one=True)
    reference = np.clip(target, lower, upper)
    unary_at_one = .5 * h * (1 - reference)**2 + h * (reference - target) * (1 - reference)
    values = prefix + suffix[::-1] - unary_at_one
    witness = int(np.argmin(values))
    fixed_lower, fixed_upper = lower.copy(), upper.copy()
    fixed_lower[witness] = fixed_upper[witness] = 1.
    primal = reconstruct_at_witness(forward_low, forward_high, reverse_low, reverse_high, witness)
    fit = finalize_quadratic(h, target, fixed_lower, fixed_upper, caps, primal, policy,
                             work={"reconstruction_path": "stored_profile_thresholds",
                                   "threshold_reconstruction_steps": len(h) - 1,
                                   "message_passes_reused": 2, "message_passes_rebuilt": 0})
    displacement = fit.x - reference
    attained = math.fsum(.5 * h * displacement**2 + h * (reference - target) * displacement)
    attained += float(np.dot(caps, np.abs(difference(fit.x))))
    error = abs(attained - values[witness])
    tolerance = policy.inner_atol + policy.inner_rtol * abs(attained)
    eligible = upper == 1
    return QuadraticWitnessProfile(witness, fit, values, math.fsum(.5 * h * (reference - target)**2),
                                   bool(fit.qualified and np.all(np.isfinite(values[eligible])) and error <= tolerance),
                                   digest.hexdigest(), dict(scope="one_common_quadratic", reconstruction_count=1,
                                   reconstruction_path="stored_profile_thresholds", message_passes=2,
                                   prefix_value_error=float(error), forward=forward_stats, reverse=reverse_stats))


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


@dataclass(frozen=True)
class AuditContext:
    """Immutable likelihood quantities with a bounded derived-loss memo.

    Numeric audit data and model/x identity are immutable. The separate memo
    stores only repeatable proposal likelihood deltas, never feasibility or TV.
    """

    model: object
    x: np.ndarray
    losses: np.ndarray
    posterior: np.ndarray
    gradient: np.ndarray
    left: np.ndarray
    right: np.ndarray
    at_kink: np.ndarray
    plateau: np.ndarray
    cuts: np.ndarray
    proposal_losses: ProposalLossMemo = field(default_factory=ProposalLossMemo,
                                             init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("x", "losses", "posterior", "gradient", "left", "right", "at_kink", "plateau", "cuts"):
            object.__setattr__(self, name, readonly(getattr(self, name)))

    def validate(self, model, x):
        if model is not self.model or not np.array_equal(x, self.x):
            raise ValueError("Audit context does not match the model and exact primal vector")


def prepare_audit(model, x, metrics=None):
    started = perf_counter()
    terms = evaluate(model, x, derivatives=True)
    left, right = one_sided_derivatives(model, x, posterior=terms.posterior)
    mass = model.slope * x[:, None]
    slopes = np.where(model.valid, model.slope, np.nan)
    at_kink = np.any((x[:, None] == model.eps / slopes) |
                     (x[:, None] == (1 - model.eps) / slopes), axis=1)
    plateau = np.any(model.valid & (((mass <= model.eps) & (model.alt[:, None] > 0)) |
                                    ((mass >= 1 - model.eps) & (model.ref[:, None] > 0))), axis=1)
    cuts = np.r_[0, np.flatnonzero(difference(x) != 0) + 1, len(x)]
    context = AuditContext(model, x, terms.loss, terms.posterior, terms.gradient, left, right,
                           at_kink, plateau, cuts)
    if metrics is not None:
        metrics["audit_context_count"] += 1
        metrics["audit_context_seconds"] += perf_counter() - started
    return context


def stationarity(model, x, q, lower, upper, caps, *, context=None):
    context = prepare_audit(model, x) if context is None else context
    context.validate(model, x)
    a = adjoint(q)
    left, right = context.left, context.right
    # At an upward clipping kink, use a valid subgradient from [left,right].
    # A downward kink still needs the explicit directional checks below.
    gradient = np.where(left <= right, np.clip(-a, left, right), context.gradient)
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


def _snap_crossed_breakpoints(model, previous, trial, lower, upper, caps, cache=None):
    """Keep exact clipping candidates when a smooth surrogate crosses a kink.

    Only objective-improving proposals survive. The outer iteration still checks
    both descent inequalities; the QP solution retains its separate gap.
    """
    slopes = np.where(model.valid, model.slope, np.nan)
    result = trial
    def value_at(values):
        loss = float(np.sum(evaluate(model, values).loss))
        return loss, loss + float(np.dot(caps, np.abs(difference(values))))

    loss, value = value_at(trial)
    for threshold in (model.eps, 1 - model.eps):
        points = threshold / slopes
        for c in range(points.shape[1]):
            point = points[:, c]
            crossed = ((point >= np.minimum(previous, trial)) & (point <= np.maximum(previous, trial)) &
                       (point >= lower) & (point <= upper))
            if np.any(crossed):
                proposal = np.where(crossed, point, result)
                if np.array_equal(proposal, result):
                    continue
                candidate_loss, candidate_value = value_at(proposal)
                if candidate_value < value:
                    result, loss, value = proposal, candidate_loss, candidate_value
    if cache is not None:
        cache.update(loss=loss, objective=value)
    return result


def local_interval_delta(block, x, caps, start, stop, new_values, old_losses=None):
    """Change in objective using only affected likelihoods and incident edges."""
    old = x[start:stop]
    new = np.broadcast_to(np.asarray(new_values, dtype=float), old.shape)
    if old_losses is None:
        old_losses = loss(block, old)
    new_losses = loss(block, new)
    delta = float(np.sum(new_losses - old_losses))
    delta += float(np.dot(caps[start:stop - 1], np.abs(difference(new)) - np.abs(difference(old))))
    if start:
        delta += caps[start - 1] * (abs(new[0] - x[start - 1]) - abs(old[0] - x[start - 1]))
    if stop < len(x):
        delta += caps[stop - 1] * (abs(x[stop] - new[-1]) - abs(x[stop] - old[-1]))
    return delta


def _kink_check(model, x, caps, lower, upper, policy, *, force_intervals=False,
                context=None, metrics=None):
    """Audit every signed fused subinterval; screen finite plateau moves locally."""
    context = prepare_audit(model, x, metrics) if context is None else context
    context.validate(model, x)
    if metrics is not None:
        metrics["audit_anchor_count"] += 1
    if not force_intervals and not np.any((context.at_kink | context.plateau) & (lower < upper)):
        return True, None
    started = perf_counter()
    direction = interval_descent(x, context.left, context.right, lower, upper, caps, policy.stationarity_tol)
    if metrics is not None:
        metrics["audit_scan_count"] += 1
        metrics["audit_scan_seconds"] += perf_counter() - started
    started = perf_counter()
    try:
        return _finite_interval_check(model, x, caps, lower, upper, context, direction)
    finally:
        if metrics is not None:
            metrics["audit_finite_search_seconds"] += perf_counter() - started


def _finite_proposals(model, x, lower, upper, context, direction):
    """The original interval/point/offset order, independent of batching."""
    if direction is not None:
        start, stop, sign, derivative, _ = direction
        old_losses = context.losses[start:stop]
        margin = 128 * np.finfo(float).eps * (1 + float(np.sum(np.abs(old_losses))))
        room = float(np.min(upper[start:stop] - x[start:stop]) if sign > 0 else
                     np.min(x[start:stop] - lower[start:stop]))
        step = min(1e-3, room)
        for _ in range(48):
            values = x[start:stop] + sign * step
            yield FiniteProposal(start, stop, values, margin, step)
            step *= .5
        return

    affected = context.plateau & (lower < upper)
    if not np.any(affected):
        return
    cuts = context.cuts
    intervals = [(i, i + 1) for i in np.flatnonzero(affected)]
    intervals.extend((int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])
                     if b - a > 1 and np.any(affected[a:b]))
    for start, stop in intervals:
        lo, hi = float(np.max(lower[start:stop])), float(np.min(upper[start:stop]))
        if lo >= hi:
            continue
        block = model.subset(np.arange(start, stop))
        old_losses = context.losses[start:stop]
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
                yield FiniteProposal(start, stop, value, margin)


def _finite_interval_check(model, x, caps, lower, upper, context, direction):
    context.validate(model, x)
    proposals = _finite_proposals(model, x, lower, upper, context, direction)
    options = {"initial_batch_size": 1} if direction is not None else {}
    for proposal, delta in iter_proposal_deltas(context, proposals, **options):
        start, stop = proposal.start, proposal.stop
        old = x[start:stop]
        new = np.broadcast_to(np.asarray(proposal.value, dtype=float), old.shape)
        # Preserve the scalar penalty operations and their addition order. All
        # generated intervals are fused, but this also handles general vectors.
        delta += float(np.dot(caps[start:stop - 1], np.abs(difference(new)) - np.abs(difference(old))))
        if start:
            delta += caps[start - 1] * (abs(new[0] - x[start - 1]) - abs(old[0] - x[start - 1]))
        if stop < len(x):
            delta += caps[stop - 1] * (abs(x[stop] - new[-1]) - abs(x[stop] - old[-1]))
        if delta < -proposal.margin and (direction is None or
                                        delta <= 1e-4 * proposal.step * direction[3]):
            trial = x.copy()
            trial[start:stop] = new
            return False, trial
    return direction is None, None


def solve_branch(model, chain, lambda_value, witness_index, start, policy=Policy()):
    """Fixed-witness nonlinear solver for the offline enumeration reference."""
    if not 0 <= witness_index < len(model) or model.upper[witness_index] != 1:
        raise ValueError("Witness must be eligible under the original float64 box")
    return _solve_outer(model, chain, lambda_value, start, policy, witness_index)


def solve_profiled(model, chain, lambda_value, start, policy=Policy()):
    """Common-surrogate MM over the union of all eligible witness constraints."""
    return _solve_outer(model, chain, lambda_value, start, policy, None)


def _solve_outer(model, chain, lambda_value, start, policy, fixed_witness):
    if not np.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and nonnegative")
    lower, upper = model.lower.copy(), model.upper.copy()
    if fixed_witness is not None:
        lower[fixed_witness] = upper[fixed_witness] = 1.
    caps = lambda_value * chain.weights
    if isinstance(start, (WarmState, PrimalWarmState)):
        start.validate(chain)
        start = start.x
    x = np.clip(np.asarray(start, dtype=float), lower, upper)
    if x.shape != lower.shape or not np.all(np.isfinite(x)) or not np.any(x == 1):
        raise ValueError("Outer start must be finite, correctly shaped and clonal feasible")
    witness_index = int(np.flatnonzero(x == 1)[0]) if fixed_witness is None else fixed_witness
    q = np.zeros(len(model) - 1)
    total_inner = accepted = backtracks = profile_calls = witness_switches = reused = 0
    status, residual, inner_gap = "outer_iteration_limit", np.inf, np.inf
    accepted_surrogate_gap = np.inf
    qualified = False
    current = objective(model, x, caps)
    initial_objective = current
    audit_context = None
    audit_metrics = dict.fromkeys(("audit_context_count", "audit_anchor_count", "audit_scan_count"), 0)
    audit_metrics.update(dict.fromkeys(("audit_context_seconds", "audit_scan_seconds",
                                       "audit_finite_search_seconds"), 0.0))
    restart_lengths, restart_decreases = [], []
    restart_witness_switches = 0
    surrogate_decrease = 0.0
    progress_tail = []
    last_inner_qualified = last_gap_qualified = last_kkt_qualified = False
    inner_kkt, inner_scale, inner_seconds = np.inf, 0.0, 0.0
    inner_algorithm, inner_work, profile_work = "not_attempted", {}, {}
    kink_seconds = 0.0
    curvature_scale = 1.0
    largest_curvature_scale = 1.0
    audit_lower, audit_upper = lower.copy(), upper.copy()
    audit_lower[witness_index] = audit_upper[witness_index] = 1.
    for outer in range(policy.outer_max_iterations):
        terms = evaluate(model, x, derivatives=True)
        # A common profile is relatively expensive. Reuse only the last accepted
        # scalar inflation, trying half of it at the next iterate; the new h,
        # target and both messages are always rebuilt and all gates still apply.
        trial_scale = max(1.0, curvature_scale / 2) if fixed_witness is None else 1.0
        h = np.maximum(terms.curvature, 1.0) * trial_scale
        successful = False
        for attempt in range(policy.max_backtracks):
            target = x - terms.gradient / h
            inner_started = perf_counter()
            if fixed_witness is None:
                profile_calls += 1
                try:
                    # Original boxes release the previous witness. Every new
                    # curvature/target gets fresh prefix and suffix messages.
                    profile = profile_quadratic_witnesses(h, target, model.lower, model.upper, caps, policy)
                except ArithmeticError as exc:
                    inner_seconds += perf_counter() - inner_started
                    inner_work = {"failure_reason": str(exc), "stage": "witness_profile"}
                    status = "inner_qualification_unresolved"
                    last_inner_qualified = last_gap_qualified = last_kkt_qualified = False
                    inner_gap = inner_kkt = np.inf
                    break
                inner, selected = profile.fit, profile.witness
                profile_work = dict(profile.diagnostics, surrogate_sha256=profile.surrogate_sha256)
                last_inner_qualified = profile.qualified
                if not profile.qualified and inner.qualified:
                    inner.work["failure_reason"] = "witness_profile_value_gate"
            else:
                selected = fixed_witness
                inner = solve_quadratic(h, target, lower, upper, caps, x, policy)
                last_inner_qualified = inner.qualified
            inner_seconds += perf_counter() - inner_started
            total_inner += inner.iterations
            inner_gap, inner_kkt, inner_scale = inner.gap, inner.kkt_residual, inner.gap_scale
            inner_algorithm, inner_work = inner.algorithm, inner.work
            last_gap_qualified = np.isfinite(inner.gap) and inner.gap <= policy.inner_atol + policy.inner_rtol * inner.gap_scale
            last_kkt_qualified = inner.kkt_residual <= policy.inner_kkt_tol
            if not last_inner_qualified:
                status = "inner_qualification_unresolved"
                break
            trial_lower, trial_upper = model.lower.copy(), model.upper.copy()
            trial_lower[selected] = trial_upper[selected] = 1.
            cache = {}
            trial = _snap_crossed_breakpoints(model, x, inner.x, trial_lower, trial_upper, caps, cache)
            unchanged = np.array_equal(trial, inner.x)
            trial_gap = inner.gap if unchanged else None
            polished = _snap_fusions(trial, trial_lower, trial_upper, min(1e-8, policy.fusion_tol / 10))
            if not np.array_equal(polished, trial):
                polish_gap, polish_scale = quadratic_gap(polished, inner.dual, h, target, trial_lower, trial_upper, caps)
                if (polish_gap <= policy.inner_atol + policy.inner_rtol * polish_scale and
                        quadratic_kkt(polished, inner.dual, h, target, trial_lower, trial_upper, caps) <= policy.inner_kkt_tol):
                    polish_loss = float(np.sum(evaluate(model, polished).loss))
                    polish_value = polish_loss + float(np.dot(caps, np.abs(difference(polished))))
                    if polish_value <= cache["objective"] + 1e-12 * (1 + current):
                        trial, trial_gap = polished, polish_gap
                        cache.update(loss=polish_loss, objective=polish_value)
            if np.array_equal(trial, inner.x):
                trial_gap = inner.gap
                reused += 1
            step = trial - x
            surrogate = float(np.sum(terms.loss) + np.dot(terms.gradient, step) + .5 * np.dot(h, step**2))
            trial_tv = float(np.dot(caps, np.abs(difference(trial))))
            margin = 128 * np.finfo(float).eps * (1 + abs(current))
            # Cached observed likelihood, majorization and surrogate descent are
            # distinct from the reused direct-QP certificate and remain mandatory.
            if (cache["loss"] <= surrogate + margin and cache["objective"] <= current + margin and
                    surrogate + trial_tv <= current + margin):
                surrogate_decrease += current - cache["objective"]
                x, q, current = trial, inner.dual, cache["objective"]
                audit_context = None
                audit_lower, audit_upper = trial_lower, trial_upper
                witness_switches += int(selected != witness_index)
                witness_index = selected
                if trial_gap is None:
                    trial_gap = quadratic_gap(x, q, h, target, audit_lower, audit_upper, caps)[0]
                accepted_surrogate_gap = max(0.0, trial_gap)
                successful = True
                accepted += 1
                curvature_scale = trial_scale
                largest_curvature_scale = max(largest_curvature_scale, trial_scale)
                break
            h *= 2
            trial_scale *= 2
            backtracks += 1
        if not successful:
            if last_inner_qualified:
                status = "majorization_unresolved"
            break
        audit_context = prepare_audit(model, x, audit_metrics)
        residual, feasible = stationarity(model, x, q, audit_lower, audit_upper, caps, context=audit_context)
        kink_started = perf_counter()
        kink_ok, restart = _kink_check(model, x, caps, audit_lower, audit_upper, policy,
                                       context=audit_context, metrics=audit_metrics)
        if fixed_witness is None and restart is None and kink_ok:
            # Any clonal-feasible interval excludes at least one extreme occupied
            # witness. Two endpoint anchors cover these interval directions.
            occupied = np.flatnonzero(x == 1)
            for anchor in dict.fromkeys((int(occupied[0]), int(occupied[-1]))):
                if anchor == witness_index:
                    continue
                check_lower, check_upper = model.lower.copy(), model.upper.copy()
                check_lower[anchor] = check_upper[anchor] = 1.
                kink_ok, restart = _kink_check(model, x, caps, check_lower, check_upper, policy,
                                               force_intervals=True, context=audit_context, metrics=audit_metrics)
                if restart is not None or not kink_ok:
                    break
        kink_seconds += perf_counter() - kink_started
        progress_tail.append({"outer_iteration": outer + 1, "objective": current,
                              "curvature_scale": curvature_scale, "witness": witness_index,
                              "stationarity_residual": residual, "interval_restart": restart is not None})
        progress_tail = progress_tail[-8:]
        if restart is not None:
            restart_lengths.append(float(np.max(np.abs(restart - x))))
            x = restart
            previous_witness = witness_index
            witness_index = int(np.flatnonzero(x == 1)[0]) if fixed_witness is None else fixed_witness
            restart_witness_switches += int(witness_index != previous_witness)
            audit_lower, audit_upper = model.lower.copy(), model.upper.copy()
            audit_lower[witness_index] = audit_upper[witness_index] = 1.
            previous_objective = current
            current = objective(model, x, caps)
            restart_decreases.append(previous_objective - current)
            audit_context = None
            curvature_scale = 1.0
            continue
        if feasible and residual <= policy.stationarity_tol and kink_ok:
            status, qualified = "qualified", True
            break
        if not kink_ok:
            status = "clipping_kink_unresolved"
            break
        if not np.any(step):
            status = "stationarity_unresolved"
            break
    if audit_context is None:
        audit_context = prepare_audit(model, x, audit_metrics)
    residual, feasible = stationarity(model, x, q, audit_lower, audit_upper, caps, context=audit_context)
    return RawFit(x, q, current, witness_index, qualified,
                  {"status": status, "inner_gap_qualified": bool(last_gap_qualified),
                   "inner_kkt_qualified": bool(last_kkt_qualified), "inner_kkt_residual": inner_kkt,
                   "inner_gap_scale": inner_scale, "inner_algorithm": inner_algorithm,
                   "last_inner_work": inner_work, "last_profile_work": profile_work,
                   "failure_reason": inner_work.get("failure_reason", status) if not qualified else None,
                   "inner_gap": float(inner_gap), "raw_branch_stationarity_qualified": qualified,
                   "accepted_surrogate_gap": accepted_surrogate_gap,
                   "stationarity_residual": residual, "clonal_feasible": feasible and x[witness_index] == 1,
                   "global_optimality_proven": False, "profile_calls": profile_calls,
                   "surrogate_witnesses_profiled": profile_calls * int(np.sum(model.upper == 1)),
                   "witness_switches": witness_switches, "inner_certificates_reused": reused,
                   "restart_witness_switches": restart_witness_switches,
                   "interval_restart_count": len(restart_lengths),
                   "interval_restart_length_sum": math.fsum(restart_lengths),
                   "interval_restart_length_min": min(restart_lengths, default=0.0),
                   "interval_restart_length_max": max(restart_lengths, default=0.0),
                   "interval_restart_objective_decrease": math.fsum(restart_decreases),
                   "surrogate_objective_decrease": surrogate_decrease,
                   "initial_objective": initial_objective, "objective_decrease": initial_objective - current,
                   "progress_tail": progress_tail, **audit_metrics,
                   "largest_curvature_scale": largest_curvature_scale,
                   "outer_iterations": outer + 1, "accepted_steps": accepted,
                   "inner_iterations": total_inner, "backtracks": backtracks,
                   "inner_solve_seconds": inner_seconds, "kink_check_seconds": kink_seconds})
