"""Bounded numerical recovery for the original complete-graph quadratic problem.

Ordering and component cuts propose candidates; only the unchanged objective,
box feasibility and full primal-dual gap/KKT gates can qualify one. The helper
performs no ordinary ADMM steps and has no model, sample or source identity rules.
"""
from dataclasses import asdict, dataclass

import torch

from . import kernels as tensor_kernels


@dataclass(frozen=True)
class PrecisionLimits:
    rounds: int = 32
    coordinate_minimizations: int = 2048
    collective_reorderings: int = 8
    cut_steps_per_sign: int = 256
    projected_flow_steps: int = 1024

    def __post_init__(self):
        ceilings = (32, 2048, 8, 256, 1024)
        for (name, value), ceiling in zip(asdict(self).items(), ceilings, strict=True):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be a positive integer no larger than {ceiling}")


@dataclass
class _Work:
    pava_rounds: int = 0
    coordinate_minimizations: int = 0
    collective_calls: int = 0
    collective_reorderings: int = 0
    cut_steps: int = 0
    projected_flow_steps: int = 0
    momentum_restarts: int = 0
    objective_evaluations: int = 0
    certificate_evaluations: int = 0
    eager_certificate_evaluations: int = 0
    compiled_certificate_evaluations: int = 0


@dataclass(frozen=True)
class PrecisionResult:
    x: torch.Tensor
    dual: torch.Tensor
    stats: torch.Tensor
    qualified: bool
    work: dict


def ordered_candidate(x, problem, order=None):
    """CUDA-tensor bounded PAVA in the original stable total-order chamber.

    No initial equality is imposed, including at exact original ties. Stable
    node-index tie order defines one valid chamber; the original full QP
    certificate, never this chamber's constrained optimum, admits a solution.
    """
    h, target, lower, upper, caps = problem
    n = x.numel()
    if order is None:
        order = torch.argsort(x, stable=True)
    valid = (order.dtype == torch.int64 and order.device == x.device and order.shape == x.shape
             and torch.equal(torch.sort(order).values, torch.arange(n, device=x.device)))
    if not valid or not bool((x[order][1:] >= x[order][:-1]).all()):
        raise ValueError('Order must be a permutation preserving every strict original x ordering')
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n, device=x.device)
    # Every unordered edge has the sign of its positions in the total order.
    # At original ties this chooses a chamber; it does not assert a true order.
    external = tensor_kernels.adjoint(caps * (rank[None, :] - rank[:, None]).sign())
    safe = torch.where(lower < upper, target, lower)
    centered = h * (safe - x) - external
    stack = []
    merges = 0
    for position in range(n):
        node = order[position]
        block = dict(first=position, last=position, w=h[node], anchor=x[node],
                     d=centered[node], lo=lower[node], hi=upper[node])
        block['center'] = (block['anchor'] + block['d'] / block['w']).clamp(block['lo'], block['hi'])
        stack.append(block)
        while len(stack) > 1 and bool(stack[-2]['center'] > stack[-1]['center']):
            right, left = stack.pop(), stack.pop()
            joined = dict(first=left['first'], last=right['last'], w=left['w'] + right['w'],
                          anchor=left['anchor'],
                          d=left['d'] + right['d'] + right['w'] * (right['anchor'] - left['anchor']),
                          lo=torch.maximum(left['lo'], right['lo']), hi=torch.minimum(left['hi'], right['hi']))
            if not bool(joined['lo'] <= joined['hi']):
                return None, dict(status='infeasible_pooled_block', merges=merges)
            joined['center'] = (joined['anchor'] + joined['d'] / joined['w']).clamp(joined['lo'], joined['hi'])
            stack.append(joined)
            merges += 1
    candidate = torch.empty_like(x)
    for block in stack:
        candidate[order[block['first']:block['last'] + 1]] = block['center']
    return candidate, dict(status='candidate_only', initial_singletons=n, pooled_blocks=len(stack), merges=merges,
        original_order_rule='Explicit permutation preserves strict current x order; only literal ties may reorder',
        original_exact_tie_adjacent_pairs=int((x[order][1:] == x[order][:-1]).sum()),
        arithmetic='CUDA float64 centered weighted sums; Python stack controls only; no CPU numeric solve',
        limit='Exactly N initial singleton blocks; at most N-1 adjacent merges')

def direct_derivative(x, direction, problem):
    """Original one-sided derivative, independently of cut normalization.

    The complete symmetric edge tensor counts every unordered edge twice;
    therefore both the signed external term and fused absolute term use .5.
    """
    h, target, lower, upper, caps = problem
    dx = tensor_kernels.differences(x)
    dd = tensor_kernels.differences(direction)
    safe = torch.where(lower < upper, target, lower)
    gradient = h * (x - safe)
    node_terms = gradient * direction
    external_terms = torch.where(dx != 0, caps * dx.sign() * dd, 0.)
    fused_terms = torch.where(dx == 0, caps * dd.abs(), 0.)
    value = node_terms.sum() + .5 * external_terms.sum() + .5 * fused_terms.sum()
    # Include the h*(x-target) subtraction/product error separately, which
    # matters for the very stiff near-zero coordinates.
    gradient_scale = (h * (x.abs() + safe.abs()) * direction.abs()).sum()
    scale = (1 + gradient_scale + node_terms.abs().sum()
             + .5 * external_terms.abs().sum() + .5 * fused_terms.sum())
    margin = 8 * (x.numel() + 2) * torch.finfo(x.dtype).eps * scale
    feasible = (torch.isfinite(direction).all() & (x >= lower).all() & (x <= upper).all()
                & ((x > lower) | (direction >= 0)).all()
                & ((x < upper) | (direction <= 0)).all())
    valid = feasible & torch.isfinite(value) & torch.isfinite(margin)
    return value, margin, dict(original_derivative=float(value), arithmetic_margin=float(margin),
        original_node_term=float(node_terms.sum()), original_external_term=float(.5 * external_terms.sum()),
        original_fused_term=float(.5 * fused_terms.sum()), box_tangent_feasible=bool(feasible),
        finite_original_derivative=bool(valid))

def attained_order(x, direction):
    """Lexicographic (x, direction, node index); never perturb numerical x."""
    order = torch.argsort(direction, stable=True)
    order = order[torch.argsort(x[order], stable=True)]
    if (not torch.equal(torch.sort(order).values, torch.arange(x.numel(), device=x.device))
            or not bool((x[order][1:] >= x[order][:-1]).all())):
        raise ArithmeticError('Attained-direction order violated original strict x order')
    return order

def component_geometry(x, linear, fc):
    """Exact equality components of the current x, without changing x or caps.

    Strictly unequal x endpoints have zero cut capacity. The cut therefore
    decomposes into independent literal-equality blocks. Scaling each block
    by its own positive scalar leaves its minimizers unchanged, while its
    complete incidence operator has norm squared at most its member count.
    """
    _, labels, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    groups = counts.numel()
    row_magnitude = torch.maximum(linear.abs(), fc.sum(1))
    scales = torch.zeros(groups, dtype=x.dtype, device=x.device)
    scales.scatter_reduce_(0, labels, row_magnitude, reduce='amax', include_self=True)
    scales.clamp_min_(1.)
    node_scales = scales[labels]
    sigma = (1. / counts[labels].to(x.dtype))[:, None]
    if not bool(torch.isfinite(scales).all()):
        raise ArithmeticError('Nonfinite component normalization')
    return labels, counts, node_scales, sigma

def component_direction(x, q, problem, kernels, policy, sign, limits, work):
    """Bounded CUDA proposal, with no lower-bound or stationarity claim.

    Each block uses the unchanged primal-dual iteration with tau=.9 and
    sigma=1/block_size. A blockwise positive normalization transforms both
    unary and edge coefficients. Only same-block edges are nonzero, so
    sigma and scale agree at their two endpoints and skew symmetry remains.
    Negative-value blocks may provide a combined candidate direction; the
    *full original QP derivative* and its original-unit arithmetic margin
    admit it independently. Failure to find descent remains unresolved.
    """
    h, target, lower, upper, caps = problem
    dx = tensor_kernels.differences(x)
    same = dx == 0
    gradient = torch.where(lower < upper, h * (x - target), 0.)
    external = tensor_kernels.adjoint(torch.where(same, 0., caps * dx.sign()))
    allowed = ((x < upper) if sign > 0 else (x > lower)).to(x.dtype)
    moving = (allowed[:, None] > 0) | (allowed[None, :] > 0)
    fc = torch.where(same & moving, caps, 0.)
    linear = torch.where(allowed > 0, sign * (gradient + external), 0.)
    labels, counts, scales, sigma = component_geometry(x, linear, fc)
    a, cc = linear / scales, fc / scales[:, None]
    dual = torch.where(same & moving, sign * q / scales[:, None], 0.)
    dual = torch.maximum(-cc, torch.minimum(cc, dual))
    dual = .5 * dual - .5 * dual.T
    # Exactly separable zero-capacity blocks have their analytical box proposal.
    separable_rows = fc.sum(1) == 0
    t = torch.where(separable_rows & (a < 0), allowed, 0.)
    extrapolated = t.clone()
    trace = []
    candidate, check = None, None
    for iteration in range(limits.cut_steps_per_sign + 1):
        if iteration % policy.check_every == 0 or iteration == limits.cut_steps_per_sign:
            edges = cc * tensor_kernels.differences(t).abs()
            row_values = a * t + .5 * edges.sum(1)
            # Row reductions use the same fixed-order device reduction as the
            # full graph; avoid atomic scatter sums for candidate selection.
            node_group_values = torch.where(same, row_values[None, :], 0.).sum(1)
            negative_components = torch.unique(labels[node_group_values < 0]).numel()
            proposal = sign * torch.where(node_group_values < 0, t, 0.)
            value, margin, check = direct_derivative(x, proposal, problem)
            valid_dual = (torch.isfinite(dual).all() & (dual.abs() <= cc).all()
                          & (dual == -dual.T).all())
            valid_primal = torch.isfinite(t).all() & (t >= 0).all() & (t <= allowed).all()
            if not bool(valid_dual & valid_primal):
                raise ArithmeticError('Component cut violated primal/dual invariants')
            accepted = (check['finite_original_derivative']
                        and bool(value + margin < -policy.inner_kkt_tol))
            trace.append(dict(iteration=iteration, negative_value_components=int(negative_components),
                original_derivative=float(value), arithmetic_margin=float(margin), accepted=accepted))
            if accepted:
                candidate = proposal
                break
        if iteration == limits.cut_steps_per_sign:
            break
        work.cut_steps += 1
        dual, t, extrapolated = kernels.directional_cut_step(
            a, cc, allowed, dual, t, extrapolated, .9, sigma)
    return candidate, dict(sign=sign, attained=candidate is not None,
        direct_descent_accepted=candidate is not None, direct_derivative=check,
        status='attained_descent' if candidate is not None else 'unresolved',
        component_count=int(counts.numel()), minimum_component_size=int(counts.min()),
        maximum_component_size=int(counts.max()), minimum_component_scale=float(scales.min()),
        maximum_component_scale=float(scales.max()), iterations=iteration, trace=trace,
        step_rule='tau=.9; sigma=1/exact_literal_component_size',
        normalization='one positive scalar per disconnected exact equality component',
        no_descent_lower_bound_qualified=False)

def collective_proposal(x, q, problem, kernels, policy, limits, work):
    work.collective_calls += 1
    details, proposals = [], []
    for sign in (1, -1):
        direction, row = component_direction(x, q, problem, kernels, policy, sign, limits, work)
        if direction is not None:
            proposals.append((row['direct_derivative']['original_derivative'], sign,
                              attained_order(x, direction), direction))
        details.append(row)
    if not proposals:
        return None, dict(directions=details, accepted=False)
    proposals.sort(key=lambda item: (item[0], -item[1]))
    value, sign, order, direction = proposals[0]
    stable = torch.argsort(x, stable=True)
    work.collective_reorderings += 1
    return order, dict(directions=details, accepted=True, chosen_sign=sign,
        chosen_original_derivative=value, changed_order_positions=int((order != stable).sum()),
        strict_original_order_preserved=True, numerical_jitter=False,
        attained_direction_min=float(direction.min()), attained_direction_max=float(direction.max()))

def coordinate_minimizer(x, node, h, target, lower, upper, caps):
    """Exact piecewise quadratic minimizer proposal; GPU float64 arithmetic.

    Original .5 dense edge sum yields row caps times |t-x_j|, with no extra
    factor of two. Original self capacity is zero. A centered derivative/root
    avoids subtracting uncentered h*target sums in high-curvature cases.
    """
    order = torch.argsort(x, stable=True)
    knots, weights = x[order], caps[node, order]
    cumulative = weights.cumsum(0)
    total = cumulative[-1]
    origin = x[node]
    gradient = h[node] * (origin - target[node])
    right_derivative = gradient + h[node] * (knots - origin) + 2 * cumulative - total
    eligible = right_derivative >= 0
    if bool(eligible.any()):
        crossing = torch.nonzero(eligible, as_tuple=True)[0][0]
        before = torch.where(crossing > 0, cumulative[(crossing - 1).clamp_min(0)], 0.)
        left = torch.where(crossing > 0, knots[(crossing - 1).clamp_min(0)], lower[node])
        right = knots[crossing]
    else:
        before = total
        left, right = knots[-1], upper[node]
    root = origin + (-gradient + total - 2 * before) / h[node]
    candidate = torch.maximum(lower[node], torch.minimum(upper[node], root.clamp(left, right)))
    delta = candidate - x
    equal = torch.where(delta == 0, caps[node], 0.).sum()
    force = (caps[node] * delta.sign()).sum()
    node_gradient = h[node] * (candidate - target[node])
    left_derivative, right_derivative = node_gradient + force - equal, node_gradient + force + equal
    # Covers centered arithmetic, original derivative subtraction/product and
    # N-term weighted prefix/force reductions. This is diagnostic only; final
    # admission remains the unchanged original complete-QP certificate.
    scale = (1 + node_gradient.abs() + gradient.abs() + (h[node] * (candidate - origin)).abs()
             + h[node] * (candidate.abs() + target[node].abs()) + caps[node].abs().sum())
    margin = 16 * (x.numel() + 2) * torch.finfo(x.dtype).eps * scale
    left_ok = (candidate == lower[node]) | (left_derivative <= margin)
    right_ok = (candidate == upper[node]) | (right_derivative >= -margin)
    valid = torch.isfinite(candidate) & (candidate >= lower[node]) & (candidate <= upper[node]) & left_ok & right_ok
    return candidate, dict(one_dimensional_bracket_qualified=bool(valid),
        left_derivative=float(left_derivative), right_derivative=float(right_derivative),
        derivative_roundoff=float(margin), displacement=float(candidate - origin))

def singleton_excess(x, node, problem):
    h, target, lower, upper, caps = problem
    if bool(lower[node] == upper[node]):
        return x.new_tensor(-float('inf'))
    delta = x[node] - x
    same = delta == 0
    gradient = h[node] * (x[node] - target[node])
    external = (caps[node] * delta.sign()).sum()
    capacity = torch.where(same, caps[node], 0.).sum()
    residual = gradient + external
    excess = torch.where(x[node] == lower[node], -residual - capacity,
        torch.where(x[node] == upper[node], residual - capacity, residual.abs() - capacity))
    margin = 8 * (x.numel() + 2) * torch.finfo(x.dtype).eps * (1 + gradient.abs() + caps[node].sum())
    return excess - margin

def violations(x, problem):
    h, target, lower, upper, caps = problem
    d = x[:, None] - x[None, :]
    same = d == 0
    gradient = h * (x - torch.where(lower < upper, target, lower))
    external = (caps * d.sign()).sum(1)
    capacity = torch.where(same, caps, 0.).sum(1)
    residual = gradient + external
    excess = torch.where(x == lower, -residual - capacity,
        torch.where(x == upper, residual - capacity, residual.abs() - capacity))
    margin = 8 * (x.numel() + 2) * torch.finfo(x.dtype).eps * (1 + gradient.abs() + caps.sum(1))
    return torch.where(lower < upper, excess - margin, -float('inf'))


def _qualified(stats, policy):
    return bool(torch.isfinite(stats).all()
        & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
        & (stats[2] <= policy.inner_kkt_tol))


def _certificate(x, dual, problem, work):
    work.certificate_evaluations += 1
    work.eager_certificate_evaluations += 1
    return tensor_kernels.gap_kkt(x, dual, *problem)


def _compiled_certificate(x, dual, problem, kernels, work):
    work.certificate_evaluations += 1
    work.compiled_certificate_evaluations += 1
    return kernels.gap_kkt(x, dual, *problem)


def _certificates_agree(eager, compiled, n):
    # Independent final gates are mandatory. This additional comparison covers
    # ordinary FP64 reduction-order error, without weakening either gate.
    margin = 128 * (n * n + 2) * torch.finfo(eager.dtype).eps * (
        1 + eager.abs() + compiled.abs())
    return bool(torch.isfinite(eager).all() & torch.isfinite(compiled).all()
                & ((eager - compiled).abs() <= margin).all())


def accelerated_flow(x, dual, problem, kernels, policy, limits, work):
    """Project every tested dual; momentum states never supply certificates."""
    from .qp import flow_polish_step, prepare_flow_polish
    stats = _certificate(x, dual, problem, work)
    if _qualified(stats, policy):
        return dual, stats
    geometry = prepare_flow_polish(x, *problem)
    extrapolated = dual
    momentum = x.new_ones(())
    for iteration in range(1, limits.projected_flow_steps + 1):
        projected = flow_polish_step(extrapolated, geometry, kernels)
        work.projected_flow_steps += 1
        restart = bool(((extrapolated - projected) * (projected - dual)).sum() > 0)
        if restart:
            work.momentum_restarts += 1
            next_momentum = x.new_ones(())
            next_extrapolated = projected
        else:
            next_momentum = (1 + torch.sqrt(1 + 4 * momentum.square())) * .5
            beta = (momentum - 1) / next_momentum
            next_extrapolated = projected + beta * (projected - dual)
        dual, extrapolated, momentum = projected, next_extrapolated, next_momentum
        if iteration == 1 or iteration % policy.check_every == 0 or iteration == limits.projected_flow_steps:
            stats = _certificate(x, dual, problem, work)
            if _qualified(stats, policy):
                break
    return dual, stats


def _validate_problem(x, dual, problem):
    h, target, lower, upper, caps = problem
    n = h.numel()
    if (n == 0 or h.ndim != 1 or x.shape != h.shape or target.shape != h.shape
            or lower.shape != h.shape or upper.shape != h.shape
            or caps.shape != (n, n) or dual.shape != caps.shape or h.dtype != torch.float64):
        raise ValueError('Invalid precision QP shapes or float64 dtype')
    if any(t.device != h.device or t.dtype != h.dtype for t in (x, dual, *problem)):
        raise ValueError('Precision QP tensors must share device and float64 dtype')
    if not bool((h > 0).all() & (lower <= upper).all() & (caps >= 0).all()
                & (caps == caps.T).all() & (caps.diagonal() == 0).all()
                & torch.isfinite(h).all() & torch.isfinite(target).all()
                & torch.isfinite(lower).all() & torch.isfinite(upper).all()
                & torch.isfinite(caps).all()):
        raise ValueError('Invalid original precision QP values')


def recover_quadratic(x, dual, h, target, lower, upper, caps, kernels, policy, *,
                      limits=PrecisionLimits()):
    """Propose a bounded repair of any finite original complete-graph QP state.

    Qualified input is allowed for callers that retain a literal accepted
    surrogate and must subsequently recheck its outer/raw gates. A rejected
    proposal returns the original state and its original qualification. Work's
    ``candidate_accepted`` distinguishes it from a newly admitted candidate.
    No ordinary ADMM solve, likelihood update or graph change occurs here.
    """
    from .qp import _objective_context, quadratic_value
    if not isinstance(limits, PrecisionLimits):
        raise ValueError('Validated precision limits required')
    problem = (h, target, lower, upper, caps)
    _validate_problem(x, dual, problem)
    work = _Work()
    original_eager = _certificate(x, dual, problem, work)
    original_stats = _compiled_certificate(x, dual, problem, kernels, work)
    original_qualified = (_qualified(original_eager, policy) and _qualified(original_stats, policy)
                          and _certificates_agree(original_eager, original_stats, x.numel()))
    if not bool(torch.isfinite(original_eager).all() & torch.isfinite(original_stats).all()):
        raise ValueError('Finite feasible original primal/dual state required')
    inputs = (x, dual, *problem)
    frozen = tuple(t.clone() for t in inputs)
    safe, reference, original_value = _objective_context(x, *problem)
    work.objective_evaluations += 1

    def objective(candidate):
        work.objective_evaluations += 1
        return quadratic_value(candidate, h, safe, caps, reference)

    def admitted(before, after):
        margin = 32 * torch.finfo(x.dtype).eps * (1 + before.abs() + after.abs())
        return bool(torch.isfinite(before) & torch.isfinite(after) & (after <= before + margin))

    current, next_order = x, None
    final_dual, final_stats = None, None
    stop = 'round_budget_exhausted'
    for round_index in range(limits.rounds):
        work.pava_rounds += 1
        candidate, _ = ordered_candidate(current, problem, next_order)
        next_order = None
        if candidate is None:
            stop = 'ordered_proposal_unavailable'
            break
        if not admitted(objective(current), objective(candidate)):
            stop = 'ordered_objective_rejected'
            break
        current = candidate
        excess = violations(current, problem)
        nodes = torch.nonzero(excess > 0, as_tuple=True)[0].tolist()
        if not nodes:
            if work.collective_reorderings < limits.collective_reorderings:
                order, _ = collective_proposal(current, dual, problem, kernels, policy, limits, work)
                if order is not None:
                    if round_index + 1 == limits.rounds:
                        stop = 'round_budget_before_collective_proposal'
                        break
                    next_order = order
                    continue
            final_dual, final_stats = accelerated_flow(
                current, dual, problem, kernels, policy, limits, work)
            stop = 'qualified' if _qualified(final_stats, policy) else 'full_certificate_unresolved'
            break
        changes = 0
        for node in nodes:
            if work.coordinate_minimizations >= limits.coordinate_minimizations:
                stop = 'coordinate_budget_exhausted'
                break
            if not bool(singleton_excess(current, node, problem) > 0):
                continue
            work.coordinate_minimizations += 1
            value, bracket = coordinate_minimizer(current, node, *problem)
            if not bracket['one_dimensional_bracket_qualified']:
                stop = 'coordinate_bracket_unresolved'
                break
            trial = current.clone()
            trial[node] = value
            if not admitted(objective(current), objective(trial)):
                stop = 'coordinate_objective_rejected'
                break
            changes += not torch.equal(trial, current)
            current = trial
        if stop != 'round_budget_exhausted':
            break
        if not changes:
            if work.collective_reorderings < limits.collective_reorderings:
                order, _ = collective_proposal(current, dual, problem, kernels, policy, limits, work)
                if order is not None and round_index + 1 < limits.rounds:
                    next_order = order
                    continue
            stop = 'no_representable_coordinate_change'
            break
    objective_admitted = admitted(original_value, objective(current))
    if stop == 'no_representable_coordinate_change' and final_dual is None and objective_admitted:
        final_dual, final_stats = accelerated_flow(
            current, dual, problem, kernels, policy, limits, work)
        stop = 'qualified' if _qualified(final_stats, policy) else 'full_certificate_unresolved'
    # Admission is independently recomputed on the actual returned tensors.
    accepted = False
    if final_dual is not None:
        final_eager = _certificate(current, final_dual, problem, work)
        final_stats = _compiled_certificate(current, final_dual, problem, kernels, work)
        accepted = (objective_admitted and _qualified(final_eager, policy)
                    and _qualified(final_stats, policy)
                    and _certificates_agree(final_eager, final_stats, x.numel()))
        if not accepted:
            stop = 'independent_admission_rejected'
    if not all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
               for a, b in zip(inputs, frozen, strict=True)):
        raise ArithmeticError('Precision proposal changed original QP tensors')
    if (work.pava_rounds > limits.rounds
            or work.coordinate_minimizations > limits.coordinate_minimizations
            or work.collective_reorderings > limits.collective_reorderings
            or work.collective_calls > limits.rounds
            or work.cut_steps > 2 * limits.rounds * limits.cut_steps_per_sign
            or work.projected_flow_steps > limits.projected_flow_steps):
        raise ArithmeticError('Precision recovery exceeded its declared work limits')
    if accepted:
        result_x, result_dual, stats = current, final_dual, final_stats
    else:
        result_x, result_dual, stats = x, dual, original_stats
    qualified = bool(accepted or original_qualified)
    recorded_work = dict(asdict(work), attempted=True, qualified=qualified,
                         candidate_accepted=bool(accepted), stop_reason=stop)
    return PrecisionResult(result_x, result_dual, stats, qualified, recorded_work)
