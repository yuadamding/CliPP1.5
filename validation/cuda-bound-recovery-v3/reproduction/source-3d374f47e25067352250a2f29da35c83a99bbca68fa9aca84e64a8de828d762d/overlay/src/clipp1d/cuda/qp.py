"""Certified unconstrained complete-graph ADMM; no forced clonal coordinate."""

from dataclasses import dataclass
import torch
from .kernels import Kernels, differences, adjoint, flow_polish_step as _flow_step
from .policy import CudaPolicy
from .partition import polish_quadratic


@dataclass
class QuadraticFit:
    x: torch.Tensor
    dual: torch.Tensor
    gap: torch.Tensor
    scale: torch.Tensor
    kkt: torch.Tensor
    qualified: bool
    iterations: int
    polish_iterations: int = 0


@dataclass(frozen=True)
class QualifiedDualWarmState:
    """A single qualified surrogate dual, scoped to one frozen graph and lambda.

    This carries an initialization only. No old gap, curvature, target or scaled
    ADMM variable is admitted as a certificate for the next surrogate.
    """
    dual: torch.Tensor
    graph_identity: object
    lambda_value: float

    def __post_init__(self):
        # Own the state independently of a retained QP result. The additional
        # snapshot is bounded to one dual and checked only when another QP starts,
        # never within ADMM iterations. It catches .data writes that bypass _version.
        object.__setattr__(self, "dual", self.dual.detach().clone())
        object.__setattr__(self, "_version", (id(self.dual), self.dual._version))
        object.__setattr__(self, "_snapshot", self.dual.detach().clone())

    @classmethod
    def from_fit(cls, fitted, graph, lambda_value):
        if not fitted.qualified:
            raise ValueError("Dual continuation requires a qualified QP result")
        graph.validate()
        return cls(fitted.dual, graph.identity, float(lambda_value))

    def for_problem(self, graph, lambda_value):
        graph.validate()
        if (self.graph_identity is not graph.identity or float(lambda_value) != self.lambda_value
                or (id(self.dual), self.dual._version) != self._version
                or not torch.equal(self.dual, self._snapshot)):
            raise ValueError("Qualified dual initialization must match its unchanged graph and lambda")
        return self.dual


def quadratic_value(x, h, target, caps, reference=None):
    # Subtract a common boxed reference before arithmetic; fixed constants vanish.
    if reference is None:
        reference = torch.zeros_like(x)
    delta = x - reference
    return (0.5 * h * delta.square() + h * (reference - target) * delta).sum() + 0.5 * (
        caps * (differences(x).abs() - differences(reference).abs())
    ).sum()


@dataclass(frozen=True)
class FlowPolishContext:
    """Owned geometry for one equality proposal, never a cross-QP cache.

    Every argument is a freshly derived tensor or an owned caps copy. Original
    candidate/problem mutations cannot change this context. It is confined to
    one proposal's repair loop; the original QP supplies every certificate.
    Tensor arguments are private implementation data and are never mutated.
    """
    _arguments: tuple


def prepare_flow_polish(x, h, target, lower, upper, caps):
    """Prepare the fixed geometry once; subsequent steps change only q."""
    same = x[:, None] == x[None, :]
    external = caps * differences(x).sign()
    free = lower < upper
    safe_target = torch.where(free, target, x)
    gradient = torch.where(free, h * (x - safe_target), 0.0)
    fixed = ~free
    at_lower = free & (x == lower)
    at_upper = free & (x == upper)
    sizes = same.sum(1)
    return FlowPolishContext((same, external, gradient, fixed,
                              at_lower, at_upper,
                              sizes, caps.detach().clone()))


def flow_polish_step(q, context, kernels=None):
    """Dispatch one pure tensor repair; no admission or geometry rebuilding."""
    function = _flow_step if kernels is None else getattr(kernels, "flow_polish_step", _flow_step)
    return function(q, *context._arguments)


def _polish_dual(x, q, h, target, lower, upper, caps):
    """One cone-aware repair, still requiring a full original certificate."""
    return flow_polish_step(q, prepare_flow_polish(x, h, target, lower, upper, caps))


def _objective_context(x, h, target, lower, upper, caps):
    safe_target = torch.where(lower < upper, target, lower)
    reference = torch.maximum(lower, torch.minimum(upper, safe_target))
    return safe_target, reference, quadratic_value(x, h, safe_target, caps, reference)


@dataclass(frozen=True)
class _EqualityProposal:
    candidate: torch.Tensor
    geometry: FlowPolishContext | None
    dual: torch.Tensor
    stats: torch.Tensor
    qualified: bool
    flow_steps: int
    incoming_stats: torch.Tensor
    candidate_unchanged: bool
    exact_edge_complementarity: bool


def _qualify_equality_proposal(candidate, q, h, target, lower, upper, caps, kernels, policy):
    """Always certify the original full QP, independently of proposal geometry."""
    stats = kernels.gap_kkt(candidate, q, h, target, lower, upper, caps)
    gate = (torch.isfinite(stats).all()
            & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
            & (stats[2] <= policy.inner_kkt_tol))
    return stats, bool(gate)


def _prepare_equality_proposal(x, q, h, target, lower, upper, caps, kernels, policy,
                               tolerance, objective_context, incoming_stats=None):
    """Prepare, objective-check and independently certify one exact candidate."""
    candidate = polish_quadratic(x, q, h, target, lower, upper, caps, tolerance)
    safe_target, reference, old_value = objective_context
    new_value = quadratic_value(candidate, h, safe_target, caps, reference)
    roundoff = 32 * torch.finfo(h.dtype).eps * (1 + old_value.abs() + new_value.abs())
    if not bool(torch.isfinite(new_value) & (new_value <= old_value + roundoff)):
        return None
    # Preserve an already valid original dual before any recovery transform.
    # incoming_stats is only supplied by this checkpoint on the identical QP,
    # x and q; it is reusable only when equality preparation leaves x unchanged.
    candidate_unchanged = torch.equal(candidate, x)
    if incoming_stats is not None and candidate_unchanged:
        stats = incoming_stats
        qualified = bool(torch.isfinite(stats).all()
                         & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
                         & (stats[2] <= policy.inner_kkt_tol))
    else:
        stats, qualified = _qualify_equality_proposal(
            candidate, q, h, target, lower, upper, caps, kernels, policy)
    if qualified:
        d = differences(candidate)
        exact_edges = bool(((d == 0) | (q == caps * d.sign())).all())
        return _EqualityProposal(candidate, None, q, stats, True, 0,
                                 stats, candidate_unchanged, exact_edges)
    original_stats = stats
    geometry = prepare_flow_polish(candidate, h, target, lower, upper, caps)
    candidate_q = flow_polish_step(q, geometry, kernels)
    if torch.equal(candidate_q, q):
        stats, qualified = original_stats, False
    else:
        stats, qualified = _qualify_equality_proposal(candidate, candidate_q, h, target, lower,
                                                     upper, caps, kernels, policy)
    return _EqualityProposal(candidate, geometry, candidate_q, stats, qualified, 1,
                             original_stats, candidate_unchanged, True)


_NO_PROPOSAL = object()


def _refine_polish(x, q, h, target, lower, upper, caps, kernels, policy, *,
                   _initial_proposal=_NO_PROPOSAL, _objective=None, _incoming_stats=None):
    """Bounded, independently certified refinement; no reporting-policy change.

    The ordinary checkpoint may supply its exact default-tolerance proposal,
    avoiding repeated grouping, geometry, objective and certificate work at that
    same checkpoint. No proposal/certificate is retained across ADMM updates or
    surrogates. Each candidate permits at most 1024 flow steps; a supplied first
    step was already counted by the caller and consumes one of those steps.
    """
    objective = _objective_context(x, h, target, lower, upper, caps) if _objective is None else _objective
    steps = 0
    qualified_fallback = None
    for divisor in (1.0, 16.0, 256.0):
        reused = divisor == 1.0 and _initial_proposal is not _NO_PROPOSAL
        proposal = _initial_proposal if reused else _prepare_equality_proposal(
            x, q, h, target, lower, upper, caps, kernels, policy,
            policy.fusion_tol / divisor, objective, _incoming_stats)
        if proposal is None:
            continue
        if not reused:
            steps += proposal.flow_steps
        candidate, candidate_q, stats = proposal.candidate, proposal.dual, proposal.stats
        if not bool(torch.isfinite(stats).all()):
            continue
        if proposal.qualified:
            if proposal.exact_edge_complementarity:
                return (candidate, candidate_q, stats), steps
            # Preserve this valid dual instead of transforming it. A gap gate
            # permits tiny nonzero differences with unsaturated edges; keep
            # trying the existing tighter equality proposals so this admission
            # does not starve exact-kink recovery in the outer raw audit.
            if qualified_fallback is None:
                qualified_fallback = (candidate, candidate_q, stats)
            continue
        # No internal flow degrees of freedom means projection cannot help.
        if not bool((proposal.geometry._arguments[-2] > 1).any()):
            continue
        checkpoint_q = candidate_q
        for repair in range(2, 1025):
            candidate_q = flow_polish_step(candidate_q, proposal.geometry, kernels)
            steps += 1
            if repair % policy.check_every == 0 or repair == 1024:
                if torch.equal(candidate_q, checkpoint_q):
                    break  # This identical candidate/dual already failed its certificate.
                stats, qualified = _qualify_equality_proposal(
                    candidate, candidate_q, h, target, lower, upper, caps, kernels, policy)
                if qualified:
                    return (candidate, candidate_q, stats), steps
                checkpoint_q = candidate_q
    return qualified_fallback, steps


def solve_qp(
    h, target, lower, upper, caps, kernels: Kernels, policy=CudaPolicy(), start=None, dual=None
):
    n = h.numel()
    if (
        n == 0
        or h.ndim != 1
        or target.shape != h.shape
        or lower.shape != h.shape
        or upper.shape != h.shape
        or caps.shape != (n, n)
        or h.dtype != torch.float64
    ):
        raise ValueError("Invalid complete-graph QP shapes or dtype")
    if any(t.device != h.device or t.dtype != h.dtype for t in (target, lower, upper, caps)):
        raise ValueError("QP tensors must share device and float64 dtype")
    valid = (
        (h > 0).all()
        & (lower <= upper).all()
        & (caps >= 0).all()
        & (caps == caps.T).all()
        & (caps.diagonal() == 0).all()
        & torch.isfinite(h).all()
        & torch.isfinite(target).all()
        & torch.isfinite(lower).all()
        & torch.isfinite(upper).all()
        & torch.isfinite(caps).all()
    )
    if not bool(valid):
        raise ValueError("Invalid complete-graph QP values")
    for name, supplied, expected in (("start", start, h), ("dual", dual, caps)):
        if supplied is not None and (
            not isinstance(supplied, torch.Tensor)
            or supplied.shape != expected.shape
            or supplied.dtype != h.dtype
            or supplied.device != h.device
            or not bool(torch.isfinite(supplied).all())
        ):
            raise ValueError(f"QP {name} must match shape, float64 dtype, device and be finite")
    x = torch.maximum(lower, torch.minimum(upper, target if start is None else start)).clone()
    q = (
        torch.zeros_like(caps)
        if dual is None
        else torch.maximum(-caps, torch.minimum(caps, dual)).clone()
    )
    q = q * 0.5 - q.T * 0.5
    q = torch.maximum(-caps, torch.minimum(caps, q))
    # Consensus is controlled by the active fused components, which can be much
    # smaller than N. Dividing curvature by the full node count underconditions
    # those components and can stall edge complementarity despite small node KKT.
    # This initial rho and residual balancing affect conditioning, never lambda.
    rho = h.median().clamp_min(torch.finfo(h.dtype).tiny)
    if n == 1 or not bool((caps > 0).any()) or not bool((lower < upper).any()):
        x = torch.maximum(lower, torch.minimum(upper, target))
        q = caps * differences(x).sign()
        stats = kernels.gap_kkt(x, q, h, target, lower, upper, caps)
        gate = (
            torch.isfinite(stats).all()
            & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
            & (stats[2] <= policy.inner_kkt_tol)
        )
        return QuadraticFit(x, q, *stats, bool(gate), 0)
    base_rho = rho.clone()
    # Every new surrogate rescales the physical dual with its new numerical rho.
    # h/target and the full certificate are always recomputed for this problem.
    v = q / rho
    z = differences(x)
    polish_iterations = 0
    if dual is not None:
        stats = kernels.gap_kkt(x, q, h, target, lower, upper, caps)
        gate = (
            torch.isfinite(stats).all()
            & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
            & (stats[2] <= policy.inner_kkt_tol)
        )
        if bool(gate):
            # A gap-qualified near-fusion can still fail the outer exact-kink
            # audit. Warm admission must offer the same numerical equality
            # recovery as an ADMM checkpoint before returning an unchanged state.
            refined, steps = _refine_polish(
                x, q, h, target, lower, upper, caps, kernels, policy, _incoming_stats=stats)
            polish_iterations += steps
            if refined is not None:
                candidate, candidate_q, stats = refined
                return QuadraticFit(candidate, candidate_q, *stats, True, 0, polish_iterations)
            return QuadraticFit(x, q, *stats, True, 0, polish_iterations)
    stats = h.new_tensor([float("inf"), 0.0, float("inf")])
    # This fixed QP coefficient never depends on the ADMM state or rho.
    safe_target = torch.where(lower < upper, target, lower)
    weighted_target = h * safe_target
    combined_step = getattr(kernels, "admm_step", None)
    for iteration in range(1, policy.inner_max_iterations + 1):
        previous_z = z
        if combined_step is not None:
            x, z, v = combined_step(h, weighted_target, lower, upper, caps, z, v, rho)
        else:
            # Preserve the injectable legacy kernel protocol used by independent
            # references. Production Kernels always has the strict combined step;
            # compiler errors propagate and never activate this branch.
            b = weighted_target + rho * adjoint(z - v)
            x = kernels.boxed_rank_one(h, b, lower, upper, rho)
            z, v = kernels.edge_update(x, v, caps, rho)
        if iteration % policy.check_every == 0 or iteration == policy.inner_max_iterations:
            q = torch.maximum(-caps, torch.minimum(caps, rho * v))
            objective = _objective_context(x, h, target, lower, upper, caps)
            proposal = _prepare_equality_proposal(
                x, q, h, target, lower, upper, caps, kernels, policy,
                policy.fusion_tol, objective)
            if proposal is not None:
                polish_iterations += proposal.flow_steps
                if proposal.qualified:
                    if proposal.exact_edge_complementarity:
                        return QuadraticFit(proposal.candidate, proposal.dual, *proposal.stats,
                                            True, iteration, polish_iterations)
                    refined, steps = _refine_polish(
                        x, q, h, target, lower, upper, caps, kernels, policy,
                        _initial_proposal=proposal, _objective=objective,
                        _incoming_stats=proposal.incoming_stats if proposal.candidate_unchanged else None)
                    polish_iterations += steps
                    candidate, candidate_q, stats = refined
                    return QuadraticFit(candidate, candidate_q, *stats,
                                        True, iteration, polish_iterations)
            # The original dual was already checked on this exact x if equality
            # preparation changed nothing. Reuse only within this checkpoint.
            stats = (proposal.incoming_stats
                     if proposal is not None and proposal.candidate_unchanged
                     else kernels.gap_kkt(x, q, h, target, lower, upper, caps))
            gate = (
                torch.isfinite(stats).all()
                & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
                & (stats[2] <= policy.inner_kkt_tol)
            )
            if bool(gate):
                refined, steps = _refine_polish(
                    x, q, h, target, lower, upper, caps, kernels, policy,
                    _initial_proposal=proposal, _objective=objective, _incoming_stats=stats)
                polish_iterations += steps
                if refined is not None:
                    candidate, candidate_q, stats = refined
                    return QuadraticFit(candidate, candidate_q, *stats, True, iteration, polish_iterations)
                return QuadraticFit(x, q, *stats, True, iteration, polish_iterations)
            if ((iteration % (64 * policy.check_every) == 0 or iteration == policy.inner_max_iterations)
                    and bool(torch.isfinite(stats).all())):
                refined, steps = _refine_polish(
                    x, q, h, target, lower, upper, caps, kernels, policy,
                    _initial_proposal=proposal, _objective=objective, _incoming_stats=stats)
                polish_iterations += steps
                if refined is not None:
                    candidate, candidate_q, stats = refined
                    return QuadraticFit(candidate, candidate_q, *stats, True, iteration, polish_iterations)
        if iteration % (4 * policy.check_every) == 0:
            # Residual balancing changes only numerical conditioning. Admission
            # above still depends exclusively on the original gap/KKT gates.
            primal_norm = (0.5 * (differences(x) - z).square().sum()).sqrt()
            dual_norm = rho * adjoint(z - previous_z).square().sum().sqrt()
            next_rho = torch.where(
                primal_norm > 10 * dual_norm,
                rho * 2,
                torch.where(dual_norm > 10 * primal_norm, rho * 0.5, rho),
            )
            next_rho = next_rho.clamp(base_rho * 1e-8, base_rho * 1e8)
            v = v * (rho / next_rho)
            rho = next_rho
    return QuadraticFit(x, q, *stats, False, policy.inner_max_iterations, polish_iterations)
