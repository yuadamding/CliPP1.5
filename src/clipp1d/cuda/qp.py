"""Certified unconstrained complete-graph ADMM; no forced clonal coordinate."""

from dataclasses import dataclass
import torch
from .kernels import Kernels, differences, adjoint
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


def quadratic_value(x, h, target, caps, reference=None):
    # Subtract a common boxed reference before arithmetic; fixed constants vanish.
    if reference is None:
        reference = torch.zeros_like(x)
    delta = x - reference
    return (0.5 * h * delta.square() + h * (reference - target) * delta).sum() + 0.5 * (
        caps * (differences(x).abs() - differences(reference).abs())
    ).sum()


def _polish_dual(x, q, h, target, lower, upper, caps):
    """Algebraic flow correction inside exact fused groups, independently audited.

    This is a candidate correction, not an alternative optimizer. A cap-active
    correction may fail the full certificate and is then simply discarded.
    """
    same = x[:, None] == x[None, :]
    d = differences(x)
    q = torch.where(same, q, caps * d.sign())
    free = lower < upper
    safe_target = torch.where(free, target, x)
    residual = torch.where(free, h * (x - safe_target), 0.0) + adjoint(q)
    group_total = torch.where(same, residual[None, :], 0.0).sum(1)
    fixed = ~free
    fixed_count = (same & fixed[None, :]).sum(1)
    at_lower = free & (x == lower)
    at_upper = free & (x == upper)
    lower_count = (same & at_lower[None, :]).sum(1)
    upper_count = (same & at_upper[None, :]).sum(1)
    normal = torch.where(
        fixed_count > 0,
        torch.where(fixed, group_total / fixed_count.clamp_min(1), 0.0),
        torch.where(
            group_total >= 0,
            torch.where(at_lower, group_total / lower_count.clamp_min(1), 0.0),
            torch.where(at_upper, group_total / upper_count.clamp_min(1), 0.0),
        ),
    )
    correction = residual - normal
    sizes = same.sum(1)
    delta = -differences(correction) / sizes[:, None]
    proposal = q + torch.where(same, delta, 0.0)
    return torch.maximum(-caps, torch.minimum(caps, proposal))


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
    # Complete-graph nonzero Laplacian eigenvalues are N. Balance H against
    # rho*N; rho is numerical conditioning, never the statistical lambda.
    rho = h.median().clamp_min(torch.finfo(h.dtype).tiny) / n
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
    v = q / rho
    z = differences(x)
    stats = h.new_tensor([float("inf"), 0.0, float("inf")])
    for iteration in range(1, policy.inner_max_iterations + 1):
        # Do not even multiply irrelevant frozen targets by h: those products
        # can overflow despite a perfectly finite free-coordinate problem.
        safe_target = torch.where(lower < upper, target, lower)
        b = h * safe_target + rho * adjoint(z - v)
        x = kernels.boxed_rank_one(h, b, lower, upper, rho)
        previous_z = z
        z, v = kernels.edge_update(x, v, caps, rho)
        if iteration % policy.check_every == 0 or iteration == policy.inner_max_iterations:
            q = torch.maximum(-caps, torch.minimum(caps, rho * v))
            # Projection only corrects edge-dual rounding; the independent gap audits it.
            candidate = polish_quadratic(x, q, h, target, lower, upper, caps, policy.fusion_tol)
            candidate_q = _polish_dual(candidate, q, h, target, lower, upper, caps)
            stats = kernels.gap_kkt(candidate, candidate_q, h, target, lower, upper, caps)
            candidate_gate = (
                torch.isfinite(stats).all()
                & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
                & (stats[2] <= policy.inner_kkt_tol)
            )
            # Polishing is only an active-set proposal. Certify both its original
            # QP and objective descent, omitting irrelevant frozen constants.
            reference = torch.maximum(lower, torch.minimum(upper, safe_target))
            old_value = quadratic_value(x, h, safe_target, caps, reference)
            new_value = quadratic_value(candidate, h, safe_target, caps, reference)
            roundoff = 32 * torch.finfo(h.dtype).eps * (1 + old_value.abs() + new_value.abs())
            candidate_gate &= torch.isfinite(new_value) & (new_value <= old_value + roundoff)
            if bool(candidate_gate):
                return QuadraticFit(candidate, candidate_q, *stats, True, iteration)
            stats = kernels.gap_kkt(x, q, h, target, lower, upper, caps)
            gate = (
                torch.isfinite(stats).all()
                & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
                & (stats[2] <= policy.inner_kkt_tol)
            )
            if bool(gate):
                return QuadraticFit(x, q, *stats, True, iteration)
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
    return QuadraticFit(x, q, *stats, False, policy.inner_max_iterations)
