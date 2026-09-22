"""Unconstrained full-graph box and one-sided directional qualification.

The device-side cut relaxation covers arbitrary signed node subsets, including
noncontiguous exact fused groups. Feasible dual lower bounds certify absence of
descent; unresolved cut optimization never becomes a successful raw certificate.
"""
from dataclasses import dataclass, field
import torch
from .kernels import differences, adjoint
from .policy import CudaPolicy


@dataclass
class Audit:
    qualified: bool
    residual: float
    direction: torch.Tensor | None
    status: str
    signed_direction_count: int = 0
    diagnostics: dict = field(default_factory=dict)


def _cut_inputs(a, caps, allowed, initial_dual):
    n = a.numel()
    arrays = (a, caps, allowed, initial_dual)
    if (a.ndim != 1 or n == 0 or caps.shape != (n, n) or allowed.shape != (n,) or
            initial_dual.shape != (n, n) or
            any(v.dtype != torch.float64 or v.device != a.device for v in arrays)):
        raise ValueError("Directional cuts require matching device float64 node/edge arrays")
    if not bool(torch.isfinite(a).all() & torch.isfinite(caps).all() &
                torch.isfinite(initial_dual).all() & ((allowed == 0) | (allowed == 1)).all() &
                (caps >= 0).all() & (caps == caps.T).all() & (caps.diagonal() == 0).all()):
        raise ValueError("Invalid finite symmetric capacities or feasible-direction mask")


def directional_cut(a, caps, allowed, initial_dual, policy=CudaPolicy(), *, diagnostics=None,
                    tolerance=None):
    """Minimize a't + sum caps_ij |t_j-t_i| over 0 <= t <= allowed.

    All quantities remain on the input device. For every feasible skew dual q,
    LB=sum min(a+D'q,0)*allowed is a lower bound on *all* binary subsets and their
    continuous relaxation. The certificate includes cancellation-sensitive
    reduction error, and attained descent has an independent primal margin.
    ``diagnostics`` receives scalar bounds and status; no fitted array is moved
    to host. An iteration limit produces (False, None), never approximate success.
    """
    _cut_inputs(a, caps, allowed, initial_dual)
    threshold = a.new_tensor(policy.stationarity_tol) if tolerance is None else torch.as_tensor(
        tolerance, dtype=a.dtype, device=a.device)
    if threshold.ndim != 0 or not bool(torch.isfinite(threshold) & (threshold > 0)):
        raise ValueError("Directional cut tolerance must be a finite positive scalar")
    n = a.numel()
    q = torch.maximum(-caps, torch.minimum(caps, initial_dual)).clone()
    q = (q - q.T) * .5
    t = torch.zeros_like(a)
    extrapolated = t.clone()
    tau, sigma = .9, 1. / max(n, 1)
    eps = torch.finfo(a.dtype).eps
    status, lower, value = "unresolved", a.new_tensor(-float("inf")), a.new_zeros(())
    for iteration in range(policy.inner_max_iterations + 1):
        if iteration % policy.check_every == 0 or iteration == policy.inner_max_iterations:
            r = a + adjoint(q)
            lower = (r.clamp_max(0.) * allowed).sum()
            # Bounding only |r| would miss cancellation in a + D'q. Account for
            # every absolute summand before the row and final reductions.
            margin = (8 * (n + 2) * eps *
                      (1 + ((a.abs() + q.abs().sum(-1)) * allowed).sum()))
            lower = lower - margin
            dual_valid = (torch.isfinite(q).all() & (q.abs() <= caps).all() & (q == -q.T).all())
            if bool(dual_valid & torch.isfinite(lower) & (lower >= -threshold)):
                status = "qualified_lower_bound"
                if diagnostics is not None:
                    diagnostics.update(status=status, iterations=iteration, lower_bound=float(lower),
                                       attained_value=float(value), roundoff_margin=float(margin))
                return True, None
            node_terms = a * t
            edge_terms = caps * differences(t).abs()
            value = node_terms.sum() + .5 * edge_terms.sum()
            primal_margin = 8 * (n + 2) * eps * (1 + node_terms.abs().sum() + .5 * edge_terms.sum())
            primal_valid = (torch.isfinite(t).all() & (t >= 0).all() & (t <= allowed).all())
            if bool(primal_valid & torch.isfinite(value) & (value + primal_margin < -threshold)):
                status = "attained_descent"
                if diagnostics is not None:
                    diagnostics.update(status=status, iterations=iteration, lower_bound=float(lower),
                                       attained_value=float(value), roundoff_margin=float(margin),
                                       primal_margin=float(primal_margin))
                return False, t
        if iteration == policy.inner_max_iterations:
            break
        q = torch.maximum(-caps, torch.minimum(caps, q + sigma * differences(extrapolated)))
        old = t
        t = torch.maximum(torch.zeros_like(t), torch.minimum(allowed, t - tau * (a + adjoint(q))))
        extrapolated = 2 * t - old
    if diagnostics is not None:
        diagnostics.update(status=status, iterations=iteration, lower_bound=float(lower),
                           attained_value=float(value), roundoff_margin=float(margin))
    return False, None


def audit_raw(model, x, q, caps, witness=None, policy=CudaPolicy()):
    """Audit the full original box, without a clonal fitting constraint.

    ``witness=None`` preserves call compatibility but no coordinate can be
    omitted from the KKT or negative-direction checks. Label 0 is assigned only
    after fitting and has no role in this certificate.
    """
    if witness is not None:
        raise ValueError("Unconstrained audit cannot fix or exempt a clonal witness")
    model.validate()
    _cut_inputs(x, caps, torch.ones_like(x), q)
    losses, grad, _, _, left, right = model.terms(x)
    valid = ((x >= model.lower).all() & (x <= model.upper).all()
             & torch.isfinite(x).all() & torch.isfinite(q).all()
             & torch.isfinite(losses).all() & (q.abs() <= caps).all()
             & (q == -q.T).all())
    if not bool(valid):
        return Audit(False, float("inf"), None, "infeasible")
    d = differences(x)
    a = adjoint(q)
    chosen = torch.maximum(left, torch.minimum(right, -a))
    grad = torch.where(left <= right, chosen, grad)
    grad = torch.where(x == model.lower, right, grad)
    grad = torch.where(x == model.upper, left, grad)
    r = grad + a
    r = torch.where(x == model.lower, r.clamp_max(0.), r)
    r = torch.where(x == model.upper, r.clamp_min(0.), r)
    r = torch.where(model.lower == model.upper, 0., r)
    node = (r.abs() / (1 + grad.abs() + a.abs())).max()
    comp = torch.where(d != 0., (q - caps * d.sign()).abs() / (1 + caps), 0.).max()
    residual = float(torch.maximum(node, comp))
    fused_caps = torch.where(d == 0., caps, 0.)
    external = adjoint(torch.where(d != 0., caps * d.sign(), 0.))
    external_absolute = torch.where(d != 0., caps, 0.).sum(-1)
    initial = torch.where(d == 0., q, 0.)
    diagnostics = dict(clonal_constraint=False,
                       coverage="all_feasible_signed_subsets_of_exact_fused_groups",
                       proof="device_dual_lower_bounds_of_complete_subset_cut_relaxation",
                       tolerance_rule="derivative_plus_tol_times_one_plus_absolute_node_and_cut_terms")
    for sign, derivative, allowed, dual in (
            (1, right, (x < model.upper).to(x.dtype), initial),
            (-1, -left, (x > model.lower).to(x.dtype), -initial)):
        # The old signed-subset acceptance rule is additive: derivative +
        # tol*(1 + absolute likelihood/external terms + fused cut capacity).
        # Incorporate all but the constant tol into the cut objective. Fixed
        # coordinates and fixed-fixed edges contribute no admission scale.
        absolute = derivative.abs() + external_absolute
        linear = torch.where(allowed > 0, derivative + sign * external +
                             policy.stationarity_tol * absolute, 0.)
        moving = (allowed[:, None] > 0) | (allowed[None, :] > 0)
        fc = torch.where(moving, (1 + policy.stationarity_tol) * fused_caps, 0.)
        dual = torch.where(moving, dual, 0.)
        scale = torch.maximum(linear.abs().max(), fc.sum(-1).max()).clamp_min(1.)
        cut = {}
        ok, direction = directional_cut(linear / scale, fc / scale, allowed, dual / scale,
                                        policy, diagnostics=cut,
                                        tolerance=policy.stationarity_tol / scale)
        cut["objective_scale"] = float(scale)
        name = "positive" if sign > 0 else "negative"
        diagnostics[name] = cut
        count = 1 if sign > 0 else 2
        if not ok:
            return Audit(False, residual, None if direction is None else sign * direction,
                         name + ("_descent" if direction is not None else "_unresolved"),
                         count, diagnostics)
    qualified = residual <= policy.stationarity_tol
    return Audit(qualified, residual, None, "qualified" if qualified else "componentwise_unresolved",
                 2, diagnostics)
