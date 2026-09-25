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


class InvalidDirectionalCut(ValueError):
    """An invalid numerical cut is evidence of nonqualification, never success."""

    def __init__(self, checks):
        super().__init__("Invalid finite symmetric capacities or feasible-direction mask")
        self.checks = checks


def _cut_inputs(a, caps, allowed, initial_dual):
    n = a.numel()
    arrays = (a, caps, allowed, initial_dual)
    if (a.ndim != 1 or n == 0 or caps.shape != (n, n) or allowed.shape != (n,) or
            initial_dual.shape != (n, n) or
            any(v.dtype != torch.float64 or v.device != a.device for v in arrays)):
        raise ValueError("Directional cuts require matching device float64 node/edge arrays")
    checks = dict(linear_finite=torch.isfinite(a).all(),
                  capacities_finite=torch.isfinite(caps).all(),
                  dual_finite=torch.isfinite(initial_dual).all(),
                  mask_binary=((allowed == 0) | (allowed == 1)).all(),
                  capacities_nonnegative=(caps >= 0).all(),
                  capacities_symmetric=(caps == caps.T).all(),
                  diagonal_zero=(caps.diagonal() == 0).all())
    if not bool(torch.stack(tuple(checks.values())).all()):
        raise InvalidDirectionalCut({name: bool(value) for name, value in checks.items()})


def _cut_lower_bound(a, q, allowed, unit):
    """Conservative negative-part reduction for a fixed feasible skew dual."""
    r = a + adjoint(q)
    gamma = 8 * (a.numel() + 2) * torch.finfo(a.dtype).eps
    # Bound each residual before taking its negative part. Positive rows whose
    # complete error interval stays positive contribute exactly zero: their
    # large coefficients must not penalize an unrelated row's lower bound.
    # Absolute a/q terms retain protection against row-sum cancellation and
    # coefficient normalization. The factor-eight slack also covers rounding
    # in this nonnegative error calculation; no cancellation occurs there.
    row_error = gamma * (a.abs() + q.abs().sum(-1))
    downward = a.new_tensor(-float("inf"))
    row_lower = torch.nextafter(r - row_error, downward)
    negative = row_lower.clamp_max(0.) * allowed
    # The remaining reduction contains only nonpositive terms. Keep the
    # original-unit constant floor and cover summation/subtraction rounding.
    lower = torch.nextafter(negative.sum() - gamma * (unit + negative.abs().sum()), downward)
    unadjusted = (r.clamp_max(0.) * allowed).sum()
    return lower, unadjusted - lower


def directional_cut(a, caps, allowed, initial_dual, policy=CudaPolicy(), *, diagnostics=None,
                    tolerance=None, roundoff_unit=1.0):
    """Minimize a't + sum caps_ij |t_j-t_i| over 0 <= t <= allowed.

    All quantities remain on the input device. For every feasible skew dual q,
    LB=sum min(a+D'q,0)*allowed is a lower bound on *all* binary subsets and their
    continuous relaxation. The certificate includes cancellation-sensitive
    reduction error, and attained descent has an independent primal margin.
    ``diagnostics`` receives scalar bounds and status; no fitted array is moved
    to host. An iteration limit produces (False, None), never approximate success.

    If all objective coefficients and the tolerance were divided by S, callers
    must also pass roundoff_unit=1/S. The error bound Gamma*(1+absolute terms)
    then becomes Gamma*(1/S+absolute scaled terms), preserving the original-unit
    constant floor. Rowwise absolute a/q terms cover cancellation and relative
    coefficient/cap errors from normalization before the negative-part map;
    the final reduction includes only these conservative negative terms.
    The primal bound still includes every absolute node/edge term. No tolerance
    is increased.
    """
    _cut_inputs(a, caps, allowed, initial_dual)
    threshold = a.new_tensor(policy.stationarity_tol) if tolerance is None else torch.as_tensor(
        tolerance, dtype=a.dtype, device=a.device)
    if threshold.ndim != 0 or not bool(torch.isfinite(threshold) & (threshold > 0)):
        raise ValueError("Directional cut tolerance must be a finite positive scalar")
    unit = torch.as_tensor(roundoff_unit, dtype=a.dtype, device=a.device)
    if unit.ndim != 0 or not bool(torch.isfinite(unit) & (unit > 0)):
        raise ValueError("Directional cut roundoff unit must be a finite positive scalar")
    if diagnostics is not None:
        diagnostics.update(roundoff_unit=float(unit), tolerance=float(threshold))
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
            lower, margin = _cut_lower_bound(a, q, allowed, unit)
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
            primal_margin = 8 * (n + 2) * eps * (unit + node_terms.abs().sum() + .5 * edge_terms.sum())
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
    with model.validated_stage():
        return _audit_raw(model, x, q, caps, policy)


def _audit_raw(model, x, q, caps, policy):
    _cut_inputs(x, caps, torch.ones_like(x), q)
    losses, grad, _, _, left, right = model.terms(x)
    valid = ((x >= model.lower).all() & (x <= model.upper).all()
             & torch.isfinite(x).all() & torch.isfinite(q).all()
             & torch.isfinite(losses).all() & (q.abs() <= caps).all()
             & (q == -q.T).all())
    if not bool(valid):
        return Audit(False, float("inf"), None, "infeasible")
    if not bool(torch.isfinite(grad).all() & torch.isfinite(left).all() &
                torch.isfinite(right).all()):
        return Audit(False, float("inf"), None, "nonfinite_likelihood_derivatives")
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
        name = "positive" if sign > 0 else "negative"
        count = 1 if sign > 0 else 2
        if not bool(torch.isfinite(scale)):
            diagnostics[name] = dict(status="nonfinite_cut_scale")
            return Audit(False, float("inf"), None, name + "_invalid_cut", count, diagnostics)
        try:
            ok, direction = directional_cut(linear / scale, fc / scale, allowed, dual / scale,
                                            policy, diagnostics=cut,
                                            tolerance=policy.stationarity_tol / scale,
                                            roundoff_unit=torch.ones_like(scale) / scale)
        except InvalidDirectionalCut as error:
            # Keep the strict cut invariant. This start cannot supply a raw
            # certificate or a descent direction. Other starts/path candidates
            # still undergo their own unchanged qualification checks.
            cut.update(status="invalid_cut_inputs", validation_checks=error.checks,
                       objective_scale=float(scale))
            diagnostics[name] = cut
            return Audit(False, float("inf"), None, name + "_invalid_cut", count, diagnostics)
        cut["objective_scale"] = float(scale)
        diagnostics[name] = cut
        if not ok:
            return Audit(False, residual, None if direction is None else sign * direction,
                         name + ("_descent" if direction is not None else "_unresolved"),
                         count, diagnostics)
    qualified = residual <= policy.stationarity_tol
    return Audit(qualified, residual, None, "qualified" if qualified else "componentwise_unresolved",
                 2, diagnostics)
