"""Bounded original scalar likelihood plus a fixed affine objective term.

This supplies numerical proposals to callers that retain their own full model
acceptance gates. It changes neither ordinary scalar dispatch nor tolerances.
"""
from contextlib import contextmanager

import torch

from .policy import CudaPolicy
from .scalar import Problems, _solve_general, posterior_variance_bound


class AffineProblem:
    """One immutable scalar group on an original-box subinterval.

    The affine coefficient is constant throughout this problem. The reference
    centers its value arithmetically; it does not change its derivative or
    optimizer. More than one likelihood row may belong to the single group.
    """

    def __init__(self, base: Problems, lower, upper, coefficient, reference):
        if base.count != 1:
            raise ValueError("Exactly one scalar group is required")
        self.base, self.model, self.count = base, base.model, 1
        values = (lower, upper, coefficient, reference)
        if any(not isinstance(value, torch.Tensor) or value.numel() != 1
               or value.dtype != torch.float64 or value.device != base.model.device for value in values):
            raise ValueError("Affine parameters must be scalar original-device float64 tensors")
        self.lower, self.upper, self.coefficient, self.reference = (
            value.detach().clone().reshape(1) for value in values)
        if not bool(torch.isfinite(torch.cat(self._values())).all()
                    & (self.lower >= base.lower).all() & (self.upper <= base.upper).all()
                    & (self.lower <= self.reference).all() & (self.reference <= self.upper).all()):
            raise ValueError("Finite original-box interval containing the reference required")
        self._snapshots = tuple(value.clone() for value in self._values())
        self._versions = self._metadata()

    def _values(self):
        return self.lower, self.upper, self.coefficient, self.reference

    def _metadata(self):
        return (id(self.base), id(self.model), self.count,
                tuple((id(value), value._version, value.shape, value.dtype, value.device, value.data_ptr())
                      for value in self._values()))

    def validate(self):
        if self._metadata() != self._versions:
            raise ValueError("Affine scalar problem changed")
        self.base.validate()
        if not all(torch.equal(first.view(torch.uint8), second.view(torch.uint8))
                   for first, second in zip(self._values(), self._snapshots, strict=True)):
            raise ValueError("Affine scalar problem changed")

    @contextmanager
    def validated_stage(self):
        with self.base.validated_stage():
            self.validate()
            try:
                yield self
            finally:
                self.validate()

    def reduce(self, *args, **kwargs):
        return self.base.reduce(*args, **kwargs)

    def loss(self, points):
        self.validate()
        return self.base.loss(points) + self.coefficient[:, None] * (points - self.reference[:, None])

    def evaluate(self, points):
        self.validate()
        loss, gradient = self.base.evaluate(points)
        return loss + self.coefficient[:, None] * (points - self.reference[:, None]), gradient + self.coefficient[:, None]

    def bounds(self, left, right):
        self.validate()
        original, mid, likelihood = self.base.bounds(left, right)
        coefficient, reference = self.coefficient[:, None], self.reference[:, None]
        endpoint = torch.where(coefficient >= 0., left, right)
        minimum = coefficient * (endpoint - reference)
        eps = torch.finfo(torch.float64).eps
        error = 16 * eps * (1 + original.abs() + coefficient.abs()
                           * (endpoint.abs() + reference.abs()) + minimum.abs())
        downward = original.new_tensor(-float("inf"))
        additive = torch.nextafter(original + minimum - error, downward)
        tilted_loss = likelihood + coefficient * (mid - reference)

        # A linear term leaves the likelihood curvature unchanged. The full
        # derivative g+c retains cancellation lost by adding separate infima.
        m, p = self.model, self.base
        lo, r = left[p.group], right[p.group]
        sl = m.slope[:, None, :]
        pl = (sl * lo[..., None]).clamp(m.eps, 1 - m.eps)
        pr = (sl * r[..., None]).clamp(m.eps, 1 - m.eps)
        safe = torch.where(m.slope > 0, m.slope, torch.ones_like(m.slope))
        lk = (torch.full_like(safe, m.eps) / safe)[:, None, :]
        hk = (torch.full_like(safe, 1 - m.eps) / safe)[:, None, :]
        valid = torch.isfinite(m.log_prior)[:, None, :]
        crossing = ((((lk >= lo[..., None]) & (lk <= r[..., None]))
                     | ((hk >= lo[..., None]) & (hk <= r[..., None]))) & valid)
        crossing = p.reduce(crossing.any(-1).to(m.alt.dtype), "max") > 0
        mass = sl * mid[p.group][..., None]
        moving = torch.where((mass > m.eps) & (mass < 1 - m.eps), sl, 0.)
        score_l = moving * (m.alt[:, None, None] / pl - m.ref[:, None, None] / (1 - pl))
        score_r = moving * (m.alt[:, None, None] / pr - m.ref[:, None, None] / (1 - pr))
        spread = (torch.where(valid, score_l, -float("inf")).amax(-1)
                  - torch.where(valid, score_r, float("inf")).amin(-1))
        spread = torch.where(valid.sum(-1) == 1, 0., spread)
        negative_curvature = p.reduce(posterior_variance_bound(
            m, pl, pr, moving, valid, spread.square() / 4))
        sensitivity = m.alt[:, None, None] / pl + m.ref[:, None, None] / (1 - pr)
        sensitivity = torch.where(valid, sensitivity, 0.).amax(-1)
        input_roundoff = 32 * eps * p.reduce(sensitivity)
        gradient = p.evaluate(mid)[1]
        radius = torch.maximum(mid - left, right - mid)
        curvature_term = .5 * negative_curvature * radius.square()
        smooth = tilted_loss - (gradient + coefficient).abs() * radius - curvature_term - input_roundoff
        arithmetic_error = 128 * eps * (
            1 + likelihood.abs() + tilted_loss.abs()
            + (gradient.abs() + coefficient.abs()) * radius
            + coefficient.abs() * (mid.abs() + reference.abs())
            + curvature_term.abs() + input_roundoff.abs())
        smooth = torch.nextafter(smooth - arithmetic_error, downward)
        lower = torch.where(crossing, additive, torch.maximum(additive, smooth))
        return lower, mid, tilted_loss

    def nearest_kink(self, left, right):
        return self.base.nearest_kink(left, right)


def solve_affine_scalar(problem: AffineProblem, policy=CudaPolicy()):
    """Use the original general search; the analytical shortcut omits the tilt."""
    if not isinstance(problem, AffineProblem):
        raise TypeError("An immutable affine scalar problem is required")
    with problem.validated_stage():
        return _solve_general(problem, policy)
