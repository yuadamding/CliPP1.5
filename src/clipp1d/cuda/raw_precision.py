"""Bounded terminal proposals admitted only by the original raw certificate.

This is auxiliary precision work, not another MM iteration or ordinary QP
solve. The caller owns the once-per-start QP allowance. A failed final raw
audit returns the original endpoint and its matching original negative audit.
"""
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter

import torch

from . import kernels as tensor_kernels
from .affine_scalar import AffineProblem, solve_affine_scalar
from .audit import Audit, audit_raw
from .kernels import differences, raw_residual
from .policy import CudaPolicy
from .qp_precision import _certificates_agree, _qualified, recover_quadratic
from .scalar import Problems


def _same_bytes(first, second):
    return (first.shape == second.shape and first.dtype == second.dtype
            and first.device == second.device
            and torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)))


@dataclass(frozen=True)
class AcceptedQPContext:
    """Literal last accepted surrogate, captured once at raw finalization.

    Only O(N) vectors are retained. The current dense dual must pass fresh
    independent eager and compiled literal-QP gates; it is not inherited here.
    A directional restart or any changed primal invalidates this context.
    """
    h: torch.Tensor
    target: torch.Tensor
    start: torch.Tensor
    x: torch.Tensor

    def __post_init__(self):
        values = (self.h, self.target, self.start, self.x)
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1
               or value.dtype != torch.float64 or value.shape != self.x.shape
               or value.device != self.x.device for value in values):
            raise ValueError("Accepted surrogate requires matching original-device FP64 vectors")
        for name in ("h", "target", "start", "x"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        object.__setattr__(self, "_snapshots", tuple(value.clone() for value in self.values()))
        object.__setattr__(self, "_metadata", tuple((id(value), value._version) for value in self.values()))

    def values(self):
        return self.h, self.target, self.start, self.x

    def validate(self):
        if (tuple((id(value), value._version) for value in self.values()) != self._metadata
                or not all(_same_bytes(value, frozen)
                           for value, frozen in zip(self.values(), self._snapshots, strict=True))):
            raise ValueError("Accepted surrogate context changed")


@dataclass(frozen=True)
class RawPrecisionLimits:
    dual_steps: int = 512
    scalar_calls: int = 8

    def __post_init__(self):
        for (name, value), ceiling in zip(asdict(self).items(), (512, 8), strict=True):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be a positive integer no larger than {ceiling}")


@dataclass(frozen=True)
class RawPrecisionResult:
    x: torch.Tensor
    dual: torch.Tensor
    objective: torch.Tensor
    audit: Audit
    accepted: bool
    work: dict
    qp_work: dict | None


def _componentwise_only(audit):
    return (not audit.qualified and audit.status == "componentwise_unresolved"
            and audit.signed_direction_count == 2
            and all(audit.diagnostics.get(sign, {}).get("status") == "qualified_lower_bound"
                    for sign in ("positive", "negative")))


def _nonincrease(before, after):
    margin = 32 * torch.finfo(before.dtype).eps * (1 + before.abs())
    return bool(torch.isfinite(before) & torch.isfinite(after) & (after <= before + margin))


def _literal_context(model, x, dual, caps, context, policy, work):
    context.validate()
    h, target, start, endpoint = context.values()
    if not _same_bytes(endpoint, x):
        return None, "accepted_endpoint_changed"
    if (h.shape != x.shape or h.device != x.device
            or not bool(torch.isfinite(torch.cat(context.values())).all() & (h > 0).all()
                        & (start >= model.lower).all() & (start <= model.upper).all())):
        return None, "invalid_accepted_surrogate"
    losses, gradient, _, _, left, right = model.terms(start)
    gradient = torch.where(start == model.lower, right, gradient)
    gradient = torch.where(start == model.upper, left, gradient)
    if not _same_bytes(start - gradient / h, target):
        return None, "literal_target_changed"
    problem = (h, target, model.lower, model.upper, caps)
    eager = tensor_kernels.gap_kkt(x, dual, *problem)
    work["context_eager_certificate_evaluations"] += 1
    compiled = model.kernels.gap_kkt(x, dual, *problem)
    work["context_compiled_certificate_evaluations"] += 1
    if not (_qualified(eager, policy) and _qualified(compiled, policy)
            and _certificates_agree(eager, compiled, x.numel())):
        return None, "accepted_surrogate_certificate_rejected"
    return (losses, gradient), "qualified_literal_context"


def recover_raw(model, x, dual, caps, policy=CudaPolicy(), *, accepted_qp_context=None,
                allow_qp_precision=True, limits=RawPrecisionLimits()):
    """Try one literal-QP repair, fixed-primal flow and exact-block scalar moves.

    Only a final negative start is eligible. Ordinary outer/QP/scalar policy
    budgets and all original acceptance thresholds remain unchanged. The
    explicitly additional work here is bounded and reported separately.
    Neither a QP nor scalar certificate substitutes for the full raw audit.
    """
    if not isinstance(limits, RawPrecisionLimits) or not isinstance(allow_qp_precision, bool):
        raise ValueError("Validated raw limits and a Boolean per-start QP allowance required")
    if accepted_qp_context is not None and not isinstance(accepted_qp_context, AcceptedQPContext):
        raise ValueError("A literal accepted surrogate context is required")
    if (x.shape != (model.n,) or dual.shape != (model.n, model.n) or caps.shape != dual.shape
            or any(value.dtype != torch.float64 or value.device != model.device for value in (x, dual, caps))):
        raise ValueError("Raw recovery requires original-device FP64 node/edge tensors")
    inputs, frozen = (x, dual, caps), tuple(value.clone() for value in (x, dual, caps))
    counter_before = dict(model.scalar_work_counters)
    work = dict(attempted=False, candidate_accepted=False, status="not_attempted", limits=asdict(limits),
                audit_calls=0, audit_seconds=0., audit_cut_iterations=0,
                scalar_calls=0, scalar_subdivisions=0, scalar_seconds=0., dual_steps=0,
                objective_evaluations=0, context_eager_certificate_evaluations=0,
                context_compiled_certificate_evaluations=0, affine_blocks=[],
                qp_context_status="not_requested", qp_seconds=0.)
    qp_work = None

    def objective(point):
        work["objective_evaluations"] += 1
        return model.loss(point).sum() + .5 * (caps * differences(point).abs()).sum()

    def audit(point, flow):
        began = perf_counter()
        result = audit_raw(model, point, flow, caps, policy=policy)
        work["audit_calls"] += 1
        work["audit_seconds"] += perf_counter() - began
        work["audit_cut_iterations"] += sum(
            result.diagnostics.get(sign, {}).get("iterations", 0) for sign in ("positive", "negative"))
        return result

    def finish(point, flow, value, certificate, accepted, status):
        work.update(candidate_accepted=accepted, status=status)
        return RawPrecisionResult(point, flow, value, certificate, accepted, work, qp_work)

    try:
        with model.validated_stage():
            original_value = objective(x)
            original_audit = audit(x, dual)
            work.update(original_residual=original_audit.residual if isfinite(original_audit.residual) else None,
                        original_residual_finite=isfinite(original_audit.residual))
            current, flow, value, certificate = x, dual, original_value, original_audit
            if original_audit.qualified or not bool(torch.isfinite(original_value)):
                return finish(x, dual, original_value, original_audit, False, "original_already_qualified_or_nonfinite")
            if (original_audit.status in ("infeasible", "nonfinite_likelihood_derivatives")
                    or original_audit.status.endswith("_invalid_cut")
                    or not isfinite(original_audit.residual)):
                return finish(x, dual, original_value, original_audit, False, "original_state_ineligible")
            work["attempted"] = True

            if allow_qp_precision and accepted_qp_context is not None:
                source, work["qp_context_status"] = _literal_context(
                    model, x, dual, caps, accepted_qp_context, policy, work)
                if source is not None:
                    h, target, start, _ = accepted_qp_context.values()
                    began = perf_counter()
                    repaired = recover_quadratic(x, dual, h, target, model.lower, model.upper,
                                                 caps, model.kernels, policy)
                    work["qp_seconds"] += perf_counter() - began
                    qp_work = repaired.work
                    if repaired.qualified and qp_work["candidate_accepted"]:
                        losses, gradient = source
                        displacement = repaired.x - start
                        likelihood = model.loss(repaired.x).sum()
                        original_start_value = objective(start)
                        increment = gradient * displacement + .5 * h * displacement.square()
                        major = losses.sum() + increment.sum()
                        penalty = .5 * (caps * differences(repaired.x).abs()).sum()
                        previous_penalty = .5 * (caps * differences(start).abs()).sum()
                        work["objective_evaluations"] += 1
                        trial_value = likelihood + penalty
                        surrogate_delta = increment.sum() + penalty - previous_penalty
                        margin = 64 * torch.finfo(x.dtype).eps * (1 + original_start_value.abs() + major.abs())
                        majorization = (torch.isfinite(trial_value) & torch.isfinite(likelihood)
                                        & torch.isfinite(major) & torch.isfinite(surrogate_delta)
                                        & torch.isfinite(margin)
                                        & (likelihood <= major + margin)
                                        & (surrogate_delta <= margin)
                                        & (trial_value <= original_start_value + margin))
                        work["qp_original_mm_pass"] = bool(majorization)
                        work["qp_terminal_objective_pass"] = _nonincrease(original_value, trial_value)
                        if bool(majorization) and work["qp_terminal_objective_pass"]:
                            current, flow, value = repaired.x, repaired.dual, trial_value
                            certificate = audit(current, flow)

            # Raw flow is useful only after the original signed subset gates
            # have already passed. It cannot erase a descent/unresolved cut.
            if _componentwise_only(certificate):
                _, nominal, _, _, left, right = model.terms(current)
                at_lower, at_upper = current == model.lower, current == model.upper
                fixed = model.lower == model.upper
                same = differences(current) == 0
                external = caps * differences(current).sign()
                sizes = same.sum(-1).to(current.dtype)
                step = model.kernels.raw_dual_step
                for iteration in range(1, limits.dual_steps + 1):
                    flow = step(flow, caps, same, external, sizes, nominal, left, right,
                                at_lower, at_upper, fixed)
                    work["dual_steps"] += 1
                    if iteration % policy.check_every == 0 or iteration == limits.dual_steps:
                        certificate = audit(current, flow)
                        if not _componentwise_only(certificate):
                            break

            if _componentwise_only(certificate):
                delta = differences(current)
                external_valid = bool(torch.where(delta != 0, flow == caps * delta.sign(), True).all())
                if external_valid:
                    _, nominal, _, _, left, right = model.terms(current)
                    residual, gradient, divergence = raw_residual(
                        flow, nominal, left, right, current == model.lower,
                        current == model.upper, model.lower == model.upper)
                    node_residual = residual.abs() / (1 + gradient.abs() + divergence.abs())
                    centers, labels = torch.unique(current, sorted=True, return_inverse=True)
                    scores = torch.zeros_like(centers)
                    scores.scatter_reduce_(0, labels, node_residual, reduce="amax", include_self=True)
                    order = torch.argsort(scores, descending=True, stable=True)
                    original_signs = delta.sign()
                    for group in order.tolist():
                        if work["scalar_calls"] >= limits.scalar_calls or not bool(scores[group] > policy.stationarity_tol):
                            break
                        indices = torch.nonzero(labels == group, as_tuple=True)[0]
                        center = current[indices[0]]
                        lower, upper = model.lower[indices].max(), model.upper[indices].min()
                        others = current[labels != group]
                        below, above = others[others < center], others[others > center]
                        if below.numel():
                            lower = torch.maximum(lower, torch.nextafter(below.max(), center.new_tensor(float("inf"))))
                        if above.numel():
                            upper = torch.minimum(upper, torch.nextafter(above.min(), center.new_tensor(-float("inf"))))
                        if not bool((lower <= center) & (center <= upper) & (lower < upper)):
                            continue
                        # Each external unordered edge is counted once. Exact
                        # internal ties contribute zero, independent of q.
                        coefficient = (caps[indices] * (center - current)[None, :].sign()).sum()
                        subset = model.subset(indices)
                        base = Problems(subset, torch.tensor([indices.numel()], dtype=torch.long, device=x.device))
                        problem = AffineProblem(base, lower, upper, coefficient, center)
                        began = perf_counter()
                        scalar = solve_affine_scalar(problem, policy)
                        scalar_qualified = bool(scalar.qualified.all())
                        work["scalar_calls"] += 1
                        work["scalar_subdivisions"] += scalar.subdivisions
                        work["scalar_seconds"] += perf_counter() - began
                        record = dict(rows=indices.numel(), initial_residual=float(scores[group]),
                                      qualified=scalar_qualified, accepted=False,
                                      subdivisions=scalar.subdivisions, original_edge_signs_preserved=False)
                        work["affine_blocks"].append(record)
                        if not record["qualified"]:
                            continue
                        trial = current.clone()
                        trial[indices] = scalar.phi[0]
                        record["original_edge_signs_preserved"] = torch.equal(differences(trial).sign(), original_signs)
                        trial_value = objective(trial)
                        record["objective_nonincrease"] = _nonincrease(value, trial_value)
                        if record["original_edge_signs_preserved"] and record["objective_nonincrease"]:
                            current, value = trial, trial_value
                            record["accepted"] = True

            # Always audit the actual final tensors. No intermediate scalar,
            # QP or prior raw certificate can qualify this endpoint.
            final_value = objective(current)
            final_audit = audit(current, flow)
            accepted = final_audit.qualified and _nonincrease(original_value, final_value)
            work.update(proposal_audit_status=final_audit.status,
                        proposal_residual=final_audit.residual if isfinite(final_audit.residual) else None,
                        proposal_residual_finite=isfinite(final_audit.residual),
                        original_objective_nonincrease=_nonincrease(original_value, final_value))
            if accepted:
                return finish(current, flow, final_value, final_audit, True, "qualified")
            return finish(x, dual, original_value, original_audit, False, "final_raw_gate_rejected")
    finally:
        work["model_work"] = {name: model.scalar_work_counters[name] - value
                              for name, value in counter_before.items()}
        if not all(_same_bytes(value, original) for value, original in zip(inputs, frozen, strict=True)):
            raise ArithmeticError("Raw precision modified its original endpoint or graph")
        if accepted_qp_context is not None:
            accepted_qp_context.validate()
