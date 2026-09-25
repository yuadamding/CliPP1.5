"""Observed-likelihood majorization over original boxes on a complete graph.

No mutation or cluster is required to attain one. Clonal designation occurs
only after independent membership refitting and never changes these solutions.
"""
from dataclasses import dataclass
from time import perf_counter
import torch
from .kernels import differences
from .qp import QualifiedDualWarmState, solve_qp
from .audit import audit_raw
from .policy import CudaPolicy, QualificationError


@dataclass
class RawFit:
    x: torch.Tensor
    dual: torch.Tensor
    objective: torch.Tensor
    witness: int | None
    qualified: bool
    diagnostics: dict


@dataclass(frozen=True)
class PrimalWarmState:
    """One immutable primal continuation tied to the exact frozen graph object."""
    x: torch.Tensor
    graph_identity: object

    def __post_init__(self):
        object.__setattr__(self, "x", self.x.detach().clone())
        object.__setattr__(self, "_version", (id(self.x), self.x._version))
        object.__setattr__(self, "_snapshot", self.x.clone())

    def validate(self, graph):
        graph.validate()
        if (self.graph_identity is not graph.identity or
                self.x.shape != graph.pilot.shape or self.x.dtype != graph.pilot.dtype or
                self.x.device != graph.pilot.device or
                (id(self.x), self.x._version) != self._version or
                not bool(torch.isfinite(self.x).all()) or not torch.equal(self.x, self._snapshot)):
            raise ValueError("Continuation must contain an unchanged finite primal bound to this graph")


def objective(model, x, caps):
    return model.loss(x).sum() + .5 * (caps * differences(x).abs()).sum()


def bound_majorization_curvature(model, x, curvature, posterior):
    """Match an active-box one-sided derivative without changing likelihood terms.

    At a represented clipping kink, ``terms`` can have zero moving curvature
    while the feasible one-sided derivative is nonzero. Add only that missing
    interior-side curvature for the surrogate. The likelihood API, pilots and
    curvature-weighted pooled starts retain their original arithmetic.
    """
    slope = model.slope
    mass = x[:, None] * slope
    safe_slope = torch.where(slope > 0, slope, torch.ones_like(slope))
    low = torch.full_like(slope, model.eps) / safe_slope
    high = torch.full_like(slope, 1 - model.eps) / safe_slope
    free = model.lower < model.upper
    inward = (((x == model.lower) & free)[:, None] & (x[:, None] == low)) | (
        ((x == model.upper) & free)[:, None] & (x[:, None] == high))
    moving = (mass > model.eps) & (mass < 1 - model.eps)
    missing = inward & ~moving & (slope > 0)
    p = mass.clamp(model.eps, 1 - model.eps)
    addition = (posterior * torch.where(missing, slope, 0.).square() *
                (model.alt[:, None] / p.square() +
                 model.ref[:, None] / (1 - p).square())).sum(-1)
    return curvature + addition


def direction_restart(model, x, direction, caps, current, policy):
    distance = torch.where(direction > 0, model.upper - x,
                           torch.where(direction < 0, x - model.lower, float("inf")))
    maximum = (distance / direction.abs().clamp_min(torch.finfo(x.dtype).tiny)).min()
    step = maximum.clamp_max(1e-3)
    for _ in range(policy.max_backtracks):
        trial = (x + step * direction).clamp(model.lower, model.upper)
        value = objective(model, trial, caps)
        margin = 32 * torch.finfo(x.dtype).eps * (1 + current.abs())
        if bool(torch.isfinite(value) & (value < current - margin)):
            return trial
        step = step * .5
    return None


def inner_certificate_diagnostics(fitted, policy):
    """Report each literal QP gate independently; neither flag admits a QP.

    The combined result retains the solver's feasibility/validity checks. A
    missing certificate has no measured values and cannot pass either gate.
    Compute the allowance on the certificate device in the admission order.
    """
    allowance = None if fitted is None else policy.inner_atol + policy.inner_rtol * fitted.scale
    gap_pass = bool(fitted is not None and torch.isfinite(fitted.gap)
                    & torch.isfinite(fitted.scale) & torch.isfinite(allowance)
                    & (fitted.gap >= 0) & (fitted.scale >= 0)
                    & (fitted.gap <= allowance))
    kkt_pass = bool(fitted is not None and torch.isfinite(fitted.kkt)
                    & (fitted.kkt >= 0)
                    & (fitted.kkt <= policy.inner_kkt_tol))
    return dict(inner_gap_pass=gap_pass, inner_kkt_pass=kkt_pass,
                inner_qp_qualified=bool(fitted is not None and fitted.qualified),
                # Retain field names consumed by older diagnostic readers,
                # with their intended individual-test meaning corrected.
                inner_gap_qualified=gap_pass, inner_kkt_qualified=kkt_pass,
                inner_gap=None if fitted is None else float(fitted.gap),
                inner_gap_scale=None if fitted is None else float(fitted.scale),
                inner_gap_allowed=None if allowance is None else float(allowance),
                inner_kkt_residual=None if fitted is None else float(fitted.kkt),
                inner_kkt_allowed=policy.inner_kkt_tol,
                inner_certificate_present=fitted is not None,
                inner_certificate_scope="last_solved_surrogate_not_raw_likelihood")


def solve_start(model, graph, lam, start, policy=CudaPolicy(), *,
                surrogate_policy="scalar_backtracking_v1", observer=None):
    with model.validated_stage(), graph.validated_stage():
        if surrogate_policy == "scalar_backtracking_v1" and observer is None:
            return _solve_start(model, graph, lam, start, policy)
        return _solve_start(model, graph, lam, start, policy,
                            surrogate_policy=surrogate_policy, observer=observer)


def coordinate_inflation(inflation, losses, gradient_step, quadratic_step, trial_losses):
    """Research policy: double only rows violating their local quadratic model.

    The conservative row margins identify proposals, never admit an iterate.
    The original aggregate majorization, QP and observed-objective gates still
    apply. If no row explains a rejected aggregate trial, retain the original
    all-coordinate backtrack instead of accepting or weakening that gate.
    """
    major = losses + gradient_step + quadratic_step
    margin = 64 * torch.finfo(losses.dtype).eps * (
        1 + losses.abs() + gradient_step.abs() + quadratic_step.abs() + trial_losses.abs())
    failed = (~torch.isfinite(trial_losses) | ~torch.isfinite(major)
              | ~torch.isfinite(margin) | (trial_losses > major + margin))
    return inflation * torch.where(failed | ~failed.any(), 2., 1.)


def _solve_start(model, graph, lam, start, policy, *,
                 surrogate_policy="scalar_backtracking_v1", observer=None):
    if surrogate_policy not in ("scalar_backtracking_v1", "coordinate_backtracking_v1"):
        raise ValueError("Unknown outer surrogate policy")
    coordinate = surrogate_policy == "coordinate_backtracking_v1"
    x = start.clamp(model.lower, model.upper).clone()
    caps = graph.weights * lam
    q = torch.zeros_like(caps)
    current = objective(model, x, caps)
    inflation = torch.ones_like(x) if coordinate else 1.
    backtracks = surrogate_calls = inner = restarts = polish_iterations = 0
    tail = []
    last_inner = None
    warm_dual = None
    lambda_literal = float(lam)
    qp_seconds = audit_seconds = 0.0
    audit_calls = dual_warm_starts = dual_warm_resets = 0

    def run_audit():
        nonlocal audit_seconds, audit_calls
        began = perf_counter()
        audited = audit_raw(model, x, q, caps, None, policy)
        # Audit and QP return through device-scalar qualification checkpoints;
        # measuring that stage needs no additional per-iteration synchronization.
        audit_seconds += perf_counter() - began
        audit_calls += 1
        return audited

    def result(audit, qualified, status, iterations, **extra):
        qualified = qualified and bool(torch.isfinite(current))
        diagnostics = dict(status=status, outer_surrogate_policy=surrogate_policy,
                           raw_stationarity_qualified=bool(audit and audit.qualified),
                           directional_stationarity_qualified=bool(audit and audit.qualified),
                           raw_branch_stationarity_qualified=bool(audit and audit.qualified),
                           directional_qualified=bool(audit and audit.qualified),
                           box_feasible=bool(((x >= model.lower) & (x <= model.upper)).all()),
                           clonal_constraint=False, global_optimality_proven=False,
                           outer_iterations=iterations, inner_iterations=inner,
                           qp_admm_iterations=inner,
                           surrogate_qp_calls=surrogate_calls, backtracks=backtracks,
                           qp_calls=surrogate_calls, qp_dual_warm_starts=dual_warm_starts,
                           qp_dual_warm_resets=dual_warm_resets,
                           qp_polish_iterations=polish_iterations,
                           qp_seconds=qp_seconds, audit_seconds=audit_seconds,
                           audit_calls=audit_calls,
                           phase_timing_scope="host wall through device-qualified QP/audit returns",
                           direction_restarts=restarts, objective_tail=tail,
                           stationarity_residual=None if audit is None else audit.residual,
                           audit_status=None if audit is None else audit.status,
                           audit_diagnostics={} if audit is None else audit.diagnostics,
                           audit_signed_direction_count=0 if audit is None else audit.signed_direction_count,
                           inner_polish_iterations=0 if last_inner is None else last_inner.polish_iterations,
                           **inner_certificate_diagnostics(last_inner, policy))
        diagnostics.update(extra)
        return RawFit(x, q, current, None, qualified, diagnostics)

    if not bool(torch.isfinite(current) & torch.isfinite(caps).all()):
        return result(None, False, "nonfinite_initial_objective", 0)
    for iteration in range(policy.outer_max_iterations):
        graph.validate_metadata()
        losses, grad, curv, posterior, left, right = model.terms(x)
        grad = torch.where(x == model.lower, right, grad)
        grad = torch.where(x == model.upper, left, grad)
        curv = bound_majorization_curvature(model, x, curv, posterior)
        if not bool(torch.isfinite(losses).all() & torch.isfinite(grad).all() & torch.isfinite(curv).all()):
            return result(None, False, "nonfinite_likelihood_surrogate", iteration)
        base_h = curv.clamp_min(1.)
        accepted = False
        for backtrack_index in range(policy.max_backtracks):
            h = base_h * inflation
            target = x - grad / h
            if not bool(torch.isfinite(h).all() & torch.isfinite(target).all()):
                return result(None, False, "nonfinite_majorization_surrogate", iteration)
            initial_dual = None if warm_dual is None else warm_dual.for_problem(graph, lambda_literal)
            began = perf_counter()
            last_inner = solve_qp(h, target, model.lower, model.upper, caps,
                                  model.kernels, policy, start=x, dual=initial_dual)
            qp_seconds += perf_counter() - began
            dual_warm_starts += initial_dual is not None
            surrogate_calls += 1
            inner += last_inner.iterations
            polish_iterations += last_inner.polish_iterations
            if not last_inner.qualified:
                if observer is not None:
                    observer("unresolved_qp", dict(
                        iteration=iteration, backtrack_index=backtrack_index,
                        inflation=inflation, base_h=base_h, h=h, target=target,
                        lower=model.lower, upper=model.upper, caps=caps,
                        start=x, dual=initial_dual, fitted=last_inner))
                return result(None, False, "surrogate_unresolved", iteration)
            # The QP can qualify even when the likelihood majorization trial is
            # rejected. Its physical dual remains a valid initialization for the
            # changed surrogate, whose certificate must be recomputed from scratch.
            warm_dual = QualifiedDualWarmState.from_fit(last_inner, graph, lambda_literal)
            trial = last_inner.x
            d = trial - x
            trial_losses = model.loss(trial)
            trial_loss = trial_losses.sum()
            major = losses.sum() + (grad * d + .5 * h * d.square()).sum()
            penalty_trial = .5 * (caps * differences(trial).abs()).sum()
            trial_value = trial_loss + penalty_trial
            surrogate_delta = ((grad * d + .5 * h * d.square()).sum() + penalty_trial -
                               .5 * (caps * differences(x).abs()).sum())
            margin = 64 * torch.finfo(x.dtype).eps * (1 + current.abs() + major.abs())
            finite_gate = (torch.isfinite(trial_value) & torch.isfinite(trial_loss) &
                           torch.isfinite(major) & torch.isfinite(surrogate_delta) & torch.isfinite(margin))
            gate = (finite_gate & (trial_loss <= major + margin) &
                    (surrogate_delta <= margin) & (trial_value <= current + margin))
            if observer is not None:
                observer("outer_trial", dict(
                    iteration=iteration, backtrack_index=backtrack_index,
                    inflation=inflation, base_h=base_h, h=h, losses=losses,
                    gradient_step=grad * d, quadratic_step=.5 * h * d.square(),
                    trial_losses=trial_losses, accepted=bool(gate),
                    finite_gate=bool(finite_gate), margin=margin,
                    majorization_slack=major + margin - trial_loss,
                    surrogate_slack=margin - surrogate_delta,
                    objective_slack=current + margin - trial_value))
            if bool(gate):
                accepted = True
                old_value = current
                x, q, current = trial, last_inner.dual, trial_value
                inflation = (inflation / 2.).clamp_min(1.) if coordinate else max(1., inflation / 2.)
                decrease = old_value - current
                tail.append(float(current))
                tail = tail[-8:]
                break
            inflation = (coordinate_inflation(inflation, losses, grad * d, .5 * h * d.square(),
                                              trial_losses) if coordinate else inflation * 2.)
            backtracks += 1
        if accepted and not bool(decrease.abs() <= 1e-10 * (1 + current.abs())):
            continue
        audit = run_audit()
        if audit.qualified:
            return result(audit, True, "qualified", iteration + 1)
        if (audit.status.endswith("_invalid_cut") or
                audit.status == "nonfinite_likelihood_derivatives"):
            # An invalid audit problem supplies neither a certificate nor a
            # usable restart direction. Retrying the same start cannot repair
            # that evidence; retain its diagnostics and try the other starts.
            return result(audit, False, "raw_audit_unresolved", iteration + 1)
        if audit.direction is not None:
            restart = direction_restart(model, x, audit.direction, caps, current, policy)
            if restart is not None:
                x = restart
                current = objective(model, x, caps)
                q = torch.zeros_like(caps)
                warm_dual = None
                inflation = torch.ones_like(x) if coordinate else 1.
                restarts += 1
                continue
        # An unchanged, gap-qualified QP state may fail the stricter raw kink
        # audit. Do not repeatedly admit that same warm state without ADMM work.
        if warm_dual is not None:
            warm_dual = None
            dual_warm_resets += 1
        if not accepted:
            break
    audit = run_audit()
    return result(audit, audit.qualified, "qualified" if audit.qualified else "outer_unresolved", iteration + 1)


def _validate_pilots(model, graph, pilots, policy):
    fields = (pilots.phi, pilots.loss, pilots.lower_bound, pilots.gap, pilots.alternative)
    if any(t.shape != (model.n,) or t.dtype != torch.float64 or t.device != model.device for t in fields):
        raise ValueError("Pilots must match the model float64 node vectors")
    if (pilots.qualified.shape != (model.n,) or pilots.qualified.dtype != torch.bool or
            pilots.qualified.device != model.device):
        raise ValueError("Pilot qualification mask must match the model")
    actual = model.loss(pilots.phi)
    margin = 256 * torch.finfo(actual.dtype).eps * (1 + actual.abs() + pilots.loss.abs())
    valid = (torch.stack([torch.isfinite(v).all() for v in fields]).all() &
             pilots.qualified.all() & (pilots.phi >= model.lower).all() &
             (pilots.phi <= model.upper).all() & (pilots.alternative >= model.lower).all() &
             (pilots.alternative <= model.upper).all() & (pilots.gap >= 0).all() &
             (pilots.gap <= policy.scalar_atol + policy.scalar_rtol * pilots.loss.abs()).all() &
             ((actual - pilots.loss).abs() <= margin).all() &
             ((pilots.loss - pilots.lower_bound - pilots.gap).abs() <= margin).all() &
             (pilots.phi == graph.pilot).all())
    if not bool(valid):
        raise QualificationError("Separable scalar pilots do not qualify for this model and graph")


def fit_lambda(model, graph, pilots, lam, previous=None, policy=CudaPolicy(), *, on_qualified=None):
    """Keep raw-objective continuation; stream qualified starts to an observer.

    The observer must not modify the RawFit or its tensors. It may retain one
    selected reference, but no history of dense dual matrices is accumulated.
    """
    with model.validated_stage(), graph.validated_stage():
        return _fit_lambda(model, graph, pilots, lam, previous, policy, on_qualified=on_qualified)


def _fit_lambda(model, graph, pilots, lam, previous, policy, *, on_qualified=None):
    lam = torch.as_tensor(lam, dtype=torch.float64, device=model.device)
    if lam.ndim != 0 or not bool(torch.isfinite(lam) & (lam >= 0)):
        raise ValueError("Penalty must be a finite nonnegative scalar")
    if graph.n != model.n or graph.weights.device != model.device:
        raise ValueError("Graph must match the likelihood node count and device")
    if graph.mutation_ids and graph.mutation_ids != model.mutation_ids:
        raise ValueError("Graph mutation identities do not match the likelihood")
    if previous is not None:
        if not isinstance(previous, PrimalWarmState):
            raise ValueError("Continuation requires a graph-bound PrimalWarmState")
        previous.validate(graph)
    _validate_pilots(model, graph, pilots, policy)
    if bool(lam == 0):
        x = pilots.phi.clone()
        caps = torch.zeros_like(graph.weights)
        value, gap = model.loss(x).sum(), pilots.gap.sum()
        if not bool(torch.isfinite(value) & torch.isfinite(gap)):
            raise QualificationError("Separable objective or aggregate scalar gap is nonfinite")
        began = perf_counter()
        audit = audit_raw(model, x, caps, caps, None, policy)
        audit_seconds = perf_counter() - began
        result = RawFit(x, caps, value, None, True,
                      dict(status="qualified_separable", separable_scalar_gap_qualified=True,
                           separable_global_gap=float(gap), clonal_constraint=False,
                           raw_stationarity_qualified=audit.qualified,
                           directional_stationarity_qualified=audit.qualified,
                           raw_branch_stationarity_qualified=audit.qualified,
                           directional_qualified=audit.qualified, box_feasible=True,
                           stationarity_residual=audit.residual, audit_status=audit.status,
                           audit_diagnostics=audit.diagnostics,
                           audit_signed_direction_count=audit.signed_direction_count,
                           **dict(inner_certificate_diagnostics(None, policy),
                                  inner_certificate_scope="not_applicable_separable_lambda_zero"),
                           inner_iterations=0, qp_admm_iterations=0, outer_iterations=0, surrogate_qp_calls=0,
                           qp_calls=0, qp_dual_warm_starts=0, qp_seconds=0.0,
                           qp_dual_warm_resets=0,
                           qp_polish_iterations=0, inner_polish_iterations=0,
                           audit_calls=1, audit_seconds=audit_seconds,
                           phase_timing_scope="host wall through device-qualified QP/audit returns",
                           starts_attempted=0, search_complete=True, global_optimality_proven=False))
        if on_qualified is not None:
            on_qualified(result, "separable")
        return result
    curvature = model.terms(pilots.phi)[2].clamp_min(1.)
    if not bool(torch.isfinite(curvature).all() & torch.isfinite(graph.weights * lam).all()):
        raise QualificationError("Complete-graph penalty caps or pilot curvature are nonfinite")
    normalized_curvature = curvature / curvature.max()
    pooled_value = (normalized_curvature * pilots.phi).sum() / normalized_curvature.sum()
    pooled = pooled_value.expand_as(pilots.phi).clamp(model.lower, model.upper)
    starts = []
    for x in (None if previous is None else previous.x, pilots.phi, pooled, pilots.alternative):
        if x is None:
            continue
        x = x.clamp(model.lower, model.upper).clone()
        if not any(torch.equal(x, old) for old in starts):
            starts.append(x)
    best, diagnostics = None, []
    for start_index, initial in enumerate(starts):
        result = solve_start(model, graph, lam, initial, policy)
        finite_objective = bool(torch.isfinite(result.objective))
        if not finite_objective:
            result.qualified = False
            result.diagnostics.update(status="nonfinite_raw_objective")
        diagnostics.append(dict(result.diagnostics, objective=float(result.objective) if finite_objective else None,
                                qualified=result.qualified))
        if result.qualified and (best is None or bool(result.objective < best.objective)):
            best = result
        if result.qualified and on_qualified is not None:
            on_qualified(result, f"start_{start_index}")
    coverage = dict(starts=diagnostics, search_complete=all(d["qualified"] for d in diagnostics),
                    starts_attempted=len(starts), starts_qualified=sum(d["qualified"] for d in diagnostics),
                    starts_unresolved=sum(not d["qualified"] for d in diagnostics),
                    search_policy="unconstrained_complete_graph_multistart_v1", clonal_constraint=False)
    # Selected-start diagnostics remain in starts; phase totals cover all attempted starts.
    for name in ("qp_calls", "qp_admm_iterations", "qp_dual_warm_starts", "qp_dual_warm_resets", "qp_polish_iterations", "qp_seconds", "audit_calls", "audit_seconds"):
        coverage[name] = sum(row[name] for row in diagnostics)
    if best is None:
        raise QualificationError("No qualified unconstrained complete-graph start", **coverage)
    best.diagnostics.update(coverage)
    return best
