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


def solve_start(model, graph, lam, start, policy=CudaPolicy()):
    with model.validated_stage(), graph.validated_stage():
        return _solve_start(model, graph, lam, start, policy)


def _solve_start(model, graph, lam, start, policy):
    x = start.clamp(model.lower, model.upper).clone()
    caps = graph.weights * lam
    q = torch.zeros_like(caps)
    current = objective(model, x, caps)
    inflation = 1.
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
        diagnostics = dict(status=status, raw_stationarity_qualified=bool(audit and audit.qualified),
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
                           inner_gap_qualified=bool(last_inner and last_inner.qualified),
                           inner_kkt_qualified=bool(last_inner and last_inner.qualified),
                           inner_gap=None if last_inner is None else float(last_inner.gap),
                           inner_gap_scale=None if last_inner is None else float(last_inner.scale),
                           inner_kkt_residual=None if last_inner is None else float(last_inner.kkt),
                           inner_polish_iterations=0 if last_inner is None else last_inner.polish_iterations,
                           inner_certificate_scope="last_solved_surrogate_not_raw_likelihood")
        diagnostics.update(extra)
        return RawFit(x, q, current, None, qualified, diagnostics)

    if not bool(torch.isfinite(current) & torch.isfinite(caps).all()):
        return result(None, False, "nonfinite_initial_objective", 0)
    for iteration in range(policy.outer_max_iterations):
        graph.validate_metadata()
        losses, grad, curv, _, left, right = model.terms(x)
        grad = torch.where(x == model.lower, right, grad)
        grad = torch.where(x == model.upper, left, grad)
        if not bool(torch.isfinite(losses).all() & torch.isfinite(grad).all() & torch.isfinite(curv).all()):
            return result(None, False, "nonfinite_likelihood_surrogate", iteration)
        base_h = curv.clamp_min(1.)
        accepted = False
        for _ in range(policy.max_backtracks):
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
                return result(None, False, "surrogate_unresolved", iteration)
            # The QP can qualify even when the likelihood majorization trial is
            # rejected. Its physical dual remains a valid initialization for the
            # changed surrogate, whose certificate must be recomputed from scratch.
            warm_dual = QualifiedDualWarmState.from_fit(last_inner, graph, lambda_literal)
            trial = last_inner.x
            d = trial - x
            trial_loss = model.loss(trial).sum()
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
            if bool(gate):
                accepted = True
                old_value = current
                x, q, current = trial, last_inner.dual, trial_value
                inflation = max(1., inflation / 2.)
                decrease = old_value - current
                tail.append(float(current))
                tail = tail[-8:]
                break
            inflation *= 2.
            backtracks += 1
        if accepted and not bool(decrease.abs() <= 1e-10 * (1 + current.abs())):
            continue
        audit = run_audit()
        if audit.qualified:
            return result(audit, True, "qualified", iteration + 1)
        if audit.direction is not None:
            restart = direction_restart(model, x, audit.direction, caps, current, policy)
            if restart is not None:
                x = restart
                current = objective(model, x, caps)
                q = torch.zeros_like(caps)
                warm_dual = None
                inflation = 1.
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


def fit_lambda(model, graph, pilots, lam, previous=None, policy=CudaPolicy()):
    with model.validated_stage(), graph.validated_stage():
        return _fit_lambda(model, graph, pilots, lam, previous, policy)


def _fit_lambda(model, graph, pilots, lam, previous, policy):
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
        return RawFit(x, caps, value, None, True,
                      dict(status="qualified_separable", separable_scalar_gap_qualified=True,
                           separable_global_gap=float(gap), clonal_constraint=False,
                           raw_stationarity_qualified=audit.qualified,
                           directional_stationarity_qualified=audit.qualified,
                           raw_branch_stationarity_qualified=audit.qualified,
                           directional_qualified=audit.qualified, box_feasible=True,
                           stationarity_residual=audit.residual, audit_status=audit.status,
                           audit_diagnostics=audit.diagnostics,
                           audit_signed_direction_count=audit.signed_direction_count,
                           inner_gap_qualified=True, inner_gap=0., inner_gap_scale=0.,
                           inner_kkt_qualified=True, inner_kkt_residual=0.,
                           inner_iterations=0, qp_admm_iterations=0, outer_iterations=0, surrogate_qp_calls=0,
                           qp_calls=0, qp_dual_warm_starts=0, qp_seconds=0.0,
                           qp_dual_warm_resets=0,
                           qp_polish_iterations=0, inner_polish_iterations=0,
                           audit_calls=1, audit_seconds=audit_seconds,
                           phase_timing_scope="host wall through device-qualified QP/audit returns",
                           starts_attempted=0, search_complete=True, global_optimality_proven=False))
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
    for initial in starts:
        result = solve_start(model, graph, lam, initial, policy)
        finite_objective = bool(torch.isfinite(result.objective))
        if not finite_objective:
            result.qualified = False
            result.diagnostics.update(status="nonfinite_raw_objective")
        diagnostics.append(dict(result.diagnostics, objective=float(result.objective) if finite_objective else None,
                                qualified=result.qualified))
        if result.qualified and (best is None or bool(result.objective < best.objective)):
            best = result
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
