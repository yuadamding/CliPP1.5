"""Control-flow regression for the per-start budget, separate from QP accuracy."""
from types import SimpleNamespace

import pytest
import torch

from clipp1d.cuda import solver
from clipp1d.cuda.audit import Audit
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.qp import QuadraticFit


@pytest.fixture
def tensor():
    if not torch.cuda.is_available():
        pytest.skip("Allocated CUDA control regression")
    return lambda value: torch.tensor(value, dtype=torch.float64, device="cuda:0")


def work():
    return dict(attempted=True, qualified=True, candidate_accepted=True, stop_reason="qualified",
                **{key: (3 if key == "projected_flow_steps" else 0)
                   for key in solver._PRECISION_COUNTERS})


def problem(tensor):
    start = tensor([.8, .8])
    graph = SimpleNamespace(weights=tensor([[0., 0.], [0., 0.]]), identity=object(),
                            validate=lambda: None, validate_metadata=lambda: None)

    def terms(x):
        gradient = x - .2
        return .5 * gradient.square(), gradient, torch.ones_like(x), None, gradient, gradient

    model = SimpleNamespace(lower=torch.zeros_like(start), upper=torch.ones_like(start),
                            kernels=None, loss=lambda x: .5 * (x - .2).square(), terms=terms)
    return model, graph, start


@pytest.mark.parametrize("first_recovered", [False, True])
def test_ordinary_qualified_qp_does_not_consume_recovery_but_attempt_does(tensor, monkeypatch, first_recovered):
    model, graph, start = problem(tensor)
    monkeypatch.setattr(solver, "bound_majorization_curvature", lambda model, x, curv, posterior: curv)
    allowed = []

    def qp(h, target, lower, upper, caps, kernels, policy, **kwargs):
        allowed.append(kwargs["_allow_precision"])
        first = len(allowed) == 1
        return QuadraticFit(tensor([.3, .3]), torch.zeros_like(caps), tensor(0. if first else 1.),
                            tensor(1.), tensor(0. if first else 1.), first, 4 if first else 2,
                            5 if first else 0, precision_work=work() if first and first_recovered else None)

    monkeypatch.setattr(solver, "solve_qp", qp)
    result = solver._solve_start(model, graph, tensor(1.), start, CudaPolicy())
    assert allowed == [True, not first_recovered]
    assert not result.qualified and result.diagnostics["status"] == "surrogate_unresolved"
    assert result.diagnostics["qp_admm_iterations"] == 6
    assert result.diagnostics["qp_polish_iterations"] == 5
    assert result.diagnostics["qp_precision_attempts"] == int(first_recovered)
    assert result.diagnostics["qp_precision_projected_flow_steps"] == (3 if first_recovered else 0)


def test_second_recovery_cannot_be_hidden_by_qp_implementation(tensor, monkeypatch):
    model, graph, start = problem(tensor)
    monkeypatch.setattr(solver, "bound_majorization_curvature", lambda model, x, curv, posterior: curv)

    def qp(h, target, lower, upper, caps, kernels, policy, **kwargs):
        return QuadraticFit(tensor([.3, .3]), torch.zeros_like(caps), tensor(0.), tensor(1.),
                            tensor(0.), True, 4, 3, precision_work=work())

    monkeypatch.setattr(solver, "solve_qp", qp)
    with pytest.raises(ArithmeticError, match="one terminal-QP recovery budget"):
        solver._solve_start(model, graph, tensor(1.), start, CudaPolicy())


def test_precision_diagnostic_totals_are_explicit_and_do_not_change_scientific_limits():
    result = solver.precision_work_diagnostics([work()])
    assert result["qp_precision_attempts"] == result["qp_precision_qualified"] == 1
    assert result["qp_precision_projected_flow_steps"] == 3
    assert CudaPolicy().inner_max_iterations == 20000
    assert CudaPolicy().outer_max_iterations == 150


def negative_audit(*args, **kwargs):
    return Audit(False, 1., None, "componentwise_unresolved", 2,
                 {sign: {"status": "qualified_lower_bound"} for sign in ("positive", "negative")})


def raw_recovery_stub(seen):
    def recover(model, x, dual, caps, policy, **kwargs):
        seen.update(kwargs)
        counters = {key: 0 for key in solver._RAW_PRECISION_COUNTERS}
        return SimpleNamespace(accepted=False, x=x, dual=dual, objective=tensor_objective(x),
            audit=negative_audit(), qp_work=None,
            work=dict(counters, attempted=True, candidate_accepted=False))
    return recover


def tensor_objective(x):
    return .5 * (x - .2).square().sum()


@pytest.mark.parametrize("prior_recovery", [False, True])
def test_raw_finalizer_receives_last_accepted_not_last_executed_qp(tensor, monkeypatch, prior_recovery):
    from clipp1d.cuda import raw_precision
    from dataclasses import replace
    model, graph, start = problem(tensor)
    monkeypatch.setattr(solver, "bound_majorization_curvature", lambda model, x, curv, post: curv)
    monkeypatch.setattr(solver, "audit_raw", negative_audit)
    calls, seen = [], {}

    def qp(h, target, lower, upper, caps, kernels, policy, **kwargs):
        calls.append(kwargs)
        first = len(calls) == 1
        return QuadraticFit(tensor([.3, .3] if first else [.9, .9]), torch.zeros_like(caps),
                            tensor(0.), tensor(1.), tensor(0.), True, 2,
                            precision_work=work() if first and prior_recovery else None)

    monkeypatch.setattr(solver, "solve_qp", qp)
    monkeypatch.setattr(raw_precision, "recover_raw", raw_recovery_stub(seen))
    result = solver._solve_start(model, graph, tensor(1.), start,
                                replace(CudaPolicy(), outer_max_iterations=2, max_backtracks=1))
    context = seen["accepted_qp_context"]
    assert torch.equal(context.start, start)
    assert torch.equal(context.x, tensor([.3, .3]))
    assert torch.equal(result.x, context.x)
    assert seen["allow_qp_precision"] is (not prior_recovery)
    assert not result.qualified and result.diagnostics["outer_iterations"] == 2
    assert result.diagnostics["raw_precision_attempts"] == 1
    assert result.diagnostics["raw_precision_accepted"] == 0


def test_directional_restart_invalidates_accepted_qp_context(tensor, monkeypatch):
    from clipp1d.cuda import raw_precision
    from dataclasses import replace
    model, graph, start = problem(tensor)
    monkeypatch.setattr(solver, "bound_majorization_curvature", lambda model, x, curv, post: curv)

    def audit(*args, **kwargs):
        result = negative_audit()
        result.direction = torch.ones_like(start)
        return result

    monkeypatch.setattr(solver, "audit_raw", audit)
    monkeypatch.setattr(solver, "direction_restart", lambda *args: tensor([.4, .4]))
    monkeypatch.setattr(solver, "solve_qp", lambda h, target, lo, hi, caps, *args, **kwargs:
        QuadraticFit(start.clone(), torch.zeros_like(caps), tensor(0.), tensor(1.), tensor(0.), True, 1))
    seen = {}
    monkeypatch.setattr(raw_precision, "recover_raw", raw_recovery_stub(seen))
    result = solver._solve_start(model, graph, tensor(1.), start,
                                replace(CudaPolicy(), outer_max_iterations=1))
    assert seen["accepted_qp_context"] is None
    assert result.diagnostics["direction_restarts"] == 1
    assert torch.equal(result.x, tensor([.4, .4]))
