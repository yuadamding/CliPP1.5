"""Qualified dual reuse and stage integrity; CPU references are not CUDA evidence."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from clipp1d.cuda.graph import build_graph
from clipp1d.cuda.kernels import Kernels, adjoint, differences
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.qp import QualifiedDualWarmState, _polish_dual, _refine_polish, quadratic_value, solve_qp


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def assert_certificate(fit, h, target, lower, upper, caps, kernels):
    policy = CudaPolicy()
    assert fit.qualified
    stats = kernels.gap_kkt(fit.x, fit.dual, h, target, lower, upper, caps)
    assert bool(torch.isfinite(stats).all())
    assert float(stats[0]) <= policy.inner_atol + policy.inner_rtol * float(stats[1])
    assert float(stats[2]) <= policy.inner_kkt_tol


@pytest.mark.parametrize("seed", range(5))
def test_cold_warm_qp_recertifies_changed_curvature_target_bounds_and_caps(seed):
    generator = torch.Generator().manual_seed(seed)
    n = 8
    h = 1.0 + 5 * torch.rand(n, generator=generator, dtype=torch.float64)
    target = torch.rand(n, generator=generator, dtype=torch.float64)
    lower, upper = torch.zeros_like(h), torch.ones_like(h)
    lower[0] = upper[0] = 0.3
    graph = build_graph(target)
    caps = graph.weights * 0.15
    kernels = Kernels("cpu")
    original = solve_qp(h, target, lower, upper, caps, kernels)
    assert_certificate(original, h, target, lower, upper, caps, kernels)
    identical = solve_qp(h, target, lower, upper, caps, kernels,
                         start=original.x, dual=original.dual)
    assert identical.iterations == 0
    assert_certificate(identical, h, target, lower, upper, caps, kernels)

    # Changed rho, target, one fixed bound, and smaller caps invalidate any old
    # certificate. Both initializations must pass the unchanged new-problem gates.
    h = h * 7
    target = target.flip(0) + 0.1
    lower[0] = upper[0] = 0.4
    caps = caps * 0.35
    cold = solve_qp(h, target, lower, upper, caps, kernels, start=original.x)
    warm = solve_qp(h, target, lower, upper, caps, kernels,
                    start=original.x, dual=original.dual)
    assert warm.iterations > 0
    for fitted in (cold, warm):
        assert_certificate(fitted, h, target, lower, upper, caps, kernels)
    torch.testing.assert_close(warm.x, cold.x, atol=2e-6, rtol=1e-8)
    torch.testing.assert_close(quadratic_value(warm.x, h, target, caps),
                               quadratic_value(cold.x, h, target, caps), atol=1e-8, rtol=1e-10)


def test_warm_scaled_dual_is_recreated_from_new_rho_and_projected_caps():
    h, target = tensor([20, 30, 40]), tensor([0.1, 0.4, 0.9])
    lower, upper = h * 0, h * 0 + 1
    caps = tensor([[0, 0.1, 0.2], [0.1, 0, 0.15], [0.2, 0.15, 0]])
    supplied = tensor([[0, 4, -3], [-4, 0, 2], [3, -2, 0]])
    start = tensor([0.9, 0.2, 0.1])
    actual = Kernels("cpu")
    calls = []

    def boxed(hh, b, lo, hi, rho):
        calls.append((b.clone(), rho.clone()))
        return actual.boxed_rank_one(hh, b, lo, hi, rho)

    kernels = SimpleNamespace(boxed_rank_one=boxed, edge_update=actual.edge_update,
                              gap_kkt=actual.gap_kkt)
    result = solve_qp(h, target, lower, upper, caps, kernels, start=start, dual=supplied)
    assert_certificate(result, h, target, lower, upper, caps, actual)
    rho = h.median()
    projected = supplied.clamp(-caps, caps)
    expected_b = h * target + rho * adjoint(differences(start) - projected / rho)
    torch.testing.assert_close(calls[0][0], expected_b, atol=0, rtol=0)
    torch.testing.assert_close(calls[0][1], rho, atol=0, rtol=0)


def test_qualified_dual_state_is_bound_to_graph_lambda_and_qualification():
    graph = build_graph(tensor([0.2, 0.8]))
    h, target = tensor([5, 8]), tensor([0.2, 0.8])
    fit = solve_qp(h, target, h * 0, h * 0 + 1, graph.weights * 0.1, Kernels("cpu"))
    state = QualifiedDualWarmState.from_fit(fit, graph, 0.1)
    assert state.for_problem(graph, 0.1) is not fit.dual
    torch.testing.assert_close(state.dual, fit.dual, atol=0, rtol=0)
    with pytest.raises(ValueError, match="qualified"):
        QualifiedDualWarmState.from_fit(replace(fit, qualified=False), graph, 0.1)
    with pytest.raises(ValueError, match="graph and lambda"):
        state.for_problem(build_graph(graph.pilot), 0.1)
    with pytest.raises(ValueError, match="graph and lambda"):
        state.for_problem(graph, 0.2)
    state.dual.add_(0.0)
    with pytest.raises(ValueError, match="unchanged"):
        state.for_problem(graph, 0.1)


def test_qualified_dual_owns_input_and_rejects_exposed_data_mutation():
    graph = build_graph(tensor([0.2, 0.8]))
    h, target = tensor([5, 8]), tensor([0.2, 0.8])
    fit = solve_qp(h, target, h * 0, h * 0 + 1, graph.weights * 0.1, Kernels("cpu"))
    original = fit.dual.clone()
    state = QualifiedDualWarmState.from_fit(fit, graph, 0.1)
    fit.dual.data[0, 1] += 0.25
    torch.testing.assert_close(state.for_problem(graph, 0.1), original, atol=0, rtol=0)
    version = state.dual._version
    state.dual.data[0, 1] += 0.25
    assert state.dual._version == version
    with pytest.raises(ValueError, match="unchanged"):
        state.for_problem(graph, 0.1)


def test_graph_stage_hot_checks_do_not_compare_device_snapshots(monkeypatch):
    graph = build_graph(tensor([0.2, 0.4, 0.8]))
    original = torch.stack
    comparisons = []

    def counting_stack(values, *args, **kwargs):
        comparisons.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", counting_stack)
    with graph.validated_stage():
        for _ in range(10):
            graph.validate()
            graph.validate_metadata()
            with graph.validated_stage():
                graph.validate()
    assert comparisons == [4, 4]  # One full entry and one full exit, independent of calls.


@pytest.mark.parametrize("mode", ["version", "data", "metadata", "exception"])
def test_graph_stage_catches_mutations_at_hot_or_exit_boundary(mode):
    graph = build_graph(tensor([0.2, 0.8]))
    with pytest.raises(ValueError, match="modified"):
        with graph.validated_stage():
            if mode == "version":
                graph.weights.add_(0.1)
                graph.validate_metadata()
            elif mode == "metadata":
                object.__setattr__(graph, "weight_rule", "changed")
                graph.validate_metadata()
            else:
                graph.weights.data[0, 1] += 0.1
                graph.validate_metadata()  # .data bypasses version; exit still catches it.
                if mode == "exception":
                    raise RuntimeError("original stage failed")
    assert graph._stage_depth == 0


def test_outer_backtracking_reuses_only_qualified_duals_and_preserves_audit_details(monkeypatch):
    from benchmarks.qualify_cuda import host_fixture
    from clipp1d.cuda.model import TensorModel
    import clipp1d.cuda.solver as solver

    model = TensorModel.from_host(host_fixture("single_support"), "cpu", compiled=False)
    initial = tensor([0.4, 0.42, 0.96])
    graph = build_graph(initial, model.mutation_ids)
    real_solve, real_loss = solver.solve_qp, TensorModel.loss
    recorded = []
    losses = 0

    def solve(*args, **kwargs):
        output = real_solve(*args, **kwargs)
        recorded.append((kwargs["dual"], output))
        return output

    def loss(self, value):
        nonlocal losses
        losses += 1
        answer = real_loss(self, value)
        # Force only the first likelihood acceptance check to backtrack; the QP
        # result itself remains fully qualified, so its dual should be retained.
        return answer + 1e6 if losses == 2 else answer

    monkeypatch.setattr(solver, "solve_qp", solve)
    monkeypatch.setattr(TensorModel, "loss", loss)
    fitted = solver.solve_start(model, graph, tensor(0.6), initial)
    assert fitted.qualified
    assert fitted.diagnostics["backtracks"] >= 1
    assert recorded[0][0] is None
    for previous, current in zip(recorded, recorded[1:]):
        assert previous[1].qualified
        assert current[0] is not previous[1].dual
        torch.testing.assert_close(current[0], previous[1].dual, atol=0, rtol=0)
    detail = fitted.diagnostics
    assert detail["qp_calls"] == len(recorded)
    assert detail["qp_dual_warm_starts"] == len(recorded) - 1
    assert detail["qp_seconds"] > 0 and detail["audit_seconds"] > 0
    assert detail["audit_calls"] >= 1
    assert detail["audit_signed_direction_count"] == 2
    for sign in ("positive", "negative"):
        cut = detail["audit_diagnostics"][sign]
        assert cut["status"] == "qualified_lower_bound"
        assert {"lower_bound", "attained_value", "roundoff_margin", "objective_scale", "iterations"} <= cut.keys()


def test_numerical_polish_recovers_near_fusions_inside_overwide_public_run():
    from clipp1d.cuda.partition import grouping

    optimum = tensor([0.3, 0.3, 0.300015, 0.30003, 0.30003])
    x = optimum + tensor([-1e-8, 1e-8, 0, -1e-8, 1e-8])
    h = torch.full_like(x, 100.0)
    caps = torch.full((5, 5), 1e-7, dtype=x.dtype)
    caps.fill_diagonal_(0)
    dual = caps * differences(optimum).sign()
    target = optimum + adjoint(dual) / h
    policy = CudaPolicy()
    # Keep the conservative public extraction rule unchanged for unequal values.
    assert grouping(x, policy.fusion_tol)[2].unique().numel() == 5
    candidate, steps = _refine_polish(x, dual, h, target, x * 0, x * 0 + 1,
                                      caps, Kernels("cpu"), policy)
    assert candidate is not None and 0 <= steps <= 3072
    torch.testing.assert_close(candidate[0], optimum, atol=2e-16, rtol=0)
    assert grouping(candidate[0], policy.fusion_tol)[2].tolist() == [0, 0, 1, 2, 2]
    assert float(candidate[2][0]) <= policy.inner_atol + policy.inner_rtol * float(candidate[2][1])
    assert float(candidate[2][2]) <= policy.inner_kkt_tol
    assert float(quadratic_value(candidate[0], h, target, caps)) <= float(quadratic_value(x, h, target, caps))


@pytest.mark.parametrize("fraction", [0.99999, 0.99])
def test_bounded_repeated_flow_projection_recovers_certificate_after_cap_clipping(fraction):
    x = tensor([0.4, 0.4, 0.4])
    h = torch.ones_like(x)
    caps = tensor([[0, 0.01, 0.01], [0.01, 0, 0.5], [0.01, 0.5, 0]])
    known_dual = tensor([[0, 0.01, 0.01], [-0.01, 0, 0.31], [-0.01, -0.31, 0]])
    target = x + adjoint(known_dual)
    dual = known_dual * fraction
    kernels, policy = Kernels("cpu"), CudaPolicy()
    initial = _polish_dual(x, dual, h, target, h * 0, h, caps)
    stats = kernels.gap_kkt(x, initial, h, target, h * 0, h, caps)
    gap_passes = float(stats[0]) <= policy.inner_atol + policy.inner_rtol * float(stats[1])
    assert gap_passes == (fraction == 0.99999)
    assert float(stats[2]) > policy.inner_kkt_tol
    candidate, steps = _refine_polish(x, dual, h, target, h * 0, h, caps, kernels, policy)
    assert candidate is not None and 1 < steps <= 3072
    assert float(candidate[2][0]) <= policy.inner_atol + policy.inner_rtol * float(candidate[2][1])
    assert float(candidate[2][2]) <= policy.inner_kkt_tol
    assert torch.equal(candidate[1], -candidate[1].T)
    assert bool((candidate[1].abs() <= caps).all())


def test_refined_proposal_cannot_cross_original_bounds_even_if_objective_improves(monkeypatch):
    import clipp1d.cuda.qp as qp

    x, h = tensor([0.5, 0.5]), tensor([1, 1])
    caps = tensor([[0, 0.1], [0.1, 0]])
    monkeypatch.setattr(qp, "polish_quadratic", lambda *args: tensor([2, 2]))
    candidate, steps = _refine_polish(x, caps * 0, h, tensor([2, 2]), h * 0, h,
                                      caps, Kernels("cpu"), CudaPolicy())
    assert candidate is None and steps == 3


def test_gap_qualified_warm_near_fusions_are_polished_before_zero_step_return():
    optimum = tensor([0.3, 0.3, 0.300015, 0.30003, 0.30003])
    x = optimum + tensor([-1e-12, 1e-12, 0, -1e-12, 1e-12])
    h = torch.full_like(x, 100.0)
    caps = torch.full((5, 5), 0.1, dtype=x.dtype)
    caps.fill_diagonal_(0)
    dual = caps * differences(optimum).sign()
    target = optimum + adjoint(dual) / h
    kernels, policy = Kernels("cpu"), CudaPolicy()
    stats = kernels.gap_kkt(x, dual, h, target, x * 0, x * 0 + 1, caps)
    assert float(stats[0]) <= policy.inner_atol + policy.inner_rtol * float(stats[1])
    assert float(stats[2]) <= policy.inner_kkt_tol
    assert x[0] != x[1] and x[3] != x[4]
    fit = solve_qp(h, target, x * 0, x * 0 + 1, caps, kernels, start=x, dual=dual)
    # Equality proposals still snap the primal, but the already valid incoming
    # dual is retained without unnecessary flow transformations.
    assert fit.qualified and fit.iterations == 0 and fit.polish_iterations == 0
    assert fit.x[0] == fit.x[1] and fit.x[3] == fit.x[4]
    torch.testing.assert_close(fit.x, optimum, atol=2e-16, rtol=0)


def test_failed_raw_audit_clears_warm_dual_before_next_surrogate(monkeypatch):
    from benchmarks.qualify_cuda import host_fixture
    from clipp1d.cuda.model import TensorModel
    import clipp1d.cuda.solver as solver

    model = TensorModel.from_host(host_fixture("single_support"), "cpu", compiled=False)
    initial = tensor([0.4, 0.42, 0.96])
    graph = build_graph(initial, model.mutation_ids)
    real_audit, real_qp = solver.audit_raw, solver.solve_qp
    calls, audit_count = [], 0

    def audit(*args, **kwargs):
        nonlocal audit_count
        result = real_audit(*args, **kwargs)
        audit_count += 1
        if audit_count == 1:
            return replace(result, qualified=False, direction=None, status="test_unresolved")
        return result

    def solve(*args, **kwargs):
        calls.append((audit_count, kwargs.get("dual")))
        return real_qp(*args, **kwargs)

    monkeypatch.setattr(solver, "audit_raw", audit)
    monkeypatch.setattr(solver, "solve_qp", solve)
    fitted = solver.solve_start(model, graph, tensor(0.6), initial)
    assert fitted.qualified and fitted.diagnostics["qp_dual_warm_resets"] == 1
    assert next(dual for count, dual in calls if count == 1) is None


def test_n64_pooled_surrogate_qualifies_without_dividing_rho_by_all_nodes():
    from benchmarks.qualify_cuda import scaling_fixture
    from clipp1d.cuda.model import TensorModel

    # Captured G failure mechanism, regenerated from exact declared counts:
    # old median(h)/N conditioning failed the production budget on this QP.
    model = TensorModel.from_host(scaling_fixture(64), "cpu", compiled=False)
    pilots = 2 * model.alt / (model.alt + model.ref)
    graph = build_graph(pilots, model.mutation_ids)
    curvature = model.terms(pilots)[2]
    weights = curvature / curvature.max()
    start = ((weights * pilots).sum() / weights.sum()).expand_as(pilots).clone()
    _, gradient, curvature, *_ = model.terms(start)
    h = curvature.clamp_min(1.0)
    target = start - gradient / h
    caps = graph.weights * 0.9009674666827776
    fitted = solve_qp(h, target, model.lower, model.upper, caps, model.kernels, start=start)
    assert fitted.iterations <= CudaPolicy().inner_max_iterations
    assert_certificate(fitted, h, target, model.lower, model.upper, caps, model.kernels)
