"""All-start ADMM accounting includes work that does not win selection."""
import pytest
import torch

from clipp1d.cuda.graph import build_graph
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import QualificationError
from clipp1d.cuda.scalar import pilot
from clipp1d.cuda import selection, solver


@pytest.fixture
def model():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    alt = torch.tensor([20., 35., 65., 80.], dtype=torch.float64)
    slope = torch.full((4, 1), .5, dtype=torch.float64)
    result = TensorModel(tuple("abcd"), alt, 200 - alt, slope,
                         torch.zeros_like(slope), torch.full_like(alt, 1e-6),
                         torch.ones_like(alt), 1e-6, Kernels("cpu"))
    yield result
    torch.set_num_threads(old_threads)


def test_all_start_total_retains_unqualified_start_work(model, monkeypatch):
    pilots = pilot(model)
    graph = build_graph(pilots.phi, mutation_ids=model.mutation_ids)
    original = solver.solve_start
    calls = []

    def with_unresolved_start(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result.diagnostics.copy())
        if len(calls) == 1:
            result.qualified = False
            result.diagnostics["status"] = "injected_unresolved_after_real_qp_work"
        return result

    monkeypatch.setattr(solver, "solve_start", with_unresolved_start)
    previous = solver.PrimalWarmState(torch.full_like(pilots.phi, .5), graph.identity)
    result = solver.fit_lambda(model, graph, pilots, .1, previous)
    assert result.qualified and not result.diagnostics["search_complete"]
    assert result.diagnostics["starts_unresolved"] == 1
    assert len(calls) >= 2 and calls[0]["inner_iterations"] > 0
    assert result.diagnostics["qp_admm_iterations"] == sum(
        row["inner_iterations"] for row in calls)
    assert all(row["qp_admm_iterations"] == row["inner_iterations"]
               for row in result.diagnostics["starts"])


def test_path_total_retains_failed_penalty_and_separable_zero(model, monkeypatch):
    original = selection.fit_lambda

    def with_unresolved_penalty(*args, **kwargs):
        result = original(*args, **kwargs)
        if float(args[3]) == .1:
            raise QualificationError("injected failure after QP work", **result.diagnostics)
        return result

    monkeypatch.setattr(selection, "fit_lambda", with_unresolved_penalty)
    result = selection.fit_tensor_model(model, lambda_values=[0., .1, 1.])
    assert result.search_status == "incomplete"
    assert result.records[0]["qp_admm_iterations"] == 0
    failed = result.records[1]
    assert failed["raw_status"] == "unresolved"
    assert failed["qp_admm_iterations"] > 0
    assert failed["qp_admm_iterations"] == failed["failure_diagnostics"]["qp_admm_iterations"]
    assert result.timings["qp_admm_iterations"] == sum(
        row["qp_admm_iterations"] for row in result.records)
    assert "unresolved" in result.timings["qp_work_scope"]
