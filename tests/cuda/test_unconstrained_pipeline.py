"""CPU tensor reference evidence; never substitutes for allocated CUDA qualification."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from clipp1d.cuda.graph import build_graph
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import grouping, partition_score, refit
from clipp1d.cuda.policy import CudaPolicy, QualificationError
from clipp1d.cuda.scalar import pilot
from clipp1d.cuda.selection import fit_tensor_model
from clipp1d.cuda.solver import PrimalWarmState, RawFit, direction_restart, fit_lambda, objective


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def model(alt=(10., 30., 11., 31.), slope=None, upper=None, mutation_ids=None):
    a = tensor(alt)
    n = a.numel()
    slope = torch.full((n, 1), .4, dtype=torch.float64) if slope is None else tensor(slope)
    valid = slope > 0
    prior = torch.where(valid, -valid.sum(-1, keepdim=True).double().log(), -float("inf"))
    return TensorModel(tuple(f"m{i:03}" for i in range(n)) if mutation_ids is None else mutation_ids,
                       a, 100. - a, slope, prior, torch.full_like(a, 1e-6),
                       torch.ones_like(a) if upper is None else tensor(upper), 1e-6, Kernels("cpu"))


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_lambda_zero_keeps_every_qualified_pilot_unconstrained():
    m = model(upper=[.6, .9, .6, .9])
    p = pilot(m)
    graph = build_graph(p.phi)
    fit = fit_lambda(m, graph, p, 0.)
    torch.testing.assert_close(fit.x, p.phi, rtol=0., atol=0.)
    assert not bool((fit.x == 1.).any())
    assert fit.witness is None and fit.qualified
    assert fit.diagnostics["clonal_constraint"] is False
    assert fit.diagnostics["separable_scalar_gap_qualified"]
    assert fit.diagnostics["separable_global_gap"] == float(p.gap.sum())


def test_lambda_zero_rejects_stale_or_unqualified_pilots():
    m = model()
    p = pilot(m)
    graph = build_graph(p.phi)
    with pytest.raises(QualificationError, match="do not qualify"):
        fit_lambda(m, graph, replace(p, qualified=torch.zeros_like(p.qualified)), 0.)
    with pytest.raises(QualificationError, match="do not qualify"):
        fit_lambda(m, graph, replace(p, gap=p.gap + 1.), 0.)
    changed = model(alt=(12., 30., 11., 31.))
    with pytest.raises(QualificationError, match="do not qualify"):
        fit_lambda(changed, graph, p, 0.)


def test_noncontiguous_refit_is_unconstrained_and_canonically_labeled():
    m = model(alt=(30., 10., 32., 12.), upper=[.9, .8, .9, .8])
    result = refit(m, tensor([.8, .3, .8, .3]))
    assert result.labels.tolist() == [0, 1, 0, 1]
    torch.testing.assert_close(result.centers, tensor([.775, .275]), rtol=0., atol=2e-15)
    assert result.clonal == 0
    assert result.centers[result.clonal] != 1.
    expected = -(m.alt * (.4 * result.phi).log() + m.ref * torch.log1p(-.4 * result.phi)).sum()
    torch.testing.assert_close(result.loss, expected, rtol=0., atol=1e-12)
    assert result.gap < 1e-6


def test_clonal_designation_ties_use_smallest_canonical_node_only():
    m = model(alt=(20., 20.), upper=[.7, .7])
    result = refit(m, tensor([.2, .8]))
    assert result.labels.tolist() == [0, 1]
    torch.testing.assert_close(result.centers, tensor([.5, .5]), atol=1e-15, rtol=0.)
    assert result.clonal == 0


def test_conservative_gpu_tolerance_policy_has_no_diameter_chaining():
    _, _, labels, counts = grouping(tensor([.100038, .1, .100019, 1., 1. - 1e-8]), 2e-5)
    assert labels.tolist() == [0, 1, 2, 3, 4]
    assert counts.tolist() == [1, 1, 1, 1, 1]
    order, _, labels, counts = grouping(tensor([.8, .2, .8, .2]), 2e-5)
    assert labels.tolist() == [0, 1, 0, 1]
    assert order.tolist() == [0, 2, 1, 3]
    assert counts.tolist() == [2, 2, 0, 0]


def test_allocation_score_matches_original_cpu_arithmetic():
    actual = float(partition_score(tensor(12.), torch.tensor([2, 3], dtype=torch.long)))
    mass = math.factorial(2) * math.factorial(1) * math.factorial(2) * math.factorial(3) / math.factorial(6)
    assert abs(actual - (24 + 2 * math.log(5) - 1.4 * math.log(mass))) < 1e-13


def test_graph_bound_continuation_rejects_other_graph_and_mutation():
    m = model()
    p = pilot(m)
    graph = build_graph(p.phi)
    warm = PrimalWarmState(p.phi, graph.identity)
    other_graph = build_graph(p.phi)
    with pytest.raises(ValueError, match="bound to this graph"):
        fit_lambda(m, other_graph, p, .1, warm)
    warm.x.data[0] += .01
    with pytest.raises(ValueError, match="unchanged"):
        fit_lambda(m, graph, p, .1, warm)
    with pytest.raises(ValueError, match="PrimalWarmState"):
        fit_lambda(m, graph, p, .1, p.phi)


def test_direction_restart_can_move_without_any_exact_one_coordinate():
    m = model(alt=(10., 10.))
    x = tensor([.5, .5])
    caps = torch.zeros((2, 2), dtype=torch.float64)
    current = objective(m, x, caps)
    restart = direction_restart(m, x, tensor([-1., -1.]), caps, current, CudaPolicy())
    assert restart is not None
    assert not bool((restart == 1.).any())
    assert objective(m, restart, caps) < current


def test_strong_penalty_fits_one_free_center_without_witness():
    m = model(alt=(10., 11., 12.), upper=[.7, .7, .7])
    p = pilot(m)
    graph = build_graph(p.phi)
    result = fit_lambda(m, graph, p, 20.)
    assert result.qualified and result.witness is None
    torch.testing.assert_close(result.x, tensor([.275, .275, .275]), atol=2e-6, rtol=0.)
    assert result.diagnostics["box_feasible"]
    assert result.diagnostics["raw_branch_stationarity_qualified"]
    assert result.diagnostics["directional_qualified"]
    assert result.diagnostics["inner_gap_qualified"]
    assert result.diagnostics["inner_kkt_qualified"]


def test_mixed_multiplicity_low_level_pipeline_keeps_original_boxes():
    m = model(alt=(15., 17., 29.), slope=[[.2, .4], [.2, .4], [.4, 0.]], upper=[.85, .85, .8])
    result = fit_tensor_model(m, lambda_values=[0., .2])
    assert result.raw.qualified and result.raw.witness is None
    assert bool((result.raw.x <= m.upper).all())
    assert bool((result.refit.phi <= m.upper).all())
    assert not bool((result.raw.x == 1.).any())
    assert result.refit.clonal == int((result.refit.centers - 1.).abs().argmin())
    assert len(result.records) == 2
    assert np.isfinite(float(result.lambda_value))


def test_unresolved_nonzero_path_is_not_reported_complete(monkeypatch):
    import clipp1d.cuda.selection as selection
    original = fit_lambda

    def unresolved(model, graph, pilots, lam, previous, policy):
        if bool(lam > 0):
            raise QualificationError("QP budget reached", inner_gap_qualified=False)
        return original(model, graph, pilots, lam, previous, policy)

    monkeypatch.setattr(selection, "fit_lambda", unresolved)
    result = fit_tensor_model(model(), lambda_values=[0., .2])
    assert float(result.lambda_value) == 0.
    assert result.search_status == "incomplete"
    assert result.records[0]["raw_status"] == result.records[0]["refit_status"] == "qualified"
    assert result.records[1]["raw_status"] == "unresolved"
    assert result.timings["raw_unresolved_penalties"] == 1


def test_unqualified_return_cannot_be_admitted(monkeypatch):
    import clipp1d.cuda.selection as selection

    def unqualified(model, graph, pilots, lam, previous, policy):
        return RawFit(pilots.phi, torch.zeros_like(graph.weights), tensor(1.), None, False,
                      {"status": "unresolved", "search_complete": False})

    monkeypatch.setattr(selection, "fit_lambda", unqualified)
    with pytest.raises(QualificationError, match="No qualified"):
        fit_tensor_model(model(), lambda_values=[0., .2])


@pytest.mark.parametrize("lam", [-1., float("nan"), float("inf")])
def test_invalid_lambda_rejected(lam):
    m = model()
    p = pilot(m)
    with pytest.raises(ValueError, match="finite nonnegative"):
        fit_lambda(m, build_graph(p.phi), p, lam)


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
def test_nonfinite_raw_objective_cannot_enter_selection(monkeypatch, bad_value):
    import clipp1d.cuda.selection as selection
    original = fit_lambda

    def broken(model, graph, pilots, lam, previous, policy):
        result = original(model, graph, pilots, 0., previous, policy)
        if bool(lam > 0):
            result.objective = tensor(bad_value)
        return result

    monkeypatch.setattr(selection, 'fit_lambda', broken)
    result = fit_tensor_model(model(), lambda_values=[0., .2])
    assert result.search_status == 'incomplete'
    assert float(result.lambda_value) == 0.
    assert result.records[1]['raw_status'] == 'unresolved'


@pytest.mark.parametrize('field,bad_value', [('score', float('nan')), ('score', float('inf')),
                                           ('gap', -1.), ('gap', float('inf'))])
def test_nonfinite_or_negative_refit_cannot_enter_selection(monkeypatch, field, bad_value):
    import clipp1d.cuda.selection as selection
    original = refit
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        if calls:
            setattr(result, field, tensor(bad_value))
        calls += 1
        return result

    monkeypatch.setattr(selection, 'refit', broken)
    result = fit_tensor_model(model(), lambda_values=[0., .2])
    assert result.search_status == 'incomplete'
    assert float(result.lambda_value) == 0.
    assert result.records[1]['raw_status'] == 'qualified'
    assert result.records[1]['refit_status'] == 'unresolved'


def test_finite_inputs_with_overflowed_penalty_caps_are_unresolved():
    m = model()
    p = pilot(m)
    graph = build_graph(p.phi)
    with pytest.raises(QualificationError, match='penalty caps'):
        fit_lambda(m, graph, p, torch.finfo(torch.float64).max)


def test_score_arithmetic_overflow_does_not_become_qualified():
    with pytest.raises(QualificationError, match='score is nonfinite'):
        partition_score(tensor(1e308), torch.tensor([1], dtype=torch.long))


@pytest.mark.parametrize('exponent', [-2000, 2000])
def test_unrepresentable_logarithmic_path_is_rejected(exponent):
    policy = replace(CudaPolicy(), path_min_exponent=exponent, path_max_exponent=exponent)
    with pytest.raises(ValueError, match='finite and strictly positive'):
        fit_tensor_model(model(), policy)


def test_exhausted_upper_boundary_is_explicit_without_changing_planned_coverage(monkeypatch):
    import clipp1d.cuda.selection as selection
    original = refit
    calls = 0

    def declining_score(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        result.score = tensor(100. - calls)
        calls += 1
        return result

    monkeypatch.setattr(selection, 'refit', declining_score)
    policy = replace(CudaPolicy(), path_min_exponent=0, path_max_exponent=0, path_extensions=2)
    result = fit_tensor_model(model(alt=(10., 10.)), policy)
    assert len(result.records) == 4
    assert result.timings['extensions'] == 2
    assert result.timings['selected_at_upper_boundary']
    assert result.timings['extension_limit_reached']
    assert result.timings['path_truncated']
    assert result.timings['planned_path_complete']
    assert result.search_status == 'complete'


def test_lambda_zero_reports_absent_surrogate_separately_from_qualified_scalar_pilots():
    m = model()
    p = pilot(m)
    result = fit_lambda(m, build_graph(p.phi), p, 0.)
    assert result.qualified and result.diagnostics['separable_scalar_gap_qualified']
    assert result.diagnostics['inner_certificate_scope'] == 'not_applicable_separable_lambda_zero'
    assert not result.diagnostics['inner_certificate_present']
    assert not result.diagnostics['inner_qp_qualified']
    assert result.diagnostics['inner_gap'] is None
