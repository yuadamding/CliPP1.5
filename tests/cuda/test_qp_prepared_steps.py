"""Prepared QP step references; CPU tracing does not qualify CUDA execution."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from clipp1d.cuda import qp
from clipp1d.cuda.kernels import (
    Kernels, StructuralCompileBank, adjoint, admm_step, boxed_rank_one,
    differences, edge_update, flow_polish_step,
)
from clipp1d.cuda.policy import CudaPolicy


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def legacy_flow(x, q, h, target, lower, upper, caps):
    """Independent transcription of the daaf50a flow projection arithmetic."""
    same = x[:, None] == x[None, :]
    q = torch.where(same, q, caps * differences(x).sign())
    free = lower < upper
    safe_target = torch.where(free, target, x)
    residual = torch.where(free, h * (x - safe_target), 0.0) + adjoint(q)
    total = torch.where(same, residual[None, :], 0.0).sum(1)
    fixed = ~free
    fixed_count = (same & fixed[None, :]).sum(1)
    at_lower, at_upper = free & (x == lower), free & (x == upper)
    lower_count = (same & at_lower[None, :]).sum(1)
    upper_count = (same & at_upper[None, :]).sum(1)
    normal = torch.where(fixed_count > 0,
        torch.where(fixed, total / fixed_count.clamp_min(1), 0.0),
        torch.where(total >= 0,
            torch.where(at_lower, total / lower_count.clamp_min(1), 0.0),
            torch.where(at_upper, total / upper_count.clamp_min(1), 0.0)))
    delta = -differences(residual - normal) / same.sum(1)[:, None]
    return torch.maximum(-caps, torch.minimum(caps, q + torch.where(same, delta, 0.0)))


def problem(n, seed=42):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (n,), generator=rng).to(torch.float64) / 3
    h = 1 + 5 * torch.rand(n, generator=rng, dtype=torch.float64)
    target = torch.rand(n, generator=rng, dtype=torch.float64)
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    if n > 1:
        lower[0] = upper[0] = x[0]
        target[0] = 1e308  # The fixed-coordinate target is scientifically irrelevant.
    caps = torch.rand(n, n, generator=rng, dtype=torch.float64)
    caps = (caps + caps.T) * .5
    caps.fill_diagonal_(0.)
    q = torch.randn(n, n, generator=rng, dtype=torch.float64)
    q = torch.maximum(-caps, torch.minimum(caps, q - q.T))
    return x, q, h, target, lower, upper, caps


@pytest.mark.parametrize('n', [1, 2, 7, 16])
def test_prepared_flow_matches_original_arithmetic_through_repeated_cap_projections(n):
    x, q, h, target, lower, upper, caps = problem(n)
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    old = q.clone()
    for _ in range(20):
        q = qp.flow_polish_step(q, context, Kernels('cpu'))
        old = legacy_flow(x, old, h, target, lower, upper, caps)
        torch.testing.assert_close(q, old, atol=0, rtol=0)
        assert bool(torch.isfinite(q).all() & (q.abs() <= caps).all() & (q == -q.T).all())


def test_prepared_geometry_owns_its_values_and_new_proposal_rebuilds_changed_problem():
    x, q, h, target, lower, upper, caps = problem(7)
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    expected = qp.flow_polish_step(q, context)
    for value in (x, h, target, lower, upper, caps):
        value.data.zero_()
    torch.testing.assert_close(qp.flow_polish_step(q, context), expected, atol=0, rtol=0)
    new = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    assert not torch.equal(qp.flow_polish_step(q, new), expected)
    with pytest.raises(FrozenInstanceError):
        context._arguments = ()


@pytest.mark.parametrize('n', [1, 2, 7, 16])
def test_complete_admm_step_matches_separate_node_edge_reference(n):
    x, q, h, target, lower, upper, caps = problem(n)
    weighted = h * torch.where(lower < upper, target, lower)
    z, v, rho = differences(x), q, tensor(.7)
    old_z, old_v = z.clone(), v.clone()
    for _ in range(4):
        x, z, v = admm_step(h, weighted, lower, upper, caps, z, v, rho)
        b = weighted + rho * adjoint(old_z - old_v)
        old_x = boxed_rank_one(h, b, lower, upper, rho)
        old_z, old_v = edge_update(old_x, old_v, caps, rho)
        for actual, reference in ((x, old_x), (z, old_z), (v, old_v)):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_both_larger_tensor_boundaries_trace_fullgraph_without_changing_cache_limits():
    limits = (torch._dynamo.config.recompile_limit, torch._dynamo.config.accumulated_recompile_limit)
    admm = StructuralCompileBank(admm_step, backend='eager')
    flow = StructuralCompileBank(flow_polish_step, backend='eager')
    for n in (1, 2, 7, 16, 7):
        x, q, h, target, lower, upper, caps = problem(n)
        inputs = (h, h * torch.where(lower < upper, target, lower), lower, upper,
                  caps, differences(x), q, tensor(.7))
        for actual, expected in zip(admm(*inputs), admm_step(*inputs)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
        inputs = (q, *context._arguments)
        torch.testing.assert_close(flow(*inputs), flow_polish_step(*inputs), atol=0, rtol=0)
    assert limits == (torch._dynamo.config.recompile_limit, torch._dynamo.config.accumulated_recompile_limit)
    assert admm.diagnostics()['eager_fallback'] is False
    assert flow.diagnostics()['eager_fallback'] is False
    assert flow.calls == admm.calls == 5


def repair_problem():
    x, h = tensor([.4, .4, .4]), tensor([1, 1, 1])
    caps = tensor([[0, .01, .01], [.01, 0, .5], [.01, .5, 0]])
    exact = tensor([[0, .01, .01], [-.01, 0, .31], [-.01, -.31, 0]])
    return x, exact * .99, h, x + adjoint(exact), h * 0, h, caps


def test_one_preparation_per_proposal_and_no_duplicate_checkpoint_certificate(monkeypatch):
    args, kernels, policy = repair_problem(), Kernels('cpu'), CudaPolicy()
    original, counts = qp.prepare_flow_polish, {'prepare': 0, 'flow': 0}

    def prepare(*args):
        counts['prepare'] += 1
        return original(*args)

    def step(*args):
        counts['flow'] += 1
        return flow_polish_step(*args)

    monkeypatch.setattr(qp, 'prepare_flow_polish', prepare)
    kernels.flow_polish_step = step
    x, q, h, target, lower, upper, caps = args
    objective = qp._objective_context(x, h, target, lower, upper, caps)
    proposal = qp._prepare_equality_proposal(*args, kernels, policy, policy.fusion_tol, objective)
    assert proposal is not None and not proposal.qualified
    assert counts == {'prepare': 1, 'flow': 1}
    result, steps = qp._refine_polish(*args, kernels, policy,
                                     _initial_proposal=proposal, _objective=objective)
    assert result is not None and steps > 1
    assert counts['prepare'] == 1 and counts['flow'] == steps + 1
    ordinary, ordinary_steps = qp._refine_polish(*args, kernels, policy)
    assert ordinary_steps == steps + 1
    for a, b in zip(result, ordinary):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('warm', [False, True])
def test_solve_qp_combined_and_separate_boundaries_keep_original_certificates(warm):
    x, q, h, target, lower, upper, caps = problem(7)
    actual = Kernels('cpu')
    separate = SimpleNamespace(boxed_rank_one=actual.boxed_rank_one,
                               edge_update=actual.edge_update, gap_kkt=actual.gap_kkt)
    fits = [qp.solve_qp(h, target, lower, upper, caps, k, start=x, dual=q if warm else None)
            for k in (actual, separate)]
    for fit in fits:
        assert fit.qualified
        stats = actual.gap_kkt(fit.x, fit.dual, h, target, lower, upper, caps)
        assert float(stats[0]) <= CudaPolicy().inner_atol + CudaPolicy().inner_rtol * float(stats[1])
        assert float(stats[2]) <= CudaPolicy().inner_kkt_tol
    assert fits[0].iterations == fits[1].iterations
    assert fits[0].polish_iterations == fits[1].polish_iterations
    torch.testing.assert_close(fits[0].x, fits[1].x, atol=0, rtol=0)
    torch.testing.assert_close(fits[0].dual, fits[1].dual, atol=0, rtol=0)
