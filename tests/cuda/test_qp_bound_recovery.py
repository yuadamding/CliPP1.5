"""Small original-QP references for adaptive box-normal dual recovery.

CPU arithmetic and compiler tracing here are not allocated-CUDA qualification.
"""
import pytest
import torch

from clipp1d.cuda import qp
from clipp1d.cuda.kernels import Kernels, StructuralCompileBank, adjoint, differences
from clipp1d.cuda.policy import CudaPolicy


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def review_problem(reflected=False):
    x = tensor([.5, .5, .5])
    h = tensor([128., 128., 128.])
    target = tensor([37 / 64, .5, 127 / 256])
    lower, upper = tensor([1e-6] * 3), tensor([.5, .5, .9])
    caps = tensor([[0, .125, 1], [.125, 0, .125], [1, .125, 0]])
    q = tensor([[0, 0, -.5], [0, 0, 0], [.5, 0, 0]])
    if reflected:
        x, target, lower, upper, q = 1 - x, 1 - target, 1 - upper, 1 - lower, -q
    return x, q, h, target, lower, upper, caps


def normal_violation(x, q, h, target, lower, upper):
    # Independent componentwise projection in original node coordinates.
    result = torch.zeros_like(x)
    residual = torch.where(lower < upper, h * (x - target), 0.) + adjoint(q)
    for i in range(x.numel()):
        if lower[i] == upper[i]:
            continue
        if x[i] == lower[i]:
            result[i] = min(residual[i], 0.)
        elif x[i] == upper[i]:
            result[i] = max(residual[i], 0.)
        else:
            result[i] = residual[i]
    return result


def qualified(args, q):
    x, _, h, target, lower, upper, caps = args
    stats = Kernels('cpu').gap_kkt(x, q, h, target, lower, upper, caps)
    p = CudaPolicy()
    return bool(torch.isfinite(stats).all()
                & (stats[0] <= p.inner_atol + p.inner_rtol * stats[1])
                & (stats[2] <= p.inner_kkt_tol)), stats


@pytest.mark.parametrize('reflected', [False, True])
def test_review_certificate_is_preserved_and_zero_flow_recovers(reflected):
    args = review_problem(reflected)
    x, exact, h, target, lower, upper, caps = args
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    assert qualified(args, exact)[0]
    torch.testing.assert_close(qp.flow_polish_step(exact, context), exact, atol=0, rtol=0)
    candidate = torch.zeros_like(exact)
    for _ in range(1024):
        candidate = qp.flow_polish_step(candidate, context)
        if qualified(args, candidate)[0]:
            break
    assert qualified(args, candidate)[0]
    torch.testing.assert_close(x, tensor([.5, .5, .5]), atol=0, rtol=0)
    assert bool((candidate == -candidate.T).all() & (candidate.abs() <= caps).all())


def test_exact_review_old_fixed_point_recovers_under_original_gates():
    args = review_problem()
    x, _, h, target, lower, upper, caps = args
    bad = tensor([[0, -.125, -1], [.125, 0, .125], [1, -.125, 0]])
    valid, stats = qualified(args, bad)
    assert not valid
    assert float(stats[0]) == .00054931640625
    assert float(stats[2]) == .15789473684210525
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    for _ in range(1024):
        bad = qp.flow_polish_step(bad, context)
        if qualified(args, bad)[0]:
            break
    assert qualified(args, bad)[0]


@pytest.mark.parametrize('reflected', [False, True])
@pytest.mark.parametrize('warm', [False, True])
def test_complete_qp_recovers_review_solution_without_relaxing_policy(reflected, warm):
    args = review_problem(reflected)
    x, q, h, target, lower, upper, caps = args
    fit = qp.solve_qp(h, target, lower, upper, caps, Kernels('cpu'),
                      start=x, dual=torch.zeros_like(q) if warm else None)
    assert fit.qualified and fit.iterations <= CudaPolicy().inner_max_iterations
    torch.testing.assert_close(fit.x, x, atol=1e-12, rtol=0)
    assert qualified((fit.x, fit.dual, h, target, lower, upper, caps), fit.dual)[0]


def test_fixed_coordinates_absorb_arbitrary_normal_without_uniform_distribution():
    x = tensor([.5, .5, .5])
    h = tensor([128.] * 3)
    target = tensor([1e308, -1e308, 127 / 256])
    lower, upper = tensor([.5, .5, 1e-6]), tensor([.5, .5, .9])
    caps = tensor([[0, .001, 1], [.001, 0, .001], [1, .001, 0]])
    exact = tensor([[0, 0, -.5], [0, 0, 0], [.5, 0, 0]])
    args = (x, exact, h, target, lower, upper, caps)
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    assert qualified(args, exact)[0]
    torch.testing.assert_close(qp.flow_polish_step(exact, context), exact, atol=0, rtol=0)
    q = torch.zeros_like(exact)
    for _ in range(1024):
        q = qp.flow_polish_step(q, context)
        if qualified(args, q)[0]:
            break
    assert qualified(args, q)[0]
    assert bool(torch.isfinite(q).all())


@pytest.mark.parametrize('seed', range(8))
def test_canonical_gradient_and_block_lipschitz_descent(seed):
    rng = torch.Generator().manual_seed(seed)
    x = tensor([.25, .25, .25, .75, .75, .75, .75])
    h = 1 + torch.rand(7, generator=rng, dtype=torch.float64)
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    lower[0] = x[0]
    upper[1] = x[1]
    lower[3] = upper[3] = x[3]
    lower[4] = x[4]
    upper[5] = x[5]
    caps = .1 + torch.rand(7, 7, generator=rng, dtype=torch.float64)
    caps = (caps + caps.T) / 2
    caps.fill_diagonal_(0.)
    q = .05 * torch.randn(7, 7, generator=rng, dtype=torch.float64)
    q = q - q.T
    same = x[:, None] == x[None, :]
    q = torch.where(same, q, caps * differences(x).sign())
    target = torch.randn(7, generator=rng, dtype=torch.float64)
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    v = normal_violation(x, q, h, target, lower, upper)
    for i, j in [(0, 1), (1, 2), (3, 4), (5, 6)]:
        direction = torch.zeros_like(q)
        direction[i, j], direction[j, i] = 1, -1
        epsilon = 1e-5
        values = [normal_violation(x, q + sign * epsilon * direction,
                                   h, target, lower, upper).square().sum() / 2
                  for sign in (-1, 1)]
        derivative = (values[1] - values[0]) / (2 * epsilon)
        torch.testing.assert_close(derivative, v[j] - v[i], atol=1e-8, rtol=1e-8)
    for _ in range(20):
        before = normal_violation(x, q, h, target, lower, upper).square().sum() / 2
        next_q = qp.flow_polish_step(q, context)
        after = normal_violation(x, next_q, h, target, lower, upper).square().sum() / 2
        # Canonical unordered edges, not the doubled skew-matrix norm.
        decrease = (.5 * same.sum(1)[:, None] * (next_q - q).square()).triu(1).sum()
        assert float(after - before + decrease) <= 1e-12 * (1 + float(before))
        assert bool((next_q == -next_q.T).all() & (next_q.abs() <= caps).all())
        torch.testing.assert_close(next_q[~same], (caps * differences(x).sign())[~same],
                                   atol=0, rtol=0)
        q = next_q


def test_stagnant_cone_subproblem_is_not_an_original_qp_certificate():
    x, h = tensor([.5, .5]), tensor([1., 1.])
    target, lower, upper = tensor([0., 0.]), tensor([0., 0.]), tensor([1., 1.])
    caps, q = tensor([[0, 1.], [1., 0]]), torch.zeros(2, 2, dtype=torch.float64)
    context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
    torch.testing.assert_close(qp.flow_polish_step(q, context), q, atol=0, rtol=0)
    assert not qualified((x, q, h, target, lower, upper, caps), q)[0]


def test_original_dual_certificate_admitted_before_geometry_or_flow(monkeypatch):
    args = review_problem()
    kernels, policy = Kernels('cpu'), CudaPolicy()

    def forbidden(*args, **kwargs):
        raise AssertionError('Already qualified original dual must not be transformed')

    monkeypatch.setattr(qp, 'prepare_flow_polish', forbidden)
    objective = qp._objective_context(args[0], *args[2:])
    result = qp._prepare_equality_proposal(*args, kernels, policy, policy.fusion_tol, objective)
    assert result.qualified and result.flow_steps == 0 and result.geometry is None
    assert result.dual is args[1]


def test_warm_admission_reuses_same_candidate_certificate_without_duplicate_check():
    x, q, h, target, lower, upper, caps = review_problem()
    kernels = Kernels('cpu')
    original, calls = kernels.gap_kkt, []

    def count(*args):
        calls.append(args)
        return original(*args)

    kernels.gap_kkt = count
    fit = qp.solve_qp(h, target, lower, upper, caps, kernels, start=x, dual=q)
    assert fit.qualified and fit.iterations == fit.polish_iterations == 0
    assert len(calls) == 1
    torch.testing.assert_close(fit.dual, q, atol=0, rtol=0)


def test_valid_near_fusion_fallback_is_preserved_when_tighter_proposals_do_not_help(monkeypatch):
    x, h = tensor([.5, .5 + 1e-12]), tensor([1., 1.])
    caps, q = tensor([[0, .01], [.01, 0]]), torch.zeros(2, 2, dtype=torch.float64)
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    args = x, q, h, x, lower, upper, caps
    assert qualified(args, q)[0]
    monkeypatch.setattr(qp, 'polish_quadratic', lambda *args: x)

    def forbidden(*args):
        raise AssertionError('A qualified dual must remain available as a fallback')

    monkeypatch.setattr(qp, 'prepare_flow_polish', forbidden)
    candidate, steps = qp._refine_polish(*args, Kernels('cpu'), CudaPolicy())
    assert candidate is not None and steps == 0
    torch.testing.assert_close(candidate[0], x, atol=0, rtol=0)
    torch.testing.assert_close(candidate[1], q, atol=0, rtol=0)


def test_unchanged_cold_checkpoint_reuses_failed_original_certificate(monkeypatch):
    x, h = tensor([.5, .5]), tensor([1., 1.])
    target, lower, upper = tensor([0., 0.]), tensor([0., 0.]), tensor([1., 1.])
    caps, q = tensor([[0, 1.], [1., 0]]), torch.zeros(2, 2, dtype=torch.float64)
    kernels = Kernels('cpu')
    original, calls = kernels.gap_kkt, []

    def count(*args):
        calls.append(args)
        return original(*args)

    kernels.gap_kkt = count
    kernels.admm_step = lambda *args: (x, q, q)
    monkeypatch.setattr(qp, 'polish_quadratic', lambda *args: x)
    fit = qp.solve_qp(h, target, lower, upper, caps, kernels,
                      CudaPolicy(inner_max_iterations=16), start=x)
    assert not fit.qualified
    # One original certificate. Identical flow outputs, raw check and the two
    # remaining tolerance proposals reuse it within this same checkpoint.
    assert len(calls) == 1


def test_cone_boundary_traces_strictly_with_original_reference_results():
    from clipp1d.cuda.kernels import flow_polish_step

    bank = StructuralCompileBank(flow_polish_step, backend='eager')
    for reflected in (False, True):
        x, q, h, target, lower, upper, caps = review_problem(reflected)
        context = qp.prepare_flow_polish(x, h, target, lower, upper, caps)
        arguments = (torch.zeros_like(q), *context._arguments)
        torch.testing.assert_close(bank(*arguments), flow_polish_step(*arguments), atol=0, rtol=0)
    assert bank.diagnostics()['eager_fallback'] is False
