"""Generic allocated-CUDA mathematical and bounded-work precision regressions."""
from dataclasses import replace
from itertools import permutations, product

import pytest
import torch

from clipp1d.cuda import kernels as tensor_kernels, qp, qp_precision as precision
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.policy import CudaPolicy


@pytest.fixture(scope='module')
def bank():
    if not torch.cuda.is_available():
        pytest.skip('Allocated CUDA required; controller execution is not qualification')
    return Kernels('cuda:0', compiled=True)


@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield


def tensor(values):
    return torch.tensor(values, dtype=torch.float64, device='cuda:0')


def gate(stats, policy=CudaPolicy()):
    return bool(torch.isfinite(stats).all()
        & (stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
        & (stats[2] <= policy.inner_kkt_tol))


def original_energy(x, problem):
    h, target, lower, upper, caps = problem
    safe = torch.where(lower < upper, target, lower)
    return .5 * (h * (x - safe).square()).sum() + .5 * (
        caps * tensor_kernels.differences(x).abs()).sum()


class CountedBank:
    def __init__(self, original):
        self.original = original
        self.counts = dict(admm_step=0, flow_polish_step=0, directional_cut_step=0, gap_kkt=0)

    def __getattr__(self, name):
        function = getattr(self.original, name)
        if name not in self.counts:
            return function
        def counted(*args, **kwargs):
            self.counts[name] += 1
            return function(*args, **kwargs)
        return counted


@pytest.mark.parametrize('fixture', [
    (.4, 1., .35, .1, .9, [.2, .7], [0., 0.], .35),
    (.5, 1., .5, 0., 1., [.2, .2, .8], [.3, .2, .1], .2),
    (.4, 3., -3., .4, .4, [.2, .8], [.1, .2], .4),
    (4e-6, 1e12, 4e-6 + 2e-11, 1e-6, 1., [2e-6, 6e-6], [3., 1.], 4e-6 + 1.8e-11),
    (.2, 2., -1., .1, .9, [.4, .6], [.1, .1], .1),
    (.6, 2., .8, 0., 1., [.2], [.3], .65),
])
def test_coordinate_against_all_knots_and_interval_roots(bank, fixture):
    origin, weight, aim, lo, hi, others, weights, expected = fixture
    x = tensor([origin, *others])
    h, target = torch.ones_like(x), torch.zeros_like(x)
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    h[0], target[0], lower[0], upper[0] = weight, aim, lo, hi
    caps = x.new_zeros((x.numel(), x.numel()))
    caps[0, 1:] = tensor(weights)
    caps[1:, 0] = caps[0, 1:].clone()
    before = [v.clone() for v in (x, h, target, lower, upper, caps)]
    candidate, bracket = precision.coordinate_minimizer(x, 0, h, target, lower, upper, caps)
    order = torch.argsort(x)
    prefix = torch.cat((x.new_zeros(1), caps[0, order].cumsum(0)))
    roots = target[0] + (caps[0].sum() - 2 * prefix) / h[0]
    alternatives = torch.cat((x, roots, lower[:1], upper[:1])).clamp(lower[0], upper[0])
    values = .5 * h[0] * (alternatives-target[0]).square() + (
        caps[0][None, :] * (alternatives[:, None]-x).abs()).sum(1)
    value = .5 * h[0] * (candidate-target[0]).square() + (caps[0]*(candidate-x).abs()).sum()
    margin = 128 * torch.finfo(x.dtype).eps * (1 + value.abs() + values.abs().max())
    assert bracket['one_dimensional_bracket_qualified']
    assert value <= values.min() + margin
    assert abs(float(candidate)-expected) <= 16 * torch.finfo(x.dtype).eps * (1 + abs(expected))
    assert all(torch.equal(a, b) for a, b in zip(before, (x, h, target, lower, upper, caps), strict=True))


def collective_problem():
    x = tensor([.5] * 4)
    h, target = torch.ones_like(x), tensor([.9, .9, .1, .1])
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    caps = x.new_full((4, 4), .1)
    caps.fill_diagonal_(0.)
    caps[0, 1] = caps[1, 0] = caps[2, 3] = caps[3, 2] = 2.
    return x, (h, target, lower, upper, caps)


def test_collective_escape_matches_exhaustive_orders_and_grid(bank):
    x, problem = collective_problem()
    dual = torch.zeros_like(problem[-1])
    for node in range(4):
        value, bracket = precision.coordinate_minimizer(x, node, *problem)
        assert value == x[node] and bracket['one_dimensional_bracket_qualified']
    chamber, _ = precision.ordered_candidate(x, problem)
    torch.testing.assert_close(chamber, x, rtol=0, atol=2e-15)
    counted = CountedBank(bank)
    result = precision.recover_quadratic(x, dual, *problem, counted, CudaPolicy())
    expected = tensor([.7, .7, .3, .3])
    assert result.qualified and result.work['candidate_accepted']
    torch.testing.assert_close(result.x, expected, rtol=0, atol=2e-15)
    assert gate(tensor_kernels.gap_kkt(result.x, result.dual, *problem))
    values = []
    for order in permutations(range(4)):
        candidate, _ = precision.ordered_candidate(x, problem, torch.tensor(order, device=x.device))
        values.append(original_energy(candidate, problem))
    grid = tensor(list(product((0., .3, .5, .7, 1.), repeat=4)))
    grid_values = .5 * ((grid-problem[1]).square()*problem[0]).sum(1) + .5 * (
        problem[-1] * (grid[:, None, :]-grid[:, :, None]).abs()).sum((1, 2))
    assert original_energy(result.x, problem) <= torch.stack(values).min() + 1e-13
    assert original_energy(result.x, problem) <= grid_values.min() + 1e-13
    assert result.work['collective_reorderings'] > 0
    assert counted.counts['admm_step'] == 0
    assert counted.counts['directional_cut_step'] == result.work['cut_steps']
    assert counted.counts['flow_polish_step'] == result.work['projected_flow_steps']
    assert counted.counts['gap_kkt'] == result.work['compiled_certificate_evaluations']
    assert result.work['certificate_evaluations'] == (
        result.work['eager_certificate_evaluations'] + result.work['compiled_certificate_evaluations'])
    torch.testing.assert_close(result.stats, bank.gap_kkt(result.x, result.dual, *problem), rtol=0, atol=0)


def test_component_cut_ignores_unrelated_capacity_scale(bank):
    x = tensor([4e-6]*4 + [.75]*4)
    h = tensor([1e12]*4 + [1.]*4)
    target = x + tensor([.4e-12, .4e-12, -.4e-12, -.4e-12, 0., 0., 0., 0.])
    lower, upper = torch.zeros_like(x), torch.ones_like(x)
    lower[6:] = upper[6:] = x[6:]
    caps = x.new_zeros((8, 8))
    caps[:4, :4] = .1
    caps[0, 1] = caps[1, 0] = caps[2, 3] = caps[3, 2] = 2.
    caps[4:, 4:] = 1e6
    caps.fill_diagonal_(0.)
    q = torch.zeros_like(caps)
    problem = (h, target, lower, upper, caps)
    small = (h[:4], target[:4], lower[:4], upper[:4], caps[:4, :4].contiguous())
    work, isolated_work = precision._Work(), precision._Work()
    direction, _ = precision.component_direction(x, q, problem, bank, CudaPolicy(), 1,
                                                precision.PrecisionLimits(), work)
    isolated, _ = precision.component_direction(x[:4], q[:4, :4].contiguous(), small,
        bank, CudaPolicy(), 1, precision.PrecisionLimits(), isolated_work)
    assert direction is not None and isolated is not None
    assert (direction[4:] == 0).all()
    torch.testing.assert_close(direction[:4], isolated, rtol=2e-14, atol=2e-14)
    assert work.cut_steps == isolated_work.cut_steps <= 256
    value, margin, check = precision.direct_derivative(x, direction, problem)
    assert check['box_tangent_feasible'] and value + margin < -CudaPolicy().inner_kkt_tol


@pytest.mark.parametrize('normal, feasible', [(False, True), (True, True), (False, False)])
def test_accelerated_flow_known_dual_and_impossible_primal(bank, normal, feasible):
    n = 16
    x = tensor([.5]*n)
    h, lower, upper = torch.ones_like(x), torch.zeros_like(x), torch.ones_like(x)
    caps, witness = x.new_zeros((n, n)), x.new_zeros((n, n))
    index = torch.arange(n-1, device=x.device)
    caps[index, index+1] = caps[index+1, index] = 1.
    flow = .1 * torch.sin((index.to(x.dtype)+1)*torch.pi/n)
    witness[index, index+1], witness[index+1, index] = flow, -flow
    cone = torch.zeros_like(x)
    if normal:
        lower[0], upper[-1], cone[0], cone[-1] = x[0], x[-1], .05, -.05
    target = x + tensor_kernels.adjoint(witness) - cone
    if not feasible:
        caps.zero_()
        witness.zero_()
        target = x + .1
    problem = (h, target, lower, upper, caps)
    assert gate(tensor_kernels.gap_kkt(x, witness, *problem)) is feasible
    saved = [v.clone() for v in (x, *problem)]
    work, counted = precision._Work(), CountedBank(bank)
    dual, stats = precision.accelerated_flow(x, torch.zeros_like(caps), problem, counted,
        CudaPolicy(), precision.PrecisionLimits(), work)
    assert gate(stats) is feasible
    assert gate(tensor_kernels.gap_kkt(x, dual, *problem)) is feasible
    assert work.projected_flow_steps == counted.counts['flow_polish_step'] <= 1024
    assert work.momentum_restarts <= work.projected_flow_steps
    assert all(torch.equal(a, b) for a, b in zip(saved, (x, *problem), strict=True))


def test_qualified_literal_qp_and_zero_flow_endpoint(bank):
    x = tensor([.2, .8])
    problem = (torch.ones_like(x), x.clone(), torch.zeros_like(x), torch.ones_like(x), x.new_zeros((2, 2)))
    result = precision.recover_quadratic(x, torch.zeros_like(problem[-1]), *problem, bank, CudaPolicy())
    assert result.qualified and result.work['candidate_accepted']
    assert torch.equal(result.x, x)
    assert result.work['projected_flow_steps'] == 0
    original = qp.solve_qp(*problem, bank, start=x)
    assert original.qualified and original.precision_work is None and original.iterations == 0


def test_short_precision_limit_retains_original_negative_state(bank):
    x, problem = collective_problem()
    q = torch.zeros_like(problem[-1])
    result = precision.recover_quadratic(x, q, *problem, bank, CudaPolicy(),
                                       limits=precision.PrecisionLimits(rounds=1))
    assert not result.qualified and not result.work['candidate_accepted']
    assert result.x is x and result.dual is q
    assert result.work['pava_rounds'] == 1
    assert result.work['projected_flow_steps'] == 0


def test_terminal_control_counts_real_admm_without_an_extra_solve(bank, monkeypatch):
    # Suppress only the old optional equality proposal to exercise terminal
    # admission after one real compiled ADMM update on an analytic two-node QP.
    monkeypatch.setattr(qp, '_prepare_equality_proposal', lambda *a, **k: None)
    monkeypatch.setattr(qp, '_refine_polish', lambda *a, **k: (None, 0))
    x = tensor([.5, .5])
    caps = tensor([[0., .05], [.05, 0.]])
    problem = (torch.ones_like(x), tensor([.2, .8]), torch.zeros_like(x), torch.ones_like(x), caps)
    policy = replace(CudaPolicy(), inner_max_iterations=1)
    first, second = CountedBank(bank), CountedBank(bank)
    negative = qp.solve_qp(*problem, first, policy, start=x, _allow_precision=False)
    fitted = qp.solve_qp(*problem, second, policy, start=x, _allow_precision=True)
    assert not negative.qualified and negative.precision_work is None
    assert fitted.qualified and fitted.precision_work['attempted']
    assert fitted.iterations == negative.iterations == first.counts['admm_step'] == second.counts['admm_step'] == 1
    assert fitted.polish_iterations == fitted.precision_work['projected_flow_steps']
    torch.testing.assert_close(fitted.x, tensor([.25, .75]), rtol=0, atol=2e-15)


def test_original_nonfinite_state_rejected_and_inputs_preserved(bank):
    x, problem = collective_problem()
    x[0] = float('nan')
    saved = x.view(torch.uint8).clone()
    with pytest.raises(ValueError, match='Finite feasible'):
        precision.recover_quadratic(x, torch.zeros_like(problem[-1]), *problem, bank, CudaPolicy())
    assert torch.equal(x.view(torch.uint8), saved)


@pytest.mark.parametrize('disagreement', ['gate', 'parity'])
def test_final_compiled_certificate_disagreement_retains_original(bank, disagreement):
    class DisagreeingBank:
        def __getattr__(self, name):
            return getattr(bank, name)

        def gap_kkt(self, *args):
            stats = bank.gap_kkt(*args).clone()
            if disagreement == 'gate':
                stats[2] = 1.
            else:
                # Both candidate gates still pass, but its independently
                # calculated objective scale is intentionally inconsistent.
                stats[1] += 1.
            return stats

    x, problem = collective_problem()
    dual = torch.zeros_like(problem[-1])
    result = precision.recover_quadratic(x, dual, *problem, DisagreeingBank(), CudaPolicy())
    assert not result.qualified and not result.work['candidate_accepted']
    assert result.x is x and result.dual is dual
    assert result.work['stop_reason'] == 'independent_admission_rejected'
    assert result.work['compiled_certificate_evaluations'] == 2


def test_compiled_component_steps_match_independent_block_updates(bank):
    x = tensor([.2, .2, .2, .7, .7, .9])
    linear, allowed = tensor([-.4, .15, .25, 2e6, -2e6, -.2]), tensor([1., 0., 1., 1., 1., 1.])
    caps = x.new_zeros((6, 6))
    caps[:3, :3], caps[3:5, 3:5] = .2, 1e6
    caps.fill_diagonal_(0.)
    labels, counts, scales, sigma = precision.component_geometry(x, linear, caps)
    a, cc = linear / scales, caps / scales[:, None]
    q, t = torch.zeros_like(cc), torch.zeros_like(x)
    extrapolated = t.clone()
    blocks = [torch.nonzero(labels == i, as_tuple=True)[0] for i in range(counts.numel())]
    references = [(q[idx[:, None], idx[None, :]].clone(), t[idx].clone(), extrapolated[idx].clone())
                  for idx in blocks]
    for _ in range(16):
        q, t, extrapolated = bank.directional_cut_step(a, cc, allowed, q, t, extrapolated, .9, sigma)
        for i, idx in enumerate(blocks):
            bq, bt, be = references[i]
            bc = cc[idx[:, None], idx[None, :]]
            next_q = (bq + (be[None, :] - be[:, None]) / idx.numel()).clamp(-bc, bc)
            next_t = (bt - .9 * (a[idx] - next_q.sum(1))).clamp_min(0.).minimum(allowed[idx])
            references[i] = next_q, next_t, 2 * next_t - bt
            for measured, expected in zip((q[idx[:, None], idx[None, :]], t[idx], extrapolated[idx]),
                                          references[i], strict=True):
                torch.testing.assert_close(measured, expected, rtol=128 * torch.finfo(x.dtype).eps,
                                           atol=128 * torch.finfo(x.dtype).eps)
        assert (q == -q.T).all() and (q.abs() <= cc).all()
        assert (t >= 0).all() and (t <= allowed).all()


def test_compiled_raw_flow_matches_independent_kink_and_box_step(bank):
    x = tensor([.2, .2, .6, .6, .6, .9])
    caps = x.new_full((6, 6), .3)
    caps.fill_diagonal_(0.)
    same = x[:, None] == x[None, :]
    external = caps * (x[None, :] - x[:, None]).sign()
    sizes = same.sum(1)
    nominal = tensor([.7, -.1, .3, -.5, 1.1, -.2])
    left, right = nominal - .25, nominal + .4
    left[2], right[2] = .9, -.8  # Original nonconvex kink convention.
    at_lower = torch.tensor([True, False, False, False, False, False], device=x.device)
    at_upper = torch.tensor([False, False, False, True, False, False], device=x.device)
    fixed = torch.tensor([False, False, False, False, True, False], device=x.device)
    dual = torch.zeros_like(caps)
    for _ in range(16):
        current = torch.where(same, dual, external)
        divergence = -current.sum(1)
        derivative = torch.minimum(right, torch.maximum(left, -divergence))
        derivative = torch.where(left <= right, derivative, nominal)
        derivative = torch.where(at_lower, right, derivative)
        derivative = torch.where(at_upper, left, derivative)
        residual = derivative + divergence
        residual = torch.where(at_lower, residual.minimum(x.new_zeros(())), residual)
        residual = torch.where(at_upper, residual.maximum(x.new_zeros(())), residual)
        residual = torch.where(fixed, 0., residual)
        correction = (residual[:, None] - residual[None, :]) / sizes[:, None]
        expected = (current + torch.where(same, correction, 0.)).clamp(-caps, caps)
        dual = bank.raw_dual_step(dual, caps, same, external, sizes, nominal, left, right,
                                 at_lower, at_upper, fixed)
        torch.testing.assert_close(dual, expected, rtol=128 * torch.finfo(x.dtype).eps,
                                   atol=128 * torch.finfo(x.dtype).eps)
        assert (dual == -dual.T).all() and (dual.abs() <= caps).all()
        assert torch.equal(dual[~same], external[~same])
    assert 'raw_dual_step' in bank.compilation_diagnostics()
