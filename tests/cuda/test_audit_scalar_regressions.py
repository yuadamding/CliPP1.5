"""Independent tensor-reference regressions; CPU tests do not qualify CUDA runs."""
import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.special import logsumexp

from clipp1d.cuda.audit import audit_raw, directional_cut
from clipp1d.cuda.kernels import Kernels, differences
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.scalar import Problems, solve_scalar


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(alt, ref, slope=None, lower=None, upper=None):
    alt, ref = tensor(alt), tensor(ref)
    n = len(alt)
    slope = torch.ones((n, 1), dtype=torch.float64) if slope is None else tensor(slope)
    valid = slope > 0
    prior = torch.where(valid, -valid.sum(-1, keepdim=True).double().log(), -float("inf"))
    return TensorModel(tuple(f"m{i}" for i in range(n)), alt, ref, slope, prior,
                       torch.full_like(alt, 1e-6) if lower is None else tensor(lower),
                       torch.ones_like(alt) if upper is None else tensor(upper),
                       1e-6, Kernels("cpu", compiled=False))


@pytest.mark.parametrize("seed", range(16))
def test_signed_cut_lower_bound_and_attained_direction_against_every_subset(seed):
    rng = np.random.default_rng(seed)
    n = 6
    unary = tensor(rng.normal(size=n))
    weights = tensor(rng.uniform(0., .8, (n, n)))
    weights = (weights + weights.T) / 2
    weights.fill_diagonal_(0)
    allowed = tensor(rng.integers(0, 2, n))
    initial = tensor(rng.normal(size=(n, n)))
    initial = initial - initial.T
    oracle = min(float(unary @ tensor(bits) + .5 * (weights * differences(tensor(bits)).abs()).sum())
                 for bits in itertools.product((0., 1.), repeat=n)
                 if all(bits[i] <= allowed[i] for i in range(n)))
    diagnostics = {}
    qualified, direction = directional_cut(unary, weights, allowed, initial, diagnostics=diagnostics)
    assert diagnostics["lower_bound"] <= oracle + 1e-12
    if qualified:
        assert oracle >= -CudaPolicy().stationarity_tol
        assert direction is None
        assert diagnostics["status"] == "qualified_lower_bound"
    elif direction is not None:
        assert bool(((direction >= 0) & (direction <= allowed)).all())
        objective = unary @ direction + .5 * (weights * differences(direction).abs()).sum()
        assert float(objective) < -CudaPolicy().stationarity_tol
    else:
        pytest.fail(f"Small complete-graph directional cut failed to resolve: {diagnostics}")


def test_cut_iteration_exhaustion_remains_unresolved():
    # The exact minimum is zero. One first-order step cannot yet route the
    # negative unary through the expensive fusion edge to its positive partner.
    weights = tensor([[0., 10.], [10., 0.]])
    diagnostics = {}
    ok, direction = directional_cut(tensor([-1., 1.]), weights, tensor([1., 1.]),
                                    torch.zeros_like(weights),
                                    replace(CudaPolicy(), inner_max_iterations=1), diagnostics=diagnostics)
    assert not ok and direction is None
    assert diagnostics["status"] == "unresolved"
    assert diagnostics["lower_bound"] < -CudaPolicy().stationarity_tol


def test_cut_lower_bound_margin_accounts_for_canceled_dual_terms():
    # q has large cancelling row terms while r is almost zero. A margin formed
    # from r alone would discard the actual row-sum arithmetic scale.
    weights = tensor([[0., 1e9, 1e9], [1e9, 0., 1e9], [1e9, 1e9, 0.]])
    dual = tensor([[0., 1e9, -1e9], [-1e9, 0., 1e9], [1e9, -1e9, 0.]])
    diagnostics = {}
    ok, direction = directional_cut(torch.zeros(3, dtype=torch.float64), weights,
                                    torch.ones(3, dtype=torch.float64), dual,
                                    replace(CudaPolicy(), inner_max_iterations=1), diagnostics=diagnostics)
    assert not ok and direction is None
    assert diagnostics["roundoff_margin"] > 1e-5
    assert diagnostics["lower_bound"] < 0


def test_stationary_solution_below_one_is_qualified_without_witness():
    m = model([2, 4], [8, 6], upper=[.8, .9])
    x = tensor([.2, .4])
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    result = audit_raw(m, x, zeros, zeros)
    assert result.qualified
    assert result.signed_direction_count == 2
    assert result.diagnostics["clonal_constraint"] is False
    with pytest.raises(ValueError, match="witness"):
        audit_raw(m, x, zeros, zeros, 0)


def test_negative_whole_fused_group_not_blocked_by_clonal_witness():
    m = model([2, 2], [8, 8], slope=[[.5], [.5]])
    x = tensor([1., 1.])
    caps = tensor([[0., 100.], [100., 0.]])
    result = audit_raw(m, x, torch.zeros_like(caps), caps)
    assert not result.qualified
    assert result.status == "negative_descent"
    assert bool((result.direction < 0).all())


def test_noncontiguous_subset_descent_and_bound_mask():
    m = model([3, 1, 3], [1, 5, 1], lower=[1e-6, .5, 1e-6])
    caps = tensor([[0., .1, 5.], [.1, 0., .1], [5., .1, 0.]])
    result = audit_raw(m, tensor([.5, .5, .5]), torch.zeros_like(caps), caps)
    assert not result.qualified
    assert result.direction is not None
    assert result.direction[0] > 0 and result.direction[2] > 0
    assert result.direction[1] == 0


def test_fixed_coordinate_derivative_cannot_hide_free_clipping_descent():
    m = model([1, 0], [9, 1e20], lower=[.5e-6, .5], upper=[1., .5])
    x = tensor([1e-6, .5])
    zeros = torch.zeros((2, 2), dtype=torch.float64)
    result = audit_raw(m, x, zeros, zeros)
    assert not result.qualified
    assert result.direction is not None
    assert result.direction[0] > 0 and result.direction[1] == 0


@pytest.mark.parametrize("point,side", [(1e-6, 1), (1 - 1e-6, -1)])
def test_clipping_one_sided_directions_and_fixed_coordinate(point, side):
    m = model([3], [7], lower=[.5e-6])
    zeros = torch.zeros((1, 1), dtype=torch.float64)
    x = tensor([point])
    result = audit_raw(m, x, zeros, zeros)
    assert not result.qualified
    assert result.direction is not None and result.direction[0] * side > 0
    fixed = model([3], [7], lower=[point], upper=[point])
    assert audit_raw(fixed, x, zeros, zeros).qualified


def test_exact_scalar_groups_skip_interval_buffers(monkeypatch):
    m = model([0, 30, 100], [100, 70, 0], slope=[[.5], [.5], [.5]])
    problems = Problems(m, torch.ones(3, dtype=torch.long))
    def forbidden(*args, **kwargs):
        raise AssertionError("Analytic scalar problems must not allocate interval bounds")
    monkeypatch.setattr(problems, "bounds", forbidden)
    result = solve_scalar(problems)
    torch.testing.assert_close(result.phi, tensor([1e-6, .6, 1.]), rtol=0, atol=1e-14)
    assert bool(result.qualified.all())
    assert result.subdivisions == 0


def test_unique_support_not_in_first_storage_column():
    m = model([30], [70], slope=[[0., .5]])
    result = solve_scalar(Problems(m, torch.ones(1, dtype=torch.long)))
    assert bool(result.qualified.all())
    assert result.phi[0] == pytest.approx(.6)
    assert result.subdivisions == 0


def test_exact_upper_clipping_plateau_uses_first_attaining_point():
    m = model([100], [0])
    result = solve_scalar(Problems(m, torch.ones(1, dtype=torch.long)))
    assert bool(result.qualified.all())
    assert result.phi[0] == 1 - m.eps


@pytest.mark.parametrize("slope", [.4, .3, 1.2])
@pytest.mark.parametrize("threshold", [1e-6, 1 - 1e-6])
def test_scalar_kinks_keep_canonical_true_division_bits(slope, threshold):
    canonical = np.float64(threshold) / np.float64(slope)
    spacing = np.spacing(canonical)
    lo, hi = canonical - 8 * spacing, canonical + 4 * spacing
    m = model([16], [84], slope=[[slope]], lower=[lo], upper=[hi])
    p = Problems(m, torch.ones(1, dtype=torch.long))
    found = p.nearest_kink(tensor([lo]), tensor([hi]))
    assert float(found[0]).hex() == float(canonical).hex()
    # The original feasibility endpoints remain unchanged, including their bits.
    assert float(m.lower[0]).hex() == float(lo).hex()
    assert float(m.upper[0]).hex() == float(hi).hex()


@pytest.mark.parametrize("slope", [.4, 1.2])
@pytest.mark.parametrize("threshold", [1e-6, 1 - 1e-6])
def test_scalar_bounds_at_exact_kink_and_nextafter_neighbors(slope, threshold):
    canonical = np.float64(threshold) / np.float64(slope)
    before = np.nextafter(canonical, -np.inf)
    after = np.nextafter(canonical, np.inf)
    m = model([16, 31], [84, 69], slope=[[slope], [slope]],
              lower=[before, before], upper=[after, after])
    p = Problems(m, torch.tensor([2], dtype=torch.long))
    for lo, hi in [(before, canonical), (canonical, after), (before, after),
                   (canonical, canonical)]:
        lower_bound = p.bounds(tensor([[lo]]), tensor([[hi]]))[0][0, 0]
        # There are at most three representable candidate CCFs in this interval.
        oracle = min(independent_loss(m, [0, 1], value)
                     for value in (before, canonical, after) if lo <= value <= hi)
        assert float(lower_bound) <= oracle + 1e-12
    result = solve_scalar(p)
    assert bool(result.qualified.all())
    assert before <= float(result.phi[0]) <= after
    oracle = min(independent_loss(m, [0, 1], value) for value in (before, canonical, after))
    assert float(result.lower_bound[0]) <= oracle + 1e-12
    assert float(result.loss[0]) <= oracle + 1e-9


@pytest.mark.parametrize("slope", [.4, 1.2])
def test_single_support_refit_at_upper_clipping_uses_canonical_boundary(slope):
    from clipp1d.cuda.partition import refit
    canonical = np.float64(1 - 1e-6) / np.float64(slope)
    after = np.nextafter(canonical, np.inf)
    m = model([100, 80], [0, 0], slope=[[slope], [slope]],
              lower=[canonical - 1e-4, canonical - 1e-4], upper=[after, after])
    result = refit(m, tensor([canonical, canonical]))
    assert result.centers.numel() == 1
    assert float(result.centers[0]).hex() == float(canonical).hex()
    assert bool(result.scalar.qualified.all())


@pytest.mark.parametrize("slope", [.4, 1.2])
@pytest.mark.parametrize("threshold", [1e-6, 1 - 1e-6])
def test_taylor_bound_is_conservative_on_adjacent_float_intervals_near_clipping(slope, threshold):
    canonical = np.float64(threshold) / np.float64(slope)
    points = [canonical]
    before = after = canonical
    for _ in range(32):
        before = np.nextafter(before, -np.inf)
        after = np.nextafter(after, np.inf)
        points.extend((before, after))
    points = np.sort(points)
    m = model([16, 31], [84, 69], slope=[[slope], [slope]],
              lower=[points[0]] * 2, upper=[points[-1]] * 2)
    p = Problems(m, torch.tensor([2], dtype=torch.long))
    for lo, hi in zip(points[:-1], points[1:]):
        bound = p.bounds(tensor([[lo]]), tensor([[hi]]))[0][0, 0]
        oracle = min(independent_loss(m, [0, 1], value) for value in (lo, hi))
        assert float(bound) <= oracle + 1e-12


def independent_loss(m, members, value):
    alt = m.alt.numpy()[members, None]
    ref = m.ref.numpy()[members, None]
    probability = np.clip(m.slope.numpy()[members] * value, m.eps, 1 - m.eps)
    joint = alt * np.log(probability) + ref * np.log1p(-probability) + m.log_prior.numpy()[members]
    return float(-logsumexp(joint, axis=1).sum())


def test_mixed_multiplicity_group_bounds_and_qualified_global_refit():
    m = model([20, 37, 0, 18, 40], [80, 63, 100, 82, 60],
              slope=[[.15, .3, .45], [.2, .4, .6], [.3, .6, 0.], [.4, 0., 0.], [.25, .5, 0.]])
    lengths = torch.tensor([2, 1, 2], dtype=torch.long)
    p = Problems(m, lengths)
    groups = [[0, 1], [2], [3, 4]]
    for lo, hi in [(1e-6, .01), (.01, .4), (.4, .7), (.7, 1.)]:
        lower = p.bounds(torch.full((3, 1), lo, dtype=torch.float64),
                         torch.full((3, 1), hi, dtype=torch.float64))[0][:, 0]
        for i, members in enumerate(groups):
            sampled = min(independent_loss(m, members, point) for point in np.linspace(lo, hi, 401))
            assert lower[i] <= sampled + 1e-10
    result = solve_scalar(p, replace(CudaPolicy(), scalar_batch_size=2))
    assert bool(result.qualified.all())
    for i, members in enumerate(groups):
        # Independent global exploration followed by a local scalar minimizer.
        def objective(value):
            return independent_loss(m, members, float(np.asarray(value).reshape(-1)[0]))
        global_fit = differential_evolution(objective, [(1e-6, 1.)], seed=17 + i,
                                            tol=1e-12, atol=1e-11, polish=True)
        oracle = min(global_fit.fun, objective(1e-6), objective(1.))
        local = minimize_scalar(objective, bounds=(max(1e-6, global_fit.x[0] - .01),
                                                  min(1., global_fit.x[0] + .01)), method="bounded",
                                options={"xatol":1e-14})
        oracle = min(oracle, local.fun)
        assert float(result.lower_bound[i]) <= oracle + 1e-9
        assert float(result.loss[i]) <= oracle + 2e-7
        assert float(result.gap[i]) <= CudaPolicy().scalar_atol + CudaPolicy().scalar_rtol * abs(float(result.loss[i]))


def test_membership_batching_preserves_scalar_values():
    m = model([12, 30, 21, 15, 39], [88, 70, 79, 85, 61],
              slope=[[.2, .4], [.3, 0.], [.25, .5], [.2, 0.], [.3, .6]])
    p = Problems(m, torch.ones(5, dtype=torch.long))
    batch = solve_scalar(p, replace(CudaPolicy(), scalar_batch_size=2))
    whole = solve_scalar(p, replace(CudaPolicy(), scalar_batch_size=64))
    assert bool(batch.qualified.all() & whole.qualified.all())
    torch.testing.assert_close(batch.loss, whole.loss, rtol=1e-12, atol=1e-9)
    torch.testing.assert_close(batch.phi, whole.phi, rtol=0., atol=1e-5)


def test_tiny_scalar_budget_does_not_promote_unresolved_mixture():
    m = model([20, 35], [80, 65], slope=[[.2, .4, .6], [.15, .3, .45]])
    p = Problems(m, torch.tensor([2], dtype=torch.long))
    policy = replace(CudaPolicy(), scalar_max_intervals=1, scalar_atol=1e-14, scalar_rtol=1e-15)
    result = solve_scalar(p, policy)
    assert not bool(result.qualified.all())
    assert result.gap[0] > policy.scalar_atol + policy.scalar_rtol * result.loss[0].abs()


@pytest.mark.parametrize("field", ["alt", "ref", "slope", "log_prior", "lower", "upper"])
def test_scalar_kernel_calls_preserve_model_mutation_guard(field):
    m = model([20], [80], slope=[[.2, .4]])
    p = Problems(m, torch.ones(1, dtype=torch.long))
    getattr(m, field).add_(.001)
    with pytest.raises(ValueError, match="modified"):
        p.evaluate(tensor([[.3]]))
    with pytest.raises(ValueError, match="modified"):
        m.subset(slice(0, 1))


def test_scalar_membership_mutation_rejected():
    m = model([20, 30], [80, 70])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    p.group[0] = 1
    with pytest.raises(ValueError, match="modified"):
        solve_scalar(p)


@pytest.mark.parametrize("lengths", [[0, 2], [1], [3], [1, -1, 2]])
def test_scalar_invalid_membership_lengths_rejected(lengths):
    m = model([20, 30], [80, 70])
    with pytest.raises(ValueError, match="lengths"):
        Problems(m, torch.tensor(lengths, dtype=torch.long))
