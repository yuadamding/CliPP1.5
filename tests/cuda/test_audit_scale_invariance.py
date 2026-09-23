"""Original-unit cut oracles; CPU references do not qualify CUDA execution."""

from dataclasses import replace
from decimal import Decimal, localcontext
import itertools

import numpy as np
import pytest
import torch

from clipp1d.cuda.audit import audit_raw, directional_cut
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exact_value(unary, caps, point):
    """Evaluate represented input floats with independent high-precision sums."""
    with localcontext() as context:
        context.prec = 100
        a = [Decimal.from_float(float(value)) for value in unary]
        x = [Decimal.from_float(float(value)) for value in point]
        value = sum((ai * xi for ai, xi in zip(a, x)), Decimal(0))
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                value += Decimal.from_float(float(caps[i, j])) * abs(x[i] - x[j])
        return +value


def exact_minimum(unary, caps, allowed):
    return min(exact_value(unary, caps, bits)
               for bits in itertools.product((0., 1.), repeat=len(unary))
               if all(bit <= permitted for bit, permitted in zip(bits, allowed)))


def scaled_cut(unary, caps, allowed, dual, scale, *, policy=CudaPolicy()):
    scale = tensor(scale)
    details = {}
    result = directional_cut(unary / scale, caps / scale, allowed, dual / scale,
                             policy, diagnostics=details,
                             tolerance=tensor(policy.stationarity_tol) / scale,
                             roundoff_unit=torch.ones_like(scale) / scale)
    return *result, details


def original_units(value, scale):
    with localcontext() as context:
        context.prec = 100
        return Decimal.from_float(float(value)) * Decimal.from_float(float(scale))


@pytest.mark.parametrize("scale", [1., 3.7, 2.3e12, 8.9e18])
def test_zero_cut_with_huge_capacities_has_no_normalization_error_floor(scale):
    # The true optimum is exactly zero and q=0 is an exact feasible witness.
    # Large unused capacities must not introduce an original-unit error floor.
    unary = torch.zeros(4, dtype=torch.float64)
    caps = torch.full((4, 4), 2.3e12, dtype=torch.float64)
    caps.fill_diagonal_(0.)
    ok, direction, details = scaled_cut(unary, caps, torch.ones_like(unary),
                                        torch.zeros_like(caps), scale,
                                        policy=replace(CudaPolicy(), inner_max_iterations=1))
    assert ok and direction is None
    assert details["iterations"] == 0
    assert details["status"] == "qualified_lower_bound"
    assert exact_minimum(unary, caps, torch.ones_like(unary)) == 0
    lower = original_units(details["lower_bound"], scale)
    assert -Decimal("1e-12") < lower < 0


def test_raw_audit_scales_its_error_unit_for_large_fused_capacities():
    alt = tensor([20., 20., 20.])
    slope = torch.full((3, 1), .5, dtype=torch.float64)
    model = TensorModel(('a', 'b', 'c'), alt, 100 - alt, slope,
                        torch.zeros_like(slope), torch.full_like(alt, 1e-6),
                        torch.ones_like(alt), 1e-6, Kernels('cpu'))
    caps = torch.full((3, 3), 1e14, dtype=torch.float64)
    caps.fill_diagonal_(0.)
    result = audit_raw(model, torch.full_like(alt, .4), torch.zeros_like(caps), caps,
                       policy=replace(CudaPolicy(), inner_max_iterations=1))
    assert result.qualified and result.signed_direction_count == 2
    for name in ('positive', 'negative'):
        details = result.diagnostics[name]
        assert details['iterations'] == 0
        assert original_units(details['lower_bound'], details['objective_scale']) > -Decimal('1e-12')


def test_exact_256_node_scaling_fixture_high_lambda_audit_qualifies_immediately():
    # This is the actual late-path N=256 failure fixture. Its single-support
    # pilots and all-fused pooled optimum have analytical count solutions;
    # neither a CPU path fit nor a substituted optimizer is needed here.
    from benchmarks.qualify_cuda import scaling_fixture
    from clipp1d.cuda.graph import build_graph
    from clipp1d.cuda.kernels import differences

    model = TensorModel.from_host(scaling_fixture(256), 'cpu', compiled=False)
    pilot = (model.alt / (model.alt + model.ref)) / model.slope[:, 0]
    graph = build_graph(pilot, model.mutation_ids)
    caps = graph.weights * 154555.71518032427
    pooled = (model.alt.sum() / (model.alt + model.ref).sum()) / model.slope[0, 0]
    x = pooled.expand(model.n).clone()
    gradient = model.terms(x)[1]
    dual = -differences(gradient) / model.n
    assert bool((dual.abs() <= caps).all())
    assert torch.equal(dual, -dual.T)
    result = audit_raw(model, x, dual, caps,
                       policy=replace(CudaPolicy(), inner_max_iterations=1))
    assert result.qualified and result.signed_direction_count == 2
    assert result.residual <= CudaPolicy().stationarity_tol
    for sign in ('positive', 'negative'):
        details = result.diagnostics[sign]
        assert details['status'] == 'qualified_lower_bound'
        assert details['iterations'] == 0
        assert original_units(details['lower_bound'], details['objective_scale']) >= -Decimal('2e-5')


@pytest.mark.parametrize("scale", [1., 3.7, 2.3e9, 7.1e15])
def test_large_cancellation_keeps_absolute_dual_error_after_normalization(scale):
    dual = tensor([[0., 1e9 + .1, -1e9 + .2],
                   [-1e9 - .1, 0., 1e9 + .3],
                   [1e9 - .2, -1e9 - .3, 0.]])
    caps = torch.full_like(dual, 2e9)
    caps.fill_diagonal_(0.)
    unary = dual.sum(-1)
    allowed = torch.ones_like(unary)
    ok, direction, details = scaled_cut(unary, caps, allowed, dual, scale,
                                        policy=replace(CudaPolicy(), inner_max_iterations=1))
    oracle = exact_minimum(unary, caps, allowed)
    assert original_units(details['lower_bound'], scale) <= oracle
    assert original_units(details['roundoff_margin'], scale) > Decimal('2e-5')
    assert not ok and direction is None
    assert details['status'] == 'unresolved'


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sign", [-1., 1.])
def test_scaled_certificates_agree_with_exhaustive_original_unit_cuts(seed, sign):
    rng = np.random.default_rng(seed)
    unary = tensor(sign * rng.normal(size=5))
    raw_caps = rng.uniform(.01, .8, size=(5, 5))
    caps = tensor((raw_caps + raw_caps.T) * .5)
    caps.fill_diagonal_(0.)
    allowed = tensor(rng.integers(0, 2, size=5))
    initial = tensor(rng.normal(size=(5, 5)))
    initial = initial - initial.T
    oracle = exact_minimum(unary, caps, allowed)
    threshold = Decimal.from_float(CudaPolicy().stationarity_tol)
    for scale in (1., 3.7, 31.):
        ok, direction, details = scaled_cut(unary, caps, allowed, initial, scale)
        assert original_units(details['lower_bound'], scale) <= oracle
        if ok:
            assert oracle >= -threshold and direction is None
        else:
            assert direction is not None, details
            assert bool(((direction >= 0) & (direction <= allowed)).all())
            assert exact_value(unary, caps, direction) < -threshold


@pytest.mark.parametrize("unit", [0., -1., float('nan'), float('inf'), [1., 1.]])
def test_invalid_error_units_cannot_disable_numerical_protection(unit):
    with pytest.raises(ValueError, match='roundoff|error|unit'):
        directional_cut(tensor([0.]), tensor([[0.]]), tensor([1.]), tensor([[0.]]),
                        roundoff_unit=tensor(unit))
