"""Independent exact-subset checks for the rowwise cut lower bound.

These CPU arithmetic references do not establish allocated-CUDA qualification.
"""

from dataclasses import replace
from decimal import Decimal, localcontext
import itertools

import numpy as np
import pytest
import torch

from clipp1d.cuda.audit import _cut_lower_bound, directional_cut
from clipp1d.cuda.policy import CudaPolicy


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def decimal(value):
    return Decimal.from_float(float(value))


def exact_bounds(a, caps, q, allowed):
    """Evaluate represented input floats, without sharing tensor reductions."""
    with localcontext() as context:
        context.prec = 400
        n = len(a)
        unary = [decimal(value) for value in a]
        residual = [unary[i] - sum((decimal(q[i, j]) for j in range(n)), Decimal(0))
                    for i in range(n)]
        dual = sum((min(value, Decimal(0)) for value, permit in zip(residual, allowed)
                    if permit), Decimal(0))
        values = []
        for bits in itertools.product((0, 1), repeat=n):
            if any(bit > permit for bit, permit in zip(bits, allowed)):
                continue
            value = sum((unary[i] for i in range(n) if bits[i]), Decimal(0))
            value += sum((decimal(caps[i, j]) for i in range(n) for j in range(i + 1, n)
                          if bits[i] != bits[j]), Decimal(0))
            values.append(value)
        return +dual, min(values)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("scale", [1., 1e-100, 1e100])
@pytest.mark.parametrize("seed", range(8))
def test_rowwise_bound_is_below_exact_dual_and_every_feasible_subset(seed, scale):
    rng = np.random.default_rng(seed)
    n = 6
    raw = np.triu(rng.normal(size=(n, n)), 1)
    q = tensor((raw - raw.T) * scale)
    extra = rng.uniform(.01, .5, size=(n, n))
    caps = q.abs() + tensor((extra + extra.T) * (.5 * scale))
    caps.fill_diagonal_(0.)
    a = tensor(rng.normal(size=n) * scale)
    allowed = tensor(rng.integers(0, 2, size=n))
    assert torch.equal(q, -q.T) and bool((q.abs() <= caps).all())
    lower, margin = _cut_lower_bound(a, q, allowed, tensor(scale))
    exact_dual, exact_minimum = exact_bounds(a, caps, q, allowed)
    assert decimal(lower) <= exact_dual <= exact_minimum
    assert float(margin) >= 0


@pytest.mark.parametrize("scale", [1., 3.7, 2.3e12, 8.9e18])
def test_cancellation_stays_conservative_in_original_units(scale):
    q = tensor([[0., 1e9 + .1, -1e9 + .2],
                [-1e9 - .1, 0., 1e9 + .3],
                [1e9 - .2, -1e9 - .3, 0.]])
    a = q.sum(-1)
    caps = torch.full_like(q, 2e9)
    caps.fill_diagonal_(0.)
    allowed = torch.ones_like(a)
    exact_dual, exact_minimum = exact_bounds(a, caps, q, allowed)
    lower, margin = _cut_lower_bound(a / scale, q / scale, allowed, tensor(1 / scale))
    with localcontext() as context:
        context.prec = 400
        assert decimal(lower) * decimal(scale) <= exact_dual <= exact_minimum
        assert decimal(margin) * decimal(scale) > Decimal("2e-5")
    details = {}
    qualified, direction = directional_cut(
        a / scale, caps / scale, allowed, q / scale,
        replace(CudaPolicy(), inner_max_iterations=1), diagnostics=details,
        tolerance=tensor(CudaPolicy().stationarity_tol / scale), roundoff_unit=tensor(1 / scale))
    assert not qualified and direction is None and details["status"] == "unresolved"


def test_positive_256_node_cut_qualifies_without_a_global_positive_row_penalty():
    a = torch.linspace(0., .05, 256, dtype=torch.float64)
    q = torch.zeros((256, 256), dtype=torch.float64)
    caps = torch.ones_like(q)
    caps.fill_diagonal_(0.)
    unit, tolerance = tensor(9.4e-8), tensor(1.88e-12)
    old_margin = 8 * (256 + 2) * torch.finfo(a.dtype).eps * (unit + a.abs().sum())
    assert float(old_margin) > float(tolerance)
    details = {}
    qualified, direction = directional_cut(
        a, caps, torch.ones_like(a), q, replace(CudaPolicy(), inner_max_iterations=1),
        diagnostics=details, tolerance=tolerance, roundoff_unit=unit)
    assert qualified and direction is None
    assert details["iterations"] == 0 and details["status"] == "qualified_lower_bound"
    # The empty subset is optimal because every unary and edge cost is nonnegative.
    assert -float(tolerance) <= details["lower_bound"] < 0
    assert details["roundoff_margin"] == -details["lower_bound"]


@pytest.mark.parametrize("negative", [-2.1e-5, -1e-3])
@pytest.mark.parametrize("scale", [1., 3.7, 1e8])
def test_small_true_negative_cut_is_never_hidden_by_large_positive_rows(negative, scale):
    a = tensor([negative, 1e8, 1e7, 0.])
    caps = torch.zeros((4, 4), dtype=torch.float64)
    allowed = torch.ones_like(a)
    details = {}
    qualified, direction = directional_cut(
        a / scale, caps, allowed, caps, replace(CudaPolicy(), inner_max_iterations=1),
        diagnostics=details, tolerance=tensor(CudaPolicy().stationarity_tol / scale),
        roundoff_unit=tensor(1 / scale))
    exact_dual, exact_minimum = exact_bounds(a, caps, caps, allowed)
    with localcontext() as context:
        context.prec = 400
        assert decimal(details["lower_bound"]) * decimal(scale) <= exact_dual == exact_minimum
    assert not qualified
    assert details["lower_bound"] < -CudaPolicy().stationarity_tol / scale


@pytest.mark.parametrize("allowed", [[0., 0., 0.], [0., 1., 1.], [1., 1., 1.]])
def test_masked_coordinates_and_tiny_negative_terms_keep_exact_bound(allowed):
    a = tensor([-1e100, -1e-100, 1e100])
    q = torch.zeros((3, 3), dtype=torch.float64)
    allowed = tensor(allowed)
    lower, margin = _cut_lower_bound(a, q, allowed, tensor(1e-100))
    exact_dual, exact_minimum = exact_bounds(a, q, q, allowed)
    assert decimal(lower) <= exact_dual == exact_minimum
    assert float(margin) >= 0


def test_subnormal_zero_bound_is_directed_down_even_when_margin_underflows():
    a = tensor([0.])
    q = tensor([[0.]])
    unit = torch.nextafter(tensor(0.), tensor(1.))
    lower, margin = _cut_lower_bound(a, q, tensor([1.]), unit)
    assert float(lower) < 0 and float(margin) > 0
