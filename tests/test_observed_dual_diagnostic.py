"""Independent gate/oracle checks for the offline observed-gradient diagnostic."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.diagnose_observed_dual import (
    edge_bounds, gradient_adjoint_interval, independent_lp, observed_adjoint_bounds,
    observed_dual, reachable_dual,
)


def context(g, left=None, right=None):
    g = np.asarray(g, dtype=float)
    return SimpleNamespace(gradient=g, left=g if left is None else np.asarray(left),
                           right=g if right is None else np.asarray(right))


@pytest.mark.parametrize('tolerance', [0., 2e-5, .1, .8])
def test_adjoint_interval_matches_normalized_gate(tolerance):
    g = np.array([-1000., -1., -1e-12, 0., 1e-12, 1., 1000.])
    lo, hi = gradient_adjoint_interval(g, tolerance)
    for a in (lo, hi):
        value = np.abs(g + a) / (1 + np.abs(g) + np.abs(a))
        np.testing.assert_allclose(value, tolerance, rtol=1e-9, atol=1e-16)
    middle = (lo + hi) / 2
    assert np.all(np.abs(g + middle) / (1 + np.abs(g) + np.abs(middle)) <= tolerance + 1e-15)
    for a in (lo - 1e-5 * (1 + abs(lo)), hi + 1e-5 * (1 + abs(hi))):
        assert np.all(np.abs(g + a) / (1 + np.abs(g) + np.abs(a)) > tolerance)


def test_observed_dual_repairs_bad_surrogate_dual_without_changing_gradient():
    # A perfectly stationary fused pair has gradient[-1,+1], hence q=-1.
    c = context([-1., 1., 0.])
    x = np.array([.4, .4, 1.])
    lower, upper = np.zeros(3), np.ones(3)
    lower[2] = 1.
    # The nonzero edge to the fixed witness is zero-weight; q[1] must remain0.
    caps = np.array([2., 0.])
    q, receipt = observed_dual(c, x, lower, upper, caps, np.zeros(2), .5)
    np.testing.assert_allclose(q, [-1., 0.], atol=1e-10)
    np.testing.assert_array_equal(c.gradient, [-1., 1., 0.])
    assert receipt['feasible_above'] < 1e-15


def test_large_retained_edge_residual_does_not_set_invalid_node_bracket():
    c = context([-1., 1., 0.])
    x = np.array([.4, .4, 1.])
    lower, upper = np.zeros(3), np.ones(3)
    lower[2] = 1.
    q, receipt = observed_dual(c, x, lower, upper, np.array([2., 0.]), np.zeros(2), 1.9)
    np.testing.assert_allclose(q, [-1., 0.], atol=1e-10)
    assert receipt['feasible_above'] < 1e-15


def test_edge_modes_distinguish_exact_constraints_from_original_gate_budget():
    x, caps, tolerance = np.array([.2, .5, .5, .1]), np.array([2., 3., 4.]), .01
    lo, hi = edge_bounds(x, caps, tolerance, 'exact')
    np.testing.assert_array_equal(lo, [2., -3., -4.])
    np.testing.assert_array_equal(hi, [2., 3., -4.])
    lo, hi = edge_bounds(x, caps, tolerance, 'edge_tolerant')
    np.testing.assert_allclose(lo, [1.97, -3., -4.])
    np.testing.assert_allclose(hi, [2., 3., -3.95])
    lo, hi = edge_bounds(x, caps, tolerance, 'full_gate')
    np.testing.assert_allclose(lo, [1.97, -3.04, -4.05])
    np.testing.assert_allclose(hi, [2.03, 3.04, -3.95])


def test_observed_dual_cannot_remove_incompatible_fused_block_sum():
    c = context([1., 1., 0.])
    x = np.array([.4, .4, 1.])
    lower, upper = np.zeros(3), np.ones(3)
    lower[2] = 1.
    caps = np.array([2., 0.])
    q, receipt = observed_dual(c, x, lower, upper, caps, np.zeros(2), .6)
    assert receipt['infeasible_below'] > .49
    np.testing.assert_array_equal(c.gradient, [1., 1., 0.])
    assert not independent_lp(c, x, lower, upper, caps, 2e-5).success
    assert q[1] == 0


def test_clipping_and_box_normal_signs():
    x = np.array([0., 1., .5, .5, 1.])
    lower, upper = np.zeros(5), np.ones(5)
    lower[-1] = 1.
    c = context([3., -4., 9., -7., 100.], left=[0., -4., -2., 4., 100.],
                right=[3., 0., 3., -7., 100.])
    lo, hi = observed_adjoint_bounds(c, x, lower, upper, 0.)
    # Lower box admits g+a>=0; upper box admits g+a<=0. Upward kink permits
    # a in[-right,-left]; downward kink uses the retained ordinary derivative.
    np.testing.assert_array_equal(lo, [-3., -np.inf, -3., 7., -np.inf])
    np.testing.assert_array_equal(hi, [np.inf, 4., 2., 7., np.inf])


def test_reachability_matches_independent_sparse_lp():
    rng = np.random.default_rng(719)
    for _ in range(30):
        n = 12
        x = rng.choice([.2, .5, 1.], n)
        lower, upper = np.zeros(n), np.ones(n)
        lower[-1] = upper[-1] = x[-1] = 1.
        caps = rng.uniform(.1, 4., n - 1)
        c = context(rng.normal(size=n))
        tolerance = rng.uniform(0., .2)
        lo, hi = observed_adjoint_bounds(c, x, lower, upper, tolerance)
        el, eh = -caps.copy(), caps.copy()
        active = np.diff(x) != 0
        el[active] = eh[active] = caps[active] * np.sign(np.diff(x)[active])
        q = reachable_dual(lo, hi, el, eh)
        lp = independent_lp(c, x, lower, upper, caps, tolerance)
        assert (q is not None) == lp.success
        if q is not None:
            a = np.r_[-q[0], q[:-1] - q[1:], q[-1]]
            assert np.all(a >= lo - 1e-12)
            assert np.all(a <= hi + 1e-12)
