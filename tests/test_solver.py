from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import osqp
import pytest
from scipy import sparse

from clipp1d.policy import Policy
from clipp1d.solver import quadratic_gap, solve_quadratic, solve_quadratic_iterative


def oracle(h, target, lower, upper, caps):
    n = len(h)
    # An independent dense chain matrix is intentionally used only in this test oracle.
    d = np.diff(np.eye(n), axis=0)
    ident = np.eye(n - 1)
    a = np.block([[d, -ident], [-d, -ident], [np.eye(n), np.zeros((n, n - 1))]])
    problem = osqp.OSQP()
    problem.setup(P=sparse.diags(np.r_[h, np.zeros(n - 1)], format="csc"),
                  q=np.r_[-h * target, caps], A=sparse.csc_matrix(a),
                  l=np.r_[np.full(2 * (n - 1), -np.inf), lower],
                  u=np.r_[np.zeros(2 * (n - 1)), upper], verbose=False,
                  eps_abs=1e-10, eps_rel=1e-10, max_iter=100000, polishing=True)
    result = problem.solve(raise_error=True)
    assert result.info.status == "solved"
    return result.x[:n]


@pytest.mark.parametrize("n,seed", [(2, 3), (4, 19), (8, 8), (6, 71)])
def test_inner_against_independent_qp(n, seed):
    rng = np.random.default_rng(seed)
    h, target = np.exp(rng.uniform(-1, 2, n)), rng.normal(.5, .7, n)
    lower, upper = np.zeros(n), rng.uniform(.65, 1, n)
    lower[1] = upper[1] = 1
    caps = rng.uniform(.05, 2, n - 1)
    expected = oracle(h, target, lower, upper, caps)
    fit = solve_quadratic(h, target, lower, upper, caps, np.zeros(n))
    assert fit.qualified and fit.gap >= 0
    assert_allclose(fit.x, expected, atol=3e-5)
    assert np.all(fit.x >= lower) and np.all(fit.x <= upper) and fit.x[1] == 1


def test_inner_exhaustion_fails_closed():
    result = solve_quadratic_iterative(np.ones(3), np.array([.1, .4, 1]), np.zeros(3), np.ones(3),
                             np.ones(2), np.zeros(3), replace(Policy(), inner_max_iterations=1))
    assert not result.qualified


@pytest.mark.parametrize("offset", [1., 1e6, 1e9])
def test_frozen_target_offset_does_not_weaken_qualification(offset):
    h, target = np.ones(3), np.array([offset, 0, 1.])
    lower, upper, caps = np.array([1., 0, 0]), np.ones(3), np.ones(2)
    result = solve_quadratic(h, target, lower, upper, caps, np.array([1., 0, 1.]))
    assert result.qualified
    assert_allclose(result.x, 1, atol=1e-6, rtol=0)
    # Independent objective with the irrelevant frozen-coordinate constant removed.
    error = .5 * (result.x[1]**2 + (result.x[2] - 1)**2) + np.abs(np.diff(result.x)).sum() - .5
    assert error <= 1e-8
    assert result.gap >= error - 1e-14


def test_gap_does_not_cancel_under_a_large_frozen_offset():
    x = np.array([1., .97606969, .97914330])
    q = np.array([-1., .9])
    # This feasible dual yields a positive gap whose value is independent of T.
    gaps = [quadratic_gap(x, q, np.ones(3), np.array([t, 0, 1.]),
                          np.array([1., 0, 0]), np.ones(3), np.ones(2))[0]
            for t in (1., 1e6, 1e9)]
    assert gaps[0] > 1e-3
    assert_allclose(gaps, gaps[0], rtol=1e-12, atol=1e-14)


def test_nonnegative_gap_matches_independent_small_primal_minus_dual():
    rng = np.random.default_rng(91)
    n = 7
    h, target = rng.uniform(.5, 3, n), rng.normal(size=n)
    lower, upper = np.zeros(n), np.ones(n)
    lower[2] = upper[2] = .5
    x = np.clip(rng.uniform(size=n), lower, upper)
    caps, q = rng.uniform(.1, 3, n - 1), rng.uniform(-.1, .1, n - 1)
    d = np.diff(np.eye(n), axis=0)
    a = d.T @ q
    z = np.clip(target - a / h, lower, upper)
    primal = .5 * np.dot(h, (x - target)**2) + np.dot(caps, np.abs(d @ x))
    dual = .5 * np.dot(h, (z - target)**2) + np.dot(a, z)
    gap, _ = quadratic_gap(x, q, h, target, lower, upper, caps)
    assert_allclose(gap, primal - dual, rtol=1e-13, atol=1e-13)
