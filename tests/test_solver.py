from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import osqp
import pytest
from scipy import sparse

from clipp1d.policy import Policy
from clipp1d.solver import solve_quadratic


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
    result = solve_quadratic(np.ones(3), np.array([.1, .4, 1]), np.zeros(3), np.ones(3),
                             np.ones(2), np.zeros(3), replace(Policy(), inner_max_iterations=1))
    assert not result.qualified
