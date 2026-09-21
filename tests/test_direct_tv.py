from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d.policy import Policy
from clipp1d.solver import profile_quadratic_witnesses, solve_quadratic, solve_quadratic_iterative
from clipp1d.tv import direct_primal, forward_messages, split_primal
from test_solver import oracle


@pytest.mark.parametrize("seed", range(20))
def test_bounded_direct_tv_against_independent_epigraph_qp(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 13
    h, target = np.exp(rng.uniform(-2, 4, n)), rng.normal(.5, 2, n)
    lower = rng.uniform(-.3, .2, n)
    upper = lower + rng.uniform(.01, 2, n)
    if seed % 2 == 0:
        lower[n // 2] = upper[n // 2] = .5
    caps = np.exp(rng.uniform(-3, 5, n - 1))
    expected = oracle(h, target, lower, upper, caps)
    result = solve_quadratic(h, target, lower, upper, caps, lower)
    assert result.qualified and result.algorithm == "bounded_weighted_tv_dp"
    assert_allclose(result.x, expected, atol=2e-6, rtol=1e-7)
    assert result.gap <= Policy().inner_atol + Policy().inner_rtol * result.gap_scale
    assert result.kkt_residual <= Policy().inner_kkt_tol


@pytest.mark.parametrize("witness", [0, 4, 8])
def test_fixed_witness_split_preserves_objective_and_dual(witness):
    rng = np.random.default_rng(23)
    h, target, caps = rng.uniform(.2, 5, 9), rng.normal(.5, .3, 9), rng.uniform(0, 2, 8)
    lower, upper = np.zeros(9), np.ones(9)
    lower[witness] = upper[witness] = 1
    full, _ = direct_primal(h, target, lower, upper, caps)
    split, stats = split_primal(h, target, lower, upper, caps)
    assert_allclose(split, full, atol=1e-13, rtol=1e-13)
    assert stats["free_segments"] == (1 if witness in (0, 8) else 2)
    result = solve_quadratic(h, target, lower, upper, caps, lower)
    assert result.qualified and result.x[witness] == 1
    assert_allclose(result.x, full, atol=1e-13)


@pytest.mark.parametrize("seed", range(8))
def test_shared_surrogate_witness_values_match_exhaustive_branches(seed):
    rng = np.random.default_rng(seed)
    n = 8
    h, target, caps = np.exp(rng.uniform(-1, 3, n)), rng.normal(.5, 1, n), rng.uniform(0, 10, n - 1)
    lower, upper = np.full(n, 1e-6), np.where(rng.random(n) < .7, 1., .7)
    upper[-1] = 1
    profile = profile_quadratic_witnesses(h, target, lower, upper, caps)
    reference = np.clip(target, lower, upper)
    branch_values = np.full(n, np.inf)
    for k in np.flatnonzero(upper == 1):
        branch_lower = lower.copy()
        branch_lower[k] = 1
        fit = solve_quadratic(h, target, branch_lower, upper, caps, lower)
        assert fit.qualified
        delta = fit.x - reference
        branch_values[k] = np.sum(.5 * h * delta**2 + h * (reference - target) * delta) + np.dot(caps, abs(np.diff(fit.x)))
    assert profile.qualified and profile.diagnostics["reconstruction_count"] == 1
    assert profile.diagnostics["scope"] == "one_common_quadratic"
    assert_allclose(profile.relative_objectives, branch_values, atol=1e-9, rtol=1e-11)
    assert branch_values[profile.witness] <= np.min(branch_values) + 1e-9
    assert profile.fit.x[profile.witness] == 1
    changed = profile_quadratic_witnesses(h, target + .01, lower, upper, caps)
    assert changed.surrogate_sha256 != profile.surrogate_sha256


def test_direct_solver_does_not_depend_on_iterative_budget_or_warm_start():
    h, target, lower, upper, caps = np.ones(4), np.array([0., .2, .8, 1.]), np.zeros(4), np.ones(4), np.ones(3)
    policy = replace(Policy(), inner_max_iterations=1)
    direct = solve_quadratic(h, target, lower, upper, caps, lower, policy, dual=caps)
    reference = solve_quadratic_iterative(h, target, lower, upper, caps, lower, policy)
    assert direct.qualified and not reference.qualified
    assert_allclose(direct.x, .5, atol=1e-14)


def test_direct_solver_fails_closed_on_certificate_error(monkeypatch):
    import clipp1d.solver as solver

    def invalid(*args):
        return np.zeros(2), {}

    monkeypatch.setattr(solver, "split_primal", invalid)
    result = solver.solve_quadratic(np.ones(2), np.array([0., 1.]), np.zeros(2), np.ones(2),
                                    np.array([1.]), np.zeros(2))
    # Polishing may repair this fused primal; force a bad certificate independently.
    monkeypatch.setattr(solver, "reconstruct_dual", lambda *args: np.array([0.]))
    result = solver.solve_quadratic(np.ones(2), np.array([0., 1.]), np.zeros(2), np.ones(2),
                                    np.array([1.]), np.zeros(2))
    assert not result.qualified


@pytest.mark.parametrize("n", [100, 1000, 4000])
def test_message_storage_and_knot_work_are_linear(n):
    rng = np.random.default_rng(51)
    h, target = np.ones(n), rng.normal(.5, 2, n)
    lower, upper = np.zeros(n), np.ones(n)
    caps = rng.uniform(.1, 2, n - 1)
    low, high, _, _, stats = forward_messages(h, target, lower, upper, caps)
    assert low.shape == high.shape == (n - 1,)
    assert stats["knots_inserted"] <= 2 * (n - 1)
    assert stats["knots_removed"] <= stats["knots_inserted"]
    assert stats["peak_live_knots"] <= 2 * (n - 1)


def test_arbitrary_frozen_bounds_zero_edges_and_singleton():
    h, target = np.ones(5), np.array([9., 0., .5, 1., -9.])
    lower, upper, caps = np.array([.3, 0., .2, 0., .7]), np.array([.3, 1., .2, 1., .7]), np.array([0., 2., 3., 0.])
    fit = solve_quadratic(h, target, lower, upper, caps, lower)
    assert fit.qualified
    assert_allclose(fit.x, oracle(h, target, lower, upper, caps), atol=1e-8)
    one = solve_quadratic(np.ones(1), np.array([5.]), np.zeros(1), np.ones(1), np.array([]), np.zeros(1))
    assert one.qualified and one.x[0] == 1


def test_large_edge_caps_do_not_inflate_reconstruction_normals():
    n = 100
    h = np.linspace(50, 500, n)
    target = np.linspace(.1, .9, n)
    lower, upper, caps = np.zeros(n), np.ones(n), np.full(n - 1, 1e10)
    lower[40] = 1
    result = solve_quadratic(h, target, lower, upper, caps, lower)
    assert result.qualified and result.kkt_residual < 1e-10
    assert_allclose(result.x, 1, atol=0, rtol=0)
