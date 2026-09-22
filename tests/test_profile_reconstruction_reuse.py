"""Selected-witness reconstruction reuses both common-surrogate message passes."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import clipp1d.solver as solver
import clipp1d.tv as tv


def recomputed_branch(h, target, lower, upper, caps, witness):
    fixed_lower, fixed_upper = lower.copy(), upper.copy()
    fixed_lower[witness] = fixed_upper[witness] = 1
    return solver.solve_quadratic(h, target, fixed_lower, fixed_upper, caps, lower)


@pytest.mark.parametrize("seed", range(16))
def test_selected_profile_matches_independent_message_reconstruction(seed):
    rng = np.random.default_rng(1850 + seed)
    n = 3 + seed
    h, target = np.exp(rng.uniform(-2, 6, n)), rng.normal(.5, 2, n)
    lower, upper = np.zeros(n), np.where(rng.random(n) < .6, 1., .65)
    upper[-1] = 1
    caps = np.exp(rng.uniform(-3, 5, n - 1))
    profile = solver.profile_quadratic_witnesses(h, target, lower, upper, caps)
    expected = recomputed_branch(h, target, lower, upper, caps, profile.witness)
    assert profile.qualified and expected.qualified
    assert_allclose(profile.fit.x, expected.x, atol=5e-14, rtol=3e-13)
    assert_allclose(profile.fit.dual, expected.dual, atol=5e-12, rtol=3e-12)
    assert profile.fit.work["reconstruction_path"] == "stored_profile_thresholds"
    assert profile.fit.work["message_passes_rebuilt"] == 0
    assert profile.fit.work["threshold_reconstruction_steps"] == n - 1


def test_profile_builds_exactly_two_passes_and_finalizes_once(monkeypatch):
    h = np.array([2., 3., 5., 7., 11.])
    target = np.array([.8, .3, .65, .2, .95])
    lower, upper, caps = np.zeros(5), np.ones(5), np.array([.2, .5, .1, .4])
    original = [v.copy() for v in (h, target, lower, upper, caps)]
    counts = {"messages": 0, "finalizations": 0}
    message, finalize = solver.forward_messages, solver.finalize_quadratic

    def counted_message(*args, **kwargs):
        counts["messages"] += 1
        return message(*args, **kwargs)

    def counted_finalize(curvature, centers, lo, hi, edge_caps, x, *args, **kwargs):
        counts["finalizations"] += 1
        assert_array_equal(curvature, h)
        assert_array_equal(centers, target)
        assert_array_equal(edge_caps, caps)
        assert np.count_nonzero(lo == hi) == 1
        assert np.all(x[lo == hi] == 1)
        return finalize(curvature, centers, lo, hi, edge_caps, x, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Selected profile must not build another primal message solve")

    monkeypatch.setattr(solver, "forward_messages", counted_message)
    monkeypatch.setattr(solver, "finalize_quadratic", counted_finalize)
    monkeypatch.setattr(solver, "solve_quadratic", forbidden)
    monkeypatch.setattr(solver, "split_primal", forbidden)
    monkeypatch.setattr(tv, "direct_primal", forbidden)
    profile = solver.profile_quadratic_witnesses(h, target, lower, upper, caps)
    assert profile.qualified
    assert counts == {"messages": 2, "finalizations": 1}
    for actual, expected in zip((h, target, lower, upper, caps), original):
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("witness", [0, 3, 6])
def test_outward_reconstruction_all_positions_and_frozen_other_nodes(witness):
    rng = np.random.default_rng(147)
    n = 7
    h, target = rng.uniform(.2, 12, n), rng.normal(.5, 1, n)
    lower, upper = np.zeros(n), np.full(n, .8)
    upper[witness] = 1  # Only this witness is eligible.
    frozen = 2 if witness != 2 else 4
    lower[frozen] = upper[frozen] = .3
    caps = np.array([0., 2., .5, 0., .3, 4.])
    profile = solver.profile_quadratic_witnesses(h, target, lower, upper, caps)
    expected = recomputed_branch(h, target, lower, upper, caps, witness)
    assert profile.witness == witness
    assert profile.qualified and expected.qualified
    assert profile.fit.x[frozen] == .3 and profile.fit.x[witness] == 1
    assert_allclose(profile.fit.x, expected.x, atol=2e-14, rtol=3e-13)
    assert np.all(np.isinf(profile.relative_objectives[upper < 1]))


@pytest.mark.parametrize("n", [1, 7])
def test_zero_cap_singletons_and_equal_profile_ties(n):
    h, target = np.ones(n), np.full(n, .5)
    lower, upper, caps = np.zeros(n), np.ones(n), np.zeros(n - 1)
    profile = solver.profile_quadratic_witnesses(h, target, lower, upper, caps)
    assert profile.qualified
    assert profile.witness == 0  # Preserve np.argmin's existing first-equal rule.
    expected = np.full(n, .5)
    expected[0] = 1
    assert_array_equal(profile.fit.x, expected)
    assert_array_equal(profile.relative_objectives, np.full(n, .125))
    assert profile.fit.gap == 0 and profile.fit.kkt_residual == 0


@pytest.mark.parametrize("witness", [-1, 3, 1.5, True])
def test_reconstruction_rejects_invalid_witness(witness):
    with pytest.raises(ValueError, match="thresholds"):
        tv.reconstruct_at_witness(np.zeros(2), np.ones(2), np.zeros(2), np.ones(2), witness)


def test_bad_reconstructed_primal_still_fails_original_admission_gates(monkeypatch):
    original = solver.reconstruct_at_witness

    def corrupted(*args, **kwargs):
        x = original(*args, **kwargs)
        x[0] = np.nan
        return x

    monkeypatch.setattr(solver, "reconstruct_at_witness", corrupted)
    fit = solver.profile_quadratic_witnesses(np.ones(3), np.full(3, .5), np.zeros(3),
                                            np.ones(3), np.zeros(2))
    assert not fit.qualified
    assert not fit.fit.qualified
    assert fit.fit.work["failure_reason"] == "quadratic_gap_gate"
