"""Wide-range regressions, with an independent exhaustive small-QP oracle."""

from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d.policy import Policy
from clipp1d.solver import profile_quadratic_witnesses, quadratic_gap, quadratic_kkt, solve_quadratic
from clipp1d.tv import reconstruct_dual


def exhaustive_qp(h, target, lower, upper, caps):
    """Enumerate fused/+/- edges; optimize each feasible block analytically.

    This exponential oracle is intentionally limited to tiny test problems. It
    uses no TV messages, production block polishing, or reconstructed duals.
    The true active set is among the enumerated candidates.
    """
    h, target = h.astype(np.longdouble), target.astype(np.longdouble)
    best, best_value = None, np.longdouble(np.inf)
    reference = np.clip(target, lower, upper)
    for signs in product((-1, 0, 1), repeat=len(h) - 1):
        cuts = [0, *(i + 1 for i, sign in enumerate(signs) if sign), len(h)]
        x = np.empty(len(h), dtype=np.longdouble)
        feasible = True
        for a, b in zip(cuts[:-1], cuts[1:]):
            lo, hi = max(lower[a:b]), min(upper[a:b])
            if lo > hi:
                feasible = False
                break
            left = np.longdouble(caps[a - 1]) * signs[a - 1] if a else 0
            right = np.longdouble(caps[b - 1]) * signs[b - 1] if b < len(h) else 0
            x[a:b] = np.clip((np.sum(h[a:b] * target[a:b]) - left + right) /
                            np.sum(h[a:b]), lo, hi)
        if not feasible or any(s * d < 0 for s, d in zip(signs, np.diff(x)) if s):
            continue
        delta = x - reference
        value = np.sum(.5 * h * delta**2 + h * (reference - target) * delta)
        value += np.dot(caps, abs(np.diff(x)))
        if value < best_value:
            best, best_value = x, value
    assert best is not None
    return best.astype(float), best_value


def test_block_rounding_is_not_concentrated_at_small_curvature_tail():
    # df44e6a raised "No feasible chain dual interval" for this four-node QP.
    # The high-curvature fused block's represented mean leaves a rounding
    # residual; assigning it all to the low-curvature tail was the defect.
    h = np.array([57778.8938759665, 11597066.50792733, 244.67328481134516, 13549.907796008734])
    target = np.array([-.64910479455994, .10259558994674367, 1.0279704138524777, -1.6792611076346486])
    caps = np.array([.04644733301109519, 177135.88431706437, 302.55959195483996])
    lower, upper = np.zeros(4), np.ones(4)
    expected, _ = exhaustive_qp(h, target, lower, upper, caps)
    fit = solve_quadratic(h, target, lower, upper, caps, lower)
    assert fit.qualified
    assert_allclose(fit.x, expected, atol=3e-15, rtol=1e-14)
    assert fit.kkt_residual <= Policy().inner_kkt_tol


def test_guard_digit_dual_does_not_certify_an_incorrect_primal_block():
    h, target = np.ones(3), np.array([0., .5, 1.])
    lower, upper, caps = np.zeros(3), np.ones(3), np.ones(2)
    x = np.full(3, .6)  # The optimum is exactly .5.
    dual = reconstruct_dual(x, h, target, lower, upper, caps)
    gap, scale = quadratic_gap(x, dual, h, target, lower, upper, caps)
    assert gap > Policy().inner_atol + Policy().inner_rtol * scale
    assert quadratic_kkt(x, dual, h, target, lower, upper, caps) > Policy().inner_kkt_tol


@pytest.mark.parametrize("seed", range(20))
def test_wide_range_qps_against_exhaustive_independent_oracle(seed):
    rng = np.random.default_rng(191 + seed)
    n = 5
    h, target = 10**rng.uniform(-2, 9, n), rng.normal(.5, 2, n)
    lower, upper = np.zeros(n), rng.uniform(.4, 1, n)
    caps = 10**rng.uniform(-2, 8, n - 1)
    if seed % 2:
        lower[2] = upper[2] = 1
    expected, _ = exhaustive_qp(h, target, lower, upper, caps)
    fit = solve_quadratic(h, target, lower, upper, caps, lower)
    assert fit.qualified, fit.work
    assert_allclose(fit.x, expected, atol=3e-12, rtol=1e-12)
    assert fit.gap <= Policy().inner_atol + Policy().inner_rtol * fit.gap_scale
    assert fit.kkt_residual <= Policy().inner_kkt_tol


@pytest.mark.parametrize("seed", range(30))
def test_wide_range_reachable_intervals_keep_block_roundoff(seed):
    rng = np.random.default_rng(seed + 7351)
    n = 100
    h, target = 10**rng.uniform(-2, 9, n), rng.normal(.5, 2, n)
    lower, upper = np.zeros(n), np.ones(n)
    caps = 10**rng.uniform(-2, 8, n - 1)
    if seed % 2:
        lower[n // 2] = upper[n // 2] = 1
    fit = solve_quadratic(h, target, lower, upper, caps, lower)
    assert fit.qualified, fit.work
    assert fit.gap <= Policy().inner_atol + Policy().inner_rtol * fit.gap_scale
    assert fit.kkt_residual <= Policy().inner_kkt_tol


def test_profile_value_accumulation_does_not_erase_small_remaining_cost():
    # Every seed is checked; cases 8, 51, 64 and others failed the old selected
    # prefix-value gate after subtracting large clipped message areas.
    rng = np.random.default_rng(882)
    for _ in range(80):
        n = 30
        h, target = 10**rng.uniform(-2, 9, n), rng.normal(.5, 2, n)
        lower, upper = np.zeros(n), np.where(rng.random(n) < .7, 1., .7)
        upper[-1] = 1
        caps = 10**rng.uniform(-2, 8, n - 1)
        fit = profile_quadratic_witnesses(h, target, lower, upper, caps)
        assert fit.qualified, fit.diagnostics


def test_profile_removes_new_unary_offset_algebraically_before_accumulation():
    # Guard digits alone are insufficient: repeatedly adding/subtracting 1.25e8
    # produced ~7e-9 error in the .005 optimum, outside the unchanged gap gate.
    n = 1000
    h, target = np.full(n, 1e9), np.full(n, .5)
    h[-1], target[-1] = .01, 1
    caps, lower, upper = np.full(n - 1, .01), np.zeros(n), np.ones(n)
    profile = profile_quadratic_witnesses(h, target, lower, upper, caps)
    expected = .005 - .5 * .01**2 / ((n - 1) * 1e9)
    assert profile.qualified
    assert profile.witness == n - 1
    assert_allclose(profile.relative_objectives[-1], expected, atol=1e-16, rtol=0)


@pytest.mark.parametrize("seed", range(8))
def test_wide_range_profile_compares_every_witness_to_independent_oracle(seed):
    rng = np.random.default_rng(518 + seed)
    n = 5
    h, target = 10**rng.uniform(-2, 9, n), rng.normal(.5, 2, n)
    lower, upper = np.zeros(n), np.ones(n)
    caps = 10**rng.uniform(-2, 8, n - 1)
    profile = profile_quadratic_witnesses(h, target, lower, upper, caps)
    assert profile.qualified
    expected = []
    # The oracle subtracts each branch's boxed minima; add back the difference
    # from the original-box reference used by the common profile.
    reference = np.clip(target.astype(np.longdouble), lower, upper)
    for witness in range(n):
        fixed = lower.copy()
        fixed[witness] = 1
        _, value = exhaustive_qp(h, target, fixed, upper, caps)
        delta = 1 - reference[witness]
        value += .5 * h[witness] * delta**2 + h[witness] * (reference[witness] - target[witness]) * delta
        expected.append(value)
    assert_allclose(profile.relative_objectives, expected, atol=2e-7, rtol=3e-12)
    selected = expected[profile.witness]
    assert selected <= min(expected) + 1e-9 * (1 + abs(selected))
