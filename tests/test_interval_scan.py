"""Bitwise comparison to the retained scalar scan, including floating ties."""

import numpy as np
import pytest

from clipp1d.intervals import _scalar_interval_descent, interval_descent


def assert_same(actual, expected):
    assert actual == expected
    if actual is not None:
        assert actual[3].hex() == expected[3].hex()
        assert actual[4].hex() == expected[4].hex()


@pytest.mark.parametrize("adversarial", [False, True])
def test_thousands_of_runs_preserve_exact_scalar_order(adversarial):
    rng = np.random.default_rng(62441 if adversarial else 234)
    special = np.array([0., -0., 1., -1., 2.**53, -2.**53, 2.**-53, -2.**-53])
    for case in range(1500):
        n = 64 + case % 129
        x = np.repeat(rng.choice([.1, .5, .9], (n + 3) // 4), 4)[:n]
        lower = np.where(rng.random(n) < .1, x, 0.)
        upper = np.where(rng.random(n) < .1, x, 1.)
        if adversarial:
            left, right = rng.choice(special, (2, n))
            caps = abs(rng.choice(special, n - 1))
            tolerance = [0., 2.**-53, 2e-5, .5][case % 4]
        else:
            left, right = rng.normal(0, 3, (2, n))
            caps = rng.uniform(0, 5, n - 1)
            tolerance = [0., 2e-5, .1][case % 3]
        args = (x, left, right, lower, upper, caps, tolerance)
        assert_same(interval_descent(*args), _scalar_interval_descent(*args))


def test_equal_merits_keep_negative_sign_and_earliest_stop():
    n = 128
    x, lower, upper = np.linspace(.1, .9, n), np.zeros(n), np.ones(n)
    assert interval_descent(x, np.ones(n), -np.ones(n), lower, upper, np.zeros(n - 1)) == (0, 1, -1, -1., 2.)


def test_equal_prefix_keys_keep_earliest_start():
    n = 128
    x, lower, upper = np.full(n, .5), np.zeros(n), np.ones(n)
    left, right = np.zeros(n), np.zeros(n)
    left[2] = 2
    assert interval_descent(x, left, right, lower, upper, np.zeros(n - 1)) == (0, 3, -1, -2., 3.)


def test_tied_merits_across_buckets_keep_original_scan_order():
    n = 128
    x, lower, upper = np.linspace(.1, .9, n), np.zeros(n), np.ones(n)
    x[:4] = .1
    left, right = np.zeros(n), np.zeros(n)
    left[3:5] = 2
    # The one-node bucket is processed first, but the four-node run has the
    # earlier tied stopping coordinate in the original increasing-index scan.
    assert interval_descent(x, left, right, lower, upper, np.zeros(n - 1)) == (0, 4, -1, -2., 3.)


def test_infeasible_coordinates_reset_both_prefixes():
    n = 128
    x, lower, upper = np.full(n, .5), np.zeros(n), np.ones(n)
    lower[63] = upper[63] = .5
    left, right = np.zeros(n), np.zeros(n)
    left[62:65] = [1e16, 0, 1.]
    args = (x, left, right, lower, upper, np.zeros(n - 1), 2e-5)
    assert_same(interval_descent(*args), _scalar_interval_descent(*args))


@pytest.mark.parametrize("n,dtype", [(0, float), (8, float), (128, np.float32)])
def test_small_or_other_precision_inputs_use_original_scalar_semantics(n, dtype):
    x, lower, upper = np.full(n, .5, dtype=dtype), np.zeros(n, dtype=dtype), np.ones(n, dtype=dtype)
    args = (x, np.ones(n, dtype=dtype), np.zeros(n, dtype=dtype), lower, upper,
            np.zeros(max(0, n - 1), dtype=dtype), 2e-5)
    assert_same(interval_descent(*args), _scalar_interval_descent(*args))
