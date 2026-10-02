import ctypes
import itertools
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom

from clipp.model import MultiplicityModel
from clipp.refitting import refit_center

D = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
INT_VECTOR = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def evaluate(native, r, n, major, total, purity, x):
    arrays = [np.ascontiguousarray(a, dtype=np.int32) for a in (r, n, major, total)]
    x = np.ascontiguousarray(x, dtype=float)
    out = np.empty((len(x), 3))
    entry = native.CliPPEvaluateCPU
    entry.argtypes = [ctypes.c_int, INT_VECTOR, INT_VECTOR, INT_VECTOR, INT_VECTOR, ctypes.c_double, D, D]
    entry.restype = ctypes.c_int
    assert entry(len(x), *arrays, purity, x, out) == 0
    return out


@pytest.mark.parametrize("purity", [0.2, 0.8, 1.0])
def test_full_support_and_endpoints(native, purity):
    r = np.array([0, 100, 25, 1000, 7])
    n = np.array([100, 100, 100, 2000, 200])
    major = np.array([1, 1, 4, 8, 2])
    total = np.array([2, 1, 5, 8, 3])
    x = np.array([0.0, 1.0, 0.7, 0.9, 1.0])
    model = MultiplicityModel(r, n, major, total, purity)
    native_values = evaluate(native, r, n, major, total, purity, x)
    independent = []
    for ri, ni, mi, ti, xi in zip(r, n, major, total, x):
        p = purity * xi * np.arange(1, mi + 1) / (2 * (1 - purity) + purity * ti)
        independent.append(logsumexp(binom.logpmf(ri, ni, p)) - np.log(mi))
    np.testing.assert_allclose(model.log_likelihood(x * purity), independent, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(native_values[:, 0], -np.array(independent), rtol=1e-10, atol=1e-9)
    post = model.posterior(x * purity)
    np.testing.assert_allclose(post.sum(axis=1), 1)
    assert (post[~model.valid] == 0).all()


def test_gradient_and_surrogate_curvature(native):
    r = np.array([0, 10, 80, 400])
    n = np.array([100, 100, 100, 1000])
    major = np.array([1, 2, 4, 8])
    total = np.array([2, 3, 4, 10])
    purity = 0.8
    x = np.array([0.2, 0.43, 0.87, 0.74])
    step = 1e-6
    values = evaluate(native, r, n, major, total, purity, x)
    finite = (
        evaluate(native, r, n, major, total, purity, x + step)[:, 0]
        - evaluate(native, r, n, major, total, purity, x - step)[:, 0]
    ) / (2 * step)
    np.testing.assert_allclose(values[:, 1], finite, rtol=3e-7, atol=3e-6)
    model = MultiplicityModel(r, n, major, total, purity)
    a = model.scale * purity
    posterior = model.posterior(x * purity)
    # Complete-data curvature, explicitly not the marginalized Hessian.
    expected = np.sum(
        posterior * (r[:, None] / x[:, None] ** 2 + (n - r)[:, None] * a * a / (1 - x[:, None] * a) ** 2),
        axis=1,
    )
    np.testing.assert_allclose(values[:, 2], np.maximum(expected, 1e-8), rtol=1e-11)
    for r0, expected_center in [(0, 0.0), (100, 0.8)]:
        center, _ = refit_center(MultiplicityModel([r0], [100], [1], [1], 0.8))
        assert center == expected_center


@pytest.mark.parametrize("n", range(1, 8))
def test_projection_exhaustive(native, n):
    native.CliPPProject.argtypes = [ctypes.c_int, D, ctypes.c_int, D]
    rng = np.random.default_rng(n)
    for x in [np.zeros(n), np.arange(n, dtype=float), rng.normal(size=n)]:
        for k in range(1, n + 1):
            z = np.zeros(max(1, n - 1))
            assert native.CliPPProject(n, x, k, z) == 0
            actual = z[: n - 1]
            dif = np.diff(x)
            losses = []
            for keep in itertools.combinations(range(n - 1), k - 1):
                proposal = np.zeros(n - 1)
                proposal[list(keep)] = dif[list(keep)]
                losses.append(np.sum((dif - proposal) ** 2))
            np.testing.assert_allclose(np.sum((dif - actual) ** 2), min(losses), atol=1e-12)
            expected = np.zeros(n - 1)
            edges = np.argsort(-np.abs(dif), kind="stable")[: k - 1]
            expected[edges] = dif[edges]
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("rho", [0.1, 10.0, 10000.0])
def test_box_qp_dense_reference(native, rho):
    native.CliPPBoxQP.argtypes = [ctypes.c_int, D, D, D, ctypes.c_double, ctypes.c_double, ctypes.c_double, D]
    rng = np.random.default_rng(72)
    for n in (1, 2, 6):
        curvature = rng.uniform(0.5, 4, n)
        rhs = rng.normal(0, 4, n)
        z = np.zeros(max(n - 1, 1))
        if n > 3:
            z[2] = 1
        d = np.diff(np.eye(n), axis=0)
        if n > 1:
            d[z[: n - 1] != 0] = 0
        matrix = np.diag(curvature) + rho * d.T @ d
        out = np.empty(n)
        assert native.CliPPBoxQP(n, curvature, rhs, z, rho, 0.0, 1.0, out) == 0
        fit = minimize(
            lambda x: (0.5 * x @ matrix @ x - rhs @ x, matrix @ x - rhs),
            np.full(n, 0.5),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, 1)] * n,
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 3000},
        )
        np.testing.assert_allclose(out, fit.x, atol=3e-6)
        gradient = matrix @ out - rhs
        assert np.max(np.abs(out - np.clip(out - gradient, 0, 1))) < 1e-8


def test_high_rho_long_shallow_ramp(native):
    native.CliPPBoxQP.argtypes = [ctypes.c_int, D, D, D, ctypes.c_double, ctypes.c_double, ctypes.c_double, D]
    n = 1000
    curvature = np.ones(n)
    rhs = np.linspace(0.499, 0.501, n)
    z = np.zeros(n - 1)
    out = np.empty(n)
    assert native.CliPPBoxQP(n, curvature, rhs, z, 1e12, 0.0, 1.0, out) == 0
    assert abs(out.mean() - 0.5) < 1e-13
    assert np.ptp(out) < 1e-9
    # Evaluate the KKT equation with extended precision, avoiding dense rho cancellation.
    g = out.astype(np.longdouble) - rhs
    dif = np.diff(out.astype(np.longdouble)) * np.longdouble(1e12)
    g[:-1] -= dif
    g[1:] += dif
    assert np.max(np.abs(g)) < 0.0003
