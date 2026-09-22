"""Independent CPU numerical oracles. These are not CUDA qualification results."""

import itertools
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from clipp1d.cuda.kernels import Kernels, boxed_rank_one, differences, adjoint, likelihood
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.graph import build_graph
from clipp1d.cuda.scalar import Problems, pilot
from clipp1d.cuda.qp import solve_qp
from clipp1d.cuda.audit import directional_cut
from clipp1d.cuda.partition import grouping
from clipp1d.cuda.selection import fit_tensor_model


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(alt=(20.0, 21.0, 48.0), slope=None, upper=None):
    a = tensor(alt)
    n = len(alt)
    s = torch.full((n, 1), 0.5, dtype=torch.float64) if slope is None else tensor(slope)
    prior = torch.where(s > 0, -(s > 0).sum(-1, keepdim=True).double().log(), -float("inf"))
    return TensorModel(
        tuple(map(str, range(n))),
        a,
        100 - a,
        s,
        prior,
        torch.full_like(a, 1e-6),
        torch.ones_like(a) if upper is None else tensor(upper),
        1e-6,
        Kernels("cpu"),
    )


@pytest.mark.parametrize("n", [1, 2, 7, 15])
def test_complete_operators(n):
    rng = np.random.default_rng(n)
    x = tensor(rng.normal(size=n))
    raw = tensor(rng.normal(size=(n, n)))
    q = raw - raw.T
    d = differences(x)
    assert torch.allclose(0.5 * (d * q).sum(), (x * adjoint(q)).sum(), atol=1e-13, rtol=1e-13)
    assert torch.allclose(adjoint(d), n * x - x.sum(), atol=1e-13, rtol=1e-13)
    graph = build_graph(x)
    assert graph.edges == n * (n - 1) // 2
    assert bool((graph.weights == graph.weights.T).all())
    if n > 1:
        assert abs(float(graph.weights.sum()) / (n * (n - 1)) - 1.0) < 1e-14
    assert bool((graph.weights.diagonal() == 0).all())


@pytest.mark.parametrize("seed", range(8))
def test_rank_one_box_update(seed):
    rng = np.random.default_rng(seed)
    n = 6
    h = np.exp(rng.normal(size=n))
    b = rng.normal(size=n)
    lo = rng.uniform(-1, -0.1, size=n)
    hi = rng.uniform(0.1, 1, size=n)
    if seed % 2:
        lo[1] = hi[1] = 0.3
    rho = 10.0 ** (seed % 5 - 2)
    mat = np.diag(h + rho * n) - rho * np.ones((n, n))
    x = boxed_rank_one(tensor(h), tensor(b), tensor(lo), tensor(hi), tensor(rho)).numpy()
    res = minimize(
        lambda z: 0.5 * z @ mat @ z - b @ z,
        (lo + hi) / 2,
        jac=lambda z: mat @ z - b,
        bounds=list(zip(lo, hi)),
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 5000},
    )
    assert 0.5 * x @ mat @ x - b @ x <= res.fun + 1e-10
    g = mat @ x - b
    pg = x - np.clip(x - g, lo, hi)
    assert np.max(abs(pg)) < 3e-12


def test_likelihood_numpy_and_one_sided():
    m = model((0.0, 20.0, 45.0), [[0.25, 0.5, 0.0], [0.2, 0.4, 0.6], [0.4, 0.0, 0.0]])
    x = tensor([1e-6 / 0.25, 0.3, 0.8])
    loss, grad, h, post, left, right = m.terms(x)
    p = np.clip(m.slope.numpy() * x.numpy()[:, None], 1e-6, 1 - 1e-6)
    joint = (
        m.alt.numpy()[:, None] * np.log(p)
        + m.ref.numpy()[:, None] * np.log1p(-p)
        + m.log_prior.numpy()
    )
    expect = -logsumexp(joint, axis=1)
    np.testing.assert_allclose(loss.numpy(), expect, rtol=1e-14, atol=1e-13)
    assert np.isfinite(post.numpy()).all()
    assert left[0] != right[0]
    for i in (1, 2):
        d = torch.zeros_like(x)
        d[i] = 1e-6
        finite = (m.terms(x + d)[0][i] - m.terms(x - d)[0][i]) / 2e-6
        assert abs(float(finite - grad[i])) < 1e-6


def test_pilot_mixtures_and_bounds():
    m = model(
        (0.0, 20.0, 45.0, 30.0),
        [[0.3, 0.6, 0.0], [0.2, 0.4, 0.6], [0.4, 0.0, 0.0], [0.15, 0.3, 0.45]],
    )
    p = pilot(m)
    assert bool(p.qualified.all())
    grid = torch.linspace(1e-6, 1.0, 5001, dtype=torch.float64)[None, :].expand(4, -1)
    losses = likelihood(m.alt, m.ref, m.slope, m.log_prior, grid, m.eps)[0]
    assert bool((p.loss <= losses.min(-1).values + 1e-7).all())
    probs = Problems(m, torch.ones(4, dtype=torch.long))
    for lo, hi in [(0.01, 0.3), (0.3, 0.6), (0.6, 0.999)]:
        lb = probs.bounds(
            torch.full((4, 1), lo, dtype=torch.float64), torch.full((4, 1), hi, dtype=torch.float64)
        )[0][:, 0]
        v = torch.linspace(lo, hi, 1001, dtype=torch.float64)[None, :].expand(4, -1)
        assert bool((lb <= probs.evaluate(v)[0].min(-1).values + 1e-10).all())


def test_complete_qp_against_explicit_epigraph():
    rng = np.random.default_rng(55)
    n = 5
    h = np.exp(rng.normal(size=n))
    target = rng.uniform(-0.2, 1.3, n)
    caps = rng.uniform(0.01, 0.2, (n, n))
    caps = (caps + caps.T) / 2
    np.fill_diagonal(caps, 0.0)
    lo = np.zeros(n)
    hi = np.ones(n)
    lo[2] = hi[2] = 1.0
    a, b = np.triu_indices(n, 1)
    e = len(a)
    D = np.eye(n)[b] - np.eye(n)[a]
    A = np.vstack((np.c_[D, -np.eye(e)], np.c_[-D, -np.eye(e)]))
    def fun(y):
        return 0.5 * np.sum(h * (y[:n] - target) ** 2) + caps[a, b] @ y[n:]
    def jac(y):
        return np.r_[h * (y[:n] - target), caps[a, b]]
    init = np.r_[np.clip(target, lo, hi), np.abs(D @ np.clip(target, lo, hi))]
    oracle = minimize(
        fun,
        init,
        jac=jac,
        bounds=list(zip(lo, hi)) + [(0, None)] * e,
        constraints=[{"type": "ineq", "fun": lambda y: -A @ y, "jac": lambda y: -A}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    fit = solve_qp(tensor(h), tensor(target), tensor(lo), tensor(hi), tensor(caps), Kernels("cpu"))
    assert fit.qualified
    assert abs(fun(np.r_[fit.x.numpy(), np.abs(D @ fit.x.numpy())]) - oracle.fun) < 2e-9
    np.testing.assert_allclose(fit.x.numpy(), oracle.x[:n], atol=2e-5, rtol=0.0)


@pytest.mark.parametrize("seed", range(4))
def test_directional_cut_against_all_subsets(seed):
    rng = np.random.default_rng(seed)
    n = 5
    a = tensor(rng.normal(size=n))
    caps = tensor(rng.uniform(0, 0.4, (n, n)))
    caps = (caps + caps.T) / 2
    caps.fill_diagonal_(0.0)
    allowed = torch.ones(n, dtype=torch.float64)
    allowed[seed] = 0.0
    exact = min(
        float(a @ tensor(s) + 0.5 * (caps * differences(tensor(s)).abs()).sum())
        for s in itertools.product([0.0, 1.0], repeat=n)
        if s[seed] == 0.0
    )
    ok, d = directional_cut(a, caps, allowed, torch.zeros_like(caps))
    if exact < -2e-5:
        assert not ok and d is not None
        assert float(a @ d + 0.5 * (caps * differences(d).abs()).sum()) < -2e-5
    else:
        assert ok and d is None


def test_memberships_noncontiguous_and_no_chaining():
    x = tensor([0.4, 0.8, 0.400001, 1.0, 0.800001])
    _, _, labels, _ = grouping(x, 2e-5)
    assert labels[0] == labels[2] and labels[1] == labels[4]
    y = tensor([0.1, 0.100019, 0.100038])
    _, _, labels, _ = grouping(y, 2e-5)
    assert len(torch.unique(labels)) == 3
    _, _, labels, _ = grouping(tensor([1.0, 1.0 - 1e-8]), 2e-5)
    assert labels[0] != labels[1]


def test_frozen_graph_mutation_rejected():
    graph = build_graph(tensor([0.2, 0.8]))
    graph.weights[0, 1] = 2.0
    with pytest.raises(ValueError, match="modified"):
        graph.validate()


def test_short_end_to_end_reference():
    m = model()
    result = fit_tensor_model(m, lambda_values=[0.0, 0.1, 0.5])
    assert result.raw.qualified
    assert result.raw.witness is None
    assert bool((result.raw.x < 1).all())
    assert result.graph.edges == 3
    assert result.refit.clonal == int((result.refit.centers - 1).abs().argmin())
    assert result.search_status in ("complete", "incomplete")
    assert len(result.records) == 3
    assert result.raw.x.device.type == "cpu"


def test_canonical_model_mutation_rejected():
    m = model()
    m.alt[0] = 13.0
    with pytest.raises(ValueError, match="modified"):
        m.terms(tensor([0.2, 0.4, 1.0]))


def test_compiler_fullgraph_trace_matches_eager_cpu():
    # Dynamo tracing is not CUDA/Inductor qualification.
    from clipp1d.cuda.kernels import edge_update, gap_kkt

    n = 4
    h = tensor([1.0, 2.0, 3.0, 4.0])
    b = tensor([0.2, 0.7, 0.5, 0.9])
    lo = torch.zeros(n, dtype=torch.float64)
    hi = torch.ones_like(lo)
    lo[0] = hi[0] = 1.0
    rho = tensor(0.3)
    traced = torch.compile(boxed_rank_one, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        traced(h, b, lo, hi, rho), boxed_rank_one(h, b, lo, hi, rho), rtol=0.0, atol=0.0
    )
    m = model()
    x = tensor([0.2, 0.4, 0.8])
    caps = torch.ones((3, 3), dtype=torch.float64) * 0.1
    caps.fill_diagonal_(0.0)
    for fn, args in [
        (likelihood, (m.alt, m.ref, m.slope, m.log_prior, x[:, None], m.eps)),
        (edge_update, (x, torch.zeros_like(caps), caps, rho)),
        (gap_kkt, (x, torch.zeros_like(caps), torch.ones_like(x), x, m.lower, m.upper, caps)),
    ]:
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        actual, expected = compiled(*args), fn(*args)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_frozen_target_constant_does_not_inflate_gap_scale():
    from clipp1d.cuda.kernels import gap_kkt

    h = tensor([1.0, 2.0])
    x = tensor([1.0, 0.4])
    lo = tensor([1.0, 0.0])
    hi = tensor([1.0, 1.0])
    caps = torch.zeros((2, 2), dtype=torch.float64)
    q = torch.zeros_like(caps)
    a = gap_kkt(x, q, h, tensor([0.0, 0.4]), lo, hi, caps)
    b = gap_kkt(x, q, h, tensor([1e100, 0.4]), lo, hi, caps)
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_strong_penalty_raw_fusion_is_optimizer_qualified():
    from clipp1d.cuda.solver import fit_lambda

    m = model()
    p = pilot(m)
    g = build_graph(p.phi)
    result = fit_lambda(m, g, p, tensor(2.0))
    assert result.qualified
    assert result.x[0] == result.x[1]
    assert result.x[2] < 1.0
    assert result.diagnostics["directional_qualified"]
