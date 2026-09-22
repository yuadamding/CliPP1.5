"""CPU mathematical reference checks; these never establish CUDA qualification."""

from dataclasses import replace

import numpy as np
import osqp
import pytest
from scipy import sparse
import torch

from clipp1d.cuda.graph import build_graph, penalty_reference
from clipp1d.cuda.kernels import Kernels, adjoint, boxed_rank_one, differences, gap_kkt
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.qp import solve_qp


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def oracle(h, target, lower, upper, caps):
    n = len(h)
    left, right = np.triu_indices(n, 1)
    m = left.size
    d = sparse.csc_matrix(
        (np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[left, right])),
        shape=(m, n),
    )
    constraints = sparse.vstack(
        (
            sparse.hstack((sparse.eye(n), sparse.csc_matrix((n, m)))),
            sparse.hstack((d, -sparse.eye(m))),
            sparse.hstack((-d, -sparse.eye(m))),
        ),
        format="csc",
    )
    qp = osqp.OSQP()
    qp.setup(
        P=sparse.diags(np.r_[h, np.zeros(m)], format="csc"),
        q=np.r_[-h * target, caps[left, right]],
        A=constraints,
        l=np.r_[lower, np.full(2 * m, -np.inf)],
        u=np.r_[upper, np.zeros(2 * m)],
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=200000,
        polishing=True,
        verbose=False,
        adaptive_rho_interval=25,
    )
    fit = qp.solve(raise_error=True)
    assert fit.info.status_val == 1
    return fit.x[:n]


@pytest.mark.parametrize("seed", range(20))
def test_complete_qp_independent_epigraph_with_bounds_and_fixed_nodes(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 9
    h = np.exp(rng.uniform(-2, 4, n))
    target = rng.normal(0.5, 2, n)
    lower = rng.uniform(-0.3, 0.2, n)
    upper = lower + rng.uniform(0.01, 2, n)
    if seed % 2 == 0:
        lower[n // 2] = upper[n // 2] = 0.5
    caps = np.exp(rng.uniform(-3, 3, (n, n)))
    caps = 0.5 * caps + 0.5 * caps.T
    np.fill_diagonal(caps, 0.0)
    expected = oracle(h, target, lower, upper, caps)
    fit = solve_qp(*map(tensor, (h, target, lower, upper, caps)), Kernels("cpu"))
    assert fit.qualified, (seed, float(fit.gap), float(fit.kkt), fit.iterations)
    np.testing.assert_allclose(fit.x.numpy(), expected, atol=3e-6, rtol=1e-7)
    assert float(fit.gap) <= CudaPolicy().inner_atol + CudaPolicy().inner_rtol * float(fit.scale)
    assert float(fit.kkt) <= CudaPolicy().inner_kkt_tol


@pytest.mark.parametrize("seed", range(20))
def test_rank_one_heterogeneous_curvature_fixed_coordinate_kkt(seed):
    rng = np.random.default_rng(seed + 701)
    n = 3 + seed % 8
    h = np.exp(rng.uniform(-8, 8, n))
    b = rng.normal(size=n) * 10
    lower = rng.uniform(-1, -0.1, n)
    upper = rng.uniform(0.1, 1, n)
    lower[n // 2] = upper[n // 2] = 0.5
    rho = np.exp(rng.uniform(-6, 6))
    x = boxed_rank_one(*map(tensor, (h, b, lower, upper, rho))).numpy()
    a = h + rho * n
    g = a * x - rho * x.sum() - b
    g[x == lower] = np.minimum(g[x == lower], 0)
    g[x == upper] = np.maximum(g[x == upper], 0)
    g[lower == upper] = 0
    scale = 1 + np.abs(a * x) + np.abs(rho * x.sum()) + np.abs(b)
    assert np.max(np.abs(g) / scale) < 2e-12


def test_gap_matches_independent_primal_dual_and_counts_each_edge_once():
    rng = np.random.default_rng(44)
    n = 7
    h = rng.uniform(0.5, 3, n)
    target = rng.normal(size=n)
    lower = np.zeros(n)
    upper = np.ones(n)
    lower[2] = upper[2] = 0.5
    x = np.clip(rng.random(n), lower, upper)
    caps = rng.uniform(0.1, 2, (n, n))
    caps = 0.5 * caps + 0.5 * caps.T
    np.fill_diagonal(caps, 0)
    q = rng.uniform(-0.1, 0.1, (n, n))
    q = 0.5 * q - 0.5 * q.T
    a = -q.sum(1)
    z = np.clip(target - a / h, lower, upper)
    d = x[None, :] - x[:, None]
    primal = 0.5 * np.dot(h, (x - target) ** 2) + 0.5 * np.sum(caps * np.abs(d))
    dual = 0.5 * np.dot(h, (z - target) ** 2) + np.dot(a, z)
    stats = gap_kkt(*map(tensor, (x, q, h, target, lower, upper, caps)))
    np.testing.assert_allclose(float(stats[0]), primal - dual, rtol=1e-12, atol=1e-12)
    assert float(stats[0]) >= 0


def test_fixed_extreme_targets_do_not_enter_solver_arithmetic_or_tolerance():
    h = tensor([8.0, 2.0, 3.0])
    target = tensor([0.7, 0.2, 0.8])
    lower = tensor([0.7, 0.0, 0.0])
    upper = tensor([0.7, 1.0, 1.0])
    caps = build_graph(target).weights * 0.4
    normal = solve_qp(h, target, lower, upper, caps, Kernels("cpu"))
    extreme = target.clone()
    extreme[0] = 1e308
    fit = solve_qp(h, extreme, lower, upper, caps, Kernels("cpu"))
    assert normal.qualified and fit.qualified
    torch.testing.assert_close(fit.x, normal.x, rtol=0, atol=0)
    torch.testing.assert_close(fit.scale, normal.scale, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind", ["nan_start", "wrong_start_shape", "float_start", "nan_dual", "wrong_dual_shape"]
)
def test_invalid_warm_vectors_fail_before_arithmetic(kind):
    h = torch.ones(3, dtype=torch.float64)
    target = tensor([0.1, 0.4, 0.9])
    lower = h * 0
    upper = h
    caps = build_graph(target).weights
    supplied = {
        "nan_start": {"start": tensor([0.1, float("nan"), 0.9])},
        "wrong_start_shape": {"start": tensor([[0.1, 0.4, 0.9]])},
        "float_start": {"start": torch.ones(3, dtype=torch.float32)},
        "nan_dual": {"dual": torch.full((3, 3), float("nan"), dtype=torch.float64)},
        "wrong_dual_shape": {"dual": torch.zeros(3, dtype=torch.float64)},
    }
    with pytest.raises(ValueError, match="QP (start|dual)"):
        solve_qp(h, target, lower, upper, caps, Kernels("cpu"), **supplied[kind])


def test_graph_canonical_identity_and_data_mutation_detection():
    graph = build_graph(tensor([0.1, 0.8, 0.3]), ("a", "b", "c"))
    graph.validate()
    assert graph.mutation_ids == ("a", "b", "c")
    graph.weights.data[0, 1] = graph.weights[0, 1] + 0.1
    with pytest.raises(ValueError, match="modified"):
        graph.validate()
    with pytest.raises(ValueError, match="canonical"):
        build_graph(tensor([0.1, 0.8]), ("z", "a"))
    with pytest.raises(ValueError, match="finite"):
        build_graph(tensor([0.1, float("nan")]))


def test_complete_reference_flow_balances_all_nodes():
    from types import SimpleNamespace

    x = tensor([0.1, 0.4, 0.7, 0.9])
    h = tensor([3.0, 7.0, 4.0, 2.0])
    model = SimpleNamespace(n=4, lower=torch.zeros_like(x), upper=torch.ones_like(x))
    graph = build_graph(x)
    ref = penalty_reference(model, graph, x, h)
    beta = (h * x).sum() / h.sum()
    g = h * (beta - x)
    g = g - g.sum() / 4
    flow = -differences(g) / 4
    torch.testing.assert_close(adjoint(flow), -g, rtol=1e-13, atol=1e-13)
    assert bool((flow.abs() <= ref * graph.weights + 1e-14).all())


def test_one_two_and_all_fixed_qps():
    for n in [1, 2, 7]:
        h = torch.ones(n, dtype=torch.float64)
        target = torch.linspace(0.1, 0.9, n, dtype=torch.float64)
        caps = build_graph(target).weights
        fit = solve_qp(h, target, h * 0, h, caps, Kernels("cpu"))
        assert fit.qualified
        torch.testing.assert_close(
            fit.x, torch.full_like(h, float(target.mean())), atol=1e-8, rtol=0
        )
        fixed = solve_qp(h, torch.full_like(h, 1e300), target, target, caps, Kernels("cpu"))
        assert fixed.qualified and float(fixed.gap) == 0 and float(fixed.scale) == 0


def test_iteration_budget_exhaustion_is_unresolved():
    h = tensor([1.0, 3.0, 7.0])
    target = tensor([0.1, 0.4, 0.9])
    lo = torch.zeros_like(h)
    hi = torch.ones_like(h)
    caps = build_graph(target).weights * 0.2
    fit = solve_qp(
        h, target, lo, hi, caps, Kernels("cpu"), replace(CudaPolicy(), inner_max_iterations=1)
    )
    assert not fit.qualified


def test_likelihood_preserves_countmodel_at_clipping_and_mixture_points():
    from clipp1d.types import CountModel
    from clipp1d.model import evaluate, one_sided_derivatives
    from clipp1d.cuda.kernels import likelihood

    eps = 1e-6
    slopes = np.array(
        [
            [0.17, 0.34, 0.51],
            [0.13, 0.26, 0.39],
            [0.49, 0.98, 0.0],
            [0.07, 0.14, 0.21],
            [0.23, 0.46, 0.69],
            [0.37, 0.74, 1.11],
        ]
    )
    valid = slopes > 0
    prior = np.where(valid, -np.log(valid.sum(1))[:, None], -np.inf)
    alt, ref = np.array([0, 1, 25, 70, 5, 90]), np.array([100, 99, 75, 30, 95, 10])
    model = CountModel(
        tuple(map(str, range(6))), alt, ref, np.full(6, eps), np.ones(6), slopes, prior, valid, eps
    )
    for x in [
        np.array([eps / 0.17, eps / 0.13, 0.4, 0.6, 0.8, (1 - eps) / 1.11]),
        np.linspace(0.1, 0.9, 6),
        np.full(6, eps),
        np.ones(6),
    ]:
        reference = evaluate(model, x, derivatives=True)
        left, right = one_sided_derivatives(model, x)
        result = likelihood(
            tensor(alt), tensor(ref), tensor(slopes), tensor(prior), tensor(x[:, None]), eps
        )
        expected = [
            reference.loss,
            reference.gradient,
            reference.curvature,
            reference.posterior,
            left,
            right,
        ]
        for actual, value in zip(result, expected):
            np.testing.assert_allclose(actual[:, 0].numpy(), value, rtol=5e-13, atol=1e-10)


def test_strict_fullgraph_trace_preserves_kernel_certificates_cpu_only():
    from clipp1d.cuda.kernels import edge_update

    h = tensor([1.0, 3.0, 7.0])
    b = tensor([0.1, 0.4, 0.9])
    lower, upper = torch.zeros_like(h), torch.ones_like(h)
    caps = build_graph(b).weights
    q = torch.zeros_like(caps)
    args = [
        (boxed_rank_one, (h, b, lower, upper, tensor(0.7))),
        (edge_update, (b, q, caps, tensor(0.7))),
        (gap_kkt, (b, q, h, b, lower, upper, caps)),
    ]
    for function, inputs in args:
        traced = torch.compile(function, backend="eager", fullgraph=True)
        torch.testing.assert_close(traced(*inputs), function(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("probe", ["lower", "upper", "below_upper", "above_upper"])
def test_full_mixture_fixture_exact_kinks_match_canonical_numpy(probe):
    from clipp1d.cuda.kernels import likelihood
    from clipp1d.model import evaluate, one_sided_derivatives
    from clipp1d.types import CountModel

    eps = 1e-6
    alt = np.array([16.0, 28.0, 37.0, 19.0, 40.0, 54.0])
    counts = np.array([1, 2, 3, 4, 2, 3])
    scale = np.array([0.4, 0.25, 0.15, 0.12, 0.35, 0.42])
    support = np.arange(1, 5)
    valid = support <= counts[:, None]
    slope = scale[:, None] * np.where(valid, support, 0)
    prior = np.where(valid, -np.log(counts[:, None]), -np.inf)
    model = CountModel(
        tuple(f"m{i:04}" for i in range(6)),
        alt,
        100 - alt,
        np.full(6, eps),
        np.minimum(1.0, (1 - eps) / (scale * counts)),
        slope,
        prior,
        valid,
        eps,
    )
    x = eps / scale if probe == "lower" else (1 - eps) / slope.max(1)
    if probe == "below_upper":
        x = np.nextafter(x, -np.inf)
    if probe == "above_upper":
        x = np.nextafter(x, np.inf)
    reference = evaluate(model, x, derivatives=True)
    left, right = one_sided_derivatives(model, x)
    expected = [
        reference.loss,
        reference.gradient,
        reference.curvature,
        reference.posterior,
        left,
        right,
    ]
    functions = [
        likelihood,
        torch.compile(likelihood, backend="eager", fullgraph=True, dynamic=True),
    ]
    for function in functions:
        outputs = function(
            tensor(alt), tensor(100 - alt), tensor(slope), tensor(prior), tensor(x[:, None]), eps
        )
        for label, actual, target in zip(
            ("loss", "gradient", "curvature", "posterior", "left", "right"), outputs, expected
        ):
            np.testing.assert_allclose(
                actual[:, 0].numpy(), target, rtol=5e-11, atol=2e-9, err_msg=f"{probe}/{label}"
            )
    if probe == "upper":
        # This is the captured GPU-a failure: scalar/Tensor reciprocal multiplication
        # misses this exact phi-space kink and invents a right derivative ~3.36e7.
        assert float(outputs[-1][0, 0]) == 0.0
        assert reference.gradient[0] > 3e7


def test_kink_graph_uses_true_division_not_scalar_reciprocal_multiply():
    from clipp1d.cuda.kernels import likelihood

    alt, ref = tensor([16.0, 28.0]), tensor([84.0, 72.0])
    slope = tensor([[0.4], [0.25]])
    prior = torch.zeros_like(slope)
    points = tensor([[(1 - 1e-6) / 0.4], [(1 - 1e-6) / 0.25]])
    graph = torch._dynamo.export(likelihood, aten_graph=True)(
        alt, ref, slope, prior, points, 1e-6
    ).graph_module
    reciprocal = [
        node for node in graph.graph.nodes if node.target == torch.ops.aten.reciprocal.default
    ]
    assert not reciprocal, "Kink divisions must retain tensor/tensor division semantics"


def test_structural_compile_families_cover_variable_widths_layouts_and_sequential_banks():
    from clipp1d.cuda.kernels import StructuralCompileBank, likelihood

    limits = (
        torch._dynamo.config.recompile_limit,
        torch._dynamo.config.accumulated_recompile_limit,
    )
    code_objects = []

    def backend(graph, inputs):
        code_objects.append(graph)
        return graph.forward

    banks = [StructuralCompileBank(likelihood, backend=backend) for _ in range(2)]
    with torch.no_grad():
        for bank in banks:
            # Proposal widths include scalar/grid/refinement widths, equal-size
            # symbolic dimensions and large intervals; views vary their layouts.
            for n, k in ((6, 4), (3, 1), (1, 4), (2, 2), (7, 3), (13, 4)):
                alt = torch.arange(n, dtype=torch.float64) + 16
                ref = 100 - alt
                slopes = torch.arange(1, k + 1, dtype=torch.float64)[None, :].expand(n, -1) * 0.12
                prior = torch.full_like(slopes, -float(np.log(k)))
                for width in (1, 2, 3, 7, 8, 13, 17, 33, 34, 35, 37, 64, 128, 4096):
                    packed = torch.full((n, width * 2), 0.37, dtype=torch.float64)
                    points = packed[:, ::2]
                    inputs = (alt, ref, slopes, prior, points, 1e-6)
                    torch.testing.assert_close(bank(*inputs), likelihood(*inputs), rtol=0, atol=0)
            assert len(bank._entries) <= bank.max_families
            assert bank.families_created <= 32
            assert bank.evictions == 0
    assert limits == (
        torch._dynamo.config.recompile_limit,
        torch._dynamo.config.accumulated_recompile_limit,
    )
    first_codes = {
        entry._torchdynamo_orig_callable.__code__ for entry in banks[0]._entries.values()
    }
    second_codes = {
        entry._torchdynamo_orig_callable.__code__ for entry in banks[1]._entries.values()
    }
    assert first_codes.isdisjoint(second_codes)
    assert all(code is not likelihood.__code__ for code in first_codes | second_codes)
    assert all(
        not hasattr(entry._torchdynamo_orig_callable, "__wrapped__")
        for bank in banks
        for entry in bank._entries.values()
    )
    assert code_objects


def test_structural_compile_family_lru_is_bounded_without_fallback():
    from clipp1d.cuda.kernels import StructuralCompileBank, likelihood

    bank = StructuralCompileBank(likelihood, backend="eager", max_families=2)
    with torch.no_grad():
        for n, k, width in ((1, 1, 1), (2, 1, 1), (2, 3, 1), (2, 3, 37), (1, 1, 1)):
            alt = torch.full((n,), 16.0, dtype=torch.float64)
            ref = 100 - alt
            slope = torch.full((n, k), 0.4, dtype=torch.float64)
            prior = torch.full_like(slope, -float(np.log(k)))
            phi = torch.full((n, width), 0.2, dtype=torch.float64)
            torch.testing.assert_close(
                bank(alt, ref, slope, prior, phi, 1e-6),
                likelihood(alt, ref, slope, prior, phi, 1e-6),
                rtol=0,
                atol=0,
            )
            assert len(bank._entries) <= 2
    diagnostics = bank.diagnostics()
    assert diagnostics["evictions"] == 3
    assert diagnostics["families_created"] == 5
    assert diagnostics["eager_fallback"] is False
    assert diagnostics["global_cache_limit_modified"] is False


def test_structural_compile_bank_replays_real_scalar_refit_pipeline_shapes():
    from clipp1d.cuda.kernels import StructuralCompileBank, likelihood
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.selection import fit_tensor_model
    from clipp1d.types import CountModel

    bank = StructuralCompileBank(likelihood, backend="eager")
    kernels = Kernels("cpu")
    kernels.likelihood = bank
    fixtures = [
        (np.array([20.0, 21.0, 48.0]), np.ones(3, dtype=int), np.full(3, 0.5)),
        (
            np.array([16.0, 28.0, 37.0, 19.0, 40.0, 54.0]),
            np.array([1, 2, 3, 4, 2, 3]),
            np.array([0.4, 0.25, 0.15, 0.12, 0.35, 0.42]),
        ),
    ]
    for alt, counts, scales in fixtures:
        n = alt.size
        support = np.arange(1, int(counts.max()) + 1)
        valid = support <= counts[:, None]
        slope = scales[:, None] * np.where(valid, support, 0)
        prior = np.where(valid, -np.log(counts[:, None]), -np.inf)
        host = CountModel(
            tuple(f"m{i:04}" for i in range(n)),
            alt,
            100 - alt,
            np.full(n, 1e-6),
            np.minimum(1.0, (1 - 1e-6) / (scales * counts)),
            slope,
            prior,
            valid,
            1e-6,
        )
        model = TensorModel.from_host(host, "cpu", compiled=False)
        object.__setattr__(model, "kernels", kernels)
        actual = fit_tensor_model(model, lambda_values=[0.0, 0.1])
        assert actual.raw.qualified and actual.search_status == "complete"
    assert bank.calls > 100
    assert bank.families_created <= bank.max_families


def test_qualifier_path_coordinate_preserves_observed_independent_reference_difference():
    from types import SimpleNamespace
    from benchmarks.qualify_cuda import path_position, closeness

    def fit_at(reference, index):
        path = [0.0] + [np.ldexp(reference, exponent) for exponent in range(-12, 13)]
        return SimpleNamespace(
            lambda_value=tensor(path[index]),
            records=[{"lambda_value": value} for value in path],
            timings={"lambda_reference": reference, "extensions": 0},
            policy=CudaPolicy(),
            model=SimpleNamespace(n=6),
        )

    eager = fit_at(71.4486657676, 12)
    compiled = fit_at(71.4486589086, 12)
    first, second = path_position(eager, "default"), path_position(compiled, "default")
    assert first["selected_coordinate"] == second["selected_coordinate"] == -1
    assert first["selected_index"] == second["selected_index"] == 12
    assert first["coordinates"] == second["coordinates"]
    assert first["actual_selected_lambda"] != second["actual_selected_lambda"]
    # The literal-lambda gate remains exact for the added shared-problem replay.
    with pytest.raises(AssertionError):
        closeness(
            first["actual_selected_lambda"], second["actual_selected_lambda"], atol=0.0, rtol=0.0
        )
    changed = path_position(fit_at(71.4486589086, 13), "default")
    assert changed["selected_coordinate"] != first["selected_coordinate"]
    broken = fit_at(71.4486589086, 12)
    broken.records[4]["lambda_value"] += 0.01
    with pytest.raises(AssertionError, match="recipe"):
        path_position(broken, "default")


@pytest.mark.parametrize("extensions", [0, 2])
def test_qualifier_exact_device_path_recipe_replays_attempt_d_rounding(monkeypatch, extensions):
    """Captured D CUDA ldexp differs from host at four exponents; no tolerance is added."""
    import math
    from types import SimpleNamespace
    from benchmarks.qualify_cuda import path_position

    reference = float.fromhex("0x1.3a8e60f2207f4p+7")
    # D's single_support eager and compiled JSON both have these exact values.
    captured = {
        -12: float.fromhex("0x1.3a8e60f2207f3p-5"),
        -9: float.fromhex("0x1.3a8e60f2207f3p-2"),
        -4: float.fromhex("0x1.3a8e60f2207f3p+3"),
        11: float.fromhex("0x1.3a8e60f2207f3p+18"),
    }
    # The second scenario extends from the rounded +11 endpoint. Its +12
    # extension must preserve the preceding ULP, not recompute device ldexp(+12).
    policy = replace(CudaPolicy(), path_max_exponent=11 if extensions else 12)
    exponents = list(range(policy.path_min_exponent, policy.path_max_exponent + 1))
    path = [0.0] + [captured.get(k, math.ldexp(reference, k)) for k in exponents]
    for _ in range(extensions):
        path.append(path[-1] * 2.0)
    fit = SimpleNamespace(
        lambda_value=tensor(path[5]),
        records=[{"lambda_value": value} for value in path],
        timings={"lambda_reference": reference, "extensions": extensions},
        policy=policy,
        model=SimpleNamespace(n=3),
    )
    calls = []

    def captured_device_ldexp(value, exponent):
        assert float(value) == reference
        k = int(exponent)
        calls.append(k)
        return value.new_tensor(captured.get(k, math.ldexp(reference, k)))

    monkeypatch.setattr(torch, "ldexp", captured_device_ldexp)
    position = path_position(fit, "default")
    assert calls == exponents
    assert position["selected_coordinate"] == -8
    assert position["recipe_recomputed_exactly"]
    assert position["host_ldexp_differences_ulps"][1] == -1.0
    if extensions:
        assert position["host_ldexp_differences_ulps"][-2:] == [-1.0, -1.0]
    else:
        assert [i for i, difference in enumerate(position["host_ldexp_differences_ulps"])
                if difference] == [1, 4, 9, 24]
    # A one-ULP change in a recorded device value still fails exact validation.
    fit.records[1]["lambda_value"] = math.nextafter(path[1], math.inf)
    with pytest.raises(AssertionError, match="exact device recipe"):
        path_position(fit, "default")


def test_qualifier_singleton_default_path_has_only_zero():
    from types import SimpleNamespace
    from benchmarks.qualify_cuda import path_position

    fit = SimpleNamespace(
        lambda_value=tensor(0.0), records=[{"lambda_value": 0.0}],
        timings={"lambda_reference": 1.0, "extensions": 0},
        policy=CudaPolicy(), model=SimpleNamespace(n=1),
    )
    result = path_position(fit, "default")
    assert result["coordinates"] == [None]
    assert result["host_ldexp_differences_ulps"] == [0.0]


def test_qualifier_shared_graph_replay_checks_nonzero_nonfused_problem(tmp_path, monkeypatch):
    import benchmarks.qualify_cuda as qualification
    from clipp1d.cuda.kernels import StructuralCompileBank, likelihood
    from clipp1d.cuda.selection import fit_tensor_model

    host = qualification.host_fixture("single_support")
    eager, compiled = Kernels("cpu"), Kernels("cpu")
    compiled.likelihood = StructuralCompileBank(likelihood, backend="eager")
    models = [
        qualification.upload(host, torch.device("cpu"), kernels) for kernels in (eager, compiled)
    ]
    fits = [fit_tensor_model(model, lambda_values=[0.0, 0.1, 0.5]) for model in models]
    events = []
    monkeypatch.setattr(qualification, "timed", lambda device, function: (function(), 0.0))
    qualification.matched_graph_cases(
        "reference_test",
        fits[0],
        fits[1],
        torch.device("cpu"),
        tmp_path,
        lambda kind, detail: events.append((kind, detail)),
    )
    parity = [event for kind, event in events if kind == "matched_graph_parity"]
    assert len(parity) >= 3
    assert any(row["lambda_value"] > 0 and row["fusion_penalty"] > 0 for row in parity)
    assert all(row["errors"]["literal_lambda"] == 0 for row in parity)
    assert len({row["weights_sha256"] for row in parity}) == 1
    assert len({row["pilot_sha256"] for row in parity}) == 1
    assert all(row["identical_graph_and_lambda"] for row in parity)
    assert list(tmp_path.glob("matched-*.json"))
