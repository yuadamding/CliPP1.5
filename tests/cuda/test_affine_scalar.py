"""Allocated-CUDA independent objectives for the shared affine scalar solver."""
import pytest


@pytest.fixture(scope="module")
def env():
    import torch
    from clipp1d.cuda.affine_scalar import AffineProblem, solve_affine_scalar
    from clipp1d.cuda.kernels import Kernels
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.policy import CudaPolicy
    from clipp1d.cuda.scalar import Problems
    if not torch.cuda.is_available():
        pytest.skip("Allocated CUDA required; a skipped test is not numerical qualification")
    kernels = Kernels(torch.device("cuda:0"), compiled=True)

    def make(coefficient, *, rows=1, mixture=False, lower=.1, upper=.9):
        a = torch.full((rows,), 9., device="cuda:0", dtype=torch.float64)
        s = a.new_tensor([[.3, .7]]).expand(rows, -1).clone() if mixture else a.new_ones((rows, 1))
        prior = s.new_full(s.shape, -.6931471805599453) if mixture else s.new_zeros(s.shape)
        model = TensorModel(tuple(str(i) for i in range(rows)), a, a.new_full(a.shape, 37.), s, prior,
                            a.new_full(a.shape, 1e-6), a.new_full(a.shape, 1 - 1e-6), 1e-6, kernels)
        base = Problems(model, torch.tensor([rows], dtype=torch.long, device=model.device))
        return AffineProblem(base, a.new_tensor(lower), a.new_tensor(upper),
                             a.new_tensor(coefficient), a.new_tensor(.4))
    return torch, CudaPolicy(), AffineProblem, solve_affine_scalar, make


@pytest.mark.parametrize("coefficient", [-11., 0., 7.])
@pytest.mark.parametrize("mixture", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_affine_bounds_cover_dense_original_group_likelihood(env, coefficient, mixture, rows):
    torch, _, _, _, make = env
    p = make(coefficient, rows=rows, mixture=mixture)
    left = p.lower[:, None] + p.lower.new_tensor([[0., .19, .55]])
    right = p.lower[:, None] + p.lower.new_tensor([[.19, .55, .8]])
    bound, mid, midloss = p.bounds(left, right)
    fractions = torch.linspace(0, 1, 1025, device=p.model.device, dtype=torch.float64)
    points = left[:, :, None] + (right - left)[:, :, None] * fractions
    losses = p.loss(points.reshape(1, -1)).reshape_as(points)
    assert bool((bound <= losses.min(-1).values).all())
    assert torch.equal(midloss, p.evaluate(mid)[0])
    assert torch.allclose(midloss, p.loss(mid), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("coefficient", [-11., 7.])
def test_affine_group_solver_matches_independent_quadratic_root(env, rows, coefficient):
    torch, policy, _, solve, make = env
    p = make(coefficient, rows=rows)
    a, n, c = p.model.alt.sum(), (p.model.alt + p.model.ref).sum(), p.coefficient[0]
    expected = 2 * a / (n + c + ((n + c).square() - 4 * c * a).sqrt())
    result = solve(p, policy)
    assert bool(result.qualified.all())
    assert torch.allclose(result.phi[0], expected, rtol=0., atol=1e-7)
    assert p.model.scalar_work_counters["analytical_groups"] == 0
    assert p.model.scalar_work_counters["general_groups"] == 1
    assert result.subdivisions <= policy.scalar_max_intervals


@pytest.mark.parametrize("mutation", ["bytes", "identity", "base"], ids=["byte-mutation", "tensor-replacement", "base-replacement"])
def test_affine_owned_parameters_reject_byte_mutation(env, mutation):
    _, _, _, solve, make = env
    p = make(7.)
    if mutation == "bytes":
        p.coefficient.data[0] += 1.
    elif mutation == "identity":
        p.coefficient = p.coefficient.clone()
    else:
        p.base = make(7.).base
    with pytest.raises(ValueError, match="changed"):
        solve(p)


def test_fixed_representable_affine_interval_is_retained(env):
    torch, policy, AffineProblem, solve, make = env
    p = make(7.)
    fixed = AffineProblem(p.base, p.reference, p.reference, p.coefficient, p.reference)
    result = solve(fixed, policy)
    assert bool(result.qualified.all())
    assert torch.equal(result.phi, fixed.reference)


@pytest.mark.parametrize("coefficient", [-11., 7.])
def test_affine_bounds_cover_exact_clipping_endpoints(env, coefficient):
    torch, _, AffineProblem, _, make = env
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda.scalar import Problems
    source = make(coefficient).model
    model = TensorModel(source.mutation_ids, source.alt, source.ref, source.alt.new_tensor([[1., 2.]]),
                        source.alt.new_zeros((1, 2)), source.lower, source.upper, source.eps, source.kernels)
    base = Problems(model, torch.ones(1, device=model.device, dtype=torch.long))
    p = AffineProblem(base, model.alt.new_tensor(.1), model.alt.new_tensor(.9),
                      model.alt.new_tensor(coefficient), model.alt.new_tensor(.4))
    kink = model.alt.new_tensor(1 - model.eps) / model.slope[0, 1]
    left = torch.stack((kink - 1e-5, kink, kink - 1e-5)).reshape(1, 3)
    right = torch.stack((kink, kink + 1e-5, kink + 1e-5)).reshape(1, 3)
    bound = p.bounds(left, right)[0]
    fraction = torch.linspace(0, 1, 1025, device=model.device, dtype=torch.float64)
    point = left[:, :, None] + (right - left)[:, :, None] * fraction
    loss = p.loss(point.reshape(1, -1)).reshape_as(point)
    assert bool((bound <= loss.min(-1).values).all())
