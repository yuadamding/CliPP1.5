"""Allocated-CUDA oracles for conservative posterior-weighted curvature bounds."""
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def env():
    import torch
    from clipp1d.cuda.kernels import Kernels
    from clipp1d.cuda.model import TensorModel
    from clipp1d.cuda import scalar
    if not torch.cuda.is_available():
        pytest.skip("Allocated CUDA required; a skipped test is not numerical qualification")
    kernels = Kernels(torch.device("cuda:0"), compiled=True)

    def make(alt, ref, slopes, priors, *, lower=1e-6, upper=1.):
        a = torch.tensor(alt, dtype=torch.float64, device="cuda:0")
        return TensorModel(tuple(str(i) for i in range(len(alt))), a, a.new_tensor(ref),
                           a.new_tensor(slopes), a.new_tensor(priors),
                           a.new_full(a.shape, lower), a.new_full(a.shape, upper), 1e-6, kernels)
    return torch, scalar, make


def variance_inputs(torch, model, left, right):
    sl = model.slope[:, None, :]
    pl = (sl * left[..., None]).clamp(model.eps, 1 - model.eps)
    pr = (sl * right[..., None]).clamp(model.eps, 1 - model.eps)
    mid = sl * (left + .5 * (right - left))[..., None]
    moving = torch.where((mid > model.eps) & (mid < 1 - model.eps), sl, 0.)
    valid = torch.isfinite(model.log_prior)[:, None, :]
    score_left = moving * (model.alt[:, None, None] / pl - model.ref[:, None, None] / (1 - pl))
    score_right = moving * (model.alt[:, None, None] / pr - model.ref[:, None, None] / (1 - pr))
    spread = (torch.where(valid, score_left, -torch.inf).amax(-1)
              - torch.where(valid, score_right, torch.inf).amin(-1))
    original = torch.where(valid.sum(-1) == 1, 0., spread.square() / 4)
    return pl, pr, moving, valid, original


@pytest.mark.parametrize("slopes,priors,lo,hi", [
    ([.3, .7], [0., 0.], .2, .8),
    ([.5, 1.9], [0., 0.], .5, .500001),
    ([.5, 1.9], [0., -1000.], .5, .500001),
    ([.5, 0., 1.9], [0., -float("inf"), -100.], .5, .500001),
    ([.5, .5], [0., 0.], .4, .6),
    ([.5], [0.], .4, .6),
    ([2., 3.], [0., 0.], .8, .9),
], ids=["balanced", "rare-extreme-score", "underflow-mass", "invalid-support", "same-score", "single", "plateau"])
def test_variance_upper_covers_dense_independent_posterior(env, slopes, priors, lo, hi):
    torch, scalar, make = env
    model = make([10.], [90.], [slopes], [priors])
    left, right = model.alt.new_tensor([[lo]]), model.alt.new_tensor([[hi]])
    pl, pr, moving, valid, original = variance_inputs(torch, model, left, right)
    bound = scalar.posterior_variance_bound(model, pl, pr, moving, valid, original)
    phi = torch.linspace(lo, hi, 257, dtype=torch.float64, device=model.device)
    raw = model.slope[:, None, :] * phi[None, :, None]
    probability = raw.clamp(model.eps, 1 - model.eps)
    logits = (model.alt[:, None, None] * probability.log()
              + model.ref[:, None, None] * torch.log1p(-probability) + model.log_prior[:, None, :])
    posterior = torch.softmax(logits, -1)
    derivative = torch.where((raw > model.eps) & (raw < 1 - model.eps), model.slope[:, None, :], 0.)
    score = derivative * (model.alt[:, None, None] / probability - model.ref[:, None, None] / (1 - probability))
    mean = (posterior * score).sum(-1, keepdim=True)
    actual = (posterior * (score - mean).square()).sum(-1)
    reference_roundoff = 64 * torch.finfo(torch.float64).eps * (1 + actual.abs())
    assert bool((actual <= bound + reference_roundoff).all())
    assert bool(torch.isfinite(bound).all() & (bound >= 0).all() & (bound <= original).all())


def test_rare_extreme_score_bound_is_strictly_tighter(env):
    torch, scalar, make = env
    model = make([10.], [90.], [[.5, 1.9]], [[0., 0.]])
    args = variance_inputs(torch, model, model.alt.new_tensor([[.5]]), model.alt.new_tensor([[.500001]]))
    bound = scalar.posterior_variance_bound(model, *args)
    assert bool((bound < args[-1] * 1e-6).all())


def test_nonfinite_envelope_retains_original_bound(env):
    torch, scalar, make = env
    model = make([10.], [90.], [[.5, 1.9]], [[0., 0.]])
    pl, pr, moving, valid, original = variance_inputs(
        torch, model, model.alt.new_tensor([[.5]]), model.alt.new_tensor([[.500001]]))
    overflow = SimpleNamespace(alt=model.alt.new_tensor([1e308]), ref=model.ref,
                               log_prior=model.log_prior, eps=model.eps)
    got = scalar.posterior_variance_bound(overflow, pl, pr, moving, valid, original)
    assert torch.equal(got.view(torch.uint8), original.view(torch.uint8))


@pytest.mark.parametrize("endpoint", [False, True], ids=["crossing", "endpoint"])
def test_clipping_crossing_retains_original_independent_bound(env, monkeypatch, endpoint):
    torch, scalar, make = env
    model = make([9.], [37.], [[1., 2.]], [[0., 0.]])
    problem = scalar.Problems(model, torch.ones(1, device=model.device, dtype=torch.long))
    kink = model.alt.new_tensor(1 - model.eps) / model.slope[0, 1]
    left = kink.reshape(1, 1) - (0. if endpoint else 1e-5)
    right = kink.reshape(1, 1) + 1e-5
    got = problem.bounds(left, right)[0]
    monkeypatch.setattr(scalar, "posterior_variance_bound", lambda m, pl, pr, moving, valid, original: original)
    expected = problem.bounds(left, right)[0]
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


def test_shared_likelihood_bound_covers_dense_mixture_and_group_sum(env):
    torch, scalar, make = env
    model = make([10., 13.], [90., 87.], [[.5, 1.9], [.4, 1.8]], [[0., 0.], [0., -1.]])
    problem = scalar.Problems(model, torch.tensor([2], device=model.device, dtype=torch.long))
    left, right = model.alt.new_tensor([[.45, .5]]), model.alt.new_tensor([[.5, .500001]])
    bound, _, _ = problem.bounds(left, right)
    fractions = torch.linspace(0, 1, 1025, device=model.device, dtype=torch.float64)
    points = left[:, :, None] + (right - left)[:, :, None] * fractions
    loss = problem.loss(points.reshape(1, -1)).reshape_as(points)
    assert bool((bound <= loss.min(-1).values).all())
