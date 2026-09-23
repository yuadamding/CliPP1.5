"""One-sided surrogate curvature references; CPU checks are not CUDA acceptance."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.solver import bound_majorization_curvature
from clipp1d.model import evaluate
from clipp1d.types import CountModel


def problem(side):
    def t(value):
        return torch.tensor(value, dtype=torch.float64)
    slope = t([[2., 0.]])
    eps = 1e-6
    value = (eps if side == 'lower' else 1 - eps) / 2
    lower, upper = (value, .9) if side == 'lower' else (1e-6, value)
    model = TensorModel(('node',), t([1836.]), t([1.]), slope,
                        t([[0., -float('inf')]]), t([lower]), t([upper]),
                        eps, Kernels('cpu'))
    return model, t([value])


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_active_kink_curvature_covers_feasible_one_sided_majorization(side):
    model, x = problem(side)
    loss, _, curvature, post, left, right = model.terms(x)
    assert float(curvature[0]) == 1e-8
    corrected = bound_majorization_curvature(model, x, curvature, post)
    p = model.eps if side == 'lower' else 1 - model.eps
    expected = 4 * (1836 / p**2 + 1 / (1 - p)**2)
    torch.testing.assert_close(corrected, curvature + expected, rtol=1e-14, atol=0)
    direction = 1 if side == 'lower' else -1
    delta = x.new_tensor([direction * model.eps / 200])
    grad = right if side == 'lower' else left
    trial_loss = model.loss(x + delta)
    margin = 64 * torch.finfo(x.dtype).eps * (1 + loss.abs())
    old_major = loss + grad * delta + .5 * curvature.clamp_min(1.) * delta.square()
    new_major = loss + grad * delta + .5 * corrected * delta.square()
    assert bool((trial_loss > old_major + margin).all())
    assert bool((trial_loss <= new_major + margin).all())
    # Likelihood outputs and the original curvature used to plan starts are intact.
    torch.testing.assert_close(model.terms(x)[2], curvature, rtol=0, atol=0)


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_fixed_coordinate_does_not_acquire_inward_curvature(side):
    model, x = problem(side)
    model = replace(model, lower=x, upper=x)
    _, _, curvature, post, _, _ = model.terms(x)
    torch.testing.assert_close(bound_majorization_curvature(model, x, curvature, post),
                               curvature, rtol=0, atol=0)


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_interior_kink_does_not_change_planned_likelihood_curvature(side):
    model, x = problem(side)
    model = replace(model, lower=torch.zeros_like(x), upper=torch.ones_like(x))
    _, _, curvature, post, _, _ = model.terms(x)
    torch.testing.assert_close(bound_majorization_curvature(model, x, curvature, post),
                               curvature, rtol=0, atol=0)


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_smooth_neighbor_never_counts_curvature_twice(side):
    model, x = problem(side)
    target = torch.full_like(x, float('inf') if side == 'lower' else -float('inf'))
    x = torch.nextafter(x, target)
    model = replace(model, **{side: x})
    _, _, curvature, post, _, _ = model.terms(x)
    torch.testing.assert_close(bound_majorization_curvature(model, x, curvature, post),
                               curvature, rtol=0, atol=0)


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_mixture_adds_only_posterior_weighted_missing_support_and_preserves_apis(side):
    model, x = problem(side)
    slope = torch.tensor([[2., 4. if side == 'lower' else .8, 0.]], dtype=x.dtype)
    alt, ref = torch.tensor([1.], dtype=x.dtype), torch.tensor([2.], dtype=x.dtype)
    p = (x[:, None]*slope[:, :2]).clamp(model.eps, 1-model.eps)
    # Calibrate valid priors to give nontrivial posterior weights at this point.
    desired = torch.tensor([[.25, .75]], dtype=x.dtype)
    log_prior = desired.log() - alt[:, None]*p.log() - ref[:, None]*torch.log1p(-p)
    log_prior -= torch.logsumexp(log_prior, dim=1, keepdim=True)
    log_prior = torch.cat((log_prior, x.new_full((1, 1), -float('inf'))), dim=1)
    model = replace(model, alt=alt, ref=ref, slope=slope, log_prior=log_prior)
    before = model.terms(x)
    loss, _, curvature, post, _, _ = before
    torch.testing.assert_close(post[:, :2], desired, atol=1e-15, rtol=1e-14)
    assert post[0, 2] == 0
    corrected = bound_majorization_curvature(model, x, curvature, post)
    edge = model.eps if side == 'lower' else 1-model.eps
    expected_addition = .25*4*(alt/edge**2 + ref/(1-edge)**2)
    torch.testing.assert_close(corrected, curvature+expected_addition, atol=0, rtol=1e-14)
    assert torch.isfinite(corrected).all()
    # This helper cannot alter the CountModel contract or the TensorModel terms
    # subsequently used for scalar pilots and curvature-weighted pooled starts.
    host = CountModel(model.mutation_ids, alt.numpy(), ref.numpy(), model.lower.numpy(),
                      model.upper.numpy(), slope.numpy(), log_prior.numpy(),
                      np.isfinite(log_prior.numpy()), model.eps)
    reference = evaluate(host, x.numpy(), derivatives=True)
    after = model.terms(x)
    for old, new in zip(before, after):
        torch.testing.assert_close(new, old, atol=0, rtol=0)
    torch.testing.assert_close(loss, torch.from_numpy(reference.loss), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(curvature, torch.from_numpy(reference.curvature), atol=1e-12, rtol=1e-14)
    torch.testing.assert_close(post, torch.from_numpy(reference.posterior), atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize('side,slope_value,moving', [
    ('lower', .0020259590509116906, False),
    ('lower', .2580127478946282, True),
    ('upper', 1.0715933998226712, False),
    ('upper', 1.0740661533334332, True),
])
def test_exact_phi_kink_handles_product_rounding_without_missing_or_double_curvature(side, slope_value, moving):
    model, _ = problem(side)
    slope = torch.tensor([[slope_value, 0.]], dtype=torch.float64)
    endpoint = model.eps if side == 'lower' else 1-model.eps
    x = torch.full((1,), endpoint, dtype=torch.float64) / slope[:, 0]
    mass = x[:, None]*slope
    assert mass[0, 0] != endpoint  # A real one-ULP division/product discrepancy.
    assert bool((mass[0, 0] > model.eps) & (mass[0, 0] < 1-model.eps)) is moving
    bounds = {'lower': x, 'upper': torch.ones_like(x)} if side == 'lower' else {
        'lower': torch.full_like(x, 1e-6), 'upper': x}
    model = replace(model, slope=slope, **bounds)
    _, _, curvature, post, _, _ = model.terms(x)
    corrected = bound_majorization_curvature(model, x, curvature, post)
    if moving:
        torch.testing.assert_close(corrected, curvature, atol=0, rtol=0)
    else:
        expected = slope_value**2*(model.alt/endpoint**2 + model.ref/(1-endpoint)**2)
        torch.testing.assert_close(corrected, curvature+expected, atol=0, rtol=1e-14)
