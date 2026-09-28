"""Independent reference and regression tests; CPU passes are not CUDA evidence."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch

from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.mixture import (MixturePolicy, criterion, expectation, fit_mixture,
                                fit_mixture_reference, maximize_centers)
from clipp1d.cuda.model import TensorModel


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model_from_counts(alt, depth, major, purity=.7, minor=None):
    alt, major = np.asarray(alt), np.asarray(major)
    minor = np.ones_like(major) if minor is None else np.asarray(minor)
    support = np.arange(1, major.max()+1)[None, :]
    valid = support <= major[:, None]
    slope = np.where(valid, purity * support / ((1-purity)*2+purity*(major+minor))[:, None], 0)
    def t(value):
        return torch.tensor(value, dtype=torch.float64)
    n = len(alt)
    return TensorModel(tuple(f"m{i:05}" for i in range(n)), t(alt), t(np.asarray(depth)-alt),
                       t(slope), t(np.where(valid, -np.log(major[:, None]), -np.inf)),
                       t(np.full(n, 1e-6)), t(np.ones(n)), 1e-6, Kernels("cpu", False))


def test_joint_posterior_and_likelihood_match_independent_enumeration():
    model = model_from_counts([0, 25, 35, 99], 100, [1, 2, 3, 4])
    centers = torch.tensor([.3, .9], dtype=torch.float64)
    weights = torch.tensor([.7, .3], dtype=torch.float64)
    enrichment = torch.tensor([.2, .8], dtype=torch.float64)
    ll, post, prior = expectation(model.alt, model.ref, model.slope,
                                 torch.isfinite(model.log_prior), centers, weights,
                                 enrichment, model.eps)
    joint = np.full((4, 2, 4), -np.inf)
    for i, a in enumerate([1, 2, 3, 4]):
        for k in range(2):
            for m in range(a):
                p = np.clip(float(model.slope[i, m]*centers[k]), model.eps, 1-model.eps)
                q = (1-float(enrichment[k]))/a + float(enrichment[k])*(m == 0)
                joint[i, k, m] = (float(model.alt[i])*np.log(p) + float(model.ref[i])*np.log1p(-p)
                                   + np.log(q) + np.log(float(weights[k])))
    norm = logsumexp(joint, axis=(1, 2))
    assert float(ll) == pytest.approx(norm.sum(), abs=1e-12)
    np.testing.assert_allclose(post.numpy(), np.exp(joint-norm[:, None, None]), atol=1e-14)
    np.testing.assert_allclose(prior.sum(-1), np.ones((4, 2)), atol=1e-15)
    assert torch.equal(prior[0, :, 0], torch.ones(2, dtype=torch.float64))


def test_uniform_branch_exactly_nests_existing_likelihood():
    model = model_from_counts([0, 25, 35, 99], 100, [1, 2, 3, 4])
    center = torch.tensor([.65], dtype=torch.float64)
    ll, post, _ = expectation(model.alt, model.ref, model.slope,
                             torch.isfinite(model.log_prior), center, torch.ones_like(center),
                             torch.zeros_like(center), model.eps)
    assert float(ll) == pytest.approx(-float(model.loss(center.expand(4)).sum()), abs=1e-12)
    torch.testing.assert_close(post[:, 0], model.terms(center.expand(4))[3])


def test_center_mstep_against_independent_scalar_optimization():
    model = model_from_counts([0, 25, 35, 99], 100, [1, 2, 3, 4])
    center = torch.tensor([.2, .8], dtype=torch.float64)
    _, post, _ = expectation(model.alt, model.ref, model.slope,
                             torch.isfinite(model.log_prior), center, center.new_tensor([.3, .7]),
                             center.new_tensor([.4, .2]), model.eps)
    result = maximize_centers(model.alt, model.ref, model.slope, post,
                              model.lower.max(), model.upper.min(), model.eps)
    for k in range(2):
        def loss(x):
            p = np.clip(model.slope.numpy()*x, model.eps, 1-model.eps)
            return -(post[:, k].numpy()*(model.alt.numpy()[:, None]*np.log(p)
                                         +model.ref.numpy()[:, None]*np.log1p(-p))).sum()
        oracle = minimize_scalar(loss, bounds=(1e-6, 1.), method="bounded",
                                 options={"xatol": 1e-13})
        assert loss(float(result[k])) <= oracle.fun + 1e-8


@pytest.mark.parametrize('major', [[1, 1, 1, 1], [1, 2, 3, 4]])
def test_blocked_center_step_is_exactly_the_original_36_updates(major):
    from clipp1d.cuda.mixture import (_center_statistics, _center_bisection_block,
                                     _center_endpoints, slope_groups)
    model = model_from_counts([0, 8, 28, 100], 100, major)
    for k in (1, 3):
        center = torch.linspace(.2, .9, k, dtype=torch.float64)
        _, posterior, _ = expectation(model.alt, model.ref, model.slope,
            torch.isfinite(model.log_prior), center, torch.ones_like(center)/k,
            torch.zeros_like(center), model.eps)
        lower, upper = model.lower.max(), model.upper.min()
        expected = maximize_centers(model.alt, model.ref, model.slope, posterior, lower, upper, model.eps)
        values, order, lengths = slope_groups(model.slope)
        successes, grouped = _center_statistics(model.alt, model.ref, posterior, order, lengths)
        lo, hi = lower.expand(k).clone(), upper.expand(k).clone()
        for _ in range(6):
            lo, hi = _center_bisection_block(successes, grouped, values, lo, hi, model.eps)
        actual = _center_endpoints(model.alt, model.ref, model.slope, posterior, lower, upper, lo, hi, model.eps)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_score_dimensions_and_shrinkage_are_explicit():
    objective, score = criterion(torch.tensor(-100.), torch.tensor([.2, .5]), 100, 2, True, 1.)
    assert float(objective) == pytest.approx(100-np.log(.8)-np.log(.5))
    assert float(score) == pytest.approx(2*float(objective)+5*np.log(100))


def test_soft_selection_recovers_weak_split_and_em_never_worsens_objective():
    rng = np.random.default_rng(1851)
    phi = np.repeat([.45, .95], 350)
    model = model_from_counts(rng.binomial(60, .35*phi), 60, np.ones(700, dtype=int))
    result = fit_mixture_reference(model, MixturePolicy(max_clusters=3))
    public = result.public_arrays(model.mutation_ids)
    assert len(public['centers']) == 2
    np.testing.assert_allclose(np.sort(public['centers'].numpy()), [.45, .95], atol=.035)
    assert float(public['centers'][0]) > float(public['centers'][1])
    for record in result.records:
        assert np.diff(record['objective_history']).max() < 1e-8
    assert result.metadata()['raw_certificate_inherited'] is False
    assert result.metadata()['production_adopted'] is False


def test_single_cluster_control_stays_single():
    rng = np.random.default_rng(117)
    model = model_from_counts(rng.binomial(100, .35*.75, 400), 100, np.ones(400, dtype=int))
    result = fit_mixture_reference(model, MixturePolicy(max_clusters=3))
    assert result.centers.numel() == 1
    assert result.adaptive is False
    assert abs(float(result.centers[0])-.75) < .02


@pytest.mark.parametrize('uniform', [False, True])
def test_learned_enrichment_distinguishes_two_generation_mechanisms(uniform):
    rng = np.random.default_rng(9831)
    major = np.repeat([1, 2, 3, 4], [30, 250, 250, 250])
    multiplicity = rng.integers(1, major+1) if uniform else np.ones_like(major)
    p = .7*.83*multiplicity / (.6 + .7*(major+1))
    model = model_from_counts(rng.binomial(250, p), 250, major)
    result = fit_mixture_reference(model, MixturePolicy(max_clusters=1),
                                    seed_centers=torch.tensor([.85], dtype=torch.float64))
    assert float(result.centers[0]) == pytest.approx(.83, abs=.025)
    if uniform:
        assert result.adaptive is False
        assert float(result.enrichment[0]) == 0
    else:
        assert result.adaptive is True
        assert float(result.enrichment[0]) > .95


def test_budget_exhaustion_and_cuda_requirement_do_not_claim_qualification():
    model = model_from_counts([10, 25, 20, 35], 100, [1, 2, 3, 4])
    result = fit_mixture_reference(model, MixturePolicy(max_clusters=2, max_iterations=1))
    assert result.status == 'iteration_budget_exhausted'
    assert not result.metadata()['selected_parameters_converged']
    with pytest.raises(ValueError, match="compiled PyTorch CUDA"):
        fit_mixture(model)
    with pytest.raises(ValueError, match="uniform"):
        changed = replace(model, log_prior=model.log_prior + .1)
        fit_mixture_reference(changed)


def test_source_identity_and_bad_policy_rejected():
    for args in ({'max_clusters': True}, {'max_clusters': 21}, {'tolerance': float('nan')},
                 {'enrichment_shrinkage': 0}, {'adaptive_multiplicity': 1}):
        with pytest.raises(ValueError):
            MixturePolicy(**args)
    model = model_from_counts([10, 25], 100, [1, 2])
    model.alt.data[0] = 11
    with pytest.raises(ValueError, match="modified"):
        fit_mixture_reference(model)
