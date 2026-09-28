"""Independent mathematical checks for the experimental prior, not CUDA evidence."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch

from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import LastPartitionCache, QualifiedPilot, refit_labels
from clipp1d.cuda.scalar import pilot

SPEC = importlib.util.spec_from_file_location('prior_experiment', Path(__file__).parents[1]/'benchmarks/prior_perturbation.py')
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def fixture_model(device='cpu', compiled=False):
    def tensor(x):
        return torch.tensor(x, dtype=torch.float64, device=device)
    support = np.array([1, 2, 3, 4])
    valid = np.arange(1, 5)[None, :] <= support[:, None]
    slope = np.array([.5, .4, .3, .28])[:, None] * np.arange(1, 5)[None, :]
    slope[~valid] = 0
    return TensorModel(tuple('abcd'), tensor([20, 120, 0, 400]), tensor([80, 280, 100, 0]),
        tensor(slope), tensor(np.where(valid, -np.log(support[:, None]), -np.inf)),
        tensor([1e-6]*4), tensor([1., 1., 1., (1-1e-6)/1.12]), 1e-6, Kernels(device, compiled))


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def independent_terms(model, point):
    a, r, s, lp = (getattr(model, k).cpu().numpy() for k in ('alt', 'ref', 'slope', 'log_prior'))
    p = np.clip(s*np.asarray(point)[:, None], model.eps, 1-model.eps)
    joint = a[:, None]*np.log(p)+r[:, None]*np.log1p(-p)+lp
    normalizer = logsumexp(joint, axis=-1)
    return -normalizer, np.exp(joint-normalizer[:, None])


@pytest.mark.parametrize('eta', [0., .01, .03, .05])
def test_support_normalization_odds_singletons_and_exact_zero(eta):
    model = fixture_model()
    xi = torch.tensor(experiment.keyed_directions('tumor', model.mutation_ids, 4)[0])
    altered = experiment.auxiliary_model(model, xi, eta)
    valid = torch.isfinite(model.log_prior)
    if eta == 0:
        assert altered is model
        assert experiment.perturb_log_prior(model.log_prior, xi, 0) is model.log_prior
    else:
        assert altered is not model
    assert torch.equal(torch.isfinite(altered.log_prior), valid)
    assert torch.equal(altered.log_prior[0], model.log_prior[0])
    torch.testing.assert_close(altered.log_prior.exp().sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-15, rtol=0)
    for i in range(4):
        delta = (altered.log_prior[i]-model.log_prior[i])[valid[i]]
        assert float(delta.max()-delta.min()) <= 2*eta+1e-15
    points = torch.tensor([.4, .5, .3, .7], dtype=torch.float64)
    assert model.loss(points)[0] == altered.loss(points)[0]


def test_common_offsets_and_row_permutation_are_invariant():
    m = fixture_model()
    noise = experiment.keyed_directions('T:a', m.mutation_ids, 4)
    order = [2, 0, 3, 1]
    permuted = experiment.keyed_directions('T:a', [m.mutation_ids[i] for i in order], 4)
    np.testing.assert_array_equal(noise[:, order], permuted)
    np.testing.assert_array_equal(noise[::2], -noise[1::2])
    assert not np.array_equal(noise[0], noise[2])
    xi = torch.tensor(noise[0])
    before = experiment.perturb_log_prior(m.log_prior, xi, .03)
    after = experiment.perturb_log_prior(m.log_prior, xi+torch.tensor([1., -2., 8., 0.])[:, None], .03)
    torch.testing.assert_close(before, after, atol=1e-15, rtol=0)


@pytest.mark.parametrize('eta', [.01, .03, .05])
def test_likelihood_identity_bounds_breakpoints_and_posteriors(eta):
    m = fixture_model()
    xi = torch.tensor(experiment.keyed_directions('identity', m.mutation_ids, 4)[1])
    aux = experiment.auxiliary_model(m, xi, eta)
    rng = np.random.default_rng(751)
    points = [m.lower.numpy(), m.upper.numpy()]
    points.extend(m.lower.numpy()+(m.upper-m.lower).numpy()*rng.random(4) for _ in range(8))
    for i in range(4):
        for slope in m.slope[i].numpy():
            if slope <= 0:
                continue
            for kink in (m.eps/slope, (1-m.eps)/slope):
                if m.lower[i] <= kink <= m.upper[i]:
                    point = (m.lower+m.upper).numpy()/2
                    point[i] = kink
                    points.append(point)
    for point in points:
        loss0, q0 = independent_terms(m, point)
        loss1, q1 = independent_terms(aux, point)
        tilt = eta*xi.numpy()
        with np.errstate(divide='ignore'):
            identity = logsumexp(m.log_prior.numpy()+tilt, axis=-1)-logsumexp(np.log(q0)+tilt, axis=-1)
        np.testing.assert_allclose(loss1-loss0, identity, atol=1e-10, rtol=1e-12)
        assert np.all(np.abs(loss1-loss0) <= 2*eta+1e-10)
        actual = aux.terms(torch.tensor(point))
        np.testing.assert_allclose(actual[0].numpy(), loss1, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(actual[3].numpy(), q1, atol=2e-13, rtol=1e-12)


def test_gradient_and_one_sided_kink_match_independent_evaluator():
    m = fixture_model()
    aux = experiment.auxiliary_model(m, torch.tensor(experiment.keyed_directions('derivative', m.mutation_ids, 4)[0]), .03)
    point = np.array([.4, .55, .37, .67])
    _, grad = aux.loss_gradient(torch.tensor(point))
    h = 1e-6
    numerical = (independent_terms(aux, point+h)[0]-independent_terms(aux, point-h)[0])/(2*h)
    np.testing.assert_allclose(grad.numpy(), numerical, atol=2e-6, rtol=1e-7)
    # Moderate non-extreme counts make represented clipping derivatives measurable.
    aux = replace(aux, alt=torch.ones(4, dtype=torch.float64)*20, ref=torch.ones(4, dtype=torch.float64)*80)
    kink = (1-aux.eps)/float(aux.slope[3, 3])
    point[3] = kink
    _, posterior = independent_terms(aux, point)
    s = aux.slope.numpy()
    p = np.clip(s*point[:, None], aux.eps, 1-aux.eps)
    score = aux.alt.numpy()[:, None]/p-aux.ref.numpy()[:, None]/(1-p)
    inside = (s*point[:, None] > aux.eps) & (s*point[:, None] < 1-aux.eps)
    left = inside.copy()
    right = inside.copy()
    left[3, 3] = True
    right[3, 3] = False
    expected_left = -(posterior*s*score*left).sum(-1)
    expected_right = -(posterior*s*score*right).sum(-1)
    for direction, reference in [(-1, expected_left), (1, expected_right)]:
        nearby = point.copy()
        nearby[3] += direction*1e-9
        delta = (independent_terms(aux, nearby)[0][3]-independent_terms(aux, point)[0][3])/(direction*1e-9)
        assert delta == pytest.approx(reference[3], abs=1e-3, rel=1e-4)


def test_immutable_models_and_cross_prior_certificates_rejected():
    m = fixture_model()
    aux = experiment.auxiliary_model(m, torch.tensor(experiment.keyed_directions('cache', m.mutation_ids, 4)[0]), .03)
    assert experiment.model_identity(m) != experiment.model_identity(aux)
    p = QualifiedPilot(aux, pilot(aux))
    cache = LastPartitionCache(aux)
    labels = torch.arange(4)
    with pytest.raises(ValueError, match='canonical model'):
        refit_labels(m, labels, pilot_reuse=p)
    with pytest.raises(ValueError, match='canonical model'):
        refit_labels(m, labels, cache=cache)
    aux.log_prior.data[0, 0] += .1
    with pytest.raises(ValueError, match='modified'):
        aux.validate(full=True)
    m.validate(full=True)


def test_alias_switch_is_a_mode_check_not_accuracy():
    m = fixture_model().subset(torch.tensor([1]))
    wells = []
    for xi in ([-1., 1., 0., 0.], [1., -1., 0., 0.]):
        aux = experiment.auxiliary_model(m, torch.tensor([xi], dtype=torch.float64), .01)
        fit = []
        for bounds in ((.28, .49), (.6, .9)):
            result = minimize_scalar(lambda value: independent_terms(aux, [value])[0][0], bounds=bounds,
                                     method='bounded', options={'xatol': 1e-14})
            fit.append((result.fun, result.x))
        wells.append(min(fit)[1])
    assert wells[0] == pytest.approx(.375, abs=1e-5)
    assert wells[1] == pytest.approx(.75, abs=1e-5)


def test_dedup_farthest_ties_and_penalty_boundaries():
    def v(x):
        return torch.tensor(x, dtype=torch.float64)
    starts = [(4, v([0, 1])), (1, v([1, 0])), (2, v([1, 0])+1e-9), (3, v([0, 0]))]
    assert [i for i, _ in experiment.distinct_starts(starts, v([0, 0]))] == [1, 4]
    records = [dict(lambda_value=x) for x in (0., 1., 2., 4., 8., 16.)]
    assert experiment.penalty_neighborhood(records, 0) == [1., 2., 4.]
    assert experiment.penalty_neighborhood(records, 4) == [2., 4., 8.]
    assert experiment.penalty_neighborhood(records, 16) == [4., 8., 16.]
    assert experiment.penalty_neighborhood(records[:2], 0) == [1.]
    assert experiment.penalty_neighborhood(records[:1], 0) == []
