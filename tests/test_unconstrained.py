"""Regression and independent-oracle checks for fitting without a clonal anchor."""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import minimize

from clipp1d.chain import build_chain
from clipp1d.fitting import fit_fixed_lambda
from clipp1d.policy import Policy
from clipp1d.scalar import compute_pilot
from clipp1d.selection import refit_partition, select_fit
from clipp1d.solver import solve_unconstrained
from clipp1d.types import NumericalQualificationError, PrimalWarmState
from conftest import count_model


@pytest.mark.parametrize('penalty', [0., .1, 2., 1000.])
def test_no_eligible_ccf_one_and_original_boxes(penalty):
    model = count_model([10, 20, 30], [90, 80, 70], upper=[.6, .7, .8])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    result = fit_fixed_lambda(model, chain, pilot, penalty)
    assert result.qualified and result.witness is None
    assert np.all(result.x >= model.lower[chain.order])
    assert np.all(result.x <= model.upper[chain.order])
    assert np.max(result.x) < 1
    assert result.diagnostics['box_feasible']
    assert result.diagnostics['clonal_constraint'] is False
    assert result.diagnostics['search_profile_calls'] == 0
    if penalty == 0:
        assert_allclose(result.x, [.25, .5, .75], rtol=0, atol=1e-14)


@pytest.mark.parametrize('penalty', [.1, 20., 1000.])
def test_convex_binomial_tv_matches_independent_epigraph_oracle(penalty):
    # SLSQP independently minimizes exact binomial loss with |difference|
    # epigraphs. Permuted input order exercises the frozen-chain permutation.
    model = count_model([35, 5, 25], [65, 95, 75], upper=[.95, .6, .8])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    ordered = model.subset(chain.order)
    p = pilot.phi[chain.order]
    caps = penalty * chain.weights
    n = len(model)

    def objective(z):
        probability = .4 * z[:n]
        return float(-np.sum(ordered.alt * np.log(probability) +
                             ordered.ref * np.log1p(-probability)) + caps @ z[n:])

    def gradient(z):
        probability = .4 * z[:n]
        return np.r_[-.4 * (ordered.alt / probability - ordered.ref / (1 - probability)), caps]

    def feasible(z):
        jumps = np.diff(z[:n])
        return np.r_[z[n:] - jumps, z[n:] + jumps]

    jacobian = np.vstack([np.c_[-np.diff(np.eye(n), axis=0), np.eye(n - 1)],
                          np.c_[np.diff(np.eye(n), axis=0), np.eye(n - 1)]])
    oracle = minimize(objective, np.r_[p, abs(np.diff(p))], jac=gradient,
                      bounds=list(zip(ordered.lower, ordered.upper)) + [(0, None)] * (n - 1),
                      method='SLSQP', constraints=[dict(type='ineq', fun=feasible, jac=lambda _: jacobian)],
                      options={'ftol': 1e-9, 'maxiter': 1000})
    assert oracle.success, oracle.message
    result = fit_fixed_lambda(model, chain, pilot, penalty)
    assert result.qualified
    assert_allclose(result.objective, oracle.fun, rtol=0, atol=2e-6)
    assert_allclose(result.x, oracle.x[:n], rtol=0, atol=3e-5)
    assert result.diagnostics['inner_gap_qualified']
    assert result.diagnostics['inner_kkt_qualified']


def test_start_at_one_is_free_to_move_down():
    model = count_model([10, 20], [90, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    result = solve_unconstrained(model, chain, .1, np.ones(2))
    assert result.qualified and np.max(result.x) < .6
    assert result.diagnostics['objective_decrease'] > 0
    assert result.witness is None


@pytest.mark.parametrize('start', [.5, [np.inf, .5], [-np.inf, .5], [np.nan, .5], [.5]])
def test_invalid_start_rejected_before_projection(start):
    model = count_model([10, 20], [90, 80])
    chain = build_chain(compute_pilot(model), model.mutation_ids)
    with pytest.raises(ValueError, match='finite correctly shaped'):
        solve_unconstrained(model, chain, .1, start)


def test_production_never_invokes_witness_profiling(monkeypatch):
    import clipp1d.clonal as legacy
    import clipp1d.solver as solver

    def forbidden(*args, **kwargs):
        pytest.fail('Unconstrained production reached a historical witness solver')

    monkeypatch.setattr(legacy, 'fit_fixed_lambda', forbidden)
    monkeypatch.setattr(solver, 'profile_quadratic_witnesses', forbidden)
    monkeypatch.setattr(solver, 'solve_profiled', forbidden)
    monkeypatch.setattr(solver, 'solve_branch', forbidden)
    model = count_model([10] * 8 + [30] * 8, [90] * 8 + [70] * 8)
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    policy = replace(Policy(), path_min_exponent=-1, path_max_exponent=1, path_extensions=0)
    _, raw, refit, _ = select_fit(model, chain, pilot, policy)
    assert_allclose(refit.centers, [.25, .75], atol=1e-12)
    assert refit.designated_clonal_block is raw.witness is None


def test_refit_can_naturally_reach_one_without_designation():
    model = count_model([50, 10], [50, 90])
    result = refit_partition(model, (0, 1, 2))
    assert result.designated_clonal_block is None
    assert_allclose(result.centers, [1, .25], atol=1e-12)


def test_pilot_gap_must_qualify_under_requested_policy():
    model = count_model([10], [90])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    stricter = replace(Policy(), scalar_atol=1e-20, scalar_rtol=0.)
    with pytest.raises(NumericalQualificationError, match='Separable pilots'):
        fit_fixed_lambda(model, chain, pilot, 0., policy=stricter)


def test_continuation_is_chain_bound_and_deduplicated():
    model = count_model([20, 20], [80, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    warm = PrimalWarmState(pilot.phi[chain.order], chain.fingerprint)
    result = fit_fixed_lambda(model, chain, pilot, .1, warm_start=warm)
    assert result.diagnostics['starts_attempted'] == 1
    assert result.diagnostics['starts_qualified'] == 1
    with pytest.raises(ValueError, match='frozen chain'):
        fit_fixed_lambda(model, chain, pilot, .1, replace(warm, chain_fingerprint='wrong'))
