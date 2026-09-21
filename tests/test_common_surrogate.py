"""Production profiles the union constraint on each shared MM quadratic."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d import clonal, solver
from clipp1d.chain import build_chain
from clipp1d.policy import Policy
from clipp1d.scalar import compute_pilot
from clipp1d.types import WarmState
from conftest import count_model


@pytest.fixture
def reference():
    path = Path(__file__).parents[1] / "benchmarks" / "reference_enumeration.py"
    spec = importlib.util.spec_from_file_location("reference_enumeration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiled_outer_releases_witness_and_keeps_original_boxes(monkeypatch):
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    ordered = model.subset(chain.order)
    original = solver.profile_quadratic_witnesses
    calls = []

    def checked(h, target, lower, upper, caps, *args, **kwargs):
        assert_allclose(lower, ordered.lower, atol=0, rtol=0)
        assert_allclose(upper, ordered.upper, atol=0, rtol=0)
        result = original(h, target, lower, upper, caps, *args, **kwargs)
        calls.append(result.witness)
        return result

    monkeypatch.setattr(solver, "profile_quadratic_witnesses", checked)
    start = WarmState(np.array([1., .75]), np.array([1e100]), chain.fingerprint)
    initial_objective = solver.objective(ordered, start.x, .1 * chain.weights)
    result = solver.solve_profiled(ordered, chain, .1, start)
    assert result.qualified and result.witness == 1
    assert calls and 1 in calls
    assert result.x[0] < .3 and result.x[1] == 1
    assert result.objective < initial_objective
    assert_allclose(ordered.lower, model.lower, atol=0, rtol=0)


def test_backtracking_rebuilds_shared_profile_from_original_boxes(monkeypatch):
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    original_profile, original_evaluate = solver.profile_quadratic_witnesses, solver.evaluate
    quadratics = []
    injection = []

    def profile(h, target, lower, upper, caps, *args, **kwargs):
        assert_allclose(lower, model.lower, atol=0, rtol=0)
        assert_allclose(upper, model.upper, atol=0, rtol=0)
        quadratics.append((h.copy(), target.copy()))
        return original_profile(h, target, lower, upper, caps, *args, **kwargs)

    def evaluate(block, phi, *args, **kwargs):
        terms = original_evaluate(block, phi, *args, **kwargs)
        if len(quadratics) == 1 and not kwargs.get("derivatives", False):
            injection.append(True)
            # Reject the first surrogate's trials, including any breakpoint
            # or fusion proposals evaluated before the acceptance gate.
            # The next quadratic must be rebuilt with increased curvature.
            return replace(terms, loss=terms.loss + 1e8)
        return terms

    monkeypatch.setattr(solver, "profile_quadratic_witnesses", profile)
    monkeypatch.setattr(solver, "evaluate", evaluate)
    result = solver.solve_profiled(model, chain, .1, np.array([1., .75]))
    assert injection and len(quadratics) >= 2
    assert_allclose(quadratics[1][0], 2 * quadratics[0][0], atol=0, rtol=0)
    assert not np.array_equal(quadratics[0][1], quadratics[1][1])
    assert result.diagnostics["backtracks"] >= 1
    assert result.qualified


def test_positive_penalty_production_uses_profiles_without_branch_enumeration(monkeypatch):
    model = count_model([10, 20, 30], [90, 80, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    original = solver.profile_quadratic_witnesses
    calls = []

    def checked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Production must not enumerate nonlinear witness branches")

    monkeypatch.setattr(solver, "profile_quadratic_witnesses", checked)
    monkeypatch.setattr(solver, "solve_branch", forbidden)
    if hasattr(clonal, "solve_branch"):
        monkeypatch.setattr(clonal, "solve_branch", forbidden)
    result = clonal.fit_fixed_lambda(model, chain, pilot, .1)
    assert result.qualified and calls
    assert result.diagnostics["search_policy"] == "common_surrogate_multistart_v1"
    assert 1 <= result.diagnostics["starts_attempted"] <= 4
    assert result.diagnostics["search_complete"]
    assert "witnesses_solved" not in result.diagnostics
    assert not result.diagnostics["global_optimality_proven"]


@pytest.mark.parametrize("dual", [1e100, -1e100])
def test_production_equal_primals_do_not_add_direct_backend_starts(dual):
    model = count_model([20, 20], [80, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    warm = WarmState(pilot.phi[chain.order], np.array([dual]), chain.fingerprint)
    result = clonal.fit_fixed_lambda(model, chain, pilot, .05, warm)
    assert result.qualified
    assert result.diagnostics["starts_attempted"] == 1
    assert result.diagnostics["search_complete"]


@pytest.mark.parametrize("penalty", [.01, .1, 2., 100.])
def test_convex_tiny_attained_objective_matches_offline_enumeration(reference, penalty):
    model = count_model([10, 20, 30], [90, 80, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    expected = reference.fit_fixed_lambda(model, chain, pilot, penalty)
    actual = clonal.fit_fixed_lambda(model, chain, pilot, penalty)
    assert actual.qualified and expected.qualified
    assert_allclose(actual.objective, expected.objective, atol=2e-7, rtol=0)
    assert_allclose(actual.x, expected.x, atol=2e-5, rtol=0)


def test_nonstationary_budget_exhaustion_is_not_profiled_qualification():
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    result = solver.solve_profiled(model, chain, .1, np.array([1., .75]),
                                   replace(Policy(), outer_max_iterations=1))
    assert not result.qualified
    assert not result.diagnostics["raw_branch_stationarity_qualified"]
    assert not result.diagnostics["global_optimality_proven"]


def test_other_occupied_witness_exposes_smooth_union_descent():
    # At [1, 1], forcing node 0 is branch-stationary: node 1 would prefer
    # CCF > 1. Nevertheless node 0 can decrease if node 1 is the witness.
    # Neither node is at a probability clipping kink or plateau.
    model = count_model([10, 50], [90, 50])
    x, caps = np.ones(2), np.array([.1])
    first_lower, first_upper = model.lower.copy(), model.upper.copy()
    first_lower[0] = first_upper[0] = 1.
    residual, feasible = solver.stationarity(model, x, np.zeros(1), first_lower, first_upper, caps)
    assert feasible and residual == 0
    second_lower, second_upper = model.lower.copy(), model.upper.copy()
    second_lower[1] = second_upper[1] = 1.
    qualified, restart = solver._kink_check(model, x, caps, second_lower, second_upper,
                                           Policy(), force_intervals=True)
    assert not qualified and restart is not None
    assert restart[0] < 1 and restart[1] == 1
    assert solver.objective(model, restart, caps) < solver.objective(model, x, caps)
