"""The offline nonlinear policy reference shares the production branch solver."""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d import solver
from clipp1d.chain import build_chain
from clipp1d.scalar import compute_pilot
from clipp1d.types import PrimalWarmState, WarmState
from conftest import count_model


@pytest.fixture
def reference():
    path = Path(__file__).parents[1] / "benchmarks" / "reference_enumeration.py"
    spec = importlib.util.spec_from_file_location("reference_enumeration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_primal_starts_ignore_dual_and_deduplicate_after_witness(reference):
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    left = WarmState(np.array([.25, .8]), np.array([1e100]), chain.fingerprint)
    right = WarmState(np.array([.25, .3]), np.array([-1e100]), chain.fingerprint)
    starts = reference.projected_primal_starts(
        (left, right, np.array([.25, 1.])), model.lower, model.upper, 1,
    )
    assert len(starts) == 1
    assert_allclose(starts[0], [.25, 1.], atol=0, rtol=0)
    assert isinstance(starts[0], np.ndarray)
    assert_allclose(left.x, [.25, .8], atol=0, rtol=0)


def test_reference_same_primal_different_dual_counts_once(monkeypatch, reference):
    model = count_model([20, 20], [80, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    original = solver.solve_branch
    seen = []

    def counted(*args, **kwargs):
        seen.append((args[3], args[4].copy()))
        assert isinstance(args[4], np.ndarray)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "solve_branch", counted)
    warm = WarmState(pilot.phi[chain.order], np.array([1e100]), chain.fingerprint)
    result = reference.fit_fixed_lambda(model, chain, pilot, .05, warm)
    assert result.qualified
    assert result.diagnostics["search_policy"] == reference.SEARCH_POLICY
    assert result.diagnostics["starts_attempted"] == len(seen) == 2
    assert [witness for witness, _ in seen] == [0, 1]
    assert result.diagnostics["witness_search_complete"]


def test_reference_releases_previous_witness(monkeypatch, reference):
    model = count_model([20, 20], [80, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    original = solver.solve_branch
    released = []

    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == 1:
            released.append(result)
        return result

    monkeypatch.setattr(solver, "solve_branch", checked)
    warm = WarmState(np.ones(2), np.array([1e100]), chain.fingerprint)
    result = reference.fit_fixed_lambda(model, chain, pilot, .05, warm)
    assert result.qualified and released
    assert all(branch.qualified and branch.x[0] < .6 and branch.x[1] == 1 for branch in released)


@pytest.mark.parametrize("penalty", [0., .1, 2.])
def test_reference_matches_explicit_convex_branch_minimum(reference, penalty):
    model = count_model([10, 20, 30], [90, 80, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    ordered = model.subset(chain.order)
    fits = [solver.solve_branch(ordered, chain, penalty, witness, pilot.phi[chain.order])
            for witness in range(len(model))]
    assert all(fit.qualified for fit in fits)
    result = reference.fit_fixed_lambda(model, chain, pilot, penalty)
    assert result.qualified
    assert_allclose(result.objective, min(fit.objective for fit in fits), atol=1e-7, rtol=0)
    assert result.diagnostics["search_policy"] == reference.SEARCH_POLICY
    assert result.diagnostics["witnesses_unresolved"] == 0


def test_reference_warm_start_requires_chain_identity(reference):
    model = count_model([10, 20], [90, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    warm = WarmState(pilot.phi, np.zeros(1), "wrong")
    with pytest.raises(ValueError, match="frozen chain"):
        reference.fit_fixed_lambda(model, chain, pilot, .1, warm)


def test_reference_accepts_primal_only_continuation(reference):
    model = count_model([10, 20], [90, 80])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    warm = PrimalWarmState(pilot.phi[chain.order], chain.fingerprint)
    result = reference.fit_fixed_lambda(model, chain, pilot, .1, warm)
    assert result.qualified
    assert result.diagnostics["search_policy"] == reference.SEARCH_POLICY
