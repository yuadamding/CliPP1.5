from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d.chain import build_chain
from clipp1d.policy import Policy
from clipp1d.scalar import _minimize_general, compute_pilot, minimize_block
from clipp1d.selection import refit_partition, select_fit
from clipp1d.solver import interval_descent, local_interval_delta, objective, solve_branch
from clipp1d.types import NumericalQualificationError, WarmState
from conftest import count_model


@pytest.mark.parametrize("tolerance", [0., 2e-5])
@pytest.mark.parametrize("seed", range(8))
def test_linear_interval_scan_matches_exhaustive_oracle(seed, tolerance):
    rng = np.random.default_rng(seed)
    x = np.array([0., 0, .3, .3, .3, 1., 1., 1.])
    lower, upper = np.zeros(8), np.ones(8)
    lower[3] = upper[3] = x[3]
    lower[6] = upper[6] = x[6]
    left, right, caps = rng.normal(size=8), rng.normal(size=8), rng.uniform(0, 2, 7)
    candidates = []
    for a in range(len(x)):
        for b in range(a + 1, len(x) + 1):
            if np.ptp(x[a:b]) != 0:
                continue
            for sign in (-1, 1):
                room = x[a:b] - lower[a:b] if sign < 0 else upper[a:b] - x[a:b]
                if np.any(room <= 0):
                    continue
                delta = np.zeros(len(x))
                delta[a:b] = sign
                jump, change = np.diff(x), np.diff(delta)
                penalty = np.where(jump == 0, caps * np.abs(change), caps * np.sign(jump) * change).sum()
                nodes = -left[a:b] if sign < 0 else right[a:b]
                cost = nodes.sum() + penalty
                scale = 1 + np.abs(nodes).sum() + np.sum(caps * np.abs(change))
                candidates.append((cost + tolerance * scale, cost))
    expected = min(candidates)
    found = interval_descent(x, left, right, lower, upper, caps, tolerance)
    if expected[0] >= 0:
        assert found is None
    else:
        assert found is not None
        assert_allclose(found[3] + tolerance * found[4], expected[0], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("start,stop", [(0, 1), (2, 4), (4, 5), (0, 5)])
def test_local_delta_matches_full_objective_without_full_evaluation(monkeypatch, start, stop):
    import clipp1d.solver as solver
    model = count_model([2, 8, 10, 20, 30], [98, 92, 90, 80, 70])
    x, caps = np.linspace(.1, .9, 5), np.array([.4, 2., 3., 1.])
    trial = x.copy()
    trial[start:stop] = .33
    expected = objective(model, trial, caps) - objective(model, x, caps)
    sizes = []
    original = solver.loss

    def counted(subset, *args, **kwargs):
        sizes.append(len(subset))
        return original(subset, *args, **kwargs)

    monkeypatch.setattr(solver, "loss", counted)
    delta = local_interval_delta(model.subset(np.arange(start, stop)), x, caps, start, stop, .33)
    assert_allclose(delta, expected, atol=1e-12)
    assert sizes and max(sizes) == stop - start


@pytest.mark.parametrize("alt,ref,slope", [([0], [10], .4), ([10], [0], 2.),
                                          ([10], [90], .4), ([10, 30], [90, 170], .4)])
def test_exact_scalar_paths_agree_with_general_search(alt, ref, slope):
    model = count_model(alt, ref, slope)
    fast, general = minimize_block(model), _minimize_general(model)
    assert fast.qualified and general.qualified
    assert fast.method == "same_slope_single_candidate_exact" and fast.evaluations == 1
    assert_allclose(fast.attained_loss, general.attained_loss, atol=1e-7, rtol=0)
    assert_allclose(fast.argmin, general.argmin, atol=1e-6, rtol=0)
    if alt == [0]:
        assert fast.argmin == model.lower[0]
    if ref == [0]:
        assert fast.argmin == (1 - model.eps) / slope


def test_lazy_search_does_not_evaluate_every_distinct_breakpoint():
    n = 512
    model = count_model(np.zeros(n), np.full(n, 100), np.linspace(.1, .5, n)[:, None])
    result = minimize_block(model)
    assert result.qualified and result.argmin == model.lower[0]
    assert result.method == "interval_search"  # heterogeneous slopes cannot be pooled
    assert result.evaluations < 100 and result.bound_evaluations < 100


def test_singleton_refits_reuse_only_source_bound_pilots(monkeypatch):
    import clipp1d.selection as selection
    model = count_model([10, 30, 40], [90, 70, 60])
    pilot = compute_pilot(model)
    original = selection.minimize_block
    calls = []

    def counted(block, *args, **kwargs):
        calls.append(block.mutation_ids)
        return original(block, *args, **kwargs)

    monkeypatch.setattr(selection, "minimize_block", counted)
    result = refit_partition(model, (0, 1, 2, 3), pilot=pilot)
    assert not calls and result.singleton_pilots_reused == 3
    changed = count_model([11, 30, 40], [89, 70, 60])
    result = refit_partition(changed, (0, 1, 2, 3), pilot=pilot)
    assert calls == [("m0",)] and result.singleton_pilots_reused == 2


def test_direct_primal_continuation_and_witness_release(monkeypatch):
    import clipp1d.solver as solver
    model = count_model([1, 300], [4, 700])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    old = WarmState(np.ones(2), np.array([100.]), chain.fingerprint)
    original = solver.solve_quadratic
    seen = []

    def inspect(*args, **kwargs):
        if not seen:
            seen.append((kwargs.get("dual"), args[2].copy(), args[3].copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "solve_quadratic", inspect)
    warm = solve_branch(model, chain, .05, 1, old)
    cold = solve_branch(model, chain, .05, 1, pilot.phi)
    assert warm.qualified and cold.qualified
    assert seen[0][0] is None
    assert seen[0][1][0] == model.lower[0] and seen[0][2][0] == model.upper[0]
    assert warm.x[0] < .6 and warm.x[1] == 1
    assert_allclose(warm.x, cold.x, atol=5e-5)
    assert_allclose(warm.objective, cold.objective, atol=1e-7)
    with pytest.raises(ValueError, match="frozen chain"):
        solve_branch(model, chain, .05, 1, replace(old, chain_fingerprint="wrong"))


def test_raw_diagnostics_survive_refit_failure(monkeypatch, make_input, tmp_path):
    import json
    import legacy_chain_api as api
    import clipp1d.selection as selection
    model = count_model([10, 40], [90, 60])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    original = selection.refit_partition

    def fail_one_block(model, cuts, *args, **kwargs):
        if len(cuts) == 2:
            raise NumericalQualificationError("deliberate refit exhaustion", scalar_gap=1.)
        return original(model, cuts, *args, **kwargs)

    monkeypatch.setattr(selection, "refit_partition", fail_one_block)
    policy = replace(Policy(), path_min_exponent=4, path_max_exponent=4, path_extensions=0)
    _, _, _, search = select_fit(model, chain, pilot, policy)
    assert search["search_status"] == "incomplete"
    failed = search["path"][1]
    assert failed["raw_status"] == "qualified" and failed["refit_status"] == "unresolved"
    assert failed["raw_diagnostics"]["raw_branch_stationarity_qualified"]
    assert np.isfinite(failed["raw_objective"]) and failed["raw_witness_mutation_id"] is None
    assert search["raw_unresolved_penalties"] == 0 and search["refit_unresolved_penalties"] == 1
    monkeypatch.setattr(api, "Policy", lambda **kwargs: policy)
    path = make_input([dict(alt_count=10, ref_count=90), dict(alt_count=40, ref_count=60)])
    result = api.fit(path, tmp_path / "incomplete")
    receipt = json.loads((tmp_path / "incomplete" / "run.json").read_text())
    assert result.search_status == receipt["search_status"] == "incomplete"
    assert receipt["status"] == "success"
    assert receipt["search"]["path"][1]["raw_status"] == "qualified"
    assert receipt["search"]["path"][1]["refit_status"] == "unresolved"
