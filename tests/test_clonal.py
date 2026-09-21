import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d.chain import build_chain
from clipp1d.io import read_tumor
from clipp1d.model import compile_model
from clipp1d.policy import Policy
from clipp1d.clonal import fit_fixed_lambda
from clipp1d.scalar import compute_pilot
from clipp1d.solver import _kink_check, objective, solve_branch, stationarity
from clipp1d.types import ClonalConstraintInfeasibleError
from conftest import count_model


def test_best_witness_not_highest_pilot_and_release():
    model = count_model([1, 300], [4, 700])
    p = compute_pilot(model)
    c = build_chain(p, model.mutation_ids)
    raw = fit_fixed_lambda(model, c, p, 0)
    assert p.phi[0] < p.phi[1] and raw.witness == 0
    next_branch = solve_branch(model.subset(c.order), c, 0, 1, raw.x)
    assert next_branch.qualified and next_branch.x[1] == 1
    assert_allclose(next_branch.x[0], .5, atol=2e-5)


def test_original_bounds_and_screening():
    model = count_model([10, 40], [90, 60])
    p = compute_pilot(model)
    c = build_chain(p, model.mutation_ids)
    weights = c.weights.copy()
    raw = fit_fixed_lambda(model, c, p, .01)
    assert raw.qualified and raw.diagnostics["witnesses_screened"] == 1
    for i in range(2):
        direct = solve_branch(model, c, .01, i, p.phi)
        assert raw.objective <= direct.objective + 1e-7
    assert np.array_equal(weights, c.weights)
    no_clonal = count_model([10], [90], upper=[1 - 1e-15])
    pilot = compute_pilot(no_clonal)
    with pytest.raises(ClonalConstraintInfeasibleError):
        fit_fixed_lambda(no_clonal, build_chain(pilot, no_clonal.mutation_ids), pilot, 0)


def test_clipped_plateau_is_escaped_before_certification():
    model = count_model([10, 40], [90, 60])
    p = compute_pilot(model)
    c = build_chain(p, model.mutation_ids)
    raw = solve_branch(model, c, 0, 1, np.array([model.eps, 1]))
    assert raw.qualified
    assert_allclose(raw.x, [.25, 1], atol=2e-5)


def test_zero_alt_exact_clipping_kink_branch():
    model = count_model([0, 40], [100, 60])
    p = compute_pilot(model)
    c = build_chain(p, model.mutation_ids)
    raw = solve_branch(model, c, .1, 1, p.phi)
    assert raw.qualified and raw.diagnostics["stationarity_residual"] < 2e-5
    assert raw.x[0] == model.eps / model.slope[0, 0]
    assert raw.x[1] == 1
    profiled = fit_fixed_lambda(model, c, p, .1)
    assert profiled.witness == 1 and profiled.diagnostics["witness_search_complete"]


def test_fused_clonal_block_cannot_hide_proper_interval_descent(make_input):
    rows = [{"mutation_id": "m0", "alt_count": 1, "ref_count": 9},
            {"mutation_id": "m1", "alt_count": 1, "ref_count": 1, "allele_b_cn": 0},
            {"mutation_id": "m2", "alt_count": 1, "ref_count": 1, "allele_b_cn": 0},
            {"mutation_id": "m3", "alt_count": 1, "ref_count": 0}]
    for row in rows:
        row["purity"] = .99999949999975
    model = compile_model(read_tumor(make_input(rows)))
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    model = model.subset(chain.order)
    x, caps = np.ones(4), 1e6 * chain.weights
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[0] = upper[0] = 1
    trial = np.array([1., 1 - 1e-7, 1 - 1e-7, 1.])
    assert objective(model, trial, caps) < objective(model, x, caps) - .1
    residual, feasible = stationarity(model, x, np.zeros(3), lower, upper, caps)
    assert feasible and residual > Policy().stationarity_tol
    accepted, restart = _kink_check(model, x, caps, lower, upper, Policy())
    assert not accepted
    if restart is not None:
        assert restart[0] == 1 and objective(model, restart, caps) < objective(model, x, caps)
    raw = solve_branch(model, chain, 1e6, 0, x)
    assert not (raw.qualified and np.array_equal(raw.x, x))


def test_lower_clipping_endpoint_uses_feasible_right_derivative():
    model = count_model([10], [90], slope=1.)
    residual, feasible = stationarity(model, model.lower, np.array([]),
                                      model.lower, model.upper, np.array([]))
    assert feasible and residual > .99
