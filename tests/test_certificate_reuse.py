from dataclasses import replace

import numpy as np

from clipp1d.chain import build_chain
from clipp1d.policy import Policy
from clipp1d.scalar import compute_pilot
from clipp1d import solver
from conftest import count_model


def test_unchanged_direct_step_reuses_gap_and_kkt_but_evaluates_likelihood(monkeypatch):
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    counts = dict(gap=0, kkt=0, likelihood=0, direct=0)
    for name, key in (("quadratic_gap", "gap"), ("quadratic_kkt", "kkt"),
                      ("evaluate", "likelihood"), ("solve_quadratic", "direct")):
        original = getattr(solver, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(solver, name, counted)
    result = solver.solve_profiled(model, chain, .01, np.array([.25, 1.]))
    assert result.qualified and counts["direct"] > 0
    assert counts["gap"] == counts["kkt"] == counts["direct"]
    assert counts["likelihood"] >= result.diagnostics["accepted_steps"]
    assert result.diagnostics["inner_certificates_reused"] == counts["direct"]


def test_failed_certificate_retains_specific_reason(monkeypatch):
    monkeypatch.setattr(solver, "quadratic_gap", lambda *args: (float("inf"), 0.))
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    result = solver.solve_profiled(model, chain, .01, np.array([.25, 1.]),
                                    replace(Policy(), outer_max_iterations=2))
    assert not result.qualified
    assert result.diagnostics["failure_reason"] == "quadratic_gap_gate"
    assert result.diagnostics["last_inner_work"]["failure_reason"] == "quadratic_gap_gate"


def test_interior_box_normal_does_not_invent_negative_gap():
    # Exact interior minimization has zero normal despite rounded subtraction.
    h = np.array([98.3, 134.91])
    target = np.array([.225, .765])
    caps = np.array([.05123456])
    q = caps.copy()
    x = target - np.array([-q[0], q[0]]) / h
    gap, _ = solver.quadratic_gap(x, q, h, target, np.zeros(2), np.ones(2), caps)
    assert np.isfinite(gap) and gap <= Policy().inner_atol


def test_common_iteration_reuses_only_curvature_scale_after_backtracking(monkeypatch):
    model = count_model([10, 30], [90, 70])
    pilot = compute_pilot(model)
    chain = build_chain(pilot, model.mutation_ids)
    real_evaluate, real_profile = solver.evaluate, solver.profile_quadratic_witnesses
    state = {"calls": 0, "outer": 0}
    seen = []

    def evaluate(*args, **kwargs):
        terms = real_evaluate(*args, **kwargs)
        if kwargs.get("derivatives", False):
            # stationarity also evaluates derivatives; profile ratios below still
            # bind to the latest current-primal curvature.
            state["base"] = np.maximum(terms.curvature, 1.)
        elif 0 < state["calls"] <= 2:
            return replace(terms, loss=terms.loss + 1e6)
        return terms

    def profile(*args, **kwargs):
        state["calls"] += 1
        seen.append(float(np.median(args[0] / state["base"])))
        return real_profile(*args, **kwargs)

    monkeypatch.setattr(solver, "evaluate", evaluate)
    monkeypatch.setattr(solver, "profile_quadratic_witnesses", profile)
    result = solver.solve_profiled(model, chain, .01, np.array([.9, 1.]))
    assert result.qualified
    assert seen[:4] == [1., 2., 4., 2.]
    assert result.diagnostics["largest_curvature_scale"] >= 4
