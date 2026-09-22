from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from clipp1d import model as likelihood, solver
from clipp1d.policy import Policy
from conftest import count_model


def test_one_likelihood_evaluation_for_stationarity_and_extreme_anchors(monkeypatch):
    model = count_model([0, 0, 0], [0, 0, 0])
    x, caps = np.ones(3), np.ones(2)
    original = likelihood.evaluate
    calls = []

    def evaluate(*args, **kwargs):
        calls.append((args[0], kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(likelihood, "evaluate", evaluate)
    monkeypatch.setattr(solver, "evaluate", evaluate)
    context = solver.prepare_audit(model, x)
    for anchor in (0, 2):
        lower, upper = model.lower.copy(), model.upper.copy()
        lower[anchor] = upper[anchor] = 1.
        assert solver.stationarity(model, x, np.zeros(2), lower, upper, caps, context=context) == (0., True)
        assert solver._kink_check(model, x, caps, lower, upper, Policy(),
                                  context=context, force_intervals=True) == (True, None)
    assert len(calls) == 1 and calls[0][1] == {"derivatives": True}


def test_context_arrays_are_immutable_and_stale_context_is_rejected():
    model = count_model([10, 40], [90, 60])
    x = np.array([1., .5])
    context = solver.prepare_audit(model, x)
    with pytest.raises(FrozenInstanceError):
        context.model = None
    for field in ("x", "losses", "posterior", "gradient", "left", "right", "at_kink", "plateau", "cuts"):
        with pytest.raises(ValueError):
            getattr(context, field).setflags(write=True)
    x[1] = .6
    with pytest.raises(ValueError, match="exact primal"):
        solver.stationarity(model, x, np.zeros(1), model.lower, model.upper, np.ones(1), context=context)
    with pytest.raises(ValueError, match="model"):
        context.validate(model.subset([0, 1]), context.x)


def test_interval_proposal_reuses_old_loss_slice(monkeypatch):
    from clipp1d import proposals
    model = count_model([10, 40], [90, 60])
    x = np.array([1., .5])
    context = solver.prepare_audit(model, x)
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[0] = upper[0] = 1.
    original = proposals.loss_at_rows
    calls = []

    def evaluate(block, rows, values):
        assert block is model
        np.testing.assert_array_equal(rows, [1])
        assert not np.array_equal(values, x[1:])
        calls.append(values.copy())
        return original(block, rows, values)

    monkeypatch.setattr(proposals, "loss_at_rows", evaluate)
    okay, restart = solver._kink_check(model, x, np.zeros(1), lower, upper, Policy(),
                                       context=context, force_intervals=True)
    assert not okay and restart is not None
    assert len(calls) == 1
    np.testing.assert_array_equal(restart, [1., .501])


@pytest.mark.parametrize("phi", [[.5, 1.], [1e-6, .2], [.4999995, .75]])
def test_reused_posterior_preserves_one_sided_derivatives(phi):
    model = count_model([10, 40], [90, 60], slope=2.)
    x = np.asarray(phi)
    posterior = likelihood.evaluate(model, x).posterior
    expected = likelihood.one_sided_derivatives(model, x)
    actual = likelihood.one_sided_derivatives(model, x, posterior=posterior)
    np.testing.assert_array_equal(actual, expected)
