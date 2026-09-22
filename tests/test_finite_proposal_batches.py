"""Decision and arithmetic regression tests against the 72ba3ed scalar path."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.special import logsumexp

from clipp1d import model as likelihood, proposals, solver
from clipp1d.proposals import FiniteProposal, iter_proposal_deltas
from clipp1d.types import CountModel
from conftest import count_model


def mixed_model(n=23, seed=29):
    rng = np.random.default_rng(seed)
    valid = np.arange(4)[None, :] < rng.integers(1, 5, n)[:, None]
    slopes = rng.uniform(.15, .7, (n, 1)) * np.arange(1, 5)[None, :]
    return CountModel(tuple(f"m{i}" for i in range(n)), rng.integers(0, 1000, n),
                      rng.integers(0, 1000, n), np.full(n, 1e-6), np.ones(n),
                      np.where(valid, slopes, 0),
                      np.where(valid, -np.log(valid.sum(axis=1)[:, None]), -np.inf),
                      valid, 1e-6)


def scalar_loss(model, phi):
    # Literal candidate arithmetic from frozen 72ba3ed/model.py:evaluate.
    mass = model.slope * phi[:, None]
    p = np.clip(mass, model.eps, 1 - model.eps)
    joint = model.alt[:, None] * np.log(p) + model.ref[:, None] * np.log1p(-p) + model.log_prior
    return -logsumexp(joint, axis=1)


def scalar_delta(model, x, caps, p, old_losses):
    old = x[p.start:p.stop]
    new = np.broadcast_to(np.asarray(p.value, dtype=float), old.shape)
    block = model.subset(np.arange(p.start, p.stop))
    delta = float(np.sum(scalar_loss(block, new) - old_losses[p.start:p.stop]))
    delta += float(np.dot(caps[p.start:p.stop - 1], np.abs(np.diff(new)) - np.abs(np.diff(old))))
    if p.start:
        delta += caps[p.start - 1] * (abs(new[0] - x[p.start - 1]) - abs(old[0] - x[p.start - 1]))
    if p.stop < len(x):
        delta += caps[p.stop - 1] * (abs(x[p.stop] - new[-1]) - abs(x[p.stop] - old[-1]))
    return delta


@pytest.mark.parametrize("seed", range(8))
def test_loss_only_and_flattened_rows_preserve_scalar_kernel_bits(seed):
    model = mixed_model(seed=seed)
    rng = np.random.default_rng(seed)
    x = rng.choice([model.eps, 0., .4999995, 1., 2.], len(model))
    np.testing.assert_array_equal(likelihood.loss(model, x), scalar_loss(model, x))
    np.testing.assert_array_equal(likelihood.evaluate(model, x).loss, scalar_loss(model, x))
    rows = rng.integers(0, len(model), 89)
    values = rng.uniform(-.1, 1.2, len(rows))
    expected = np.array([scalar_loss(model.subset([i]), np.array([v]))[0]
                         for i, v in zip(rows, values)])
    np.testing.assert_array_equal(likelihood.loss_at_rows(model, rows, values), expected)


def test_loss_only_does_not_construct_posterior(monkeypatch):
    model = count_model([1, 2], [3, 4])
    monkeypatch.setattr(likelihood, "_log_kernel", lambda *a: (None, None, None, np.array([-1., -2.])))
    def forbidden(*args, **kwargs):
        raise AssertionError("posterior exponential must not be evaluated")
    monkeypatch.setattr(likelihood.np, "exp", forbidden)
    np.testing.assert_array_equal(likelihood.loss(model, np.ones(2)), [1., 2.])


@pytest.mark.parametrize("length", [1, 17, 64, 65, 130])
def test_batch_reductions_and_duplicate_reuse_are_exact(monkeypatch, length):
    model = mixed_model(length)
    x = np.linspace(.03, .9, length)
    context = solver.prepare_audit(model, x)
    items = [FiniteProposal(i, min(i + 7, length), v, 0.)
             for i in range(length) for v in (.25, .25, .500001)]
    original = proposals.loss_at_rows
    seen = []
    def counted(model, rows, values):
        seen.append(len(rows))
        return original(model, rows, values)
    monkeypatch.setattr(proposals, "loss_at_rows", counted)
    expected = [float(np.sum(scalar_loss(model.subset(np.arange(p.start, p.stop)),
                                         np.full(p.stop - p.start, p.value)) - context.losses[p.start:p.stop]))
                for p in items]
    actual = list(iter_proposal_deltas(context, items))
    assert [delta.hex() for _, delta in actual] == [delta.hex() for delta in expected]
    assert [p for p, _ in actual] == items
    assert sum(seen) == sum(p.stop - p.start for p in items[::3]) * 2
    calls = len(seen)
    assert list(iter_proposal_deltas(context, items)) == actual
    assert len(seen) == calls


def test_singletons_batch_across_mutations_and_large_intervals_remain_bounded(monkeypatch):
    n = proposals.MAX_BATCH_ROWS + 10
    model = count_model(np.ones(n), np.ones(n))
    context = solver.prepare_audit(model, np.full(n, .3))
    items = [FiniteProposal(i, i + 1, .4, 0.) for i in range(100)]
    items += [FiniteProposal(0, n, .5, 0.), FiniteProposal(0, n, .6, 0.)]
    original = proposals.loss_at_rows
    batches = []
    def counted(model, rows, values):
        batches.append(rows.copy())
        return original(model, rows, values)
    monkeypatch.setattr(proposals, "loss_at_rows", counted)
    assert len(list(iter_proposal_deltas(context, items))) == len(items)
    assert [len(rows) for rows in batches] == [64, 36, n, n]
    np.testing.assert_array_equal(batches[0], np.arange(64))


def test_memo_is_bounded_context_owned_and_does_not_cache_boxes(monkeypatch):
    monkeypatch.setattr(proposals, "MAX_CACHE_ENTRIES", 3)
    model = count_model([1, 1], [9, 9])
    x = np.array([1., .3])
    context = solver.prepare_audit(model, x)
    items = [FiniteProposal(1, 2, v, 0.) for v in (.1, .2, .3, .4)]
    list(iter_proposal_deltas(context, items))
    assert len(context.proposal_losses._values) == 3
    assert not solver.prepare_audit(model, x).proposal_losses._values
    assert not replace(context, x=x + .01).proposal_losses._values
    with pytest.raises(FrozenInstanceError):
        context.proposal_losses = proposals.ProposalLossMemo()
    lower, upper = model.lower.copy(), model.upper.copy()
    lower[1] = upper[1] = .3
    assert list(solver._finite_proposals(model, x, lower, upper, context, None)) == []
    with pytest.raises(ValueError, match="exact primal"):
        solver._finite_interval_check(model, x + .01, np.zeros(1), lower, upper, context, None)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("directional", [False, True])
def test_decisions_and_restart_vector_match_scalar_arithmetic(seed, directional):
    model = mixed_model(13, seed)
    rng = np.random.default_rng(seed)
    x = rng.choice([1e-6, .3, .5, 1.], len(model))
    x[:3] = .3 if directional else 1e-6
    caps = rng.uniform(0, 100, len(model) - 1)
    context = solver.prepare_audit(model, x)
    lower, upper = model.lower.copy(), model.upper.copy()
    direction = (0, 3, 1, -100., 1.) if directional else None
    items = list(solver._finite_proposals(model, x, lower, upper, context, direction))
    expected, accepted = (direction is None, None), None
    for index, p in enumerate(items):
        delta = scalar_delta(model, x, caps, p, context.losses)
        if delta < -p.margin and (direction is None or delta <= 1e-4 * p.step * direction[3]):
            trial = x.copy()
            trial[p.start:p.stop] = p.value
            expected, accepted = (False, trial), index
            break
    actual = solver._finite_interval_check(model, x, caps, lower, upper, context, direction)
    assert actual[0] == expected[0]
    if accepted is None:
        assert actual[1] is None
    else:
        np.testing.assert_array_equal(actual[1], expected[1])


def test_original_clipping_order_and_first_acceptance(monkeypatch):
    model = mixed_model(3)
    model = replace(model, slope=np.tile([.25, .5, .75, 1.], (3, 1)),
                    valid=np.ones((3, 4), dtype=bool), log_prior=np.full((3, 4), -np.log(4)))
    x = np.full(3, model.eps)
    context = solver.prepare_audit(model, x)
    items = list(solver._finite_proposals(model, x, model.lower, model.upper, context, None))
    first = [p for p in items if p.start == 0 and p.stop == 1]
    points = [1e-6, 1e-6 / .75, 2e-6]
    expected = [float(np.clip(p + offset, model.eps, 1.))
                for p in points for offset in (-1e-5, 0., 1e-5)]
    assert [p.value for p in first] == expected
    assert len({p.key() for p in first}) < len(first)
    # A much better later candidate must not replace the first passing one.
    def deltas(context, candidates, **kwargs):
        for i, p in enumerate(candidates):
            yield p, 0. if i < 2 else (-1. if i == 2 else -1000.)
    monkeypatch.setattr(solver, "iter_proposal_deltas", deltas)
    okay, restart = solver._finite_interval_check(model, x, np.zeros(2), model.lower, model.upper,
                                                  context, None)
    assert not okay
    np.testing.assert_array_equal(restart, [first[2].value, x[1], x[2]])
