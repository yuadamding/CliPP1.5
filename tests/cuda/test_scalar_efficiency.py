"""Numerical and work-count regressions for reduced scalar CUDA entry points.

CPU tensor references and eager Dynamo traces are not actual CUDA qualification.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from clipp1d.cuda.kernels import Kernels, StructuralCompileBank, likelihood, loss_gradient, loss_only
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.scalar import Problems, _golden_wells, solve_scalar


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(alt, slopes):
    alt = torch.tensor(alt, dtype=torch.float64)
    slope = torch.tensor(slopes, dtype=torch.float64)
    valid = slope > 0
    prior = torch.where(valid, -valid.sum(-1, keepdim=True).double().log(), -float('inf'))
    return TensorModel(tuple(f'm{i:03}' for i in range(len(alt))), alt, 100 - alt, slope, prior,
                       torch.full_like(alt, 1e-6), torch.ones_like(alt), 1e-6,
                       Kernels('cpu', compiled=False))


@pytest.mark.parametrize('compiled', [False, True])
def test_reduced_likelihoods_match_full_arithmetic_at_clipping_and_neighbors(compiled):
    m = model([16, 37, 0], [[.4, .8, 1.2], [.3, .6, 0.], [.5, 0., 0.]])
    base = np.array([.4, .3, .5])
    probes = np.stack((np.full(3, .3), 1e-6 / base, (1 - 1e-6) / base,
                       np.nextafter(1e-6 / base, -np.inf),
                       np.nextafter((1 - 1e-6) / base, np.inf)), axis=1)
    points = torch.tensor(probes, dtype=torch.float64)
    args = (m.alt, m.ref, m.slope, m.log_prior, points, m.eps)
    expected = likelihood(*args)
    loss_fn = StructuralCompileBank(loss_only, backend='eager') if compiled else loss_only
    pair_fn = StructuralCompileBank(loss_gradient, backend='eager') if compiled else loss_gradient
    torch.testing.assert_close(loss_fn(*args), expected[0], rtol=0., atol=0.)
    loss, gradient = pair_fn(*args)
    torch.testing.assert_close(loss, expected[0], rtol=0., atol=0.)
    torch.testing.assert_close(gradient, expected[1], rtol=0., atol=0.)


def test_analytical_groups_evaluate_only_their_attained_point():
    m = model([0, 30, 48, 100], [[.5], [.5], [.5], [.5]])
    p = Problems(m, torch.ones(4, dtype=torch.long))
    before = m.integrity_counters['full_checks']
    result = solve_scalar(p)
    assert bool(result.qualified.all())
    torch.testing.assert_close(result.phi, torch.tensor([1e-6, .6, .96, 1.], dtype=torch.float64), rtol=0., atol=0.)
    counters = m.scalar_work_counters
    assert counters['scalar_rows_evaluated'] == 4
    assert counters['scalar_proposals_evaluated'] == 4
    assert counters['loss_only_calls'] == 1
    assert counters['loss_gradient_calls'] == counters['full_terms_calls'] == 0
    assert counters['analytical_groups'] == 4 and counters['general_groups'] == 0
    assert m.integrity_counters['full_checks'] - before == 2


def test_mixed_analytical_lanes_are_removed_from_general_batches(monkeypatch):
    m = model([20, 35, 12, 40], [[.5, 0., 0.], [.15, .3, .45],
                                [.25, 0., 0.], [.2, .4, .6]])
    calls = []
    original = m.kernels.loss_only
    def counted(*args):
        calls.append((args[0].numel(), args[4].shape[1]))
        return original(*args)
    monkeypatch.setattr(m.kernels, 'loss_only', counted)
    result = solve_scalar(Problems(m, torch.ones(4, dtype=torch.long)))
    assert bool(result.qualified.all())
    assert calls[0] == (2, 1)
    assert all(rows == 2 for rows, _ in calls)
    assert m.scalar_work_counters['analytical_groups'] == 2
    assert m.scalar_work_counters['general_groups'] == 2
    for i in range(4):
        one = m.subset(slice(i, i + 1))
        reference = solve_scalar(Problems(one, torch.ones(1, dtype=torch.long)))
        torch.testing.assert_close(result.loss[i], reference.loss[0], rtol=1e-13, atol=1e-10)
        torch.testing.assert_close(result.phi[i], reference.phi[0], rtol=0., atol=1e-8)


def test_packed_golden_wells_preserve_full_grid_wells_and_reduce_work():
    m = model([20, 37], [[.15, .3, .45], [.2, .4, .6]])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    seeds = torch.linspace(1e-6, 1., 35, dtype=torch.float64)[None, :].expand(2, -1)
    with p.validated_stage():
        values = p.loss(seeds)
        at = values.argmin(-1)
        best = seeds.gather(1, at[:, None])[:, 0]
        local = (values[:, 1:-1] <= values[:, :-2]) & (values[:, 1:-1] <= values[:, 2:])
        before = m.scalar_work_counters['scalar_proposals_evaluated']
        packed, packed_loss, present = _golden_wells(p, seeds, values, best)
        work = m.scalar_work_counters['scalar_proposals_evaluated'] - before
        assert present
        width = int(local.sum(-1).max())
        assert width < local.shape[1]
        assert work == m.n * width * 97
        # Independent prior full-grid golden reference; only true local wells
        # are compared, in their original seed-column tie order.
        left, right = seeds[:, :-2].clone(), seeds[:, 2:].clone()
        ratio = (5. ** .5 - 1) / 2
        for _ in range(48):
            c, d = right - ratio * (right - left), left + ratio * (right - left)
            take = p.loss(c) <= p.loss(d)
            left, right = torch.where(take, left, c), torch.where(take, d, right)
        expected = (left + right) * .5
        expected_loss = p.loss(expected)
        for row in range(2):
            occupied = torch.isfinite(packed_loss[row])
            torch.testing.assert_close(packed[row, occupied], expected[row, local[row]], rtol=0., atol=0.)
            torch.testing.assert_close(packed_loss[row, occupied], expected_loss[row, local[row]], rtol=0., atol=0.)


def test_no_actual_wells_skips_all_golden_evaluations():
    m = model([100], [[.2, .4]])
    p = Problems(m, torch.ones(1, dtype=torch.long))
    seeds = torch.linspace(1e-6, 1., 33, dtype=torch.float64)[None, :]
    with p.validated_stage():
        values = p.loss(seeds)
        before = m.scalar_work_counters['scalar_proposals_evaluated']
        _, losses, present = _golden_wells(p, seeds, values, torch.ones(1, dtype=torch.float64))
        assert not present and bool(torch.isinf(losses).all())
        assert m.scalar_work_counters['scalar_proposals_evaluated'] == before


def test_general_scalar_stage_checks_full_integrity_only_at_boundaries():
    m = model([20, 37], [[.15, .3, .45], [.2, .4, .6]])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    before = dict(m.integrity_counters)
    result = solve_scalar(p, replace(CudaPolicy(), scalar_max_intervals=256))
    assert bool(result.qualified.all())
    assert m.scalar_work_counters['loss_only_calls'] >= 97
    assert m.integrity_counters['full_checks'] - before['full_checks'] == 2
    assert m.integrity_counters['metadata_checks'] - before['metadata_checks'] > 97


def test_nested_model_stages_do_not_repeat_snapshot_reads():
    m = model([20, 30], [[.5], [.5]])
    x = torch.tensor([.4, .6], dtype=torch.float64)
    with m.validated_stage():
        with m.validated_stage():
            for _ in range(10):
                m.loss(x)
                m.loss_gradient(x)
                m.terms(x)
    assert m.integrity_counters['full_checks'] == 2
    assert m.integrity_counters['metadata_checks'] > 30


@pytest.mark.parametrize('stage', [False, True])
def test_data_mutation_is_rejected_before_direct_or_staged_return(stage):
    m = model([20], [[.5]])
    x = torch.tensor([.4], dtype=torch.float64)
    if stage:
        with pytest.raises(ValueError, match='modified'):
            with m.validated_stage():
                m.alt.data[0] += 1.
                m.loss(x)
    else:
        m.alt.data[0] += 1.
        with pytest.raises(ValueError, match='modified'):
            m.loss(x)


def test_version_mutation_is_detected_inside_stage_without_snapshot():
    m = model([20], [[.5]])
    with pytest.raises(ValueError, match='modified'):
        with m.validated_stage():
            m.alt.add_(1.)
            m.loss(torch.tensor([.4], dtype=torch.float64))


def test_exceptional_stage_exit_still_reconciles_data_and_releases_context():
    m = model([20], [[.5]])
    with pytest.raises(ValueError, match='modified'):
        with m.validated_stage():
            m.alt.data[0] += 1.
            raise RuntimeError('interrupt numerical stage')
    assert m._stage_depth == 0
    with pytest.raises(ValueError, match='modified'):
        m.loss(torch.tensor([.4], dtype=torch.float64))


def test_direct_scalar_membership_data_mutation_guard():
    m = model([20, 30], [[.5], [.5]])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    p.group.data[0] = 1
    with pytest.raises(ValueError, match='modified'):
        p.loss(torch.tensor([[.4], [.6]], dtype=torch.float64))


def test_subset_counters_accumulate_at_the_original_model():
    m = model([20, 30, 40], [[.5], [.5], [.5]])
    one, two = m.subset(slice(0, 1)), m.subset(slice(1, 3))
    assert one.scalar_work_counters is m.scalar_work_counters is two.scalar_work_counters
    assert one.integrity_counters is m.integrity_counters is two.integrity_counters
    solve_scalar(Problems(one, torch.ones(1, dtype=torch.long)))
    solve_scalar(Problems(two, torch.ones(2, dtype=torch.long)))
    assert m.scalar_work_counters['scalar_proposals_evaluated'] == 3


def test_segment_length_checks_are_reused_only_in_validated_stage(monkeypatch):
    m = model([20, 30, 40], [[.5], [.5], [.5]])
    p = Problems(m, torch.tensor([1, 2], dtype=torch.long))
    original = torch.segment_reduce
    checks = []
    def observed(*args, **kwargs):
        checks.append(kwargs.get('unsafe', False))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'segment_reduce', observed)
    expected = p.reduce(m.alt)
    with p.validated_stage():
        actual = p.reduce(m.alt)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert checks == [False, True]
    p.lengths.data[0] = 2
    with pytest.raises(ValueError, match='modified'):
        with p.validated_stage():
            p.reduce(m.alt)


def test_clean_stage_numerical_calls_do_not_read_tensor_scalars():
    m = model([20, 30], [[.2, .4], [.15, .3]])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    points = torch.tensor([[.3, .4], [.5, .6]], dtype=torch.float64)
    with p.validated_stage():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            for _ in range(4):
                p.loss(points)
                p.evaluate(points)
    reads = {event.key: event.count for event in profile.key_averages()
             if event.key in ('aten::item', 'aten::_local_scalar_dense')}
    assert reads == {}


def test_membership_data_write_during_work_prevents_stage_return():
    m = model([20, 30], [[.5], [.5]])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError, match='modified'):
        with p.validated_stage():
            p.group.data[0] = 1
            p.loss(torch.tensor([[.3], [.6]], dtype=torch.float64))
    assert p._stage_depth == 0 and m._stage_depth == 0


def test_scalar_hot_loop_never_mutates_model_or_membership_inputs():
    m = model([20, 37], [[.15, .3, .45], [.2, .4, .6]])
    p = Problems(m, torch.ones(2, dtype=torch.long))
    inputs = (m.alt, m.ref, m.slope, m.log_prior, m.lower, m.upper,
              p.lengths, p.group, p.lower, p.upper)
    before = [(value.clone(), value._version) for value in inputs]
    result = solve_scalar(p)
    assert bool(result.qualified.all())
    for value, (snapshot, version) in zip(inputs, before):
        assert value._version == version
        torch.testing.assert_close(value, snapshot, rtol=0., atol=0.)
