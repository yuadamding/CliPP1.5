"""Small CPU contract/math references; these tests never claim CUDA evidence."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import replay_failed_qp as replay
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.policy import CudaPolicy


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def problem():
    return dict(h=tensor([128.] * 3), target=tensor([37/64, .5, 127/256]),
                lower=tensor([1e-6] * 3), upper=tensor([.5, .5, .9]),
                caps=tensor([[0, .125, 1], [.125, 0, .125], [1, .125, 0]]),
                start=None, dual=None)


def test_exact_state_json_preserves_float64_bits_and_hashes(tmp_path):
    array = np.array([0., -0., np.nextafter(0., 1.), np.nextafter(0., -1.),
                      np.finfo(float).max, -np.finfo(float).max, np.nextafter(.5, 1.)])
    path = tmp_path / 'state.json'
    descriptor = replay.save_tensor(path, torch.tensor(array), tmp_path)
    receipt = dict(_receipt_path=tmp_path / 'receipt.json', artifacts={descriptor['path']: descriptor['sha256']})
    actual = replay.capture.load_tensor(receipt, descriptor, 'cpu').numpy()
    assert np.array_equal(actual.view(np.uint64), array.view(np.uint64))
    broken = deepcopy(descriptor)
    broken['tensor_sha256'] = '0' * 64
    with pytest.raises(AssertionError, match='roundtrip'):
        replay.capture.load_tensor(receipt, broken, 'cpu')
    path.write_text(path.read_text().replace('float64', 'float32'))
    with pytest.raises(AssertionError, match='hash'):
        replay.capture.load_tensor(receipt, descriptor, 'cpu')


def test_state_json_is_exclusive_and_scalar_shape_preserved(tmp_path):
    path = tmp_path / 'scalar.json'
    descriptor = replay.save_tensor(path, tensor(.5), tmp_path)
    assert descriptor['shape'] == []
    with pytest.raises(FileExistsError):
        replay.save_tensor(path, tensor(.5), tmp_path)


def test_three_node_counterexample_driver_has_honest_cpu_scope():
    kernels = Kernels('cpu')
    result = replay.review_counterexample(torch.device('cpu'), kernels, kernels)
    assert result['status'] == 'passed'
    assert result['numerical_execution'] == 'CPU reference test only'
    assert result['exact_valid_dual_preserved'] and result['legacy_uniform_bad_fixed_point']
    assert len(result['cone_recoveries']) == 4
    assert all(row['original_certificate']['independently_qualified'] for row in result['cone_recoveries'])


def test_legacy_reference_preserves_the_reviewed_bad_fixed_point():
    p = problem()
    x = tensor([.5] * 3)
    valid = tensor([[0, 0, -.5], [0, 0, 0], [.5, 0, 0]])
    bad, result = replay.repair_candidate(x, valid, p, Kernels('cpu'), CudaPolicy(), 'legacy_uniform')
    assert result['before_certificate']['independently_qualified']
    assert result['stop'] == 'stationary_unresolved'
    assert result['after_certificate']['gap'] == .00054931640625
    assert result['after_certificate']['kkt'] == .15789473684210525
    assert not result['after_certificate']['independently_qualified']
    assert torch.equal(bad, tensor([[0, -.125, -1], [.125, 0, .125], [1, -.125, 0]]))


def record():
    return dict(context=dict(source_sha256=replay.CAPTURE_SOURCE, policy=asdict(CudaPolicy())),
                problem={key: None for key in replay.PROBLEM_KEYS + ('start', 'dual')},
                returned=dict(qualified=False, iterations=20000))


@pytest.mark.parametrize('fault', ['source', 'policy', 'missing', 'qualified', 'iterations'])
def test_record_binding_rejects_wrong_source_policy_or_qp(fault):
    value = record()
    if fault == 'source':
        value['context']['source_sha256'] = '0' * 64
    elif fault == 'policy':
        value['context']['policy']['inner_rtol'] *= 2
    elif fault == 'missing':
        del value['problem']['dual']
    elif fault == 'qualified':
        value['returned']['qualified'] = True
    else:
        value['returned']['iterations'] += 1
    with pytest.raises(AssertionError):
        replay.check_problem(value)


@pytest.mark.parametrize('profile_name', ['normalized276da', 'bound8627'])
def test_current_profile_replay_requires_explicitly_reviewed_source_identity(profile_name):
    value = record()
    value['context']['source_sha256'] = replay.capture.PROFILES[profile_name]['source_sha256']
    assert replay.check_problem(value) == CudaPolicy()
    value['context']['source_sha256'] = 'f' * 64
    with pytest.raises(AssertionError, match='explicitly pinned'):
        replay.check_problem(value)


def test_original_none_initialization_is_not_replaced_with_terminal_state(monkeypatch):
    def load(receipt, descriptor, device):
        assert descriptor is not None
        return descriptor
    monkeypatch.setattr(replay.capture, 'load_tensor', load)
    value = record()
    value['problem'].update({key: tensor([1.]) for key in replay.PROBLEM_KEYS})
    actual = replay.load_problem({}, value, 'cpu')
    assert actual['start'] is None and actual['dual'] is None


def reconstruction_fixture(monkeypatch):
    monkeypatch.setattr(replay.capture, 'load_tensor', lambda receipt, descriptor, device: descriptor)
    p = problem()
    x = tensor([.5] * 3)
    q = tensor([[0, -.125, -1], [.125, 0, .125], [1, -.125, 0]])
    kernels = Kernels('cpu')
    stats = replay.stats_for(x, q, p, kernels)
    value = dict(returned=dict(x=x, q=q, gap=stats[0], scale=stats[1], kkt=stats[2]),
                 terminal_admm=dict(x=x.clone(), q=q.clone()),
                 original_compiled_certificate_replay=dict(exact_bits_equal=True, stats=stats.clone()),
                 decomposition=dict(returned=replay.capture.decompose_gap(x, q, *(p[key] for key in replay.PROBLEM_KEYS))))
    return p, stats, value, kernels


def test_terminal_capture_proof_and_eager_reconstruction_remain_exact(monkeypatch):
    p, stats, value, kernels = reconstruction_fixture(monkeypatch)
    result = replay.restore_baseline({}, value, p, kernels, CudaPolicy(), 'cpu')
    assert result['original_capture_compiled_proof_exact']
    assert result['eager_certificate_exactly_reconstructed'] and result['terminal_admm_equals_returned']
    assert result['fresh_compiled_bits_equal'] and result['compiled_gate_classifications_unchanged']
    value['returned']['gap'] = torch.nextafter(stats[0], tensor(float('inf')))
    with pytest.raises(AssertionError, match='stored terminal bits'):
        replay.restore_baseline({}, value, p, kernels, CudaPolicy(), 'cpu')


@pytest.mark.parametrize('fault', ['proof', 'eager'])
def test_changed_original_proof_or_saved_eager_certificate_fails_closed(monkeypatch, fault):
    p, stats, value, kernels = reconstruction_fixture(monkeypatch)
    if fault == 'proof':
        value['original_compiled_certificate_replay']['exact_bits_equal'] = False
    else:
        value['decomposition']['returned']['certificate']['gap'] = float(torch.nextafter(stats[0], tensor(float('inf'))))
    with pytest.raises(AssertionError):
        replay.restore_baseline({}, value, p, kernels, CudaPolicy(), 'cpu')


def test_fresh_compiled_reduction_difference_reported_without_new_tolerance(monkeypatch):
    p, stats, value, _ = reconstruction_fixture(monkeypatch)
    fresh = torch.nextafter(stats, torch.full_like(stats, float('inf')))
    kernels = SimpleNamespace(gap_kkt=lambda *args: fresh)
    result = replay.restore_baseline({}, value, p, kernels, CudaPolicy(), 'cpu')
    assert not result['fresh_compiled_bits_equal']
    assert result['eager_certificate_exactly_reconstructed']
    assert result['compiled_gate_classifications_unchanged']
    assert result['compiled_comparison']['signed_deltas'] == (fresh - stats).tolist()
    assert result['compiled_comparison']['stored_bits'] != result['compiled_comparison']['fresh_bits']
    assert not result['returned_certificate']['independently_qualified']
    assert not result['fresh_compiled_certificate']['independently_qualified']


@pytest.mark.parametrize('fault', ['gap', 'kkt', 'nonfinite'])
def test_fresh_compiled_component_gate_change_rejected_even_if_still_unqualified(monkeypatch, fault):
    p, stats, value, _ = reconstruction_fixture(monkeypatch)
    fresh = stats.clone()
    if fault == 'gap':
        fresh[0] = 0
    elif fault == 'kkt':
        fresh[2] = 0
    else:
        fresh[1] = float('nan')
    kernels = SimpleNamespace(gap_kkt=lambda *args: fresh)
    with pytest.raises(AssertionError, match='gate classifications'):
        replay.restore_baseline({}, value, p, kernels, CudaPolicy(), 'cpu')


def test_diagnostic_completion_does_not_promote_candidate_only_success():
    receipt = dict(captured_failed_qps=2, replays=[dict(qualified=True), dict(qualified=False,
                   candidate_repairs=[dict(after_certificate=dict(independently_qualified=True))])])
    replay.finish_diagnostics(receipt, False)
    assert receipt['status'] == 'passed' and receipt['diagnostic_status'] == 'completed'
    assert not receipt['all_replays_qualified'] and receipt['unresolved_qps'] == 1
    assert receipt['scientific_status'] == 'unresolved_original_start_replays'
    with pytest.raises(AssertionError, match='every original-start'):
        replay.finish_diagnostics(receipt, True)
    receipt['replays'].pop()
    with pytest.raises(AssertionError, match='coverage'):
        replay.finish_diagnostics(receipt, False)


def test_public_replay_rejects_cpu_before_loading_any_capture(monkeypatch):
    monkeypatch.setattr(replay.capture, 'load_capture', lambda *args: pytest.fail('CPU run loaded a capture'))
    with pytest.raises(ValueError, match='CUDA'):
        replay.run(SimpleNamespace(device='cpu'), {}, Path('.'), lambda *args: None)


def test_current_source_identity_is_required(monkeypatch):
    monkeypatch.setattr(replay, 'source_provenance', lambda: dict(source_sha256='1' * 64))
    with pytest.raises(AssertionError, match='explicit'):
        replay.check_source(None)
    with pytest.raises(AssertionError, match='differs'):
        replay.check_source('0' * 64)
