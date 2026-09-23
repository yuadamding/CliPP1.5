"""Attribution bookkeeping and CPU references, never allocated-CUDA evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import attribute_qp_cuda as benchmark
from clipp1d.cuda import qp
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.policy import CudaPolicy


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_problem():
    return dict(name='tiny', recipe='CPU bookkeeping test', arrays=dict(
        h=np.array([3., 4., 5., 6.]), target=np.array([.15, .18, .81, .82]),
        lower=np.full(4, 1e-6), upper=np.ones(4),
        caps=(np.ones((4, 4)) - np.eye(4)) * .2,
        start=np.array([.15, .18, .81, .82])))


@pytest.mark.parametrize('name', ['resource64', 'resource256', 'mixed64'])
def test_literal_fixtures_are_deterministic_and_original_qps_are_valid(name):
    problem = benchmark.fixture(name)
    assert benchmark.fixture_identity(problem) == benchmark.fixture_identity(benchmark.fixture(name))
    a = problem['arrays']
    assert a['h'].ndim == 1 and np.all(a['h'] > 0)
    assert all(value.dtype == np.float64 and np.all(np.isfinite(value)) for value in a.values())
    assert np.array_equal(a['caps'], a['caps'].T)
    assert np.all(a['caps'] >= 0) and np.all(np.diag(a['caps']) == 0)
    assert np.all((a['lower'] <= a['start']) & (a['start'] <= a['upper']))
    before = benchmark.fixture_identity(problem)['input_sha256']
    a['h'][0] = np.nextafter(a['h'][0], np.inf)
    assert benchmark.fixture_identity(problem)['input_sha256'] != before


def test_mixed_fixture_contains_actual_heterogeneous_supports_and_boxes():
    problem = benchmark.fixture('mixed64')
    d = problem['derivation']
    assert set(d['support_counts']) == {1, 2, 3, 4}
    assert len(set(np.array(d['alt']) + np.array(d['ref']))) > 1
    assert np.any(problem['arrays']['upper'] < 1)
    assert len(np.unique(problem['arrays']['h'])) > 10


def test_attribution_covers_updates_without_changing_the_qualified_problem():
    device = torch.device('cpu')
    arrays = benchmark.upload(tiny_problem(), device)
    snapshots = {key: value.clone() for key, value in arrays.items()}
    kernels = Kernels(device)
    expected = benchmark.solve(qp.solve_qp, arrays, kernels)
    baseline = benchmark.qualify(expected, arrays, snapshots, kernels)
    originals = qp.polish_quadratic, qp.quadratic_value, kernels.gap_kkt
    with benchmark.attribution(kernels) as (solver, stages):
        actual, detail = benchmark.measured_profile(
            lambda: benchmark.solve(solver, arrays, kernels), device, stage_counts=stages.counts)
    assert originals == (qp.polish_quadratic, qp.quadratic_value, kernels.gap_kkt)
    observed = benchmark.qualify(actual, arrays, snapshots, kernels)
    assert actual.iterations == expected.iterations
    assert observed == baseline
    assert detail['stage_operation_counts']['admm_update'] == actual.iterations
    assert detail['stages']['admm_update']['calls'] >= actual.iterations
    assert detail['stages']['equality_proposal']['calls'] > 0
    assert detail['stages']['dual_flow_repair']['calls'] > 0
    assert detail['stages']['certificate']['calls'] > 0
    assert detail['numerical_execution'] == 'CPU reference test only'
    assert detail['device_events'] == detail['kernel_dispatches'] == 0
    assert all(row['device_kernel_seconds'] == 0 for row in detail['stages'].values())


def test_hooks_restore_after_numerical_exception():
    kernels = Kernels('cpu')
    originals = qp.polish_quadratic, qp.quadratic_value, kernels.gap_kkt
    with pytest.raises(RuntimeError, match='injected'):
        with benchmark.attribution(kernels):
            raise RuntimeError('injected')
    assert originals == (qp.polish_quadratic, qp.quadratic_value, kernels.gap_kkt)


def test_unrecognized_solver_structure_fails_closed_and_restores(monkeypatch):
    def unknown_solver():
        return None
    monkeypatch.setattr(qp, 'solve_qp', unknown_solver)
    kernels = Kernels('cpu')
    original = kernels.gap_kkt
    with pytest.raises(ValueError, match='ADMM loop'):
        with benchmark.attribution(kernels):
            pytest.fail('Unknown solver must not receive partial attribution')
    assert kernels.gap_kkt is original


def test_overlapping_stage_ranges_are_rejected():
    stages = benchmark.Stages()
    with pytest.raises(RuntimeError, match='double count'):
        with stages.stage('admm_update', 'outer'):
            with stages.stage('certificate', 'inner'):
                pass
    assert stages.active is None


def kineto_event(name, device, start, end, *, identifier=1, linked=0, annotation=False, thread=1):
    return SimpleNamespace(name=lambda: name, device_type=lambda: device,
                           start_ns=lambda: start, end_ns=lambda: end,
                           correlation_id=lambda: identifier, linked_correlation_id=lambda: linked,
                           is_user_annotation=lambda: annotation,
                           start_thread_id=lambda: thread, end_thread_id=lambda: thread)


def kineto_profile(events):
    # Recursive FunctionEvent.device_time_total must never be consulted: in
    # torch 2.9 it includes the attached GPU annotation span as a fake kernel.
    def forbidden():
        raise AssertionError('Do not aggregate recursive FunctionEvent device totals')
    return SimpleNamespace(profiler=SimpleNamespace(kineto_results=SimpleNamespace(events=lambda: events)),
                           events=forbidden, key_averages=lambda: [])


def test_cuda_annotation_mirrors_do_not_count_as_work_or_duplicate_cpu_stage():
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    name = benchmark.PREFIX + 'admm_update'
    events = [kineto_event(name, cpu, 0, 100_000, annotation=True),
              kineto_event('aten::op', cpu, 5_000, 90_000, identifier=2),
              kineto_event(name, cuda, 10_000, 90_000, linked=1, annotation=True),
              kineto_event('other annotation', cuda, 10_000, 90_000, linked=2, annotation=True),
              kineto_event(name, cuda, 10_000, 90_000, linked=1, annotation=False),
              kineto_event('real_kernel', cuda, 10_000, 20_000, linked=2),
              kineto_event('Memcpy DtoH', cuda, 70_000, 75_000, linked=2),
              kineto_event('outside_kernel', cuda, 95_000, 99_000, linked=999)]
    detail = benchmark.profile_summary(kineto_profile(events), {'admm_update': 1})
    assert detail['discarded_cuda_user_annotations'] == 3
    assert detail['kernel_dispatches'] == 2
    assert detail['device_events'] == 3
    assert detail['correlated_cuda_work_events'] == 2
    assert detail['uncorrelated_cuda_work_events'] == 1
    assert detail['device_kernel_and_copy_seconds'] == pytest.approx(19e-6)
    assert detail['unattributed_device_seconds'] == pytest.approx(4e-6)
    row = detail['stages']['admm_update']
    assert row['calls'] == 1
    assert row['host_range_seconds'] == pytest.approx(100e-6)
    assert row['device_kernel_seconds'] == pytest.approx(15e-6)


def test_profile_aggregation_rejects_overlapping_stage_ownership():
    cpu = torch.autograd.DeviceType.CPU
    events = [kineto_event(benchmark.PREFIX + 'admm_update', cpu, 0, 100, annotation=True),
              kineto_event(benchmark.PREFIX + 'certificate', cpu, 10, 20, identifier=2, annotation=True)]
    with pytest.raises(AssertionError, match='Overlapping'):
        benchmark.profile_summary(kineto_profile(events))


def test_profile_rejects_missing_cpu_ranges():
    cpu = torch.autograd.DeviceType.CPU
    event = kineto_event(benchmark.PREFIX + 'admm_update', cpu, 0, 100, annotation=True)
    with pytest.raises(AssertionError, match='coverage'):
        benchmark.profile_summary(kineto_profile([event]), {'admm_update': 2})


@pytest.mark.parametrize('with_stage', [False, True])
def test_duplicate_cpu_ids_require_unanimous_stage_and_count_cuda_work_once(with_stage):
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    events = [kineto_event('op', cpu, 10, 20, identifier=2),
              kineto_event('duplicate', cpu, 11, 19, identifier=2),
              kineto_event('unreferenced', cpu, 110, 120, identifier=3),
              kineto_event('unreferenced duplicate', cpu, 210, 220, identifier=3),
              kineto_event('kernel', cuda, 15, 25, linked=2)]
    if with_stage:
        events.append(kineto_event(benchmark.PREFIX + 'admm_update', cpu, 0, 100, annotation=True))
    detail = benchmark.profile_summary(kineto_profile(events))
    assert detail['duplicate_cpu_correlation_ids'] == 2
    assert detail['cuda_work_with_unanimous_duplicate_owners'] == 1
    assert detail['kernel_dispatches'] == detail['device_events'] == 1
    assert detail['device_kernel_and_copy_seconds'] == pytest.approx(10e-9)
    assert detail['stages']['admm_update']['device_kernel_seconds'] == pytest.approx(10e-9 if with_stage else 0)


@pytest.mark.parametrize('other_stage', [None, 'certificate', 'admm_update'])
def test_conflicting_cpu_correlations_fail_with_candidate_diagnostics(other_stage):
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    events = [kineto_event(benchmark.PREFIX + 'admm_update', cpu, 0, 100, annotation=True),
              kineto_event('first owner', cpu, 10, 20, identifier=2),
              kineto_event('conflicting owner', cpu, 210, 220, identifier=2),
              kineto_event('kernel', cuda, 215, 225, linked=2)]
    if other_stage:
        events.append(kineto_event(benchmark.PREFIX + other_stage, cpu, 200, 300, identifier=3, annotation=True))
    with pytest.raises(AssertionError, match='Ambiguous CUDA work stage ownership') as error:
        benchmark.profile_summary(kineto_profile(events))
    assert 'first owner' in str(error.value) and 'conflicting owner' in str(error.value)
    assert 'linked_correlation_id' in str(error.value)


def test_certificate_rechecks_state_and_detects_input_mutation():
    arrays = benchmark.upload(tiny_problem(), torch.device('cpu'))
    snapshots = {key: value.clone() for key, value in arrays.items()}
    kernels = Kernels('cpu')
    result = benchmark.solve(qp.solve_qp, arrays, kernels)
    benchmark.qualify(result, arrays, snapshots, kernels)
    bad = replace(result, dual=torch.zeros_like(result.dual), qualified=True)
    with pytest.raises(AssertionError, match='certificate'):
        benchmark.qualify(bad, arrays, snapshots, kernels)
    arrays['target'].data[0] += .01
    with pytest.raises(AssertionError, match='modified'):
        benchmark.qualify(result, arrays, snapshots, kernels)


def receipts():
    certificate = dict(qualified=True, gap=1e-14, gap_scale=1., kkt=1e-9,
                       objective=1., minimum_curvature=1., x=[.5])
    case = dict(fixture=dict(name='tiny', input_sha256='literal-input'),
                latency_samples=[dict(seconds=2., certificate=deepcopy(certificate))],
                warmups=[dict(seconds=3., certificate=deepcopy(certificate))],
                dispatch_profile=dict(kernel_dispatches=10, certificate=deepcopy(certificate)),
                attribution=dict(kernel_dispatches=10, certificate=deepcopy(certificate)))
    receipt = dict(schema=benchmark.SCHEMA, status='passed', numerical_execution='CUDA float64', script_sha256='common-script',
                   policy=asdict(CudaPolicy()), controls=dict(fixtures=['tiny'], repeats=1, warmups=1),
                   environment=dict(gpu='test-only'), fixtures=[case],
                   source=dict(source_sha256='baseline'))
    current = deepcopy(receipt)
    current['source']['source_sha256'] = 'current'
    current['fixtures'][0]['latency_samples'][0]['seconds'] = 1.
    return receipt, current


def test_comparison_uses_only_uninstrumented_latency_and_separate_dispatches():
    baseline, current = receipts()
    baseline['fixtures'][0]['attribution']['instrumented_wall_seconds'] = 1000.
    current['fixtures'][0]['attribution']['instrumented_wall_seconds'] = 1e-6
    comparison = benchmark.compare_receipts(baseline, current)['comparisons'][0]
    assert comparison['baseline_over_current_latency'] == 2.
    assert comparison['baseline_kernel_dispatches'] == 10


@pytest.mark.parametrize('fault', ['cpu', 'failed', 'schema', 'missing_case', 'duplicate_case', 'input',
                                  'policy', 'script', 'environment', 'missing_sample',
                                  'unqualified_sample', 'bad_gap', 'missing_dispatch', 'bad_profile',
                                  'bad_state', 'objective'])
def test_comparison_rejects_incomplete_or_unmatched_evidence(fault):
    baseline, current = receipts()
    case = current['fixtures'][0]
    if fault == 'schema':
        current['schema'] = 'clipp1d.cuda.qp_attribution.v1'
    elif fault == 'cpu':
        current['numerical_execution'] = 'CPU reference test only'
    elif fault == 'failed':
        current['status'] = 'failed'
    elif fault == 'missing_case':
        current['fixtures'] = []
    elif fault == 'duplicate_case':
        current['fixtures'].append(deepcopy(case))
    elif fault == 'input':
        case['fixture']['input_sha256'] = 'different'
    elif fault == 'policy':
        current['policy']['inner_atol'] *= 2
    elif fault == 'script':
        current['script_sha256'] = 'different'
    elif fault == 'environment':
        current['environment']['gpu'] = 'different'
    elif fault == 'missing_sample':
        case['latency_samples'] = []
    elif fault == 'unqualified_sample':
        case['latency_samples'][0]['certificate']['qualified'] = False
    elif fault == 'bad_gap':
        case['latency_samples'][0]['certificate']['gap'] = 1.
    elif fault == 'missing_dispatch':
        case['dispatch_profile']['kernel_dispatches'] = 0
    elif fault == 'bad_profile':
        case['attribution']['certificate']['gap'] = float('nan')
    elif fault == 'bad_state':
        case['latency_samples'][0]['certificate']['x'] = [float('nan')]
    elif fault == 'objective':
        case['latency_samples'][0]['certificate']['objective'] = 2.
    with pytest.raises(ValueError):
        benchmark.compare_receipts(baseline, current)


def test_public_runner_rejects_cpu_before_any_measurement(tmp_path):
    args = SimpleNamespace(device='cpu')
    with pytest.raises(ValueError, match='CUDA'):
        benchmark.run(args, {}, tmp_path, lambda *args: pytest.fail('No CPU run events allowed'))
