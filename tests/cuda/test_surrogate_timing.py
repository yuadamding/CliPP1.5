"""Timing boundaries and source admission; these doubles do not qualify CUDA."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from benchmarks import time_surrogate_cuda as timing


def test_timing_has_no_observer_or_candidate_tracer(monkeypatch):
    events = []
    @contextmanager
    def strategy(name, journal=None):
        assert journal is None
        events.append('strategy')
        yield ['counts']
    def timed(device, function):
        events.append('sync-start')
        value = function()
        events.append('sync-stop')
        return value, 3.0
    monkeypatch.setattr(timing.study, 'strategy', strategy)
    monkeypatch.setattr(timing.common, 'timed', timed)
    monkeypatch.setattr(timing.selection, 'fit_tensor_model', lambda model: events.append('fit') or 'result')
    monkeypatch.setattr(timing.mixed, 'trace_candidates', lambda *a: pytest.fail('tracing enabled'))
    assert timing.timed_fit(None, timing.study.CANDIDATE, None) == ('result', 3.0, ['counts'])
    assert events == ['strategy', 'sync-start', 'fit', 'sync-stop']


def test_publication_and_summary_are_outside_timer(monkeypatch, tmp_path):
    events = []
    fit = SimpleNamespace(search_status='complete')
    calls = timing.study.StartLog()
    calls.append({'qualified': True})
    calls.attempts.append(dict(returned=True, qualified=True))
    monkeypatch.setattr(timing.torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(timing.common, 'timed', lambda device, function: (function(), .01))
    monkeypatch.setattr(timing.common, 'upload', lambda *a: events.append('upload'))
    monkeypatch.setattr(timing, 'timed_fit', lambda *a: (events.append('timed-fit-complete') or fit, 2., calls))
    monkeypatch.setattr(timing.common, 'fit_summary', lambda *a: events.append('summary') or {'pilot_and_graph': {'weights': [1]}})
    monkeypatch.setattr(timing, 'publish', lambda *a: events.append('publish') or {})
    result = timing.trial(tmp_path/'trial.json', timing.study.CANDIDATE, 1, None, None, None, None)
    assert events == ['upload', 'timed-fit-complete', 'summary', 'publish']
    assert result['publication_qualified'] and result['fit_seconds'] == 2.
    assert 'weights' not in result['summary']['pilot_and_graph']


def coverage_fixture(tmp_path):
    artifact = tmp_path/'coverage-artifact.json'
    artifact.write_text('{}')
    value = dict(schema=timing.study.SCHEMA, status='passed', cuda_available=True,
                 source={'source_sha256': 'source'}, fixture_family='below_one', nodes=64,
                 policy=asdict(timing.study.CudaPolicy()), candidate_policy=timing.study.CANDIDATE,
                 control_policy=timing.study.CONTROL, helpers=timing.study.helpers(),
                 artifacts={artifact.name: timing.common.sha(artifact)},
                 trials=[dict(surrogate_policy=timing.study.CANDIDATE, status='passed',
                              fixture_recipe=timing.mixed.fixtures.RECIPE, input_sha256='input',
                              full_path=dict(search_status='complete', final_export_and_publication_qualified=True))])
    return value, artifact


@pytest.mark.parametrize('change', ['source', 'size', 'failed', 'publication', 'recipe', 'artifact'])
def test_timing_rejects_unqualified_or_changed_coverage(tmp_path, change):
    value, artifact = coverage_fixture(tmp_path)
    if change == 'source':
        value['source']['source_sha256'] = 'other'
    elif change == 'size':
        value['nodes'] = 256
    elif change == 'failed':
        value['status'] = 'failed'
    elif change == 'publication':
        value['trials'][0]['full_path']['final_export_and_publication_qualified'] = False
    elif change == 'recipe':
        value['helpers']['fixtures'] = 'changed'
    elif change == 'artifact':
        artifact.write_text('changed')
    path = tmp_path/'coverage.json'
    path.write_text(json.dumps(value))
    args = SimpleNamespace(coverage=path, coverage_sha256=timing.common.sha(path), fixture='below_one', nodes=64)
    with pytest.raises(AssertionError):
        timing.coverage(args, {'source_sha256': 'source'})


def test_original_coverage_driver_identity_is_preserved(tmp_path):
    value, artifact = coverage_fixture(tmp_path)
    value['helpers']['surrogate_driver'] = 'original-diagnostic-driver'
    path = tmp_path/'coverage.json'
    path.write_text(json.dumps(value))
    args = SimpleNamespace(coverage=path, coverage_sha256=timing.common.sha(path), fixture='below_one', nodes=64)
    assert timing.coverage(args, {'source_sha256': 'source'})['original_helpers'] == value['helpers']


def test_speed_ratios_exclude_cold_and_incomplete_pairs():
    def row(warmup, complete, ratio):
        return dict(warmup=warmup, complete_pair=complete, speedup_control_over_candidate=ratio,
                    candidate_seconds=3., control_seconds=6.)
    rows = [row(True, True, 100.), row(False, False, 200.), row(False, True, 2.)]
    result = timing.summarize(rows)
    assert result['complete_warm_pairs'] == 1 and result['median_paired_speedup'] == 2.
    assert timing.summarize(rows[:2])['median_paired_speedup'] is None
    assert timing.order(0) == timing.order(2) == tuple(reversed(timing.order(1)))


def test_failed_trial_keeps_receipt(monkeypatch, tmp_path):
    def failed(*a):
        raise RuntimeError('unexpected failure')
    monkeypatch.setattr(timing.common, 'upload', failed)
    monkeypatch.setattr(timing.torch.cuda, 'reset_peak_memory_stats', lambda *a: None)
    monkeypatch.setattr(timing.common, 'timed', lambda device, function: (function(), .01))
    out = tmp_path/'trial.json'
    with pytest.raises(RuntimeError):
        timing.trial(out, timing.study.CANDIDATE, 1, None, None, None, None)
    assert json.loads(out.read_text())['status'] == 'failed'


def test_publication_retains_required_phase_measurements(monkeypatch, tmp_path):
    fit = SimpleNamespace(timings=dict(pilot_seconds=1., graph_build_seconds=2., path_seconds=3.,
                                      refit_seconds=4., stage_integrity_seconds=.01),
                          records=[], model=SimpleNamespace(device='cuda:0'))
    data = SimpleNamespace(input_sha256='input')
    seen = {}
    def export(fitted, provenance, **kwargs):
        seen.update(provenance)
        return SimpleNamespace()
    monkeypatch.setattr(timing, '_export', export)
    monkeypatch.setattr(timing, '_validate_result', lambda *a: None)
    monkeypatch.setattr(timing, '_publish', lambda exported, *a, **kw: exported)
    monkeypatch.setattr(timing.mixed, 'load_json', lambda *a: {})
    monkeypatch.setattr(timing.common, 'validate_public_search', lambda *a: None)
    monkeypatch.setattr(timing.common, 'validate_public_measurements', lambda *a: {})
    journal = timing.mixed.Journal(tmp_path/'trial.json')
    (journal.root/'public-output').mkdir()
    timing.publish(fit, data, journal, timing.study.CANDIDATE, 0., .2)
    assert seen['phase_seconds'] == dict(fit.timings, input_preparation_seconds=0.,
                                       device_upload_and_compile_seconds=.2)
    assert seen['outer_surrogate_policy'] == timing.study.CANDIDATE
