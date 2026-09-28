"""Cross-scheduler ownership and activation regressions for Experimental."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

OPS = Path(__file__).resolve().parents[1]/'benchmarks/mixture_pool'


def load(name):
    spec = importlib.util.spec_from_file_location('mixture_pool_'+name, OPS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load('common')
saved = {name: sys.modules.get(name) for name in ('common', 'lsf_helpers', 'queueing')}
try:
    sys.modules['common'] = common
    queueing = load('queueing')
    sys.modules['queueing'] = queueing
    helpers = load('lsf_helpers')
    sys.modules['lsf_helpers'] = helpers
    controller, lifecycle = load('controller'), load('lifecycle')
finally:
    for name, original in saved.items():
        if original is None:
            del sys.modules[name]
        else:
            sys.modules[name] = original


def test_concurrent_scheduler_claims_are_disjoint_and_skip_imports(tmp_path):
    (tmp_path/'claims').mkdir()
    (tmp_path/'payload').mkdir()
    units = [dict(key=f'{i:05}', case_id=str(i)) for i in range(40)]
    common.write(tmp_path/'payload/UNITS.json', units)
    common.write(tmp_path/'payload/IMPORTED.json', {'00000': {}})
    common.write(tmp_path/'payload/RUN_PLAN.json', dict(frozen=True))
    # An interrupted mkdir before claim publication must never be stolen.
    (tmp_path/'claims/00001').mkdir()
    owners = [dict(scheduler='lsf' if i < 22 else 'kubernetes', worker=i) for i in range(36)]
    def drain(owner):
        claimed = []
        while (task := queueing.claim(tmp_path, owner)) is not None:
            claimed.append(task['key'])
        return claimed
    with ThreadPoolExecutor(max_workers=36) as pool:
        claims = sum(list(pool.map(drain, owners)), [])
    assert len(claims) == len(set(claims)) == 38
    assert set(claims) == {u['key'] for u in units[2:]}
    assert queueing.claim(tmp_path, owners[0]) is None


def test_stop_prevents_new_claims(tmp_path):
    (tmp_path/'STOP').write_text('user stopped')
    assert queueing.claim(tmp_path, {}) is None


def test_unknown_scheduler_jobs_keep_capacity_and_foreign_jobs_fail(monkeypatch):
    active = {'12': dict(job_name='ours')}
    monkeypatch.setattr(controller, 'command', lambda *a: dict(code=1, stdout='', stderr='network'))
    assert controller.states_for(active)[0] == {'12': 'UNKNOWN'}
    monkeypatch.setattr(controller, 'command', lambda *a: dict(code=0, stdout='12 another RUN ours', stderr=''))
    with pytest.raises(ValueError, match='Foreign'):
        controller.states_for(active)


def test_atomic_activation_tests_entire_bound_identity():
    bound = dict(metadata=dict(uid='uid', resourceVersion='v2', annotations=dict(seadragon_run_id='ours')),
        spec=dict(suspend=True, parallelism=1, selector={'bound': True}, template={'complete_spec': True}))
    received = []
    fake = SimpleNamespace(name='ours', job=lambda: bound, validate=lambda a, b: a,
                           pods=lambda uid: [], kube=lambda args: received.append(args) or bound)
    lifecycle.Lifecycle.patch(fake, bound, dict(suspend=False))
    import json
    patch = json.loads(received[0][received[0].index('-p')+1])
    tests = {v['path']: v['value'] for v in patch if v['op'] == 'test'}
    assert tests == {'/metadata/uid': 'uid', '/metadata/resourceVersion': 'v2',
        '/metadata/annotations/seadragon_run_id': 'ours', '/spec/suspend': True,
        '/spec/parallelism': 1, '/spec/selector': {'bound': True}, '/spec/template': {'complete_spec': True}}
    fake.pods = lambda uid: [dict(already_running=True)]
    with pytest.raises(ValueError, match='suspended, empty'):
        lifecycle.Lifecycle.patch(fake, bound, dict(suspend=False))


def test_profile_wrapper_preserves_successful_exit_and_records_memory(tmp_path, monkeypatch):
    import runpy
    script = tmp_path/'fit.py'
    script.write_text('import sys\nassert sys.argv[1:] == ["--fit-setting", "unchanged"]\nraise SystemExit(0)\n')
    output = tmp_path/'memory.json'
    fake = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
        get_device_properties=lambda index: SimpleNamespace(name='test', total_memory=1000),
        max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(sys, 'argv', ['profile_fit.py', str(output), str(script), '--fit-setting', 'unchanged'])
    runpy.run_path(str(OPS/'profile_fit.py'), run_name='__main__')
    assert common.read(output)['peak_reserved_bytes'] == 200
