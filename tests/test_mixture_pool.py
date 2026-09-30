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
saved = {name: sys.modules.get(name) for name in ('common', 'lsf_helpers', 'queueing', 'controller')}
try:
    sys.modules['common'] = common
    queueing = load('queueing')
    sys.modules['queueing'] = queueing
    helpers = load('lsf_helpers')
    sys.modules['lsf_helpers'] = helpers
    controller, lifecycle = load('controller'), load('lifecycle')
    sys.modules['controller'] = controller
    managed_controller = load('managed_controller')
    expansion_contract = load('expansion_contract')
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
    owners = [dict(scheduler='lsf' if i < 22 else 'kubernetes',
                   family='lsf' if i < 22 else 'a100', worker=i) for i in range(36)]
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


def test_memory_routing_excludes_oversized_and_qualification_tasks(tmp_path):
    (tmp_path/'claims').mkdir()
    (tmp_path/'payload').mkdir()
    units = [dict(key='reference', case_id='reference', qualification_only=True),
             dict(key='oversized', case_id='oversized', eligible_families=[]),
             dict(key='large', case_id='large', eligible_families=['h100']),
             dict(key='small', case_id='small', eligible_families=['lsf', 'a100', 'h100'])]
    common.write(tmp_path/'payload/UNITS.json', units)
    common.write(tmp_path/'payload/IMPORTED.json', {})
    common.write(tmp_path/'payload/RUN_PLAN.json', {'frozen': True})
    assert queueing.claim(tmp_path, dict(scheduler='lsf'))['key'] == 'small'
    assert queueing.claim(tmp_path, dict(scheduler='kubernetes', family='a100')) is None
    assert queueing.claim(tmp_path, dict(scheduler='kubernetes', family='h100'))['key'] == 'large'
    assert not (tmp_path/'claims/oversized').exists()
    assert not (tmp_path/'claims/reference').exists()


def test_expansion_defers_only_tasks_that_need_fresh_seed_qualification(tmp_path):
    (tmp_path/'claims').mkdir()
    (tmp_path/'payload').mkdir()
    tasks = [dict(key='fresh', requires_seed=True), dict(key='ready', requires_seed=False)]
    for t in tasks:
        t['case_id'] = t['key']
    common.write(tmp_path/'payload/UNITS.json', tasks)
    common.write(tmp_path/'payload/IMPORTED.json', {})
    common.write(tmp_path/'payload/RUN_PLAN.json', {'frozen': True})
    owner = dict(scheduler='lsf')
    def ready(task):
        return expansion_contract.eligible(task, False)
    assert queueing.claim(tmp_path, owner, eligible=ready)['key'] == 'ready'
    assert queueing.claim(tmp_path, owner, eligible=ready) is None
    assert not (tmp_path/'claims/fresh').exists()
    assert queueing.claim(tmp_path, owner, eligible=lambda task: expansion_contract.eligible(task, True))['key'] == 'fresh'


def test_fresh_seed_gate_requires_current_plan_and_inventory(tmp_path):
    (tmp_path/'payload').mkdir()
    q = tmp_path/'qualification/lsf/QUALIFIED.json'
    q.parent.mkdir(parents=True)
    common.write(tmp_path/'payload/RUN_PLAN.json', {'plan': 'current'})
    common.write(tmp_path/'payload/INVENTORY.json', {'source': 'current'})
    assert expansion_contract.fresh_seeds_qualified(tmp_path) is False
    value = dict(status='passed', plan_sha256=common.sha(tmp_path/'payload/RUN_PLAN.json'),
                 inventory_sha256=common.sha(tmp_path/'payload/INVENTORY.json'))
    common.write(q, value)
    assert expansion_contract.fresh_seeds_qualified(tmp_path) is True
    value['inventory_sha256'] = 'old'
    common.write(q, value, replace=True)
    with pytest.raises(ValueError, match='qualification'):
        expansion_contract.fresh_seeds_qualified(tmp_path)


def test_unknown_scheduler_jobs_keep_capacity_and_foreign_jobs_fail(monkeypatch):
    active = {'12': dict(job_name='ours')}
    monkeypatch.setattr(controller, 'command', lambda *a: dict(code=1, stdout='', stderr='network'))
    assert controller.states_for(active)[0] == {'12': 'UNKNOWN'}
    monkeypatch.setattr(controller, 'command', lambda *a: dict(code=0, stdout='12 another RUN ours', stderr=''))
    with pytest.raises(ValueError, match='Foreign'):
        controller.states_for(active)


def test_managed_capacity_stays_reserved_until_all_parent_owners_exit(tmp_path, monkeypatch):
    parent = tmp_path/'parent'
    (parent/'payload').mkdir(parents=True)
    (parent/'receipts/case').mkdir(parents=True)
    contract = parent/'payload/RUN_PLAN.json'
    common.write(contract, dict(max_submitted=15, reserved_lsf_slots=1))
    p = dict(main_campaign=dict(root=str(parent), plan_sha256=common.sha(contract)))
    assert managed_controller.capacity(tmp_path, p) == 1
    common.write(parent/'receipts/controller-terminal.json', dict(queue_exhausted=True, active={}))
    owner = dict(pid=99999999, host=managed_controller.os.uname().nodename, start_ticks='1')
    common.write(parent/'receipts/controller-process.json', dict(identity=owner))
    monkeypatch.setattr(managed_controller, 'identity', lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    common.write(parent/'receipts/case/accepted.json', dict(job_id='123'))
    assert managed_controller.capacity(tmp_path, p) == 1
    reconciled = parent/'receipts/case/reconciled.json'
    common.write(reconciled, dict(scheduler_state='UNKNOWN'))
    assert managed_controller.capacity(tmp_path, p) == 1
    common.write(reconciled, dict(scheduler_state='DONE'), replace=True)
    assert managed_controller.capacity(tmp_path, p) == 15
    contract.write_text('{}')
    with pytest.raises(ValueError, match='authority'):
        managed_controller.capacity(tmp_path, p)


def test_managed_admission_requires_384_gb_and_one_exclusive_gpu():
    accepted = dict(job_id='123', job_name='managed', wall_minutes=4320,
                    memory_gb=384, gpu_model='NVIDIAA40')
    raw = ('Job <123>, Job Name <managed>, User <yding4>, Status <PSUSP>, '
           'Queue <egpu>, Command <worker>, Requested Resources <rusage[mem=384] span[hosts=1]>, '
           'Requested GPU <num=1:mode=exclusive_process:gmodel=NVIDIAA40>\n'
           ' RUNLIMIT\n 4320.0 min\n MEMLIMIT\n 384 G\n')
    assert helpers.verify_admission(raw, accepted, 'worker') == 'PSUSP'
    for old, new in [('384 G', '32 G'), ('mem=384', 'mem=32'), ('num=1:', 'num=2:')]:
        with pytest.raises(ValueError):
            helpers.verify_admission(raw.replace(old, new), accepted, 'worker')


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
