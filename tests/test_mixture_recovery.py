"""No retry can silently replace a success or change its scientific identity."""
import importlib.util
from pathlib import Path
import sys

import pytest

OPS = Path(__file__).resolve().parents[1]/'benchmarks/mixture_pool'


def load(name):
    spec = importlib.util.spec_from_file_location('terminal_recovery_'+name, OPS/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


names = ['common', 'case_helpers', 'queueing', 'worker']
saved = {name: sys.modules.get(name) for name in names}
try:
    for name in names:
        sys.modules[name] = load(name)
    common, worker = sys.modules['common'], sys.modules['worker']
    recovery = load('recovery')
finally:
    for name, original in saved.items():
        if original is None:
            del sys.modules[name]
        else:
            sys.modules[name] = original


def prepare(tmp_path):
    for name in ['payload', 'receipts/00001', 'claims/00001']:
        (tmp_path/name).mkdir(parents=True)
    common.write(tmp_path/'payload/UNITS.json', [dict(key='00001', case_id='sample')])
    terminal = tmp_path/'receipts/00001/terminal.json'
    claim = tmp_path/'claims/00001/claim.json'
    common.write(terminal, dict(case_id='sample', status='execution_failure'))
    common.write(claim, dict(owner='retired'))
    return dict(key='00001', prior_status='execution_failure', purpose='failed_case_recovery',
                terminal_sha256=common.sha(terminal), claim_sha256=common.sha(claim))


def test_changed_terminal_or_claim_cannot_authorize_retry(tmp_path):
    item = prepare(tmp_path)
    recovery.verify_predecessor(tmp_path, item)
    (tmp_path/'claims/00001/claim.json').write_text('{}')
    with pytest.raises(ValueError, match='predecessor identity'):
        recovery.verify_predecessor(tmp_path, item)


def test_success_cannot_be_reclassified_as_failed_case(tmp_path):
    item = prepare(tmp_path)
    path = tmp_path/'receipts/00001/terminal.json'
    path.write_text('{"case_id":"sample", "status":"validated"}')
    item.update(prior_status='validated', terminal_sha256=common.sha(path))
    with pytest.raises(ValueError, match='terminal failed'):
        recovery.verify_predecessor(tmp_path, item)


def test_recovery_execute_refuses_unbound_output_root(tmp_path, monkeypatch):
    control = tmp_path/'control/new'
    control.mkdir(parents=True)
    common.write(control/'GENERATION.json', dict(recovery=dict(root=str(tmp_path/'recoveries/new'), plan_sha256='bound')))
    monkeypatch.setattr(worker, 'generation', lambda root: control)
    with pytest.raises(ValueError, match='recovery output authority'):
        worker.execute(tmp_path, dict(key='00001'), {}, {}, {}, recovery=tmp_path/'outputs')
