"""Process admission and independent worker reconciliation regression coverage."""
import hashlib
import os
from pathlib import Path

import pytest

from benchmarks import run_complete_graph_cpu as runner


def stat_text(start='123', state='R'):
    fields = [state, '1', '1234', '1234'] + ['0'] * 16
    fields[19] = start
    return '1234 (worker) ' + ' '.join(fields)


def fake_proc(monkeypatch, commands, stats=None):
    commands = iter(commands)
    stats = iter(stats) if stats is not None else None
    monkeypatch.setattr(Path, 'read_bytes', lambda p: next(commands))
    monkeypatch.setattr(Path, 'read_text', lambda p: next(stats) if stats else stat_text())
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)


def test_identity_waits_past_empty_exec_window(monkeypatch):
    fake_proc(monkeypatch, [b'', b'python\0worker\0'])
    result = runner.identity(1234, ['python', 'worker'])
    assert result['cmdline_sha256'] == hashlib.sha256(b'python\0worker\0').hexdigest()
    assert result['start_ticks'] == '123'


def test_identity_never_admits_persistently_empty_command(monkeypatch):
    fake_proc(monkeypatch, [b''])
    with pytest.raises(TimeoutError, match='nonempty'):
        runner.identity(1234, timeout=0)


def test_identity_rejects_reused_pid(monkeypatch):
    fake_proc(monkeypatch, [b'python\0'], [stat_text('123'), stat_text('124')])
    with pytest.raises(RuntimeError, match='identity changed'):
        runner.identity(1234)


def test_identity_rejects_wrong_nonempty_command(monkeypatch):
    fake_proc(monkeypatch, [b'foreign\0'])
    with pytest.raises(RuntimeError, match='admitted argv'):
        runner.identity(1234, ['python'])


def test_identity_rejects_zombie(monkeypatch):
    fake_proc(monkeypatch, [b''], [stat_text(state='Z'), stat_text(state='Z')])
    with pytest.raises(ProcessLookupError):
        runner.identity(1234)


def test_identity_matches_real_current_process():
    expected = Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    result = runner.identity(os.getpid(), expected)
    assert result['pid'] == os.getpid()
    assert result['cmdline_sha256'] != hashlib.sha256(b'').hexdigest()


def setup_controller(monkeypatch, tmp_path):
    for name in ['receipts', 'logs', 'results']:
        (tmp_path/name).mkdir()
    cases = [dict(key=k, priority=i, retained_mutations=1, wall_minutes=0) for i, k in enumerate(['a', 'b'])]
    runner.write(tmp_path/'cases.json', cases)
    (tmp_path/'payload.zip').write_bytes(b'fixture')
    plan = dict(cpus=list(range(25)), workers=25, worker_rss_limit_bytes=10**9,
                host_headroom_bytes=0, payload=str(tmp_path/'payload.zip'),
                payload_sha256=runner.digest(tmp_path/'payload.zip'),
                cpu_keys=['a', 'b'], memory_budget_bytes=10**12,
                deadline_utc='2100-01-01T00:00:00+00:00')
    runner.write(tmp_path/'PLAN.json', plan)
    monkeypatch.setattr(runner, 'verify', lambda _: plan)
    monkeypatch.setattr(runner, 'mem_available', lambda: 10**12)
    monkeypatch.setattr(runner.os, 'sched_getaffinity', lambda _: set(range(25)))
    monkeypatch.setattr(runner.signal, 'signal', lambda *a: None)
    return plan


def test_worker_identity_error_does_not_block_other_reconciliation(monkeypatch, tmp_path):
    setup_controller(monkeypatch, tmp_path)
    generation = [0]
    events = []
    real_write = runner.write

    def write(path, value, replace=False):
        events.append(path.name)
        return real_write(path, value, replace)

    monkeypatch.setattr(runner, 'write', write)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: generation.__setitem__(0, generation[0]+1))
    monkeypatch.setattr(runner.time, 'monotonic', lambda: generation[0] * 10)

    def identity(pid, expected_argv=None):
        if pid == 1234 and generation[0] == 1:
            raise RuntimeError('injected worker identity mismatch')
        return dict(pid=pid, pgrp=pid, session=pid)

    monkeypatch.setattr(runner, 'identity', identity)
    original_read_text = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **kw: 'VmRSS: 1 kB\n' if str(p).startswith('/proc/') else original_read_text(p, *a, **kw))

    class Process:
        def __init__(self, argv, **kwargs):
            self.key = argv[argv.index('--key')+1]
            self.pid = 1234 if self.key == 'a' else 1235
            task = tmp_path/'results'/self.key
            (task/'output').mkdir(parents=True)
            runner.write(task/'validated.json', dict(output_sha256={}))
            runner.write(task/'terminal.json', dict(status='validated_complete', validated_sha256=runner.digest(task/'validated.json')))

        def poll(self):
            return None if self.key == 'a' and generation[0] < 2 else 0

    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    assert runner.control(tmp_path) == 1  # Integrity error prevents new admission.
    assert events.index('a.controller-error.json') < events.index('b.reconciled.json') < events.index('a.reconciled.json')
    terminal = runner.read(tmp_path/'receipts/controller-terminal.json')
    assert terminal['counts'] == {'validated_complete': 2}
    assert terminal['halt_reason']['error'] == 'injected worker identity mismatch'


def test_identity_admission_failure_stops_unregistered_child(monkeypatch, tmp_path):
    setup_controller(monkeypatch, tmp_path)
    events = []

    def identity(pid, expected_argv=None):
        if expected_argv is not None:
            raise TimeoutError('injected empty command')
        return dict(pid=pid, pgrp=pid, session=pid)

    monkeypatch.setattr(runner, 'identity', identity)

    class Process:
        pid = 1234
        returncode = -15

        def __init__(self, *args, **kwargs):
            events.append('spawn')

        def terminate(self):
            events.append('terminate')

        def wait(self, timeout):
            events.append('wait')

    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    assert runner.control(tmp_path) == 1
    assert events == ['spawn', 'terminate', 'wait']
    assert not list((tmp_path/'receipts').glob('*.accepted.json'))
    assert runner.read(tmp_path/'receipts/a.admission-failed.json')['returncode'] == -15
    assert runner.read(tmp_path/'receipts/controller-error.json')['error_type'] == 'TimeoutError'


def test_worker_exit_racing_timeout_is_reaped_without_identity_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(Path, 'read_text', lambda p: 'VmRSS: 1 kB\n')
    def exited(_):
        raise ProcessLookupError('worker exited between poll and identity')
    monkeypatch.setattr(runner, 'identity', exited)
    process = SimpleNamespace(pid=1234, poll=lambda: None)
    active = dict(process=process, identity={}, began=0, timeout=0)
    assert runner.poll_worker(tmp_path, dict(worker_rss_limit_bytes=10**12), 'case', active) is None
    assert 'stop_reason' not in active
