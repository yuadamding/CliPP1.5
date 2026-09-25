"""Frozen local CPU pool: one process/core per case, disjoint from LSF ownership."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()


def identity(pid, expected_argv=None, timeout=5.0):
    """Capture one live process after exec, never an empty/transitional command."""
    proc = Path('/proc')/str(pid)
    deadline = time.monotonic() + timeout
    expected = None if expected_argv is None else b'\0'.join(os.fsencode(v) for v in expected_argv) + b'\0'
    birth = None
    while True:
        before = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        command = (proc/'cmdline').read_bytes()
        after = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        birth = before[19] if birth is None else birth
        if before[19] != birth or after[19] != birth:
            raise RuntimeError('Process identity changed during capture')
        if before[0] in ('Z', 'X') or after[0] in ('Z', 'X'):
            raise ProcessLookupError(f'Process {pid} has exited')
        if command and (before[2], before[3]) == (after[2], after[3]):
            if expected is not None and command != expected:
                raise RuntimeError('Process command does not match admitted argv')
            return dict(pid=pid, start_ticks=birth, cmdline_sha256=hashlib.sha256(command).hexdigest(),
                        pgrp=int(after[2]), session=int(after[3]))
        if time.monotonic() >= deadline:
            raise TimeoutError(f'No stable nonempty command for process {pid}')
        time.sleep(0.01)


def verify(root):
    plan = read(root/'PLAN.json')
    assert str(root) == plan['local_root'] and sys.executable == plan['python']
    assert digest(root/'cases.json') == plan['cases_sha256']
    assert digest(root/'PARTITION.json') == plan['partition_sha256']
    partition = read(root/'PARTITION.json')
    assert set(plan['cpu_keys']) == set(partition['cpu_keys'])
    assert not set(plan['cpu_keys']) & (set(partition['gpu_keys']) | set(partition['already_owned_keys']))
    for name, sha in read(root/'inventory.json').items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p = root/name
        assert p.is_file() and not p.is_symlink() and digest(p) == sha, name
    return plan


def worker(root, key, cpu):
    # Must run before NumPy/Torch are imported by adapter or validator modules.
    sys.path.insert(0, str(root/'source'))
    sys.path.insert(0, str(root/'source/src'))
    from benchmarks.benchmark_chain import configure_execution
    controls = configure_execution(cpus=[cpu], threads=1)
    plan = verify(root)
    assert key in plan['cpu_keys'] and cpu in plan['cpus']
    case = next(c for c in read(root/'cases.json') if c['key'] == key)
    task = root/'results'/key
    task.mkdir()
    began = time.monotonic()
    from benchmarks.cohort_failures import failure_record
    phase = 'setup'
    terminal = dict(key=key, status='setup_failure', started_utc=now())
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from clipp1d.api import source_provenance
        from clipp1d.cuda.policy import QualificationError
        from benchmarks.cohort_staging import stage_case_input
        from benchmarks.fit_complete_graph_cpu import fit
        from benchmarks.validate_cpu_cohort import validate
        source = source_provenance()
        assert source['source_sha256'] == plan['source_sha256']
        assert {k: source[k] for k in plan['numerical_environment']} == plan['numerical_environment']
        write(task/'startup.json', dict(identity=identity(os.getpid()), controls=controls,
              source=source, base_commit=plan['base_commit'], plan_sha256=digest(root/'PLAN.json'),
              cpu=cpu, utc=now(), threads=torch.get_num_threads()))
        with zipfile.ZipFile(plan['payload']) as bundle:
            input_file = stage_case_input(task, case, bundle)
            truth = bundle.read(case['truth_member'])
            assert hashlib.sha256(truth).hexdigest() == case['truth_sha256']
            with (task/'truth.tsv').open('xb') as stream:
                stream.write(truth)
        phase = 'fit'
        try:
            result = fit(input_file, task/'output', max_major_cn=4,
                         memory_budget_bytes=plan['memory_budget_bytes'],
                         expected_source_sha256=plan['source_sha256'])
        except (QualificationError, MemoryError) as error:
            terminal.update(status='resource_failure' if isinstance(error, MemoryError) else 'scientific_failure',
                            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            return 20
        phase = 'validation'
        validated = validate(case, task/'output', plan['source_sha256'], result)
        validated.update(utc=now(), base_commit=plan['base_commit'], input_sha256=case['input_sha256'],
                         operation_metrics=result.operation_metrics, backend='cpu',
                         cpu_adapter_sha256=result.provenance['cpu_adapter_sha256'])
        write(task/'validated.json', validated)
        terminal.update(status='validated_'+result.search_status, validated_sha256=digest(task/'validated.json'))
        return 0
    except BaseException as error:
        terminal.update(failure_record(error, phase))
        return 1
    finally:
        terminal.update(finished_utc=now(), elapsed_seconds=time.monotonic()-began)
        write(task/'terminal.json', terminal)


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('No host memory headroom measurement')


def borrowed_cpus(plan):
    if not plan.get('draining_parent'):
        return set()
    parent = Path(plan['draining_parent'])
    progress = read(parent/'receipts/controller-progress.json')
    assert progress['phase'] in ('halted_draining', 'halted') and progress['halt_reason']
    accepted = {p.name.split('.')[0] for p in (parent/'receipts').glob('*.accepted.json')}
    assert accepted == set(plan['draining_parent_accepted']), 'Draining parent admitted new work'
    if not (parent/'receipts/controller-terminal.json').exists():
        assert identity(plan['draining_parent_owner']['pid']) == plan['draining_parent_owner']
    return {a['cpu'] for a in progress['active'].values()}


def deferred_parent_outcome(plan, key):
    """Wait for the preserved worker, then reuse success or permit one retry."""
    parent = Path(plan['draining_parent'])
    receipt = parent/'receipts'/f'{key}.reconciled.json'
    if not receipt.exists():
        return None
    outcome = read(receipt)['outcome']
    status = outcome['status']
    if status in {'validated_complete', 'validated_incomplete'}:
        task = parent/'results'/key
        assert digest(task/'validated.json') == outcome['validated_sha256']
        for name, sha in read(task/'validated.json')['output_sha256'].items():
            assert digest(task/'output'/name) == sha
        action = 'reuse'
    else:
        from benchmarks.cohort_failures import ISOLATED_FAILURES
        assert status in ISOLATED_FAILURES, f'Parent integrity failure: {key}: {status}'
        permitted = plan.get('deferred_parent_retry_keys')
        action = 'retry' if permitted is None or key in permitted else 'retain_failure'
    return dict(action=action, status=status, root=str(parent), receipt_sha256=digest(receipt))


def poll_worker(root, plan, key, a):
    """Reconcile one worker independently, preserving strict process/output checks."""
    process = a['process']
    code = process.poll()
    if code is None:
        rss = 0
        try:
            status_lines = Path('/proc', str(process.pid), 'status').read_text().splitlines()
        except FileNotFoundError:
            return None  # Exit raced poll(); reap with the next poll.
        for line in status_lines:
            if line.startswith('VmRSS:'):
                rss = int(line.split()[1])*1024
        if not a.get('stopping') and (time.monotonic()-a['began'] > a['timeout'] or rss > plan['worker_rss_limit_bytes']):
            try:
                assert identity(process.pid) == a['identity']
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return None  # Exit raced the timeout; poll/reap on the next pass.
            a['stop_reason'] = 'resource_timeout' if time.monotonic()-a['began'] > a['timeout'] else 'resource_failure'
            a['stopping'] = time.monotonic()
        elif a.get('stopping') and time.monotonic()-a['stopping'] > 15:
            try:
                assert identity(process.pid) == a['identity']
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return None
        return None
    task = root/'results'/key
    terminal = read(task/'terminal.json') if (task/'terminal.json').exists() else dict(status='execution_failure', error='No worker terminal receipt')
    if a.get('stop_reason'):
        terminal = dict(terminal, application_status=terminal['status'], status=a['stop_reason'])
    if terminal['status'].startswith('validated_'):
        assert code == 0 and digest(task/'validated.json') == terminal['validated_sha256']
        v = read(task/'validated.json')
        for name, sha in v['output_sha256'].items():
            assert digest(task/'output'/name) == sha
    record = dict(outcome=terminal, returncode=code, identity=a['identity'], utc=now())
    receipt = root/'receipts'/f'{key}.reconciled.json'
    if receipt.exists():
        prior = read(receipt)
        assert all(prior[k] == record[k] for k in ('outcome', 'returncode', 'identity'))
    else:
        write(receipt, record)
    return terminal


def control(root):
    plan = verify(root)
    sys.path.insert(0, str(root/'source'))
    lock = (root/'receipts/controller.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert len(plan['cpus']) == plan['workers'] == 25
    assert set(plan['cpus']) <= set(os.sched_getaffinity(0))
    assert mem_available() >= (plan['workers']-len(borrowed_cpus(plan)))*plan['worker_rss_limit_bytes'] + plan['host_headroom_bytes']
    assert digest(plan['payload']) == plan['payload_sha256']
    cases = [c for c in read(root/'cases.json') if c['key'] in set(plan['cpu_keys'])]
    assert len(cases) == len(plan['cpu_keys']) and cases == sorted(cases, key=lambda c: c['priority'])
    deferred = {c['key']: c for c in cases if c['key'] in plan.get('deferred_parent_keys', [])}
    active, finished, queue = {}, {}, [c for c in cases if c['key'] not in deferred]
    halted = None
    write(root/'receipts/controller-process.json', dict(identity=identity(os.getpid()), utc=now(), plan_sha256=digest(root/'PLAN.json')))
    def stop(signum, frame):
        nonlocal halted
        halted = dict(reason='Requested controller stop; draining admitted workers', signal=signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def snapshot():
        write(root/'receipts/controller-progress.json', dict(utc=now(), phase='halted_draining' if halted and active else 'halted' if halted else 'complete' if not active and not queue and not deferred else 'running',
              active={key:dict(pid=a['process'].pid, cpu=a['cpu'], identity=a['identity'], started_utc=a['started_utc'],
                              elapsed_seconds=time.monotonic()-a['began']) for key, a in active.items()},
              finished_counts=dict(Counter(finished.values())), finished=len(finished), planned=len(cases),
              remaining=len(queue), deferred_parent=len(deferred), max_workers=25, borrowed_active=len(borrowed_cpus(plan)), halt_reason=halted), replace=True)
    while active or queue or deferred:
        try:
            for key, case in list(deferred.items()):
                outcome = deferred_parent_outcome(plan, key)
                if outcome is None:
                    continue
                write(root/'receipts'/f'{key}.parent-resolved.json', outcome)
                if outcome['action'] == 'retry':
                    queue.append(case)
                    queue.sort(key=lambda c: c['priority'])
                del deferred[key]
            for key, a in list(active.items()):
                try:
                    terminal = poll_worker(root, plan, key, a)
                    if terminal is None:
                        continue
                    finished[key] = terminal['status']
                    a['out'].close()
                    a['err'].close()
                    del active[key]
                    if terminal['status'] not in {'validated_complete', 'validated_incomplete', 'scientific_failure', 'resource_failure', 'resource_timeout', 'execution_failure'}:
                        halted = halted or dict(key=key, outcome=terminal)
                except Exception as error:
                    failure = dict(key=key, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), utc=now())
                    path = root/'receipts'/f'{key}.controller-error.json'
                    if not path.exists():
                        write(path, failure)
                    halted = halted or failure
                    # Hold this slot, but still reap/reconcile other admitted workers.
            borrowed = borrowed_cpus(plan)
            assert not borrowed & {a['cpu'] for a in active.values()}
            cpus = [cpu for cpu in plan['cpus'] if cpu not in borrowed | {a['cpu'] for a in active.values()}]
            if datetime.now(timezone.utc) >= datetime.fromisoformat(plan['deadline_utc']):
                halted = halted or dict(reason='Inherited cohort deadline exhausted')
            while not halted and queue and cpus:
                case = queue[0]
                estimate = 32*8*case['retained_mutations']**2+64*4096*8*4
                assert estimate <= plan['memory_budget_bytes']
                if mem_available() < plan['host_headroom_bytes'] + estimate + 1024**3:
                    break
                cpu = cpus.pop(0)
                key = case['key']
                write(root/'receipts'/f'{key}.intent.json', dict(cpu=cpu, utc=now(), plan_sha256=digest(root/'PLAN.json')))
                out = (root/'logs'/f'{key}.out').open('xb')
                err = (root/'logs'/f'{key}.err').open('xb')
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
                argv = [sys.executable, '-B', str(root/'source/benchmarks/run_complete_graph_cpu.py'),
                        'worker', str(root), '--key', key, '--cpu', str(cpu)]
                process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=err, start_new_session=True)
                try:
                    owned = identity(process.pid, expected_argv=argv)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
                    write(root/'receipts'/f'{key}.admission-failed.json', dict(pid=process.pid, argv=argv, returncode=process.returncode, traceback=traceback.format_exc(), utc=now()))
                    out.close()
                    err.close()
                    raise
                assert owned['pgrp'] == owned['session'] == process.pid
                active[key] = dict(process=process, identity=owned, cpu=cpu, began=time.monotonic(), started_utc=now(),
                                   timeout=case['wall_minutes']*60, out=out, err=err)
                queue.pop(0)
                write(root/'receipts'/f'{key}.accepted.json', dict(identity=owned, cpu=cpu, utc=now(), argv=argv))
            snapshot()
        except BaseException as error:
            failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), utc=now())
            if not (root/'receipts/controller-error.json').exists():
                write(root/'receipts/controller-error.json', failure)
            halted = halted or failure
            snapshot()
        if halted and not active:
            break
        time.sleep(2)
    write(root/'receipts/controller-terminal.json', dict(complete=not halted and not queue and not active and not deferred,
          counts=dict(Counter(finished.values())), halt_reason=halted, utc=now()))
    return int(bool(halted))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['control', 'worker'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--key')
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    return control(root) if args.action == 'control' else worker(root, args.key, args.cpu)


if __name__ == '__main__':
    raise SystemExit(main())
