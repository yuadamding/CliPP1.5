"""One extra LSF slot, allocated qualification, three publication canaries, panel."""
from collections import Counter
import fcntl
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

from common import (GPU_MODELS, admitted_gpu_model, command, identity, now, plan,
                    qualified_gpu_model, read, sha, validate_outputs, verify, write)


def field(pattern, raw):
    values = re.findall(pattern, raw, re.M)
    if len(values) != 1:
        raise ValueError(f'Ambiguous scheduler field: {pattern}')
    return values[0]


def verify_admission(raw, accepted, worker_command):
    model = accepted['gpu_model']
    if model not in GPU_MODELS.values():
        raise ValueError('Unsupported requested GPU model')
    expected = [(r'^Job <(\d+)>', accepted['job_id']),
        (r'Job Name <([^>]+)>', accepted['job_name']), (r'User <([^>]+)>', 'yding4'),
        (r'Queue <([^>]+)>', 'egpu'), (r'Command <([^>]+)>', worker_command),
        (r'Requested Resources <([^>]+)>', 'rusage[mem=32] span[hosts=1]'),
        (r'Requested GPU <([^>]+)>', f'num=1:mode=exclusive_process:gmodel={model}')]
    for pattern, value in expected:
        if field(pattern, raw) != value:
            raise ValueError('Different admitted scheduler resources/identity')
    if (float(field(r'^ MEMLIMIT\s*\n\s*(\d+(?:\.\d+)?) G\b', raw)) != 32 or
            float(field(r'^ RUNLIMIT\s*\n\s*(\d+(?:\.\d+)?) min\b', raw)) != accepted['wall_minutes'] or
            re.findall(r',\s*(\d+)\s+Task\(s\)', raw) not in ([], ['1'])):
        raise ValueError('Different admitted memory, runtime or CPU count')
    state = field(r'Status <([^>]+)>', raw)
    if state not in ('PEND', 'PSUSP'):
        raise ValueError('Unexpected held job state')
    return state


def scheduler_state(accepted):
    q = command(['bjobs', '-a', '-noheader', '-o', 'jobid user stat job_name:200', accepted['job_id']])
    rows = [r.split() for r in q['stdout'].splitlines() if r.strip()]
    if not rows:
        return 'UNKNOWN', q
    if (len(rows) != 1 or len(rows[0]) != 4 or
            rows[0][0] != accepted['job_id'] or rows[0][1] != 'yding4' or
            rows[0][3] != accepted['job_name']):
        raise ValueError('Foreign or ambiguous scheduler identity')
    return rows[0][2], q


def terminal_log_state(path, accepted):
    """Aged-out jobs need their own identity-bound scheduler report, not absence."""
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        raw = stream.read().decode(errors='replace')
    if (f'Subject: Job {accepted["job_id"]}: <{accepted["job_name"]}>' not in raw or
            not re.search(r'Job <'+re.escape(accepted['job_name'])+r'> was submitted .*by user <yding4>', raw) or
            'Results reported at ' not in raw or 'Terminated at ' not in raw):
        return None
    if 'Successfully completed.' in raw:
        return 'DONE'
    if re.search(r'Exited with exit code|Exited with signal termination|Exited by signal|TERM_(?:MEMLIMIT|RUNLIMIT)', raw):
        return 'EXIT'
    return None


def sizing(task, canaries):
    if not canaries:
        return 60
    # Includes actual interpreter startup, compile, fit, selector and validation.
    # Bound runtime only; it does not alter the EM or candidate budgets.
    n = task['retained_mutations']
    observed = max(c['elapsed_seconds'] for c in canaries)
    largest_n = max(c['retained_mutations'] for c in canaries)
    minutes = max(10, math.ceil(5*observed*max(1, n/largest_n)/60))
    if minutes > 180:
        raise ValueError('Measured workload exceeds frozen 180-minute admission ceiling')
    return minutes


def submit(root, p, task, minutes):
    model = admitted_gpu_model(root)
    key = task['key']
    folder = root/'receipts'/key
    folder.mkdir()  # Never repeat a submitted/uncertain case.
    name = p['job_prefix']+key
    worker = f'{p["python"]} -B {root}/payload/ops/worker.py {root} {key}'
    argv = ['bsub', '-H', '-q', 'egpu', '-J', name, '-n', '1', '-M', '32',
        '-R', 'rusage[mem=32] span[hosts=1]', '-gpu', f'num=1:mode=exclusive_process:gmodel={model}',
        '-W', str(minutes), '-cwd', str(root), '-oo', str(root/'lsf-logs'/f'{key}.%J.out'),
        '-eo', str(root/'lsf-logs'/f'{key}.%J.err'), worker]
    write(folder/'submit-intent.json', dict(argv=argv, utc=now(), plan_sha256=sha(root/'payload/RUN_PLAN.json')))
    response = command(argv, 120)
    write(folder/'submit-response.json', response)
    ids = re.findall(r'Job <(\d+)> is submitted', response['stdout'])
    if len(ids) == 1:
        accepted = dict(key=key, job_id=ids[0], job_name=name, initially_held=True,
            wall_minutes=minutes, cpu=1, memory_gb=32, gpus=1, gpu_model=model,
            plan_sha256=sha(root/'payload/RUN_PLAN.json'))
        write(folder/'accepted.json', accepted)
    if response['code'] != 0 or len(ids) != 1 or 'queue <egpu>' not in response['stdout']:
        raise RuntimeError('Ambiguous submission; reconcile before any new action')
    observations = []
    deadline = time.monotonic()+300
    try:
        while True:
            q = command(['bjobs', '-UF', accepted['job_id']])
            if q['code'] != 0:
                raise RuntimeError('Could not verify held admission')
            state = verify_admission(q['stdout'], accepted, worker)
            observation = dict(job_id=accepted['job_id'], state=state, raw=q['stdout'], utc=now())
            observations.append(observation)
            if state == 'PSUSP':
                write(folder/'admitted.json', observation)
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Leave held job for reconciliation')
            time.sleep(5)
    finally:
        write(folder/'admission-observations.json', observations)
    write(folder/'release-intent.json', dict(job_id=accepted['job_id'], utc=now()))
    response = command(['bresume', accepted['job_id']])
    write(folder/'release-response.json', response)
    if response['code'] != 0 or accepted['job_id'] not in response['stdout']:
        raise RuntimeError('Uncertain release; never submit a duplicate')
    return accepted


def completion(root, active, task, state, query):
    if state not in ('DONE', 'EXIT'):
        return None
    folder = root/'receipts'/task['key']
    terminal_path = folder/'terminal.json'
    terminal = read(terminal_path) if terminal_path.is_file() else dict(
        key=task['key'], job_id=active['job_id'], status='missing_application_terminal')
    if terminal['key'] != active['key'] or terminal['job_id'] != active['job_id']:
        raise ValueError('Terminal identity mismatch')
    if terminal['status'] == 'validated':
        if state != 'DONE':
            raise ValueError('Validated application did not exit successfully')
        validated = root/'outputs'/task['key']/'VALIDATED.json'
        if sha(validated) != terminal['validated_sha256'] or read(validated)['authority'] != validate_outputs(root, task):
            raise ValueError('Changed validated output')
    record = dict(accepted=active, scheduler_state=state, query=query,
                  terminal=terminal, utc=now())
    write(folder/'reconciled.json', record)
    return terminal


def qualified(root, p):
    qroot = Path(p['qualification']['root'])
    accepted = read(qroot/'receipts/ACCEPTED.json')
    if accepted['job_id'] != p['qualification']['job_id'] or accepted['archive_sha256'] != p['qualification']['archive_sha256']:
        raise ValueError('Different qualification job/source')
    change = p['qualification'].get('resource_change')
    if change and sha(change['path']) != change['sha256']:
        raise ValueError('Changed qualification resource-transition receipt')
    state, query = scheduler_state(accepted)
    if state == 'UNKNOWN':
        state = terminal_log_state(qroot/'logs/stdout.log', accepted) or state
    path = qroot/'output/QUALIFICATION.json'
    if path.exists() and read(path)['status'] != 'passed':
        raise ValueError('CUDA qualification failed; panel remains unsubmitted')
    if state == 'EXIT':
        raise ValueError('Qualification exited unsuccessfully')
    if state != 'DONE':
        return False, state
    q = read(path)
    if (q['status'] != 'passed' or q['job_id'] != accepted['job_id'] or
            q['inventory_sha256'] != p['qualification']['inventory_sha256'] or
            len(q['paired_cases']) != 3):
        raise ValueError('Incomplete qualification evidence')
    before = read(qroot/'payload/INVENTORY.json')
    after = read(root/'payload/INVENTORY.json')
    for name, digest in before.items():
        if name.startswith('source/src/') or name in (
                'source/benchmarks/run_mixture_experiment.py', 'source/benchmarks/mixture_structural_guard.py'):
            if after.get(name) != digest:
                raise ValueError('Panel numerical source differs from qualified source')
    model = qualified_gpu_model(qroot, q)
    if model not in p['qualified_gpu_models']:
        raise ValueError('Qualified GPU outside frozen execution contract')
    write(root/'receipts/qualification-admitted.json', dict(qualification_sha256=sha(path),
          job_id=accepted['job_id'], gpu_model=model, query=query, utc=now()))
    return True, state


def controller(root):
    p = plan(root)
    lock = (root/'receipts/controller.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(30):
        if (root/'receipts/controller-process.json').exists():
            break
        time.sleep(1)
    if identity(os.getpid()) != read(root/'receipts/controller-process.json')['identity']:
        raise ValueError('Unbound controller identity')
    verify(root, full=True)
    tasks = read(root/'payload/UNITS.json')
    if len(tasks) != p['cases'] or [u['key'] for u in tasks] != [f'{i:05}' for i in range(len(tasks))]:
        raise ValueError('Different workload inventory')
    if [u['publication_canary'] for u in tasks[:3]] != [True]*3 or any(u['publication_canary'] for u in tasks[3:]):
        raise ValueError('Different publication qualification contract')
    write(root/'receipts/controller-ready.json', dict(identity=identity(os.getpid()), utc=now()))
    started = time.monotonic()
    active, finished, canaries = None, [], []
    phase, state = 'cuda_qualification', None
    try:
        while True:
            if (root/'STOP').exists() or time.monotonic()-started > 90*86400:
                raise RuntimeError('Study stop/deadline: no new submissions')
            if phase == 'cuda_qualification':
                ready, state = qualified(root, p)
                if ready:
                    phase, state = 'publication_canaries', None
                elif time.monotonic()-started > 72*3600:
                    raise TimeoutError('CUDA qualification wait budget exceeded')
            if phase != 'cuda_qualification':
                task = tasks[len(finished)] if len(finished) < len(tasks) else None
                if active:
                    state, query = scheduler_state(active)
                    if state == 'UNKNOWN':
                        state = terminal_log_state(root/'lsf-logs'/f'{task["key"]}.{active["job_id"]}.out', active) or state
                    result = completion(root, active, task, state, query)
                    if result is not None:
                        finished.append(result)
                        active, state = None, None
                        if task['publication_canary']:
                            if result['status'] != 'validated' or result['canary']['status'] != 'passed':
                                raise ValueError('Publication canary failed; panel remains unsubmitted')
                            canaries.append(dict(elapsed_seconds=result['elapsed_seconds'], retained_mutations=task['retained_mutations']))
                        elif result['status'] in ('setup_failure', 'output_validation_failure'):
                            raise ValueError('Infrastructure/output validation failure; stop refills')
                        if len(finished) >= 3 and all(r['status'] != 'validated' for r in finished[-3:]):
                            raise ValueError('Three consecutive case failures; stop refills')
                if len(finished) == 3 and phase == 'publication_canaries':
                    write(root/'receipts/panel-admitted.json', dict(canaries=canaries, utc=now(), max_submitted=1))
                    phase = 'panel'
                if not active and len(finished) < len(tasks):
                    task = tasks[len(finished)]
                    active = submit(root, p, task, 60 if task['publication_canary'] else sizing(task, canaries))
                    state = 'PEND'
                status = 'complete' if len(finished) == len(tasks) else 'running'
            else:
                status = 'waiting_for_cuda_qualification'
            progress = dict(status=status, phase=phase, planned=len(tasks), completed=len(finished),
                validated=sum(r['status'] == 'validated' for r in finished),
                outcomes=dict(Counter(r['status'] for r in finished)), active=active, scheduler_state=state,
                max_submitted=1, utc=now(), controller_identity=identity(os.getpid()))
            write(root/'PROGRESS.json', progress, replace=True)
            if status == 'complete':
                write(root/'COMPLETE.json', dict(progress, accuracy_accepted=False,
                    production_adopted=False, failures=[r for r in finished if r['status'] != 'validated']))
                return
            time.sleep(60)
    except BaseException as exc:
        # An unknown or Pending job retains the slot. No cancellation/retry here.
        write(root/'receipts/controller-halted.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            active=active, finished=len(finished), outcomes=dict(Counter(r['status'] for r in finished)), utc=now()))
        raise


if __name__ == '__main__':
    controller(Path(sys.argv[1]).resolve())
