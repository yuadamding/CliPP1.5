



import math






import re




import time




from common import (GPU_MODELS, admitted_gpu_model, command, now, read, sha, validate_outputs, write)


def field(pattern, raw):
    values = re.findall(pattern, raw, re.M)
    if len(values) != 1:
        raise ValueError(f'Ambiguous scheduler field: {pattern}')
    return values[0]


def verify_admission(raw, accepted, worker_command):
    model = accepted['gpu_model']
    memory = accepted.get('memory_gb', 32)
    if memory not in (32, 384):
        raise ValueError('Unsupported host memory contract')
    if model not in GPU_MODELS.values():
        raise ValueError('Unsupported requested GPU model')
    expected = [(r'^Job <(\d+)>', accepted['job_id']),
        (r'Job Name <([^>]+)>', accepted['job_name']), (r'User <([^>]+)>', 'yding4'),
        (r'Queue <([^>]+)>', 'egpu'), (r'Command <([^>]+)>', worker_command),
        (r'Requested Resources <([^>]+)>', f'rusage[mem={memory}] span[hosts=1]'),
        (r'Requested GPU <([^>]+)>', f'num=1:mode=exclusive_process:gmodel={model}')]
    for pattern, value in expected:
        if field(pattern, raw) != value:
            raise ValueError('Different admitted scheduler resources/identity')
    if (float(field(r'^ MEMLIMIT\s*\n\s*(\d+(?:\.\d+)?) G\b', raw)) != memory or
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
    memory = p.get('host_memory_gb', 32)
    if memory not in (32, 384):
        raise ValueError('Unsupported host memory contract')
    key = task['key']
    folder = root/'receipts'/key
    folder.mkdir()  # Never repeat a submitted/uncertain case.
    name = p['job_prefix']+key
    entry = root/'payload/ops/worker.py'
    generation_sha = p.get('worker_generation_sha256')
    if generation_sha is not None:
        from pathlib import Path
        entry = Path(p['worker_entrypoint'])
        control = entry.parent
        if (control.resolve() != control or not control.is_relative_to(root/'control') or
                sha(control/'GENERATION.json') != generation_sha or
                read(control/'GENERATION.json')['inventory'].get(entry.name) != sha(entry)):
            raise ValueError('Unbound scalar worker generation')
    worker = f'{p["python"]} -B {entry} {root} {key}'
    argv = ['bsub', '-H', '-q', 'egpu', '-J', name, '-n', '1', '-M', str(memory),
        '-R', f'rusage[mem={memory}] span[hosts=1]', '-gpu', f'num=1:mode=exclusive_process:gmodel={model}',
        '-W', str(minutes), '-cwd', str(root), '-oo', str(root/'lsf-logs'/f'{key}.%J.out'),
        '-eo', str(root/'lsf-logs'/f'{key}.%J.err'), worker]
    write(folder/'submit-intent.json', dict(argv=argv, utc=now(), plan_sha256=sha(root/'payload/RUN_PLAN.json')))
    response = command(argv, 120)
    write(folder/'submit-response.json', response)
    ids = re.findall(r'Job <(\d+)> is submitted', response['stdout'])
    if len(ids) == 1:
        accepted = dict(key=key, job_id=ids[0], job_name=name, initially_held=True,
            wall_minutes=minutes, cpu=1, memory_gb=memory, gpus=1, gpu_model=model,
            plan_sha256=sha(root/'payload/RUN_PLAN.json'))
        if generation_sha is not None:
            accepted.update(worker_entrypoint=str(entry), worker_generation_sha256=generation_sha)
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
