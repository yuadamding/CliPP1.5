"""Rolling 22-slot scalar LSF owner sharing claims with the two GPU pools."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from common import command, identity, now, plan, read, sha, verify, write
from lsf_helpers import completion, submit, terminal_log_state
from queueing import claim


def states_for(active):
    states = {j: 'UNKNOWN' for j in active}
    if not active:
        return states, None
    query = command(['bjobs', '-a', '-noheader', '-o', 'jobid user stat job_name:200', *active])
    for line in query['stdout'].splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError('Ambiguous scheduler reply')
        job, user, state, name = fields
        if job not in active or user != 'yding4' or name != active[job]['job_name']:
            raise ValueError('Foreign scheduler identity')
        states[job] = state
    return states, query


def controller(root):
    p = plan(root)
    lock = (root/'receipts/controller.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(30):
        if (root/'receipts/controller-process.json').exists():
            break
        time.sleep(1)
    owner = identity(os.getpid())
    if owner != read(root/'receipts/controller-process.json')['identity']:
        raise ValueError('Unbound controller')
    verify(root, full=True)
    q = read(root/'receipts/qualification-admitted.json')
    if q['gpu_model'] != 'NVIDIAA40' or q['source_verified'] is not True:
        raise ValueError('Missing imported A40 qualification')
    tasks = {t['key']: t for t in read(root/'payload/UNITS.json')}
    write(root/'receipts/controller-ready.json', dict(identity=owner, utc=now()))
    active, finished = {}, {}
    phase = 'new_worker_canary'
    deadline = time.monotonic()+p['deadline_seconds']
    try:
        while time.monotonic() < deadline:
            states, query = states_for(active)
            for job, a in list(active.items()):
                state = states[job]
                if state == 'UNKNOWN':
                    state = terminal_log_state(root/'lsf-logs'/f'{a["key"]}.{job}.out', a) or state
                t = completion(root, a, tasks[a['key']], state, query)
                if t is None:
                    continue
                finished[a['key']] = t['status']
                del active[job]
                if t['status'] != 'validated':
                    raise RuntimeError('Case failure; stop LSF refills without duplicating remaining owners')
                if phase == 'new_worker_canary':
                    write(root/'receipts/parallel-admitted.json', dict(key=a['key'], job_id=job,
                          terminal_sha256=sha(root/'receipts'/a['key']/'terminal.json'), cap=22, utc=now()))
                    phase = 'parallel'
            capacity = 1 if phase == 'new_worker_canary' else p['max_submitted']
            exhausted = False
            while len(active) < capacity and not (root/'STOP').exists():
                task = claim(root, dict(scheduler='lsf', controller=owner))
                if task is None:
                    exhausted = True
                    break
                a = submit(root, p, task, p['case_wall_minutes'])
                active[a['job_id']] = a
            progress = dict(utc=now(), phase=phase, active=active, states=states,
                finished=len(finished), outcomes=dict(Counter(finished.values())),
                imported=len(read(root/'payload/IMPORTED.json')), max_submitted=22,
                controller_identity=owner, stopped=(root/'STOP').exists())
            write(root/'PROGRESS.json', progress, replace=True)
            if not active and (exhausted or (root/'STOP').exists()):
                write(root/'receipts/controller-terminal.json', dict(progress, queue_exhausted=exhausted))
                return
            time.sleep(10)
        raise TimeoutError('Controller deadline')
    except BaseException as exc:
        write(root/'receipts/controller-halted.json', dict(error=repr(exc), active=active,
              traceback=traceback.format_exc(), utc=now()))
        raise


if __name__ == '__main__':
    controller(Path(sys.argv[1]).resolve())
