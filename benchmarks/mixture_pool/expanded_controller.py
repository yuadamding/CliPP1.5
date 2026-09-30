"""Fill ordinary LSF slots while preserving its original qualification job."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from common import identity, now, plan, read, sha, verify, write
from controller import states_for
from expansion_contract import contract, eligible, fresh_seeds_qualified
from lsf_helpers import completion, submit, terminal_log_state
from queueing import claim


def controller(root):
    control = Path(__file__).resolve().parent
    g = contract(root, control)
    p = plan(root)
    if p['max_submitted'] != 15 or p['reserved_lsf_slots'] != 1:
        raise ValueError('Different shared LSF ceiling')
    cap = 14
    lock = (root/'receipts/controller.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    e = root/'receipts'/control.name
    for _ in range(30):
        if (e/'process.json').exists():
            break
        time.sleep(1)
    owner = identity(os.getpid())
    if owner != read(e/'process.json')['identity']:
        raise ValueError('Unbound expansion controller')
    transition = read(e/'PREDECESSOR_STOPPED.json')
    old = read(root/'receipts/controller-process.json')['identity']
    if transition['identity'] != old or not transition['children_preserved']:
        raise ValueError('Different predecessor owner')
    try:
        if identity(old['pid']) == old and (Path('/proc')/str(old['pid'])/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
            raise ValueError('Predecessor still refilling')
    except FileNotFoundError:
        pass
    verify(root, full=True)
    tasks = {t['key']: t for t in read(root/'payload/UNITS.json')}
    active, finished = {}, {}
    for path in (root/'receipts').glob('*/accepted.json'):
        a = read(path)
        if a['plan_sha256'] != g['plan_sha256']:
            raise ValueError('Different adopted job plan')
        if not path.with_name('reconciled.json').exists():
            active[a['job_id']] = a
    if set(active) != set(transition['adopted_job_ids']) or len(active) > cap:
        raise ValueError('Changed inherited slots')
    launch_time = datetime.fromisoformat(read(root/'receipts/controller-process.json')['utc'])
    remaining = p['deadline_seconds'] - (datetime.now(timezone.utc)-launch_time).total_seconds()
    if remaining <= 0:
        raise TimeoutError('Inherited deadline exhausted')
    deadline = time.monotonic()+remaining
    p = dict(p, worker_entrypoint=str(control/'expanded_worker.py'),
             worker_generation_sha256=sha(control/'GENERATION.json'))
    write(e/'ready.json', dict(identity=owner, adopted=list(active), cap=cap, utc=now()))
    try:
        while time.monotonic() < deadline:
            states, query = states_for(active)
            for job, a in list(active.items()):
                state = states[job]
                if state == 'UNKNOWN':
                    state = terminal_log_state(root/'lsf-logs'/f'{a["key"]}.{job}.out', a) or state
                result = completion(root, a, tasks[a['key']], state, query)
                if result is None:
                    continue
                finished[a['key']] = result['status']
                del active[job]
                if result['status'] != 'validated':
                    raise RuntimeError('Case failure; preserve active jobs and halt refills')
            fresh = fresh_seeds_qualified(root)
            exhausted = False
            while len(active) < cap and not (root/'STOP').exists():
                task = claim(root, dict(scheduler='lsf', controller=owner),
                             eligible=lambda task: eligible(task, fresh))
                if task is None:
                    exhausted = fresh
                    break
                a = submit(root, p, task, task['wall_minutes'])
                active[a['job_id']] = a
            progress = dict(utc=now(), phase='full_expansion', active=active, states=states,
                finished=len(finished), outcomes=dict(Counter(finished.values())),
                max_submitted=15, case_cap=cap, reserved_lsf_slots=1,
                fresh_seeds_qualified=fresh, controller_identity=owner, stopped=(root/'STOP').exists())
            write(root/'PROGRESS.json', progress, replace=True)
            if not active and (exhausted or (root/'STOP').exists()):
                write(root/'receipts/controller-terminal.json', dict(progress, queue_exhausted=exhausted))
                return
            time.sleep(10)
        raise TimeoutError('Inherited controller deadline')
    except BaseException as exc:
        write(e/'halted.json', dict(error=repr(exc), active=active, traceback=traceback.format_exc(), utc=now()))
        raise


if __name__ == '__main__':
    controller(Path(sys.argv[1]).resolve())
