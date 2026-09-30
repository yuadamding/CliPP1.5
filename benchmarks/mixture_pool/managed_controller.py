"""Use the one reserved LSF slot, then inherit capacity after its owner exits."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import sys
import time

from common import identity, now, plan, read, sha, verify, write
from controller import states_for
from lsf_helpers import completion, submit, terminal_log_state
from queueing import claim


def capacity(root, p):
    parent = Path(p['main_campaign']['root'])
    if sha(parent/'payload/RUN_PLAN.json') != p['main_campaign']['plan_sha256']:
        raise ValueError('Different parent capacity authority')
    contract = read(parent/'payload/RUN_PLAN.json')
    if contract['max_submitted'] != 15 or contract['reserved_lsf_slots'] != 1:
        raise ValueError('The main campaign did not reserve this LSF slot')
    terminal = parent/'receipts/controller-terminal.json'
    if not terminal.exists():
        return 1
    done = read(terminal)
    if not done.get('queue_exhausted') or done['active']:
        return 1
    owner = read(parent/'receipts/controller-process.json')['identity']
    if owner['host'] != os.uname().nodename:
        raise ValueError('Capacity handoff must observe its exact controller host')
    try:
        if identity(owner['pid']) == owner and (Path('/proc')/str(owner['pid'])/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
            return 1
    except FileNotFoundError:
        pass
    for accepted in (parent/'receipts').glob('*/accepted.json'):
        reconciled = accepted.with_name('reconciled.json')
        if not reconciled.is_file() or read(reconciled)['scheduler_state'] not in ('DONE', 'EXIT'):
            return 1
    return 15


def main(root):
    p = plan(root)
    if p['schema'] != 'clipp1d.experimental_managed_seeds.v1':
        raise ValueError('Wrong managed experiment plan')
    lock = (root/'receipts/controller.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for _ in range(30):
        if (root/'receipts/controller-process.json').exists():
            break
        time.sleep(1)
    owner = identity(os.getpid())
    if owner != read(root/'receipts/controller-process.json')['identity']:
        raise ValueError('Unbound managed controller')
    verify(root, full=True)
    capacity(root, p)
    write(root/'receipts/controller-ready.json', dict(identity=owner, utc=now()))
    tasks = {t['key']: t for t in read(root/'payload/UNITS.json')}
    active, finished = {}, {}
    deadline = time.monotonic()+p['deadline_seconds']
    try:
        while time.monotonic() < deadline:
            states, query = states_for(active)
            for job, accepted in list(active.items()):
                state = states[job]
                if state == 'UNKNOWN':
                    state = terminal_log_state(root/'lsf-logs'/f'{accepted["key"]}.{job}.out', accepted) or state
                result = completion(root, accepted, tasks[accepted['key']], state, query)
                if result is None:
                    continue
                finished[accepted['key']] = result['status']
                del active[job]
                if result['status'] != 'validated':
                    raise RuntimeError('Managed case failed; preserve evidence and stop refills')
            qualified = root/'qualification/lsf/QUALIFIED.json'
            admitted = False
            if qualified.exists():
                q = read(qualified)
                if (q['status'] != 'passed' or q['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
                        q['inventory_sha256'] != sha(root/'payload/INVENTORY.json') or
                        q['extra']['raw_parity']['status'] != 'passed' or q['extra']['managed_capacity']['status'] != 'passed'):
                    raise ValueError('Invalid managed allocation qualification')
                admitted = True
            cap = capacity(root, p) if admitted else 1
            exhausted = False
            while len(active) < cap and not (root/'STOP').exists():
                task = claim(root, dict(scheduler='lsf', controller=owner))
                if task is None:
                    exhausted = True
                    break
                accepted = submit(root, p, task, task['wall_minutes'])
                active[accepted['job_id']] = accepted
            progress = dict(utc=now(), active=active, states=states, effective_cap=cap,
                qualified=admitted, finished=len(finished), outcomes=dict(Counter(finished.values())),
                parent_scope='one reserved slot until parent scalar controller finishes; then at most 15', owner=owner)
            write(root/'PROGRESS.json', progress, replace=True)
            if not active and (exhausted or (root/'STOP').exists()):
                write(root/'receipts/controller-terminal.json', dict(progress, queue_exhausted=exhausted))
                return
            time.sleep(10)
        raise TimeoutError('Managed controller deadline')
    except BaseException as exc:
        write(root/'receipts/controller-halted.json', dict(error=repr(exc), active=active, utc=now()))
        raise


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
