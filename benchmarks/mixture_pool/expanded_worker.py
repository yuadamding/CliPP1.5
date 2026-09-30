"""Reuse the frozen worker after checking the scope of its admission."""
from pathlib import Path
import os
import sys

from common import plan, read, sha, unit
from expansion_contract import contract, eligible, fresh_seeds_qualified


def main(root, key):
    control = Path(__file__).resolve().parent
    contract(root, control)
    p = plan(root)
    task = unit(root, key)
    folder = root/'receipts'/key
    a = read(folder/'accepted.json')
    if (a['job_id'] != os.environ.get('LSB_JOBID') or a['key'] != key or
            a['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
            read(folder/'admitted.json')['job_id'] != a['job_id'] or
            a['worker_entrypoint'] != str(Path(__file__).resolve()) or
            a['worker_generation_sha256'] != sha(control/'GENERATION.json') or
            a['memory_gb'] != p['host_memory_gb']):
        raise ValueError('Unbound expanded worker')
    if not eligible(task, fresh_seeds_qualified(root)):
        raise ValueError('Existing-seed qualification cannot admit a fresh seed')
    # Scientific execution and its generation remain the exact frozen worker.
    sys.path.insert(0, str(root/'payload/ops'))
    import worker
    if Path(worker.__file__).resolve() != root/'payload/ops/worker.py':
        raise ValueError('Different frozen worker')
    owner = dict(scheduler='lsf', job_id=a['job_id'], job_name=a['job_name'],
                 expansion_generation=str(control), expansion_sha256=sha(control/'GENERATION.json'))
    env, startup = worker.setup(root, 'lsf', owner)
    result = worker.execute(root, task, owner, env, startup)
    return 0 if result['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1]).resolve(), sys.argv[2]))
