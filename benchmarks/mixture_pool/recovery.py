"""Explicit, immutable retries of terminal units; original claims are untouched."""
import os
from pathlib import Path
import sys
import time

from common import now, plan, read, sha, unit, validate_outputs, write
from case_helpers import run_child
import worker


def contract(root):
    control = worker.generation(root)
    binding = read(control/'GENERATION.json')['recovery']
    base = Path(binding['root'])
    if (base.resolve() != base or not base.is_relative_to(root/'recoveries') or
            sha(base/'RECOVERY_PLAN.json') != binding['plan_sha256']):
        raise ValueError('Unbound recovery plan')
    record = read(base/'RECOVERY_PLAN.json')
    if record['parent_plan_sha256'] != sha(root/'payload/RUN_PLAN.json'):
        raise ValueError('Changed scientific plan')
    return control, base, record


def verify_predecessor(root, item):
    key = item['key']
    old = root/'receipts'/key/'terminal.json'
    if sha(old) != item['terminal_sha256'] or sha(root/'claims'/key/'claim.json') != item['claim_sha256']:
        raise ValueError('Changed predecessor identity')
    terminal = read(old)
    if terminal['case_id'] != unit(root, key)['case_id'] or terminal['status'] != item['prior_status']:
        raise ValueError('Different terminal case')
    if item['purpose'] == 'failed_case_recovery':
        if terminal['status'] != 'execution_failure':
            raise ValueError('Only a terminal failed case can be recovered')
    elif item['purpose'] == 'shutdown_replay':
        validated = root/'outputs'/key/'VALIDATED.json'
        if (terminal['status'] != 'validated' or sha(validated) != terminal['validated_sha256'] or
                read(validated)['authority'] != validate_outputs(root, unit(root, key))):
            raise ValueError('Unbound shutdown replay output')
    else:
        raise ValueError('Unsupported recovery purpose')


def run(root, family, selected_key=None):
    control, base, record = contract(root)
    p = plan(root)
    if family == 'lsf':
        item, = [item for item in record['units'] if item['key'] == selected_key and item['family'] == family]
        accepted = read(base/'receipts'/selected_key/'accepted.json')
        if (accepted['job_id'] != os.environ.get('LSB_JOBID') or accepted['key'] != selected_key or
                accepted['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
                read(base/'receipts'/selected_key/'admitted.json')['job_id'] != accepted['job_id']):
            raise ValueError('Different scalar recovery admission')
        owner = dict(scheduler='lsf', job_id=accepted['job_id'], job_name=accepted['job_name'])
    else:
        index = int(os.environ['JOB_COMPLETION_INDEX'])
        items = [item for item in record['units'] if item['family'] == family]
        if family != 'h100' or not 0 <= index < len(items) <= p['pools'][family]['workers']:
            raise ValueError('Unknown recovery index')
        item = items[index]
        owner = dict(scheduler='kubernetes', family=family, index=index, pod_uid=os.environ['POD_UID'])
        proof = root/'evidence'/family/('pod-'+owner['pod_uid']+'.admitted.json')
        until = time.monotonic()+180
        while not proof.exists() and time.monotonic() < until:
            time.sleep(1)
        admitted = read(proof)
        if admitted['pod_uid'] != owner['pod_uid'] or admitted['image'] != p['image']:
            raise ValueError('Unbound recovery Pod')
        owner['job_uid'] = admitted['job_uid']
    owner.update(operational_generation=str(control), generation_sha256=sha(control/'GENERATION.json'),
                 recovery_plan_sha256=sha(base/'RECOVERY_PLAN.json'))
    verify_predecessor(root, item)
    env, startup = worker.setup(root, family, owner)
    if family != 'lsf':
        write(root/'evidence'/family/('pod-'+owner['pod_uid']+'.hardware.json'), startup)
    # Capture the failure under the original compiler contract, in a separate
    # diagnostic process/cache. It cannot publish or substitute a fitted result.
    if item.get('qualify_family'):
        out = base/'diagnostics'/item['key']
        out.parent.mkdir(exist_ok=True)
        diag_cache = base/'diagnostics'/('cache-'+family)
        diag_cache.mkdir()
        diag_env = dict(env, TORCHINDUCTOR_CACHE_DIR=str(diag_cache/'inductor'),
                        TRITON_CACHE_DIR=str(diag_cache/'triton'))
        run_child([sys.executable, '-B', str(control/'diagnose_case.py'), str(root), item['key'], str(out)],
                  diag_env, base/'diagnostics'/(item['key']+'.log'), time.monotonic()+1200)
        worker.qualify(root, family, owner, env, startup)
        if (out/'STATE.json').exists():
            stress = out/'fixed-stress'
            stress.mkdir()
            run_child(worker.compiler_command(root, [sys.executable, '-B', str(control/'stress_posterior.py'),
                      str(stress), str(out/'STATE.json')]), env, stress/'stress.log', time.monotonic()+1200)
        write(base/(family+'-READY.json'), dict(owner=owner, utc=now(),
            qualification_sha256=sha(root/'qualification'/worker.qualification_key(root, family)/'QUALIFIED.json'),
            generation_sha256=sha(control/'GENERATION.json'),
            diagnosis_sha256=sha(out/'DIAGNOSIS.json')))
    else:
        ready = base/(family+'-READY.json')
        until = time.monotonic()+1800
        while not ready.exists() and time.monotonic() < until:
            time.sleep(1)
        if read(ready)['generation_sha256'] != owner['generation_sha256']:
            raise ValueError('Different qualified execution generation')
    key = item['key']
    (base/'claims'/key).mkdir()
    write(base/'claims'/key/'claim.json', dict(key=key, case_id=unit(root, key)['case_id'], owner=owner,
        plan_sha256=sha(root/'payload/RUN_PLAN.json'), predecessor=item, utc=now()))
    terminal = worker.execute(root, unit(root, key), owner, env, startup, recovery=base)
    if terminal['status'] != 'validated':
        raise RuntimeError('Recovery failed; retained terminal must be diagnosed before another attempt')
    write(base/'receipts'/key/'worker-complete.json', dict(owner=owner,
        terminal_sha256=sha(base/'receipts'/key/'terminal.json'), utc=now()))


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
