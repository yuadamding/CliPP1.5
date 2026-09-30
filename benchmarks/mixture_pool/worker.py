"""Shared scientific execution; scheduler identities remain separate."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from common import now, plan, read, sha, unit, validate_outputs, verify, write
from case_helpers import check_publication_canary, run_child
from queueing import claim


def generation(root):
    control = Path(__file__).resolve().parent
    if not (control/'GENERATION.json').is_file():
        return None
    if not control.is_relative_to(root/'control'):
        raise ValueError('Unexpected worker generation')
    record = read(control/'GENERATION.json')
    if record['plan_sha256'] != sha(root/'payload/RUN_PLAN.json'):
        raise ValueError('Different operational-generation plan')
    for name, digest in record['inventory'].items():
        path = control/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or sha(path) != digest:
            raise ValueError('Changed operational generation')
    return control


def qualification_key(root, family):
    control = generation(root)
    return family if control is None else family+'-'+control.name


def compiler_command(root, argv):
    control = generation(root)
    settings = (root/'payload/COMPILER_SETTINGS.json' if control is None else control/'GENERATION.json')
    if not settings.exists() or 'compiler_settings' not in read(settings):
        return argv
    if argv[0] != sys.executable:
        raise ValueError('Compiler entry requires the bound interpreter')
    helper = root/'payload/ops/compiler_entry.py' if control is None else control/'compiler_entry.py'
    return [sys.executable, '-B', str(helper), str(settings), *argv[1:]]


def setup(root, family, owner):
    p = plan(root)
    verify(root)
    if sys.executable != p['python']:
        raise ValueError('Wrong interpreter')
    packages = '\n'.join(sorted(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze', '--disable-pip-version-check'], text=True).splitlines()))+'\n'
    if (hashlib.sha256(packages.encode()).hexdigest() != p['environment_sha256'] or
            sha(p['compiler']['path']) != p['compiler']['sha256']):
        raise ValueError('Changed environment/compiler')
    token = owner['pod_uid'] if family != 'lsf' else owner['job_id']
    cache = root/'runtime'/family/token
    cache.mkdir(parents=True)
    (cache/'tmp').mkdir()
    # PyTorch/Triton use content-addressed, atomically published caches. Separate
    # device families while reusing compiled kernels across scalar LSF jobs.
    control = generation(root)
    cache_name = 'compiled'
    if control is not None and 'compiler_settings' in read(control/'GENERATION.json'):
        cache_name += '-'+sha(control/'GENERATION.json')[:16]
    elif (root/'payload/COMPILER_SETTINGS.json').exists():
        cache_name += '-'+sha(root/'payload/COMPILER_SETTINGS.json')[:16]
    shared = root/'runtime'/family/cache_name
    for name in ('inductor', 'triton'):
        (shared/name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root/'payload/source/src'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', USER='yding4', LOGNAME='yding4',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        TORCHINDUCTOR_COMPILE_THREADS='1', CC=p['compiler']['path'],
        TORCHINDUCTOR_CACHE_DIR=str(shared/'inductor'), TRITON_CACHE_DIR=str(shared/'triton'),
        TMPDIR=str(cache/'tmp'), XDG_CACHE_HOME=str(cache/'tmp'))
    model = p['hardware'][family]
    probe = ('import torch,json,clipp1d; assert torch.cuda.is_available(); '
        'assert torch.cuda.device_count()==1; '
        f'assert torch.cuda.get_device_name(0)=={model["name"]!r}; '
        f'assert torch.cuda.get_device_properties(0).total_memory>={model["minimum_bytes"]}; '
        f'assert torch.cuda.get_device_capability(0)>={tuple(model["minimum_capability"])!r}; '
        'assert torch.ones(4,device="cuda",dtype=torch.float64).sum().item()==4; '
        'print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__, '
        'cuda=torch.version.cuda,total_bytes=torch.cuda.get_device_properties(0).total_memory, '
        'capability=torch.cuda.get_device_capability(0),package=clipp1d.__file__)))')
    runtime = json.loads(subprocess.check_output([sys.executable, '-B', '-c', probe],
                        env=env, text=True, timeout=120))
    if Path(runtime['package']).resolve().parent != root/'payload/source/src/clipp1d':
        raise ValueError('Wrong loaded source')
    return env, dict(runtime=runtime, owner=owner, host=os.uname().nodename,
                     environment_sha256=p['environment_sha256'], utc=now())


def execute(root, task, owner, env, startup, *, qualification=None, recovery=None):
    key = task['key']
    base = root if qualification is None else root/'qualification'/qualification
    if recovery is not None:
        control = generation(root)
        if qualification is not None or control is None:
            raise ValueError('Recovery requires a distinct bound operational generation')
        contract = read(control/'GENERATION.json')['recovery']
        if (recovery.resolve() != recovery or not recovery.is_relative_to(root/'recoveries') or
                str(recovery) != contract['root'] or
                sha(recovery/'RECOVERY_PLAN.json') != contract['plan_sha256']):
            raise ValueError('Different recovery output authority')
        base = recovery
    receipt_dir = base/'receipts'/key
    receipt_dir.mkdir(parents=True, exist_ok=qualification is None)
    started = time.monotonic()
    terminal = dict(key=key, case_id=task['case_id'], owner=owner,
                    status='setup_failure', started_utc=now())
    if 'job_id' in owner:
        terminal['job_id'] = owner['job_id']
    try:
        write(receipt_dir/'startup.json', startup)
        if qualification is None:
            binding = read(base/'claims'/key/'claim.json')
            if (binding['key'] != key or binding['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
                    binding['owner']['scheduler'] != owner['scheduler']):
                raise ValueError('Different case ownership')
            if owner['scheduler'] == 'kubernetes' and binding['owner'] != owner:
                raise ValueError('Different Pod owner')
        output = base/'outputs'/key
        output.mkdir(parents=True)
        logs = base/'case-logs'
        logs.mkdir(exist_ok=True)
        minutes = (read(receipt_dir/'accepted.json')['wall_minutes']
                   if owner['scheduler'] == 'lsf' and qualification is None else task.get('wall_minutes', 60))
        deadline = started + minutes*60-120
        terminal['status'] = 'execution_failure'
        if task.get('requires_seed'):
            helper = ('run_managed_seed.py' if plan(root).get('seed_memory_backend') == 'cuda_managed'
                      else 'seeding.py')
            argv = compiler_command(root, [sys.executable, '-B', str(root/'payload/ops'/helper),
                                           str(root), key, str(output)])
            write(receipt_dir/'seed-command.json', dict(argv=argv, utc=now()))
            run_child(argv, env, logs/f'{key}.seed.log', deadline)
            from seeding import prepared_manifest
            manifest = prepared_manifest(root, task, output)
        else:
            manifest = root/'payload'/task['manifest']
            if sha(manifest) != task['manifest_sha256']:
                raise ValueError('Changed manifest')
        for helper, mode, extra in [('run_mixture_experiment.py', 'mixture', []),
                ('apply_mixture_structural_guard.py', 'guarded', ['--result-dirs', str(output/'mixture')])]:
            argv = [sys.executable, '-B', str(root/'payload/ops/profile_fit.py'),
                    str(receipt_dir/(mode+'-memory.json')), str(root/'payload/source/benchmarks'/helper),
                    '--manifest', str(manifest), '--outdir', str(output/mode), '--device', 'cuda:0', *extra]
            argv = compiler_command(root, argv)
            write(receipt_dir/(mode+'-command.json'), dict(argv=argv, utc=now()))
            run_child(argv, env, logs/f'{key}.{mode}.log', deadline)
        terminal['status'] = 'output_validation_failure'
        authority = validate_outputs(root, task, base/'outputs')
        if task['publication_canary']:
            terminal['canary'] = check_publication_canary(root, task, base/'outputs')
        verify(root)
        write(output/'VALIDATED.json', dict(case_id=task['case_id'], authority=authority,
            canary=terminal.get('canary'), plan_sha256=sha(root/'payload/RUN_PLAN.json'), owner=owner, utc=now()))
        terminal.update(status='validated', validated_sha256=sha(output/'VALIDATED.json'))
    except subprocess.TimeoutExpired as exc:
        terminal.update(status='resource_timeout', error=repr(exc))
    except BaseException as exc:
        terminal.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        terminal.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started)
        write(receipt_dir/'terminal.json', terminal)
    return terminal


def compare_component(root, task, base):
    expected = root/'payload/expected'/task['key']
    actual = base/'outputs'/task['key']/'mixture/00000'
    contract = read(expected/'MIXTURE_EXPECTATION.json')
    name = 'mixture_mutation_clusters.tsv'
    if sha(expected/('component_'+name)) != contract['table_sha256']:
        raise ValueError('Changed component reference')
    def rows(path):
        with path.open() as stream:
            data = list(csv.DictReader(stream, delimiter='\t'))
        result = {r['mutation_id']: r for r in data}
        if len(result) != len(data) or not data:
            raise ValueError('Invalid component population')
        return result
    a, b = rows(expected/('component_'+name)), rows(actual/name)
    if set(a) != set(b):
        raise ValueError('Component population mismatch')
    delta = max(abs(float(a[i]['mixture_ccf'])-float(b[i]['mixture_ccf'])) for i in a)
    receipt = read(actual/'EXPERIMENT.json')
    score_delta = receipt['score']-contract['score']
    if (not math.isfinite(delta) or delta > 1e-7 or not math.isfinite(score_delta) or abs(score_delta) > 1e-6 or
            any(a[i]['cluster_label'] != b[i]['cluster_label'] or a[i]['multiplicity'] != b[i]['multiplicity'] for i in a) or
            receipt['status'] != contract['status'] or receipt['adaptive'] != contract['adaptive']):
        raise ValueError('Cross-device component mismatch')
    return dict(case_id=task['case_id'], max_ccf_delta=delta, score_delta=score_delta)


def qualify(root, family, owner, env, startup):
    key = qualification_key(root, family)
    base = root/'qualification'/key
    base.mkdir(parents=True)
    xml = base/'tests.xml'
    control = generation(root)
    test_env = dict(env, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    dependency_root = root/'payload/ops' if control is None else control
    if (dependency_root/'test_dependencies').is_dir():
        test_env['PYTHONPATH'] = str(dependency_root/'test_dependencies')+os.pathsep+env['PYTHONPATH']
    settings = root/'payload/COMPILER_SETTINGS.json' if control is None else control/'GENERATION.json'
    if settings.exists() and 'compiler_settings' in read(settings):
        run_child(compiler_command(root, [sys.executable, '-B', str(dependency_root/'stress_posterior.py'), str(base)]),
                  env, base/'stress.log', time.monotonic()+1800)
        if read(base/'STRESS.json')['status'] != 'passed':
            raise ValueError('Repeated compiled posterior stress failed')
    run_child(compiler_command(root, [sys.executable, '-m', 'pytest', '-q',
               str(root/'payload/source/tests/test_mixture_cuda.py'), '--junitxml='+str(xml)]),
              test_env, base/'tests.log', time.monotonic()+1800)
    tree = ET.parse(xml).getroot()
    suites = [tree] if tree.tag == 'testsuite' else list(tree.iter('testsuite'))
    totals = {k: sum(int(n.attrib.get(k, 0)) for n in suites) for k in ('tests', 'errors', 'failures', 'skipped')}
    if totals != dict(tests=3, errors=0, failures=0, skipped=0):
        raise ValueError('Allocated CUDA tests did not all pass')
    pairs = []
    for task in read(root/'payload/UNITS.json')[:3]:
        result = execute(root, task, owner, env, startup, qualification=key)
        if result['status'] != 'validated' or result['canary']['status'] != 'passed':
            raise ValueError('GPU-family publication qualification failed')
        pairs.append(compare_component(root, task, base))
    # The largest observed input is a separate capacity case, not an accuracy gate.
    tasks = read(root/'payload/UNITS.json')
    if plan(root).get('seed_stage_canary_key'):
        seeded = unit(root, plan(root)['seed_stage_canary_key'])
        seeded_result = execute(root, seeded, owner, env, startup, qualification=key)
        if seeded_result['status'] != 'validated':
            raise ValueError('Complete-graph seed-stage publication canary failed')
    capacity_tasks = [t for t in tasks if not t.get('requires_seed') and
                      family in t.get('eligible_families', ['lsf', 'a100', 'h100'])]
    largest = max(capacity_tasks, key=lambda t: t['retained_mutations'])
    if largest['key'] not in [t['key'] for t in read(root/'payload/UNITS.json')[:3]]:
        capacity = execute(root, largest, owner, env, startup, qualification=key)
        if capacity['status'] != 'validated':
            raise ValueError('Largest-input capacity case failed')
    memory = [read(p) for p in (base/'receipts').glob('*/*-memory.json')]
    extra = {}
    if plan(root).get('seed_memory_backend') == 'cuda_managed':
        helper = root/'payload/ops/managed_seed_probe.py'
        for mode in ('standard', 'compare', 'capacity'):
            run_child(compiler_command(root, [sys.executable, '-B', str(helper), str(root), str(base), mode]),
                      env, base/(mode+'.log'), time.monotonic()+3600)
        extra = dict(raw_parity=read(base/'RAW_PARITY.json'), managed_capacity=read(base/'MANAGED_CAPACITY.json'))
        if extra['raw_parity']['status'] != 'passed' or extra['managed_capacity']['status'] != 'passed':
            raise ValueError('Managed seed parity/capacity qualification failed')
    write(base/'QUALIFIED.json', dict(status='passed', family=family, owner=owner,
        runtime=startup['runtime'], tests=totals, pairs=pairs, largest_input_n=largest['retained_mutations'],
        peak_reserved_bytes=max(m['peak_reserved_bytes'] for m in memory),
        plan_sha256=sha(root/'payload/RUN_PLAN.json'), inventory_sha256=sha(root/'payload/INVENTORY.json'),
        comparison='A40 component outputs and independent CPU guarded publication references', extra=extra, utc=now()))


def kubernetes(root, family):
    p = plan(root)
    index = int(os.environ['JOB_COMPLETION_INDEX'])
    if family not in ('a100', 'h100') or not 0 <= index < p['pools'][family]['workers']:
        raise ValueError('Unknown worker pool/index')
    owner = dict(scheduler='kubernetes', family=family, index=index, pod_uid=os.environ['POD_UID'])
    control = generation(root)
    if control is not None:
        owner.update(operational_generation=str(control), generation_sha256=sha(control/'GENERATION.json'))
    # Attach-only supervisor must bind the actual admitted Pod and image first.
    proof = root/'evidence'/family/('pod-'+owner['pod_uid']+'.admitted.json')
    until = time.monotonic()+180
    while not proof.exists() and time.monotonic() < until:
        time.sleep(1)
    admitted = read(proof)
    if admitted['pod_uid'] != owner['pod_uid'] or admitted['image'] != p['image']:
        raise ValueError('Unbound Kubernetes Pod')
    owner['job_uid'] = admitted['job_uid']
    env, startup = setup(root, family, owner)
    write(root/'evidence'/family/('pod-'+owner['pod_uid']+'.hardware.json'), startup)
    if index == 0:
        qualify(root, family, owner, env, startup)
    else:
        q = read(root/'qualification'/qualification_key(root, family)/'QUALIFIED.json')
        if q['status'] != 'passed' or q['plan_sha256'] != sha(root/'payload/RUN_PLAN.json'):
            raise ValueError('Missing exact GPU-family qualification')
    completed = []
    while (task := claim(root, owner)) is not None:
        result = execute(root, task, owner, env, startup)
        completed.append(dict(key=task['key'], status=result['status']))
        if result['status'] != 'validated':
            raise RuntimeError('Case failed; stop this worker, preserve its claim')
    write(root/'evidence'/family/('pod-'+owner['pod_uid']+'.complete.json'),
          dict(owner=owner, completed=completed, queue_exhausted=True, utc=now()))


def lsf(root, key):
    folder = root/'receipts'/key
    a = read(folder/'accepted.json')
    if (a['job_id'] != os.environ.get('LSB_JOBID') or a['key'] != key or
            a['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
            read(folder/'admitted.json')['job_id'] != a['job_id']):
        raise ValueError('Unbound LSF worker')
    owner = dict(scheduler='lsf', job_id=a['job_id'], job_name=a['job_name'])
    env, startup = setup(root, 'lsf', owner)
    if plan(root).get('lsf_qualify_before_first'):
        qualified = root/'qualification'/qualification_key(root, 'lsf')/'QUALIFIED.json'
        if not qualified.exists():
            if (root/'receipts/parallel-admitted.json').exists():
                raise ValueError('Missing qualification after parallel admission')
            qualify(root, 'lsf', owner, env, startup)
        q = read(qualified)
        if (q['status'] != 'passed' or q['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
                q['inventory_sha256'] != sha(root/'payload/INVENTORY.json')):
            raise ValueError('Different allocated LSF qualification')
    result = execute(root, unit(root, key), owner, env, startup)
    return 0 if result['status'] == 'validated' else 1


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    if sys.argv[2] == 'kubernetes':
        kubernetes(root, sys.argv[3])
    else:
        raise SystemExit(lsf(root, sys.argv[2]))
