"""One GPU, one frozen component fit and its independently identified selector."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import (GPU_MODELS, admitted_gpu_model, now, plan, read, sha, unit,
                    validate_outputs, verify, write)


def check_publication_canary(root, task):
    expected = root/'payload/expected'/task['key']
    contract = read(expected/'EXPECTATION.json')
    name = 'mixture_mutation_clusters.tsv'
    if sha(expected/name) != contract['table_sha256']:
        raise ValueError('Changed publication reference')
    target = root/'outputs'/task['key']/'guarded/00000'
    def table(path):
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        result = {r['mutation_id']: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError('Empty or duplicate publication IDs')
        return result
    a, b = table(expected/name), table(target/name)
    if set(a) != set(b):
        raise ValueError('Publication population mismatch')
    deltas = [abs(float(a[n]['mixture_ccf'])-float(b[n]['mixture_ccf'])) for n in a]
    delta = max(deltas)
    if (not all(math.isfinite(d) for d in deltas) or delta > contract['ccf_atol'] or
            any(a[n]['cluster_label'] != b[n]['cluster_label'] or
                a[n]['multiplicity'] != b[n]['multiplicity'] for n in a)):
        raise ValueError('CPU/CUDA publication mismatch')
    receipt = read(target/'EXPERIMENT.json')
    if (receipt['status'] != contract['status'] or
            receipt['original_baseline_preserved'] != contract['original_baseline_preserved'] or
            receipt['structural_decision']['selected_family'] != contract['selected_family']):
        raise ValueError('CPU/CUDA selection identity mismatch')
    return dict(status='passed', max_ccf_delta=delta, mutations=len(a),
                expectation_sha256=sha(expected/'EXPECTATION.json'))


def run_child(argv, env, log, deadline):
    with log.open('x') as stream:
        proc = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        def terminate(signum, frame):
            raise InterruptedError(f'Received signal {signum}')
        old = {s: signal.signal(s, terminate) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = proc.wait(timeout=max(1, deadline-time.monotonic()))
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)
    if code:
        raise RuntimeError(f'Child exited {code}: {log.name}')


def worker(root, key):
    p, task = plan(root), unit(root, key)
    receipt_dir = root/'receipts'/key
    accepted, admission = read(receipt_dir/'accepted.json'), read(receipt_dir/'admitted.json')
    if (accepted['job_id'] != os.environ.get('LSB_JOBID') or
            admission['job_id'] != accepted['job_id'] or accepted['key'] != key or
            accepted['plan_sha256'] != sha(root/'payload/RUN_PLAN.json') or
            sys.executable != p['python']):
        raise ValueError('Unexpected worker identity')
    started = time.monotonic()
    terminal = dict(key=key, case_id=task['case_id'], job_id=accepted['job_id'],
                    status='setup_failure', started_utc=now())
    try:
        verify(root)
        packages = '\n'.join(sorted(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze', '--disable-pip-version-check'], text=True).splitlines()))+'\n'
        if (hashlib.sha256(packages.encode()).hexdigest() != p['environment_sha256'] or
                sha(p['compiler']['path']) != p['compiler']['sha256']):
            raise ValueError('Environment or compiler changed')
        model = admitted_gpu_model(root)
        if model != accepted['gpu_model'] or model not in p['qualified_gpu_models']:
            raise ValueError('Worker GPU request differs from qualified hardware')
        device_name, = [name for name, token in GPU_MODELS.items() if token == model]
        source = root/'payload/source'
        env = dict(os.environ, PYTHONPATH=str(source/'src'), PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            TORCHINDUCTOR_COMPILE_THREADS='1', CC=p['compiler']['path'],
            TORCHINDUCTOR_CACHE_DIR=str(root/'runtime/inductor'), TRITON_CACHE_DIR=str(root/'runtime/triton'),
            TMPDIR=str(root/'runtime/tmp'))
        for name in ('inductor', 'triton', 'tmp'):
            (root/'runtime'/name).mkdir(exist_ok=True)
        probe = ('import torch,json,clipp1d; assert torch.cuda.is_available(); '
            'assert torch.cuda.device_count()==1; '
            f'assert torch.cuda.get_device_name(0)=={device_name!r}; '
            'assert torch.ones(4,device="cuda",dtype=torch.float64).sum().item()==4; '
            'print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__, '
            'cuda=torch.version.cuda,properties=str(torch.cuda.get_device_properties(0)), '
            'gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),"uuid",None)), '
            'free_total=torch.cuda.mem_get_info(),package=clipp1d.__file__)))')
        runtime = json.loads(subprocess.check_output([sys.executable, '-B', '-c', probe], env=env,
                                                     text=True, timeout=120))
        if Path(runtime['package']).resolve().parent != source/'src/clipp1d':
            raise ValueError('Wrong loaded package')
        write(receipt_dir/'startup.json', dict(runtime=runtime, host=os.uname().nodename,
            job_id=accepted['job_id'], environment_sha256=p['environment_sha256'], utc=now()))
        manifest = root/'payload'/task['manifest']
        if sha(manifest) != task['manifest_sha256']:
            raise ValueError('Changed task manifest')
        output = root/'outputs'/key
        output.mkdir()
        deadline = started+accepted['wall_minutes']*60-120
        terminal['status'] = 'execution_failure'
        for helper, mode, extra in [('run_mixture_experiment.py', 'mixture', []),
                ('apply_mixture_structural_guard.py', 'guarded', ['--result-dirs', str(output/'mixture')])]:
            argv = [sys.executable, '-B', str(source/'benchmarks'/helper), '--manifest', str(manifest),
                    '--outdir', str(output/mode), '--device', 'cuda:0', *extra]
            write(receipt_dir/(mode+'-command.json'), dict(argv=argv, utc=now()))
            run_child(argv, env, root/'case-logs'/f'{key}.{mode}.log', deadline)
        terminal['status'] = 'output_validation_failure'
        authority = validate_outputs(root, task)
        if task['publication_canary']:
            terminal['canary'] = check_publication_canary(root, task)
        verify(root)
        write(output/'VALIDATED.json', dict(case_id=task['case_id'], authority=authority,
            canary=terminal.get('canary'), plan_sha256=sha(root/'payload/RUN_PLAN.json'), utc=now()))
        terminal.update(status='validated', validated_sha256=sha(output/'VALIDATED.json'))
    except subprocess.TimeoutExpired as exc:
        terminal.update(status='resource_timeout', error=repr(exc))
    except BaseException as exc:
        terminal.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        terminal.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started)
        write(receipt_dir/'terminal.json', terminal)
    return 0 if terminal['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(worker(Path(sys.argv[1]).resolve(), sys.argv[2]))
