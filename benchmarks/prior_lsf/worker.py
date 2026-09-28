"""One admitted study task; sequential CUDA work and no automatic retries."""
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import accepted, cases, digest, now, plan, read, verify_files, write
from validator import validate


def child(root, key):
    accepted(root, key)
    sys.path[:0] = [str(root/'source/src'), str(root/'source/benchmarks')]
    task = root/'results'/key
    try:
        if key == 'qualification':
            from qualify_prior_perturbation import qualify
            qualify(root/'datasets/mechanisms', task)
        else:
            case = next(c for c in cases(root) if c['key'] == key)
            source = root/case['input_file']
            assert digest(source) == case['input_sha256']
            if case['mode'] == 'comparison':
                from prior_perturbation import run_case
                run_case(source, task, tumor_id=case['case_id'])
            elif case['mode'] in ('discovery', 'factorial'):
                from prior_perturbation_diagnostics import discovery_case, factorial_case
                if case['mode'] == 'discovery':
                    discovery_case(source, case['case_id'], task)
                else:
                    factorial_case(source, case['case_id'], root/'results'/case['baseline_key'], task)
            else:
                from time_prior_perturbation import measure_case
                measure_case(source, case['case_id'], task)
    except Exception as error:
        from cohort_failures import failure_record
        write(root/'receipts'/key/'failure.json', dict(failure_record(error, 'fit'), utc=now(),
            traceback=traceback.format_exc()))
        raise


def worker(root, key):
    p = plan(root)
    assert sys.executable == p['python']
    a, receipt = accepted(root, key)
    write(receipt/'execution.json', dict(job_id=a['job_id'], pid=os.getpid(), host=os.uname().nodename, utc=now()))
    start = time.monotonic()
    terminal = dict(key=key, job_id=a['job_id'], status='setup_failure', utc_started=now())
    try:
        verify_files(root)
        freeze = '\n'.join(sorted(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze', '--disable-pip-version-check'], text=True).splitlines()))+'\n'
        assert hashlib.sha256(freeze.encode()).hexdigest() == p['environment_sha256']
        assert digest(p['compiler']['path']) == p['compiler']['sha256']
        env = dict(os.environ, PYTHONPATH=str(root/'source/src'), PYTHONDONTWRITEBYTECODE='1',
            CC=p['compiler']['path'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1',
            TORCHINDUCTOR_CACHE_DIR=str(root/'runtime/compiler-cache/inductor'),
            TRITON_CACHE_DIR=str(root/'runtime/compiler-cache/triton'))
        # A short-lived probe exits before the fit, leaving only one CUDA owner.
        probe = """import torch,json,clipp1d
from clipp1d.api import source_provenance
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'L40' in torch.cuda.get_device_name(0)
assert torch.ones(4,device='cuda',dtype=torch.float64).sum().item()==4
print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),free_total=torch.cuda.mem_get_info(),source=source_provenance(),package=clipp1d.__file__)))"""
        import json
        runtime = json.loads(subprocess.check_output([sys.executable, '-B', '-c', probe], env=env, text=True, timeout=120))
        assert runtime['source']['source_sha256'] == p['source_sha256']
        assert Path(runtime['package']).resolve().parent == root/'source/src/clipp1d'
        write(receipt/'startup.json', dict(runtime=runtime, job_id=a['job_id'], environment_sha256=p['environment_sha256'], utc=now()))
        case = None if key == 'qualification' else next(c for c in cases(root) if c['key'] == key)
        timeout = (120 if case is None else case['wall_minutes'])*60-180
        argv = [sys.executable, '-B', str(root/'worker.py'), 'child', str(root), key]
        write(receipt/'command.json', dict(argv=argv, timeout_seconds=timeout,
            environment_overrides={k:env[k] for k in ('PYTHONPATH', 'CC', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS',
                'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')}))
        terminal['status'] = 'execution_failure'
        with (root/'case-logs'/f'{key}.out').open('xb') as out, (root/'case-logs'/f'{key}.err').open('xb') as err:
            proc = subprocess.Popen(argv, env=env, cwd=root/'source', stdout=out, stderr=err, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                terminal['status'], code = 'resource_timeout', None
        terminal['returncode'] = code
        task = root/'results'/key
        if code == 0 and key == 'qualification':
            q = read(task/'QUALIFICATION.json')
            assert q['status'] == 'passed' and q['source']['source_sha256'] == p['source_sha256']
            terminal.update(status='qualified', receipt_sha256=digest(task/'QUALIFICATION.json'))
        elif code == 0:
            terminal['status'] = 'output_validation_failure'
            v = validate(root, case)
            write(task/'validated.json', dict(v, utc=now()))
            assert validate(root, case) == v
            terminal.update(status='validated_'+v['search_status'], validated_sha256=digest(task/'validated.json'))
        elif (receipt/'failure.json').exists():
            failure = read(receipt/'failure.json')
            terminal.update(status=failure['status'], failure_sha256=digest(receipt/'failure.json'))
    except BaseException as error:
        terminal.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        terminal.update(utc_finished=now(), elapsed_seconds=time.monotonic()-start)
        write(receipt/'terminal.json', terminal)
    return 0 if terminal['status'] in ('qualified', 'validated_complete', 'validated_incomplete',
        'scientific_failure', 'resource_failure', 'resource_timeout', 'execution_failure') else 1


if __name__ == '__main__':
    action, root, key = sys.argv[1:]
    sys.exit((child if action == 'child' else worker)(Path(root), key))
