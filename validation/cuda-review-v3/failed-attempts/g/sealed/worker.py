"""One frozen-source, allocated PyTorch CUDA qualification; no cohort launch."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def compare_profiles(baseline, current):
    assert baseline['status'] == current['status'] == 'passed'
    assert baseline['script_sha256'] == current['script_sha256']
    assert baseline['source']['source_sha256'] == '50c4d2a16b361bf34943a91350f664ac0a354c84d89e36c67ff03f59c37e0ee7'
    def by_kind(receipt, kind):
        return {x['name']: x for x in receipt['cases'] if x['kind'] == kind}
    old_fixture, new_fixture = (by_kind(r, 'fixture_started') for r in (baseline, current))
    old, new = (by_kind(r, 'profile_qualified') for r in (baseline, current))
    assert set(old) == set(new) == {'analytical32', 'mixed6'}
    comparisons = {}
    for name in old:
        assert old_fixture[name]['fixture'] == new_fixture[name]['fixture']
        a, b = old[name], new[name]
        pa, pb = a['result'], b['result']
        assert len(pa['loss']) == len(pb['loss'])
        assert all(pa['qualified']) and all(pb['qualified'])
        maximum_difference = 0.0
        for x, y, gx, gy in zip(pa['loss'], pb['loss'], pa['gap'], pb['gap']):
            assert all(math.isfinite(v) for v in (x, y, gx, gy))
            margin = gx + gy + 128 * sys.float_info.epsilon * (1 + abs(x) + abs(y))
            assert abs(x-y) <= margin, (name, x, y, margin)
            maximum_difference = max(maximum_difference, abs(x-y))
        key = 'aten::_local_scalar_dense'
        before, after = a['scalar_read_proxy_counts'][key], b['scalar_read_proxy_counts'][key]
        assert after < before, (name, before, after)
        comparisons[name] = dict(fixture_sha256=old_fixture[name]['fixture']['model_sha256'],
            baseline_scalar_reads=before, current_scalar_reads=after,
            reduction_fraction=1-after/before, max_loss_difference=maximum_difference,
            current_work_counters=b['counter_deltas'])
    return dict(status='passed', cases=comparisons,
                scope='Matched CUDA scalar fits; CPU profiler scalar-read counts are synchronization proxies, not measured GPU synchronization duration or end-to-end speedup')


def main(root):
    assert root.resolve() == root and re.fullmatch(r'clipp2_clipp1d_cuda_20260922[a-z]', root.name)
    assert root.stat().st_uid == os.getuid() == 307469
    plan = json.loads((root / 'PREPARED.json').read_bytes())
    assert str(root) == plan['remote_root'] and root.name == plan['run_id']
    assert sys.executable == plan['python']
    accepted = json.loads((root / 'receipts/accepted.json').read_bytes())
    assert accepted['job_id'] == os.environ['LSB_JOBID']
    assert accepted['job_name'] == plan['job_name']
    assert accepted['plan_sha256'] == digest(root / 'PREPARED.json')
    started = time.time()
    write(root / 'receipts/execution.json', dict(job_id=accepted['job_id'], host=os.uname().nodename,
                                              pid=os.getpid(), started_unix=started))
    result = dict(passed=False, job_id=accepted['job_id'], plan_sha256=digest(root / 'PREPARED.json'))
    try:
        inventory = json.loads((root / 'inventory.json').read_bytes())
        for name, expected in inventory.items():
            path = root / name
            assert path.is_file() and not path.is_symlink() and digest(path) == expected, name
        packages = '\n'.join(sorted(subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze', '--disable-pip-version-check'], text=True,
        ).splitlines())) + '\n'
        assert hashlib.sha256(packages.encode()).hexdigest() == plan['environment_sha256']
        compiler = Path(plan['compiler']['path'])
        assert digest(compiler) == plan['compiler']['sha256']
        source = root / 'source/src'
        cache = root / 'results/compiler-cache'
        cache.mkdir()
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', CC=str(compiler),
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1',
                           TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'))
        probe = '''import json,sys
from pathlib import Path
import torch,clipp1d
assert Path(clipp1d.__file__).resolve().parent==Path(sys.argv[1])/'clipp1d'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'L40' in torch.cuda.get_device_name(0)
assert torch.ones(4,device='cuda',dtype=torch.float64).sum().item()==4
print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,source=str(Path(clipp1d.__file__).resolve()))))
'''
        runtime = json.loads(subprocess.check_output([sys.executable, '-B', '-c', probe, str(source)],
                                                    env=environment, text=True, timeout=120))
        write(root / 'receipts/startup.json', dict(**runtime, job_id=accepted['job_id'],
              environment_sha256=plan['environment_sha256'], compiler_sha256=digest(compiler),
              inventory_sha256=digest(root / 'inventory.json'), dirty_overlay=True))
        profiles = {}
        for label, tree in [('baseline', 'baseline_source'), ('current', 'source')]:
            profile_cache = root / f'results/profile-cache-{label}'
            profile_cache.mkdir()
            profile_environment = dict(environment, PYTHONPATH=str(root / tree / 'src'),
                TORCHINDUCTOR_CACHE_DIR=str(profile_cache / 'inductor'),
                TRITON_CACHE_DIR=str(profile_cache / 'triton'))
            destination = root / f'results/scalar-profile-{label}.json'
            command = [sys.executable, '-B', str(root / 'source/benchmarks/profile_scalar_cuda.py'),
                       '--device', 'cuda:0', '--out', str(destination), '--timeout-seconds', '180']
            write(root / f'receipts/profile-command-{label}.json', dict(argv=command,
                  source_directory=str(root / tree), numerical_timeout_seconds=180, subprocess_timeout_seconds=210))
            with (root / f'case-logs/profile-{label}.out').open('xb') as stdout, (root / f'case-logs/profile-{label}.err').open('xb') as stderr:
                completed = subprocess.run(command, cwd=root / tree, env=profile_environment,
                                           stdout=stdout, stderr=stderr, timeout=210)
            assert completed.returncode == 0, f'{label} profile failed; preserve all evidence'
            profiles[label] = json.loads(destination.read_bytes())
        comparison = compare_profiles(profiles['baseline'], profiles['current'])
        write(root / 'results/scalar-profile-comparison.json', comparison)
        result['scalar_profile_comparison_sha256'] = digest(root / 'results/scalar-profile-comparison.json')
        command = [sys.executable, '-B', str(root / 'source/benchmarks/qualify_cuda.py'),
                   '--device', 'cuda:0', '--out', str(root / 'results/qualification.json'),
                   '--path-grid', 'default', '--resource-n', '256', '--resource-iterations', '20000',
                   '--timeout-seconds', '2700']
        write(root / 'receipts/command.json', dict(argv=command, numerical_timeout_seconds=2700, subprocess_timeout_seconds=2820))
        with (root / 'case-logs/qualification.out').open('xb') as stdout, (root / 'case-logs/qualification.err').open('xb') as stderr:
            completed = subprocess.run(command, cwd=root / 'source', env=environment,
                                       stdout=stdout, stderr=stderr, timeout=2820)
        result['returncode'] = completed.returncode
        assert completed.returncode == 0, 'CUDA qualification exited nonzero; preserve result and logs'
        qualification = json.loads((root / 'results/qualification.json').read_bytes())
        assert qualification['status'] == 'passed', qualification
        assert qualification['source']['source_sha256'] == profiles['current']['source']['source_sha256']
        for name, expected in inventory.items():
            assert digest(root / name) == expected, name
        result.update(passed=True, qualification_sha256=digest(root / 'results/qualification.json'))
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        result.update(elapsed_seconds=time.time()-started, scope='isolated_cuda_qualification_not_cohort_performance')
        write(root / 'receipts/terminal.json', result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1])))
