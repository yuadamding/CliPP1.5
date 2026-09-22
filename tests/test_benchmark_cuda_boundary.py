"""Historical CPU run admission must reject current CUDA-only production source."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_runner(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'benchmarks'))
    spec = importlib.util.spec_from_file_location('four_cohort_boundary_runner', REPO/'benchmarks/run_four_cohorts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_rejects_cuda_before_creating_run_root(tmp_path):
    root = tmp_path/'no-run'
    completed = subprocess.run([sys.executable, str(REPO/'benchmarks/run_four_cohorts.py'),
                                'freeze', '--root', str(root), '--prepared', str(tmp_path/'unused.json')],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode != 0
    assert 'CUDA-only complete-graph' in completed.stderr
    assert not root.exists()


def test_controller_rejects_cuda_before_owner_capacity_or_children(tmp_path, monkeypatch):
    runner = load_runner(monkeypatch)
    package = tmp_path/'frozen/src/clipp1d'
    package.mkdir(parents=True)
    shutil.copyfile(REPO/'src/clipp1d/api.py', package/'api.py')
    (tmp_path/'plan.json').write_text(json.dumps({'source': {}, 'cases': []}))
    args = SimpleNamespace(root=tmp_path, owner='controller')
    with pytest.raises(ValueError, match='CUDA-only complete-graph'):
        runner.controller(args)
    assert not (tmp_path/'controller.lock').exists()
    assert not (tmp_path/'controller-capacity.json').exists()


def test_outer_diagnosis_rejects_cuda_before_output_creation(tmp_path):
    output = tmp_path/'no-diagnosis'
    completed = subprocess.run([sys.executable, str(REPO/'benchmarks/diagnose_outer_progress.py'),
                                '--frozen-package', str(REPO/'src/clipp1d'),
                                '--input-file', str(tmp_path/'unused.tsv'), '--outdir', str(output)],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode != 0
    assert 'CUDA-only complete-graph' in completed.stderr
    assert not output.exists()


def test_guard_accepts_historical_test_api_and_rejects_missing_or_partial_package(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'benchmarks'))
    from benchmark_chain import require_historical_cpu_package
    package = tmp_path/'clipp1d'
    package.mkdir()
    with pytest.raises(ValueError, match='api.py'):
        require_historical_cpu_package(package)
    (package/'api.py').write_text('def fit(path):\n    pass\n')
    with pytest.raises(ValueError, match='pinned historical chain'):
        require_historical_cpu_package(package)
    shutil.copyfile(REPO/'tests/legacy_chain_api.py', package/'api.py')
    require_historical_cpu_package(package)
