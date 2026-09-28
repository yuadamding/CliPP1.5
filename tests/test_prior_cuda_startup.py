"""Cold child entry points and the canary-to-panel receipt transition."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

BENCHMARKS = Path(__file__).parents[1]/'benchmarks'
sys.path.insert(0, str(BENCHMARKS))


@pytest.mark.parametrize('entry', ['comparison', 'timing'])
def test_cold_cuda_initialization_precedes_memory_reset_and_upload(entry, monkeypatch, tmp_path):
    import prior_perturbation as experiment
    import time_prior_perturbation as timing
    module = experiment if entry == 'comparison' else timing
    events = []
    device = torch.device('cuda:0')
    monkeypatch.setattr(module, 'require_cuda', lambda value: device)

    def synchronize(selected):
        assert selected == device
        events.append('initialized')

    def reset(selected):
        assert selected == device
        assert events == ['initialized'], 'Allocator reset before CUDA initialization'
        events.append('reset')

    def upload(*args, **kwargs):
        assert events == ['initialized', 'reset'], 'Model allocations escaped peak measurement'
        raise StopIteration('upload boundary')

    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(module, 'prepare_model', upload)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    with pytest.raises(StopIteration, match='upload boundary'):
        if entry == 'comparison':
            experiment.run_case(tmp_path/'input.tsv', tmp_path/'output', tumor_id='case')
        else:
            timing.measure_once(tmp_path/'input.tsv', 'case', tmp_path/'output', 'B')


def load_controller(monkeypatch):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    common = load('prior_startup_common', BENCHMARKS/'prior_lsf/common.py')
    monkeypatch.setitem(sys.modules, 'common', common)
    return load('prior_startup_controller', BENCHMARKS/'prior_lsf/controller.py')


def test_cold_child_runs_before_parent_cuda_context(monkeypatch, tmp_path):
    import qualify_prior_perturbation as qualification
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    (inputs/'FIT_MANIFEST.json').write_text(json.dumps(dict(cases=[dict(case_id='mechanism-00')])))
    def parent_probe(*args):
        pytest.fail('Parent must not initialize CUDA before exclusive-device child finishes')
    def child(*args):
        raise RuntimeError('stop at cold child')
    monkeypatch.setattr(qualification, 'require_cuda', parent_probe)
    monkeypatch.setattr(torch.cuda, 'get_device_name', parent_probe)
    monkeypatch.setattr(qualification, 'qualify_cold_processes', child)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    with pytest.raises(RuntimeError, match='stop at cold child'):
        qualification.qualify(inputs, tmp_path/'output')
    assert json.loads((tmp_path/'output/FAILURE.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('corrupt', [False, True])
def test_panel_admission_consumes_worker_qualification_receipt(monkeypatch, tmp_path, corrupt):
    controller = load_controller(monkeypatch)
    qualification = tmp_path/'results/qualification/QUALIFICATION.json'
    qualification.parent.mkdir(parents=True)
    qualification.write_text(json.dumps(dict(status='passed')))
    terminal = tmp_path/'receipts/qualification/terminal.json'
    terminal.parent.mkdir(parents=True)
    terminal.write_text(json.dumps(dict(status='qualified', receipt_sha256=controller.digest(qualification))))
    canary = tmp_path/'results/development-0000/validated.json'
    canary.parent.mkdir()
    canary.write_text(json.dumps(dict(search_status='incomplete')))
    if corrupt:
        qualification.write_text(json.dumps(dict(status='failed')))
        with pytest.raises(AssertionError):
            controller.admit_panel(tmp_path, 'development-0000', 'validated_incomplete', 1)
        assert not (tmp_path/'receipts/panel-admitted.json').exists()
    else:
        controller.admit_panel(tmp_path, 'development-0000', 'validated_incomplete', 1)
        record = json.loads((tmp_path/'receipts/panel-admitted.json').read_text())
        assert record['qualification_sha256'] == controller.digest(qualification)
        assert record['canary_sha256'] == controller.digest(canary)
        assert record['canary_search_status'] == 'validated_incomplete'
        assert record['max_submitted'] == 1
