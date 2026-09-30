"""Worker admission/process/result controls; no numerical module is imported."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'benchmarks/qualify_repository_improvements.py'
SPEC = importlib.util.spec_from_file_location('repository_qualification_metadata', PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_evidence():
    nodes = ['tests/a.py::test_one', 'tests/b.py::test_two[x]']
    value = dict(collected=nodes, passed_calls=list(nodes), nonpassing=[])
    assert worker.validate_test_report(value, ['tests/a.py', 'tests/b.py'], nodes)['tests'] == 2


@pytest.mark.parametrize('damage', ['missing_call', 'duplicate_collection', 'skip', 'missing_file', 'missing_cuda'])
def test_every_collected_test_and_required_cuda_identity_is_required(damage):
    nodes = ['tests/a.py::test_one', 'tests/b.py::test_two']
    value = dict(collected=list(nodes), passed_calls=list(nodes), nonpassing=[])
    files, required = ['tests/a.py', 'tests/b.py'], [nodes[0]]
    if damage == 'missing_call':
        value['passed_calls'].pop()
    elif damage == 'duplicate_collection':
        value['collected'].append(nodes[0])
    elif damage == 'skip':
        value['nonpassing'].append(dict(nodeid=nodes[0], outcome='skipped'))
    elif damage == 'missing_file':
        files.append('tests/c.py')
    else:
        required.append('tests/a.py::test_absent')
    with pytest.raises(ValueError):
        worker.validate_test_report(value, files, required)


@pytest.mark.parametrize('status', [dict(returncode=1, timed_out=False, observation={'qualification_passed': True}),
    dict(returncode=0, timed_out=True, observation={'qualification_passed': True}),
    dict(returncode=0, timed_out=False, observation=None),
    dict(returncode=0, timed_out=False, observation={'qualification_passed': False})])
def test_child_process_failure_cannot_be_erased_by_positive_receipt(status):
    assert not worker.valid_child(status)


def make_plan(tmp_path):
    root = tmp_path/'payload'
    root.mkdir()
    for directory in ('source/src/clipp1d', 'assets/baseline/src/clipp1d', 'source/tests', 'test_vendor'):
        (root/directory).mkdir(parents=True)
    for directory in ('source', 'assets/baseline'):
        (root/directory/'src/clipp1d/__init__.py').write_text('BOUND = True\n')
    (root/'source/tests/a.py').write_text('def test_one(): pass\n')
    (root/'input.tsv').write_text('input metadata fixture\n')
    (root/'test_vendor/pytest.py').write_text('VERSION = 1\n')
    manifest = {'files': {'pytest.py': worker.sha(root/'test_vendor/pytest.py')}}
    (root/'TEST_ENVIRONMENT.json').write_text(json.dumps(manifest))
    item = dict(path='input.tsv', sha256=worker.sha(root/'input.tsv'))
    plan = dict(schema='clipp1d.repository_improvement_plan.v1', mode='tests', worker_sha256=worker.sha(PATH),
        saved_partition_or_solver_state_inputs=False,
        baseline=dict(root='assets/baseline', **worker.source_inventory(root/'assets/baseline')),
        candidate=dict(root='source', **worker.source_inventory(root/'source')),
        assets={'source/tests/a.py': worker.sha(root/'source/tests/a.py')},
        configuration=dict(max_major_cn=4, partition_search=True, generic_partition_grouping=False,
            baseline_partition_policy={}, candidate_partition_policy={'refit_relocations': True}),
        timeouts=dict(tests=30, fit=30), synthetic_inputs=[item],
        tests=dict(vendor_root='test_vendor', vendor_manifest='TEST_ENVIRONMENT.json',
            vendor_manifest_sha256=worker.sha(root/'TEST_ENVIRONMENT.json'), paths=['tests/a.py'],
            allocated_cuda_files=['tests/a.py'], allocated_cuda_required_nodes=['tests/a.py::test_one'],
            cpu_reference_files=[]))
    return root, plan


def test_hash_bound_source_vendor_tests_and_original_policy_plan(tmp_path):
    root, plan = make_plan(tmp_path)
    worker.validate_plan(root, plan, 'tests')
    plan['configuration']['candidate_partition_policy']['minimum_decrease'] = 1.
    with pytest.raises(ValueError, match='partition policy'):
        worker.validate_plan(root, plan, 'tests')


@pytest.mark.parametrize('damage', ['source', 'vendor', 'test', 'input', 'worker', 'saved_state'])
def test_plan_rejects_changed_bound_assets(tmp_path, damage):
    root, plan = make_plan(tmp_path)
    if damage in ('source', 'vendor', 'test', 'input'):
        path = {'source':'source/src/clipp1d/__init__.py','vendor':'test_vendor/pytest.py',
                'test':'source/tests/a.py','input':'input.tsv'}[damage]
        (root/path).write_text('changed\n')
    elif damage == 'worker':
        plan['worker_sha256'] = '0'*64
    else:
        plan['saved_partition_or_solver_state_inputs'] = True
    with pytest.raises(ValueError):
        worker.validate_plan(root, plan, 'tests')


def test_timeout_persists_failure_and_aborts_before_any_next_phase(tmp_path, monkeypatch):
    root, plan = make_plan(tmp_path)
    path = root/'plan.json'
    path.write_text(json.dumps(plan))
    commands = []

    def expire(command, **kwargs):
        commands.append(command)
        assert not kwargs.get('start_new_session', False)
        assert all('\0' not in arg for arg in command)
        raise worker.subprocess.TimeoutExpired(command, 30)

    monkeypatch.setattr(worker.subprocess, 'run', expire)
    out = tmp_path/'out'
    with pytest.raises(TimeoutError, match='abort all further phases'):
        worker.run_child(root, path, worker.sha(path), out, 'tests')
    assert len(commands) == 1
    assert json.loads((out/'subprocess-status.json').read_text()) == dict(returncode=None, timed_out=True)


def test_successful_child_command_has_isolated_source_and_no_null_args(tmp_path, monkeypatch):
    root, plan = make_plan(tmp_path)
    path = root/'plan.json'
    path.write_text(json.dumps(plan))

    def finish(command, **kwargs):
        assert command[1:4] == ['-I', '-B', '-c']
        assert '--expected-source-sha256' in command
        paths = json.loads(command[5])
        assert paths == [str(root/'test_vendor'), str(root/'source/src'), str(root/'source')]
        assert 'PYTHONPATH' not in kwargs['env']
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker.subprocess, 'run', finish)
    result = worker.run_child(root, path, worker.sha(path), tmp_path/'out', 'tests')
    assert result['returncode'] == 0 and result['observation'] is None
    assert not worker.valid_child(result)


def test_output_inventory_matches_standard_descriptor_contract(tmp_path):
    (tmp_path/'one.json').write_text('{}\n')
    result = worker.artifacts(tmp_path)
    assert result == [dict(path='one.json', sha256=worker.sha(tmp_path/'one.json'), size=3)]


def test_confined_paths_reject_traversal_and_symlink(tmp_path):
    for name in ('../other', '/absolute'):
        with pytest.raises(ValueError):
            worker.owned(tmp_path, name)
    (tmp_path/'link').symlink_to(tmp_path/'other')
    with pytest.raises(ValueError):
        worker.owned(tmp_path, 'link')


def test_scope_cannot_classify_same_file_as_cuda_and_cpu_reference(tmp_path):
    root, plan = make_plan(tmp_path)
    plan = deepcopy(plan)
    plan['tests']['cpu_reference_files'] = ['tests/a.py']
    with pytest.raises(ValueError, match='scope must reconcile'):
        worker.validate_plan(root, plan, 'tests')


def runtime_layout(tmp_path, monkeypatch):
    root = tmp_path / 'payload'
    root.mkdir()
    temporary = tmp_path / 'runtime' / 'SHARED_TESTS' / 'tmp'
    temporary.mkdir(parents=True)
    out = tmp_path / 'outputs' / 'SHARED_TESTS' / 'qualification' / 'tests'
    out.mkdir(parents=True)
    monkeypatch.setenv('TMPDIR', str(temporary))
    return root, temporary, out


def test_pytest_runtime_is_fresh_outside_outputs_and_preserves_prior_evidence(tmp_path, monkeypatch):
    root, temporary, out = runtime_layout(tmp_path, monkeypatch)
    prior = temporary / 'prior-pytest'
    prior.mkdir()
    (prior / 'evidence.txt').write_text('retained\n')
    basetemp = worker.pytest_runtime_directory(root, out)
    assert basetemp.parent.is_dir() and not basetemp.exists()
    assert basetemp.parent.parent == temporary
    assert not basetemp.is_relative_to(tmp_path / 'outputs')
    assert (prior / 'evidence.txt').read_text() == 'retained\n'
    receipt = json.loads((out / 'pytest-runtime.json').read_text())
    assert receipt['basetemp'] == str(basetemp)
    assert receipt['worker_tmpdir'] == str(temporary)
    assert receipt['task_key'] == 'SHARED_TESTS'
    assert receipt['retained_for_diagnosis']


def test_pytest_runtime_symlinks_do_not_enter_scientific_artifacts(tmp_path, monkeypatch):
    root, temporary, out = runtime_layout(tmp_path, monkeypatch)
    basetemp = worker.pytest_runtime_directory(root, out)
    basetemp.mkdir()
    (basetemp / 'test_one0').mkdir()
    (basetemp / 'test_onecurrent').symlink_to(basetemp / 'test_one0')
    (basetemp / 'intentional-broken-link').symlink_to(temporary / 'absent')
    assert (basetemp / 'intentional-broken-link').is_symlink()
    assert not any(path.is_symlink() for path in out.rglob('*'))
    assert [item['path'] for item in worker.artifacts(out)] == ['pytest-runtime.json']


@pytest.mark.parametrize('damage', ['missing', 'relative', 'other_run', 'outputs',
    'wrong_shape', 'other_task', 'tmp_symlink', 'parent_symlink', 'output_symlink',
    'traversal', 'absent'])
def test_pytest_runtime_rejects_unbound_or_unsafe_temp_paths(tmp_path, monkeypatch, damage):
    root, temporary, out = runtime_layout(tmp_path, monkeypatch)
    if damage == 'missing':
        monkeypatch.delenv('TMPDIR')
    elif damage == 'relative':
        monkeypatch.setenv('TMPDIR', 'runtime/SHARED_TESTS/tmp')
    elif damage == 'other_run':
        monkeypatch.setenv('TMPDIR', str(tmp_path / 'other' / 'runtime' / 'SHARED_TESTS' / 'tmp'))
    elif damage == 'outputs':
        monkeypatch.setenv('TMPDIR', str(out))
    elif damage == 'wrong_shape':
        monkeypatch.setenv('TMPDIR', str(temporary / 'extra'))
    elif damage == 'other_task':
        monkeypatch.setenv('TMPDIR', str(tmp_path / 'runtime' / 'OTHER_TASK' / 'tmp'))
    elif damage == 'tmp_symlink':
        temporary.rmdir()
        temporary.symlink_to(out, target_is_directory=True)
    elif damage == 'parent_symlink':
        temporary.rmdir()
        temporary.parent.rmdir()
        temporary.parent.symlink_to(out, target_is_directory=True)
    elif damage == 'output_symlink':
        out.rmdir()
        out.symlink_to(temporary, target_is_directory=True)
    elif damage == 'traversal':
        monkeypatch.setenv('TMPDIR', str(temporary / '..' / 'tmp'))
    else:
        temporary.rmdir()
    with pytest.raises(ValueError):
        worker.pytest_runtime_directory(root, out)
    assert not (out / 'pytest-runtime.json').exists()


def test_failed_pair_preserves_execution_scope_without_adoption_claims(tmp_path):
    failed = dict(returncode=1, timed_out=False, observation=None)
    report = worker.compare_pair(failed, failed, tmp_path / 'baseline', tmp_path / 'candidate')
    assert report['qualification_scope'] == 'execution_certificates_and_shared_model_contract'
    assert report['improvement_adoption_qualified'] is False
    assert report['selected_outputs_promoted'] is False


def test_failed_test_child_receipt_preserves_actual_nonpassing_evidence(tmp_path, monkeypatch):
    root, plan = make_plan(tmp_path)
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / 'out'
    out.mkdir()
    evidence = dict(collected=['tests/a.py::test_one'], passed_calls=[],
        nonpassing=[dict(nodeid='tests/a.py::test_one', when='call', outcome='failed')])

    def fail_tests(root, plan, out):
        worker.write_json(out / 'test-results.json', evidence)
        raise ValueError('All collected tests must pass with zero skips')

    monkeypatch.setattr(worker, 'child_tests', fail_tests)
    monkeypatch.setattr(worker.sys, 'argv', [str(PATH), '--child', 'tests',
        '--payload-root', str(root), '--plan', str(plan_path), '--out', str(out),
        '--expected-plan-sha256', worker.sha(plan_path),
        '--expected-source-sha256', plan['candidate']['source_sha256']])
    assert worker.main() == 1
    report = json.loads((out / 'observation.json').read_text())
    assert report['qualification_passed'] is False
    assert report['test_results'] == evidence
    assert report['qualification_scope'] == 'execution_certificates_and_shared_model_contract'
    assert report['improvement_adoption_qualified'] is False
