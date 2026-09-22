"""Dependency and admission tests; never launch OCCAMS or contact Kubernetes."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1]/'benchmarks/run_occams.py'
    spec = importlib.util.spec_from_file_location('occams_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def predecessor(tmp_path, runner):
    root = tmp_path/'predecessor'
    root.mkdir()
    runner.write(root/'plan.json', {'cases': [1, 2]})
    return dict(root=str(root), plan_sha256=runner.sha(root/'plan.json'), cases=2, owners=[])


def finish(runner, binding, audit=True):
    root = Path(binding['root'])
    runner.write(root/'summary-final.json', {'done': 2})
    complete = dict(cases=2, status={'success': 1, 'fit_failure': 1},
                    summary_sha256=runner.sha(root/'summary-final.json'))
    runner.write(root/'COMPLETE.json', complete)
    if audit:
        (root/'audited-case-metrics.tsv').write_text('audited\n')
        runner.write(root/'final-audit.json', dict(status='passed', cases=2,
                     terminal_status=complete['status'], plan_sha256=binding['plan_sha256'],
                     summary_sha256=complete['summary_sha256'],
                     case_metrics_sha256=runner.sha(root/'audited-case-metrics.tsv')))


def test_requires_completion_and_audit(runner, predecessor):
    assert runner.predecessor_state(predecessor) == 'waiting_for_cases'
    finish(runner, predecessor, audit=False)
    assert runner.predecessor_state(predecessor) == 'waiting_for_audit'


def test_numerical_failures_can_finish_but_live_owner_still_blocks(runner, predecessor):
    finish(runner, predecessor)
    owner = dict(pid=os.getpid(), proc_start_ticks=Path('/proc/self/stat').read_text()
                 .rsplit(')', 1)[1].split()[19])
    predecessor['owners'] = [owner]
    assert runner.predecessor_state(predecessor) == 'waiting_for_owners_to_exit'
    owner['proc_start_ticks'] = '-1'  # PID reuse does not identify the old owner.
    assert runner.predecessor_state(predecessor) == 'ready'


@pytest.mark.parametrize('artifact', ['HALTED.json', 'finalization-failure.json'])
def test_failure_never_releases_gate(runner, predecessor, artifact):
    finish(runner, predecessor)
    runner.write(Path(predecessor['root'])/artifact, {'failure': True})
    with pytest.raises(ValueError, match='requires attention'):
        runner.predecessor_state(predecessor)


@pytest.mark.parametrize('artifact', ['plan.json', 'summary-final.json', 'audited-case-metrics.tsv'])
def test_modified_receipt_never_releases_gate(runner, predecessor, artifact):
    finish(runner, predecessor)
    with (Path(predecessor['root'])/artifact).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='changed|mismatch'):
        runner.predecessor_state(predecessor)


def test_incomplete_case_count_never_releases_gate(runner, predecessor):
    finish(runner, predecessor)
    path = Path(predecessor['root'])/'COMPLETE.json'
    complete = runner.load(path)
    complete['status']['success'] = 0
    runner.write(path, complete, replace=True)
    with pytest.raises(ValueError, match='count mismatch'):
        runner.predecessor_state(predecessor)


def make_plan(runner, root, predecessor, n=0, workers=1):
    root.mkdir()
    inputs = root/'input.tsv'
    inputs.write_text('dummy; not a numerical input\n')
    plan = dict(python=sys.executable, frozen_files={}, predecessor=predecessor,
                workers=workers, cpus=sorted(os.sched_getaffinity(0))[:workers],
                cases=[dict(tumor_id=f'case-{i}', retained_mutations=n-i,
                       input_path=str(inputs), input_sha256=runner.sha(inputs)) for i in range(n)])
    runner.write(root/'plan.json', plan)
    return SimpleNamespace(root=root, plan_sha256=runner.sha(root/'plan.json')), plan


def test_waiting_controller_cannot_submit_and_second_owner_is_rejected(
        runner, predecessor, tmp_path, monkeypatch):
    args, _ = make_plan(runner, tmp_path/'run', predecessor, n=1)

    def no_submit(*args, **kwargs):
        pytest.fail('Work admitted before dependency release')

    class EndObservation(Exception):
        pass

    monkeypatch.setattr(runner, 'execute', no_submit)
    monkeypatch.setattr(runner.time, 'sleep', lambda _: (_ for _ in ()).throw(EndObservation()))
    with pytest.raises(EndObservation):
        runner.controller(args)
    assert runner.load(args.root/'status.json')['state'] == 'waiting_for_cases'
    assert not (args.root/'activation.json').exists()
    assert not (args.root/'runs').exists()
    with pytest.raises(FileExistsError):
        runner.controller(args)


def test_scheduler_has_bounded_distinct_cpus_and_quick_first_order(
        runner, predecessor, tmp_path, monkeypatch):
    finish(runner, predecessor)
    args, plan = make_plan(runner, tmp_path/'run', predecessor, n=7, workers=2)
    admitted, occupied, peak = [], set(), []
    lock = threading.Lock()

    def execute(root, plan, digest, index, cpu):
        with lock:
            assert cpu not in occupied
            occupied.add(cpu)
            admitted.append(index)
            peak.append(len(occupied))
        time.sleep(.015)
        with lock:
            occupied.remove(cpu)
        return dict(status='fit_failure', index=index, tumor_id=plan['cases'][index]['tumor_id'])

    def summary(root, plan, terminals, *, final=False):
        if final:
            runner.write(root/'summary-final.json', {'cases': len(terminals)})
            (root/'case-metrics.tsv').write_text('dummy\n')
        return {'status': {'fit_failure': len(terminals)}}

    monkeypatch.setattr(runner, 'execute', execute)
    monkeypatch.setattr(runner, 'summarize', summary)
    monkeypatch.setitem(sys.modules, 'benchmark_chain', SimpleNamespace(configure_execution=lambda **kw: None))
    runner.controller(args)
    # OS scheduling may swap the first two workers' entry into execute().
    assert set(admitted[:2]) == {5, 6}
    assert sorted(admitted) == list(range(7))
    assert max(peak) == 2
    assert runner.load(args.root/'COMPLETE.json')['cases'] == 7
    assert runner.load(args.root/'activation.json')['workers'] == 2


def test_validation_failure_stops_admission_and_drains(runner, predecessor, tmp_path, monkeypatch):
    finish(runner, predecessor)
    args, _ = make_plan(runner, tmp_path/'run', predecessor, n=5, workers=1)
    admitted = []

    def execute(root, plan, digest, index, cpu):
        admitted.append(index)
        return dict(status='validation_failure', index=index, tumor_id=plan['cases'][index]['tumor_id'])

    monkeypatch.setattr(runner, 'execute', execute)
    monkeypatch.setattr(runner, 'summarize', lambda *a, **kw: None)
    runner.controller(args)
    assert len(admitted) == 1
    assert (args.root/'HALTED.json').exists()
    assert not (args.root/'COMPLETE.json').exists()


def test_frozen_source_drift_blocks_owner_creation(runner, predecessor, tmp_path):
    args, plan = make_plan(runner, tmp_path/'run', predecessor)
    (args.root/'frozen').mkdir()
    source = args.root/'frozen/source.py'
    source.write_text('original\n')
    plan['frozen_files'] = {'source.py': runner.sha(source)}
    runner.write(args.root/'plan.json', plan, replace=True)
    args.plan_sha256 = runner.sha(args.root/'plan.json')
    source.write_text('changed\n')
    with pytest.raises(ValueError, match='Frozen source changed'):
        runner.controller(args)
    assert not (args.root/'owner.lock').exists()


def test_fresh_worker_validates_outputs_and_measures_agreement(runner, tmp_path, make_input):
    from clipp1d.api import source_provenance
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path/'smoke'
    shutil.copytree(repo/'src/clipp1d', root/'frozen/src/clipp1d',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root/'frozen/benchmarks').mkdir()
    for name in ('run_occams.py', 'benchmark_chain.py', 'compare_clipp2.py'):
        shutil.copyfile(repo/'benchmarks'/name, root/'frozen/benchmarks'/name)
    path = make_input([{'mutation_id': '0001'}], metadata='##tumor_id=synthetic\n')
    baseline = root/'baseline'
    baseline.mkdir()
    clusters = baseline/'clusters.tsv'
    clusters.write_text('tumor_id\tmutation_id\tcluster_label\tphi_01\nsynthetic\t0001\t0\t1\n')
    case = dict(tumor_id='synthetic', input_path=str(path), input_sha256=runner.sha(path),
                retained_mutations=1,
                retained_ids_sha256=hashlib.sha256(json.dumps(['0001']).encode()).hexdigest(),
                baseline=dict(directory=str(baseline), clusters_filename=clusters.name,
                              files={clusters.name: runner.sha(clusters)}))
    plan = dict(python=sys.executable, source=source_provenance(), source_commit='synthetic',
                clipp2={'source_commit': 'synthetic'}, cases=[case],
                frozen_files={str(p.relative_to(root/'frozen')): runner.sha(p)
                              for p in (root/'frozen').rglob('*.py')})
    runner.write(root/'plan.json', plan)
    directory = root/'runs/synthetic'
    directory.mkdir(parents=True)
    result = subprocess.run([sys.executable, '-B', str(root/'frozen/benchmarks/run_occams.py'),
                             'worker', '--root', str(root), '--plan-sha256', runner.sha(root/'plan.json'),
                             '--index', '0', '--cpu', str(max(os.sched_getaffinity(0)))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    terminal = runner.load(directory/'terminal.json')
    assert terminal['status'] == 'success'
    metric = runner.load(directory/'metrics.json')
    assert metric['clipp2_agreement']['partition_ari'] == 1
    assert metric['clipp2_agreement']['ccf_mean_absolute_difference'] == 0
    assert runner.sha(directory/'fit/run.json') == terminal['run_sha256']
    startup = runner.load(directory/'startup.json')
    assert len(startup['controls']['selected_cpus']) == 1
    assert set(startup['controls']['thread_environment'].values()) == {'1'}
    # Reconciliation fails on a changed public output instead of accepting success.
    (directory/'fit/cluster_centers.tsv').write_text('corrupted\n')
    sys.path.insert(0, str(root/'frozen/benchmarks'))
    try:
        with pytest.raises(AssertionError):
            runner.case_metrics(root, plan, case, directory)
    finally:
        sys.path.pop(0)
