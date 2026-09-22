"""Current free fits remain valid throughout the frozen benchmark audit."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_single_case_free_fit_evaluation_and_final_audit(tmp_path, make_input, monkeypatch):
    from legacy_chain_api import fit
    from legacy_chain_api import source_provenance

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'benchmarks'))
    runner = importlib.import_module('run_four_cohorts')
    finalizer = importlib.import_module('finalize_four_cohorts')
    source = make_input([dict(mutation_id='m1', alt_count=25, ref_count=75)])
    root = tmp_path/'panel'
    directory = root/'runs/synthetic/case'
    directory.mkdir(parents=True)
    truth = root/'truth.tsv'
    truth.write_text('mutation_id\ttrue_cluster\ttrue_ccf\ttrue_multiplicity\tmajor_cn\tminor_cn\n'
                     'm1\t0\t0.625\t1\t1\t1\n')
    result = fit(source, directory/'fit')
    case = dict(dataset='synthetic', case_id='case', retained_mutations=1, purity=.8, mean_depth=100,
                input_path=str(source), input_sha256=runner.sha(source),
                original_input_path=str(source), original_input_sha256=runner.sha(source),
                truth_path=str(truth), truth_sha256=runner.sha(truth), truth_sources={})
    metrics = runner.evaluate(result, case, directory)
    for item in metrics.values():
        assert item['designated_clonal_fraction'] == 1
        assert item['all_exact_one_fraction'] == 0
        assert item['smf_error'] == 1
        assert item['estimated_smf'] == 0
        assert item['smf_definition'] == 'nearest_to_one_l2_v1'
        assert item['mae'] == pytest.approx(0)
    package = Path(importlib.import_module('clipp1d').__file__).parent
    manifest = source_provenance()
    frozen = root/'frozen/src/clipp1d'
    frozen.mkdir(parents=True)
    for name in manifest['source_files']:
        (frozen/name).write_bytes((package/name).read_bytes())
    plan = dict(cases=[case], source=manifest, runner_files={})
    runner.write(root/'plan.json', plan)
    runner.write(directory/'startup.json', dict(plan_sha256=runner.sha(root/'plan.json'),
                 controls=dict(selected_cpus=[29], thread_environment={'OMP_NUM_THREADS': '1'})))
    terminal = dict(status='success', fit_seconds=1., proposal_seconds=0., max_rss_bytes=1,
                    search_status=result.search_status, selected_origin=result.candidate_provenance['origin'],
                    score_gain=result.candidate_provenance['raw_reference']['refit_score']-result.selection_score,
                    ari_gain=0., run_sha256=runner.sha(directory/'fit/run.json'),
                    metrics_sha256=runner.sha(directory/'metrics.json'))
    runner.write(directory/'terminal.json', terminal)
    runner.summarize(root, plan, [dict(terminal, dataset='synthetic', case_id='case')], final=True)
    runner.write(root/'COMPLETE.json', dict(cases=1, status={'success': 1},
                 summary_sha256=runner.sha(root/'summary-final.json')))
    checked_plan, summary, records = finalizer.audit(root)
    assert len(records) == 1
    assert json.loads((root/'final-audit.json').read_text())['status'] == 'passed'
    finalizer.report(root, checked_plan, summary, records)
    report = (root/'REPORT.md').read_text()
    assert 'c4f3d3e54ac94f1b6707463784ed9c85a86f698d' not in report
    assert '24 CPU workers' not in report
    centers = directory/'fit/cluster_centers.tsv'
    centers.write_text(centers.read_text().removesuffix('1\n')+'0\n')
    receipt = runner.load(directory/'fit/run.json')
    receipt['table_sha256']['cluster_centers.tsv'] = runner.sha(centers)
    runner.write(directory/'fit/run.json', receipt, replace=True)
    terminal['run_sha256'] = runner.sha(directory/'fit/run.json')
    runner.write(directory/'terminal.json', terminal, replace=True)
    with pytest.raises(AssertionError):
        finalizer.audit(root)


def test_importing_historical_qualification_has_no_execution_side_effects(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'benchmarks'))
    module = importlib.import_module('qualify_partition_integration')
    assert callable(module.main)


@pytest.mark.parametrize('script,extra', [
    ('diagnose_full_stationarity_gate.py', ['--observed-directory']),
    ('diagnose_snap_mechanism.py', ['--diagnosis-directory']),
    ('prototype_penalized_polish.py', []),
])
def test_historical_diagnosis_rejects_current_package_even_with_assertions_disabled(tmp_path, script, extra):
    repo = Path(__file__).parents[1]
    output = tmp_path/'never-created'
    command = [sys.executable, '-O', str(repo/'benchmarks'/script),
               '--frozen-package', str(repo/'src/clipp1d'), '--fixture-directory', str(tmp_path),
               '--outdir', str(output), '--cpu', str(min(os.sched_getaffinity(0)))]
    if extra:
        command += [*extra, str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert 'Expected historical constrained frozen 72ba3ed package' in result.stderr
    assert not output.exists()
