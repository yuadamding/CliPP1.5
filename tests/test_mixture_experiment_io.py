"""Publication/identity checks for the explicit experimental runner."""
import importlib.util
from pathlib import Path
import sys

import pytest
import pandas as pd

BENCHMARKS = Path(__file__).parents[1] / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location('mixture_runner', BENCHMARKS / 'run_mixture_experiment.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_manifest_rejects_truth_and_changed_artifacts(tmp_path):
    data, seed = tmp_path/'input.tsv', tmp_path/'seed.tsv'
    data.write_text('input')
    seed.write_text('seed')
    case = dict(case_id='case', input_path=str(data), input_sha256=runner.sha(data),
                seed_path=str(seed), seed_sha256=runner.sha(seed))
    runner.validate_case(case)
    with pytest.raises(ValueError, match='only ID'):
        runner.validate_case(dict(case, truth_path='truth.tsv'))
    data.write_text('changed')
    with pytest.raises(ValueError, match='Changed input'):
        runner.validate_case(case)


def test_receipt_no_clobber_and_explicit_cpu_gate(tmp_path):
    path = tmp_path/'receipt.json'
    runner.write_json(path, {'test': 1})
    with pytest.raises(FileExistsError):
        runner.write_json(path, {'test': 2})
    for extra in (['--device', 'cpu'], ['--cpu-reference']):
        with pytest.raises(SystemExit):
            runner.main(['--manifest', 'unused', '--outdir', 'unused', *extra])


def test_preserved_partition_never_inherits_rejected_mixture_parameters():
    from apply_mixture_structural_guard import preserved_metadata
    baseline = pd.DataFrame(dict(cluster_label=[1, 0, 1, 0], partition_ccf=[.4, .95, .4, .95]))
    parent = dict(centers=[.7], weights=[1.], enrichment=[.8], score=123., log_likelihood=-60.)
    metadata = preserved_metadata(parent, baseline)
    assert metadata['centers'] == [.95, .4]
    assert metadata['baseline_label_order'] == [0, 1]
    assert metadata['weights'] is metadata['enrichment'] is metadata['score'] is None
    assert metadata['status'] == 'preserved_baseline'
    assert parent['centers'] == [.7]
