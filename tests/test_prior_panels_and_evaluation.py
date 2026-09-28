"""The controlled observation law, truth boundary and statistical unit."""
import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'benchmarks'))
from prior_perturbation_panels import DESIGNS, generate_panel  # noqa: E402
from evaluate_prior_perturbation import ari, summarize  # noqa: E402


def test_real_four_arm_serialization_and_evaluator_roundtrip(tmp_path, monkeypatch):
    """Explicit CPU component harness; this cannot qualify allocated CUDA."""
    import torch
    import prior_perturbation as experiment
    from prior_perturbation_panels import write_case
    from evaluate_prior_perturbation import case_metrics
    torch.set_num_threads(1)
    dataset = tmp_path/'data'
    (dataset/'inputs').mkdir(parents=True)
    (dataset/'truth').mkdir()
    case, truth = write_case(dataset, 'blind-test', np.array([20, 48, 21, 46]),
        np.array([80, 52, 79, 54]), np.ones(4), np.ones(4), 1.,
        np.array([1, 0, 1, 0]), np.array([.4, 1., .4, 1.]), np.ones(4))
    case.update(truth, depth=100, purity=1., cna_rate=0., design_index=1)
    monkeypatch.setattr(experiment, 'require_cuda', lambda device: torch.device('cpu'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 0)
    result = experiment.run_case(dataset/case['input_file'], tmp_path/'results'/case['case_id'],
                                 tumor_id=case['case_id'], compiled=False)
    measured = case_metrics(case, dataset, tmp_path/'results')
    assert measured['metrics']['P']['score'] <= measured['metrics']['B']['score']
    # Check the actual serialized source/output schema consumed by the LSF
    # supervisor, including unchanged raw files and nested artifact hashes.
    directory = Path(__file__).parents[1]/'benchmarks/prior_lsf'
    spec = importlib.util.spec_from_file_location('study_common_test', directory/'common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    monkeypatch.setitem(sys.modules, 'common', common)
    spec = importlib.util.spec_from_file_location('study_validator_test', directory/'validator.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    (tmp_path/'PREPARED.json').write_text(json.dumps(dict(source_sha256=result['source']['source_sha256'])))
    validated = validator.validate(tmp_path, dict(case, key=case['case_id'], mode='comparison'))
    assert validated['output_sha256']['RESULT.json']


def test_independent_alias_diagnostic_records_both_bracketed_wells(tmp_path):
    from prior_perturbation_panels import generate_mechanisms
    from prior_perturbation import prepare_model
    from qualify_prior_perturbation import alias_diagnostic
    import torch
    torch.set_num_threads(1)
    root = tmp_path/'inputs'
    generate_mechanisms(root)
    _, model = prepare_model(root/'inputs/mechanism-02.tsv', 'cpu', False)
    records = alias_diagnostic(model, tmp_path, 'cpu')
    assert len(records) == 3
    assert records[0]['pilot'] == pytest.approx(.75, abs=.001)
    assert records[2]['pilot'] == pytest.approx(.375, abs=.001)


def test_performance_gate_uses_all_paired_repeats_and_separates_temperature(tmp_path):
    from time_prior_perturbation import aggregate
    root = tmp_path/'timing'
    root.mkdir()
    cases = []
    for n in (256, 1000, 4000):
        for i in range(3):
            case_id = f'{n}-{i}'
            cases.append(dict(case_id=case_id, n=n, input_sha256='a'*64))
            folder = root/case_id
            folder.mkdir()
            records = [dict(repeat=r, arm=a, seconds=10 if a == 'B' else 15,
                gpu='same', source=dict(source_sha256='b'*64),
                peak_device_allocated_bytes=100, peak_device_reserved_bytes=200)
                for r in range(4) for a in ('B', 'P', 'D')]
            (folder/'TIMING.json').write_text(json.dumps(dict(input_sha256='a'*64, records=records)))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(cases=cases)))
    aggregate(manifest, root, tmp_path/'summary')
    report = json.loads((tmp_path/'summary/PERFORMANCE.json').read_text())
    assert len(report['comparisons']) == 72
    assert report['median_total_runtime_ratio'] == 1.5
    assert report['maximum_peak_memory_ratio'] == 1


def test_controlled_panel_has_exact_cells_and_blind_inputs(tmp_path):
    root = tmp_path/'development'
    assert generate_panel(root, 'development') == 108
    fit = json.loads((root/'FIT_MANIFEST.json').read_text())
    evaluation = json.loads((root/'EVALUATION_MANIFEST.json').read_text())
    cells = Counter(tuple(c[key] for key in ('depth', 'purity', 'cna_rate', 'design_index')) for c in evaluation['cases'])
    assert len(cells) == 108 and set(cells.values()) == {1}
    assert evaluation['generation_seed'] == 2026092601
    assert all(sum(sizes) == 400 for _, sizes in DESIGNS)
    assert fit['truth_in_fit_manifest'] is False
    assert all(set(c) == {'case_id', 'input_file', 'input_sha256', 'n'} for c in fit['cases'])
    with pytest.raises(FileExistsError):
        generate_panel(root, 'development')


def test_ari_agrees_with_independent_library():
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(99)
    for _ in range(40):
        x, y = rng.integers(0, 4, (2, 100))
        assert ari(x, y) == pytest.approx(adjusted_rand_score(x, y), abs=1e-15)
    assert ari([0, 0], [1, 1]) == 1
    assert ari([0, 0], [1, 2]) == 0


def test_tumor_bootstrap_pairs_all_arms_and_requires_complete_evidence():
    records = []
    for cell in range(2):
        for replicate in range(3):
            b = dict(ari=.4, ccf_mae=.1, ccf_rmse=.15, seconds=10., exact_k=True,
                     undercluster=False, overcluster=False, false_single=False,
                     false_split=False, n=400, score=500.)
            p = dict(b, ari=.5+replicate*.01, ccf_mae=.09, score=490.)
            records.append(dict(cell=[cell], metrics=dict(R=b, B=b, P=p, D=b)))
    summary = summarize(records, 6)
    assert summary['paired']['B']['ci95'][0] > 0
    assert summary['mae_reduction_percent'] == pytest.approx(10)
    assert summary['bootstrap_resamples'] == 10000
    assert summary['paired_tumors'] == 6
    assert summary['adoption_supported'] is False
    assert summary['gates']['memory'] is None
    assert summarize(records, 7)['gates']['execution_reliability'] is False


def test_evaluator_only_has_truth_access():
    path = Path(__file__).parents[1]/'benchmarks/prior_perturbation.py'
    spec = importlib.util.spec_from_file_location('protocol_boundary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import inspect
    assert 'truth' not in inspect.signature(module.run_case).parameters
    assert 'true_k' not in inspect.signature(module.additional_search).parameters
