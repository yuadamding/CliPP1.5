"""Population identity and complete comparison controls for the research screen."""
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from evaluate_mixture_experiment import checked_comparators, evaluate
from mixture_artifacts import ArtifactPaths


def test_comparison_requires_every_method_and_exact_mutation_identity():
    ids = ['001', '002']
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    metrics = dict(true_k=2, selected_k=1, ari=0., ccf_mae=.1, true_smf=.5,
                   estimated_smf=0., smf_abs_error=.5)
    common = dict(cohort_id='cohort', population_id='matched', n_mutations=2,
                  mutation_ids_sha256=digest, **metrics)
    table = pd.DataFrame([dict(common, case_id=c, method_id=m)
                          for c in ('a', 'b') for m in ('clipp15_baseline', 'comparator')])
    external = checked_comparators(table, 'cohort', 'a', ids, metrics)
    assert external.method_id.tolist() == ['comparator']
    with pytest.raises(ValueError, match='inventory'):
        checked_comparators(table.drop(index=1), 'cohort', 'a', ids, metrics)
    with pytest.raises(ValueError, match='inventory'):
        checked_comparators(pd.concat([table, table.iloc[[0]]]), 'cohort', 'a', ids, metrics)
    with pytest.raises(ValueError, match='population'):
        checked_comparators(table, 'cohort', 'a', ['001', '003'], metrics)
    with pytest.raises(ValueError, match='baseline'):
        checked_comparators(table, 'cohort', 'a', ids, dict(metrics, ari=1.))


def test_remote_tree_mapping_keeps_identity_and_rejects_escaping_paths(tmp_path):
    root = tmp_path/'imported'
    root.mkdir()
    mapping = tmp_path/'mapping.json'
    mapping.write_text(json.dumps(dict(schema='clipp1d.artifact_tree_mapping.v1',
        remote_root='/remote/study', local_root=str(root))))
    paths = ArtifactPaths(mapping)
    assert paths('/remote/study/payload/input.tsv') == root/'payload/input.tsv'
    assert paths('/remote/study-other/file') == Path('/remote/study-other/file')
    assert paths('/local/truth.tsv') == Path('/local/truth.tsv')
    with pytest.raises(ValueError, match='escapes'):
        paths('/remote/study/../other/file')
    (root/'linked').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        paths('/remote/study/linked/file')
    assert ArtifactPaths()('/local/truth.tsv') == Path('/local/truth.tsv')


def test_diploid_only_single_cluster_panel_is_unassessed_not_perfect_multiplicity(tmp_path, monkeypatch):
    """Exercise publication when an actual cohort has no CNA or multi-K target."""
    from types import SimpleNamespace
    import evaluate_mixture_experiment as evaluator

    def write(path, value):
        path.write_text(json.dumps(value))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    inp, seed, truth = [tmp_path/name for name in ('input.tsv', 'seed.tsv', 'truth.tsv')]
    inp.write_text('The input reader is replaced by a diploid-only fixture.\n')
    monkeypatch.setattr(evaluator, 'read_tumor', lambda path: SimpleNamespace(retained=[
        SimpleNamespace(states=[(1., 1, 1)], mutation_id=name) for name in ['a', 'b']]))
    seed.write_text('mutation_id\tcluster_label\tpartition_ccf\tmultiplicity\na\t0\t1\t1\nb\t0\t1\t1\n')
    truth.write_text('mutation_id\ttrue_cluster\ttrue_ccf\ttrue_multiplicity\na\t0\t1\t1\nb\t0\t1\t1\n')
    ids = tmp_path/'ids.json'
    write(ids, {'case': ['a', 'b']})
    identity = hashlib.sha256(json.dumps(['a', 'b']).encode()).hexdigest()
    common = dict(case_id='case', cohort_id='cohort', population_id='matched', n_mutations=2,
        mutation_ids_sha256=identity, true_k=1, selected_k=1, ari=1., ccf_mae=0.,
        true_smf=0., estimated_smf=0., smf_abs_error=0.)
    comparators = tmp_path/'comparators.tsv'
    pd.DataFrame([dict(common, method_id='clipp15_baseline', method='CliPP1.5 baseline'),
                  dict(common, method_id='comparator', method='Comparator')]).to_csv(comparators, sep='\t', index=False)
    case = dict(case_id='case', input_path=str(inp), input_sha256=digest(inp),
                seed_path=str(seed), seed_sha256=digest(seed))
    inference = tmp_path/'inference.json'
    write(inference, dict(cases=[case], policy={}))
    evaluation = tmp_path/'evaluation.json'
    write(evaluation, dict(inference_sha256=digest(inference), cases=[dict(case_id='case',
        cohort='cohort', truth_path=str(truth), truth_sha256=digest(truth), truth_format='normalized',
        matched_ids_path=str(ids), matched_ids_sha256=digest(ids))], comparator_metrics=str(comparators),
        comparator_metrics_sha256=digest(comparators), scope='test fixture'))
    output = tmp_path/'fit'
    result = output/'00000'
    result.mkdir(parents=True)
    table = result/'mixture_mutation_clusters.tsv'
    table.write_text(seed.read_text().replace('partition_ccf', 'mixture_ccf'))
    write(output/'BINDING.json', dict(source_inventory={'fixture': 'bound'}, policy={}))
    write(output/'COMPLETE.json', dict(source_unchanged=True, manifest_unchanged=True,
        failures=[], results=[dict(case_id='case', output=str(result))]))
    write(result/'EXPERIMENT.json', dict(inputs=case, policy={}, output_sha256={table.name: digest(table)},
        adaptive=False, status='em_fixed_point', elapsed_seconds=1., candidates=[]))
    destination = tmp_path/'evaluation'
    evaluate(inference, evaluation, [output], destination)
    summary = json.loads((destination/'SUMMARY.json').read_text())
    for entry in summary['supplied_cn_multiplicity'].values():
        assert entry['eligible'] == 0 and entry['coverage'] is None and entry['macro_f1'] is None
    assert summary['development_nonregression_checks']['cohort/matched']['multi_mean_ari'] is None
    assert summary['development_nonregression_checks']['cohort/matched']['mean_ari'] is True
    assert summary['development_supplied_cn_multiplicity_checks']['cohort']['macro_f1'] is None
    assert summary['development_all_pass'] is False
    assert summary['development_unassessed_checks']
