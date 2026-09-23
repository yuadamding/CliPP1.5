"""Strict CPU output validation using the same cohort metrics and identity contract."""
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from clipp1d.cuda.policy import CudaPolicy


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(case, output, expected_source_sha256, result=None):
    from benchmarks.compare_clipp2 import metrics, read_rows
    from benchmarks.cohort_staging import staged_input_path
    run = read(output/'run.json')
    assert run['schema'] == 'clipp1d.cuda.run.v3' and run['status'] == 'success'
    source = run['provenance']
    assert source['source_sha256'] == expected_source_sha256
    assert source['input_sha256'] == case['input_sha256']
    assert source['backend'] == 'cpu' and source['dtype'] == 'float64'
    assert source['clonal_constraint'] is False and source['clonal_label_rule'] == 'nearest_to_one_l2_v1'
    assert source['policy'] == asdict(CudaPolicy()) and source['max_major_cn'] == 4
    assert source['compiled_inference'] is False and source['execution_scope'] == 'cpu_numerical_reference' and source['numerical_device'] == 'cpu' and source['cpu_numeric_fallback'] is False
    names = {'mutation_clusters.tsv','mutation_multiplicity.tsv','cluster_centers.tsv'}
    assert set(run['table_sha256']) == names
    assert {f.name for f in output.iterdir()} == names | {'run.json'}
    for name, sha in run['table_sha256'].items():
        assert digest(output/name) == sha
    rows = read_rows(output/'mutation_clusters.tsv')
    calls = read_rows(output/'mutation_multiplicity.tsv')
    ids = sorted(rows)
    assert len(ids) == case['retained_mutations'] and set(calls) == set(rows)
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == case['retained_ids_sha256']
    assert all(r['tumor_id'] == case['tumor_id'] and r['sample_id'] == case['sample_id'] for r in rows.values())
    with (output/'cluster_centers.tsv').open() as stream:
        centers = list(csv.DictReader(stream, delimiter='\t'))
    labels = {r['cluster_label'] for r in rows.values()}
    center = {r['cluster_label']: r for r in centers}
    assert len(centers) == len(labels) and set(center) == labels
    assert labels == set(map(str, range(len(labels))))
    for label, c in center.items():
        members = [row for row in rows.values() if row['cluster_label'] == label]
        assert int(c['cluster_size']) == len(members)
        assert all(float(r['refitted_ccf']) == float(c['refitted_ccf']) for r in members)
        assert c['designated_clonal'] == ('1' if label == '0' else '0')
    assert abs(float(center['0']['refitted_ccf'])-1) == min(abs(float(c['refitted_ccf'])-1) for c in centers)
    lineage = run['candidate_provenance']
    assert lineage['raw_qualified'] is True and lineage['refit_qualified'] is True
    assert lineage['candidate_family'] == 'complete_graph_path'
    assert lineage['graph_sha256'] == run['graph_sha256'] and lineage['raw_certificate'] == run['raw_diagnostics']
    records = run['search']
    complete = all(r.get('raw_status') == r.get('refit_status') == 'qualified' and r.get('search_complete',False) for r in records)
    assert run['search_status'] == ('complete' if complete else 'incomplete')
    qualified = [r for r in records if r.get('raw_status') == r.get('refit_status') == 'qualified']
    winner = min(qualified, key=lambda r:(r['score'],r['clusters'],r['lambda_value']))
    assert (run['selection_score'],len(centers),run['selected_lambda']) == (winner['score'],winner['clusters'],winner['lambda_value'])
    if result is not None:
        from clipp1d.cuda_api import _validate_result
        from clipp1d.io import read_tumor
        _validate_result(result, read_tumor(staged_input_path(output.parent, case)), records)
    truth_path = output.parent/'truth.tsv'
    assert digest(truth_path) == case['truth_sha256']
    truth = read_rows(truth_path)
    scored = {i:row for i,row in rows.items() if i in truth}
    assert set(rows)-set(truth) == set(case['truth_unmatched_retained'])
    assert len(scored)/len(rows) == case['truth_coverage']
    adapted = {i:dict(row,multiplicity_call=row['refitted_multiplicity_call']) for i,row in calls.items()}
    metric = metrics(scored, truth, adapted, 'refitted_ccf')
    metric.update(true_smf=sum(float(truth[i]['true_ccf']) < 1-1e-12 for i in scored)/len(scored),
                  estimated_smf=sum(row['cluster_label'] != '0' for row in scored.values())/len(scored),
                  full_retained_estimated_smf=sum(row['cluster_label'] != '0' for row in rows.values())/len(rows),
                  truth_matched=len(scored), truth_unmatched=len(rows)-len(scored),
                  true_k=len({truth[i]['true_cluster'] for i in scored}), full_retained_selected_k=len(centers),
                  ccf_estimator='final_refit', multiplicity_estimator='final_refit',
                  smf_definition='fraction outside designated closest-to-one public label 0; truth CCF < 1-1e-12',
                  search_status=run['search_status'])
    return dict(metrics=metric, output_sha256={name:digest(output/name) for name in names|{'run.json'}},
                source_sha256=source['source_sha256'], search_status=run['search_status'])
