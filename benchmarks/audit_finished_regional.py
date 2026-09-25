"""Read-only finished-case audit for the receipt-bound simulation summaries."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(case, record, *, read_rows, metrics):
    root = Path(record['source_root'])
    pool = record['pool']
    assert pool in ('cpu', 'lsf', 'regional_a100')
    cpu = pool == 'cpu'
    task = root / ('output/cases' if pool == 'regional_a100' else 'results') / case['key']
    if cpu:
        terminal_path = root / 'receipts' / (case['key'] + '.reconciled.json')
    elif pool == 'lsf':
        terminal_path = root / 'receipts' / case['key'] / 'reconciled.json'
    else:
        terminal_path = task / 'terminal.json'
    terminal = json.loads(terminal_path.read_bytes())
    if pool in ('cpu', 'lsf'):
        terminal = terminal['outcome']
    assert terminal['status'] == record['status']
    assert terminal['status'].startswith('validated_')
    assert digest(task / 'validated.json') == terminal['validated_sha256']
    validated = json.loads((task / 'validated.json').read_bytes())
    output = task / 'output'
    for name, sha in validated['output_sha256'].items():
        assert Path(name).name == name and digest(output / name) == sha
    run = json.loads((output / 'run.json').read_bytes())
    assert run['status'] == 'success'
    source = run['provenance']
    assert source['source_sha256'] == validated['source_sha256']
    assert source['input_sha256'] == case['input_sha256']
    assert source['clonal_constraint'] is False
    assert source['clonal_label_rule'] == 'nearest_to_one_l2_v1'
    assert source['backend'] == ('cpu' if cpu else 'cuda')
    truth_path = task / 'truth.tsv'
    assert digest(truth_path) == case['truth_sha256']
    truth = read_rows(truth_path)
    rows = read_rows(output / 'mutation_clusters.tsv')
    assert len(rows) == case['retained_mutations']
    assert hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest() == case['retained_ids_sha256']
    assert set(rows) - set(truth) == set(case['truth_unmatched_retained'])
    assert all(row['tumor_id'] == case['tumor_id'] and row['sample_id'] == case['sample_id']
               for row in rows.values())
    calls = read_rows(output / 'mutation_multiplicity.tsv')
    assert set(calls) == set(rows)
    with (output / 'cluster_centers.tsv').open() as f:
        centers = list(csv.DictReader(f, delimiter='\t'))
    labels = {r['cluster_label'] for r in rows.values()}
    assert labels == {c['cluster_label'] for c in centers}
    zero, = [c for c in centers if c['cluster_label'] == '0']
    distance = abs(float(zero['refitted_ccf']) - 1)
    assert distance == min(abs(float(c['refitted_ccf']) - 1) for c in centers)
    tied = [c['cluster_label'] for c in centers if abs(float(c['refitted_ccf']) - 1) == distance]
    for center in centers:
        members = [r for r in rows.values() if r['cluster_label'] == center['cluster_label']]
        assert len(members) == int(center['cluster_size'])
        assert all(float(r['refitted_ccf']) == float(center['refitted_ccf']) for r in members)
        assert center['designated_clonal'] == ('1' if center['cluster_label'] == '0' else '0')
    scored = {k: row for k, row in rows.items() if k in truth}
    adapted = {k: dict(v, multiplicity_call=v['refitted_multiplicity_call']) for k, v in calls.items()}
    metric = metrics(scored, truth, adapted, 'refitted_ccf')
    for key in ('ari', 'ccc', 'rmse'):
        assert np.isclose(metric[key], validated['metrics'][key], rtol=1e-12, atol=1e-12), key
    assert metric['cna_only_multiplicity'] == validated['metrics']['cna_only_multiplicity']
    actual = np.array([float(truth[k]['true_ccf']) for k in sorted(scored)])
    estimated = np.array([float(scored[k]['refitted_ccf']) for k in sorted(scored)])
    true_smf = float(np.mean(actual < 1 - 1e-12))
    estimated_smf = sum(r['cluster_label'] != '0' for r in scored.values()) / len(scored)
    assert true_smf == validated['metrics']['true_smf']
    assert estimated_smf == validated['metrics']['estimated_smf']
    assert run['search_status'] == validated['search_status'] == record['status'][10:]
    cna_rate = case.get('metadata', {}).get('cna_event_rate')
    metric.update(key=case['key'], tumor_id=case['tumor_id'], sample_id=case['sample_id'],
                  dataset=case['dataset'], backend=source['backend'],
                  source_sha256=source['source_sha256'], source_root=str(root),
                  terminal_sha256=digest(terminal_path), validated_sha256=digest(task / 'validated.json'),
                  search_status=run['search_status'], true_k=len({truth[k]['true_cluster'] for k in scored}),
                  true_smf=true_smf, estimated_smf=estimated_smf,
                  smf_absolute_error=abs(estimated_smf-true_smf),
                  ccf_mae=float(np.mean(abs(estimated-actual))),
                  truth_matched=len(scored), truth_unmatched=len(rows)-len(scored),
                  cna_event_rate=float(cna_rate) if cna_rate is not None else None,
                  mean_depth=case['mean_depth'], clonal_cluster='0',
                  clonal_ccf=float(zero['refitted_ccf']), clonal_distance=distance,
                  clonal_ties=tied, clonal_tie_count=len(tied),
                  clonal_mutations=int(zero['cluster_size']),
                  subclonal_mutations=len(rows)-int(zero['cluster_size']),
                  full_retained_mutations=len(rows),
                  elapsed_seconds=validated.get('operation_metrics', {}).get('elapsed_seconds'),
                  ccf_sums=dict(n=len(actual), truth=float(actual.sum()), estimate=float(estimated.sum()),
                                truth2=float((actual**2).sum()), estimate2=float((estimated**2).sum()),
                                product=float((actual*estimated).sum()),
                                absolute_error=float(abs(estimated-actual).sum()),
                                squared_error=float(((estimated-actual)**2).sum())))
    return metric
