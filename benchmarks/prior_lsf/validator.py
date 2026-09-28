"""Serialized study readback; no truth, fitting or CUDA work in the supervisor."""
from pathlib import Path
from common import digest, read


def public(folder, source_sha):
    receipt = read(folder/'run.json')
    assert receipt['status'] == 'success'
    assert receipt['provenance']['source_sha256'] == source_sha
    for name, expected in receipt['table_sha256'].items():
        assert Path(name).name == name and digest(folder/name) == expected
    return receipt


def validate(root, case):
    task = root/'results'/case['key']
    source_sha = read(root/'PREPARED.json')['source_sha256']
    mode = case['mode']
    status = 'complete'
    if mode == 'comparison':
        record = read(task/'RESULT.json')
        assert record['input_sha256'] == case['input_sha256']
        assert record['truth_used'] is False and record['original_raw_preserved']
        assert record['source']['source_sha256'] == source_sha
        baseline = None
        for arm in ('R', 'B', 'P', 'D'):
            receipt = public(task/arm, source_sha)
            tables = {name: receipt['table_sha256'][name] for name in (
                'mutation_clusters.tsv', 'cluster_centers.tsv', 'mutation_multiplicity.tsv')}
            if baseline is None:
                baseline = tables
            assert tables == baseline, 'Experimental arm changed a primary raw table'
        for arm in ('P', 'D'):
            search = read(task/f'{arm}-search/SEARCH.json')
            assert search['original_prior_restored'] and search['extra_birth_generations'] == 0
            assert search['selected']['score'] <= search['baseline']['score']
            assert search['attempted_raw_solves'] <= 12
            assert search['baseline']['score'] == record['result_scores']['B']
            if search['status'] != 'complete':
                status = 'incomplete'
    elif mode == 'discovery':
        record = read(task/'DISCOVERY.json')
        assert record['input_sha256'] == case['input_sha256']
        assert record['truth_used'] is False and record['fusion_solves'] == 0
        assert set(record['results']) == {'0.01', '0.03', '0.05'}
        if any(r['status'] != 'complete' for r in record['results'].values()):
            status = 'incomplete'
    elif mode == 'factorial':
        record = read(task/'FACTORIAL.json')
        assert record['truth_used'] is False and record['comparing_different_objective_values_forbidden']
        if any(r['status'] != 'qualified' for r in record['rows']):
            status = 'incomplete'
    elif mode == 'performance':
        record = read(task/'TIMING.json')
        assert record['input_sha256'] == case['input_sha256'] and len(record['records']) == 12
        for row in record['records']:
            sub = task/f"{row['repeat']}-{row['arm']}"
            public(sub/'B', source_sha)
            if row['arm'] != 'B':
                public(sub/row['arm'], source_sha)
    else:
        raise ValueError(mode)
    inventory = {}
    for path in sorted(task.rglob('*')):
        name = path.relative_to(task)
        if name.parts[0].startswith('cache-') or name.name == 'validated.json':
            continue
        assert not path.is_symlink()
        if path.is_file():
            inventory[str(name)] = digest(path)
    return dict(search_status=status, output_sha256=inventory, input_sha256=case['input_sha256'],
                source_sha256=source_sha, mode=mode, truth_used=False)
