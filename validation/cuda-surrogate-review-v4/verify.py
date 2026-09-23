"""Read-only integrity and receipt-chain checks; no CUDA inference or cluster access.

Run with Python's standard library. ZIP members are read without extracting code.
The cohort payload and other cohort outputs are intentionally external.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(z, name):
    return json.loads(z.read(name))


def package(z, prefix):
    names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.py'))
    assert names
    h = hashlib.sha256()
    for name in names:
        h.update(f'{name.removeprefix(prefix)}\0{sha(z.read(name))}\n'.encode())
    return h.hexdigest()


def verify_experiment(z, row):
    manifest = read(z, 'IMPORT_MANIFEST.json')
    for name, entry in manifest['files'].items():
        data = z.read('imported/'+name)
        assert sha(data) == entry['sha256'] and len(data) == entry['bytes']
    for name, expected in read(z, 'sealed/inventory.json').items():
        assert sha(z.read('sealed/'+name)) == expected
    accepted = read(z, 'imported/receipts/accepted.json')
    terminal = read(z, 'imported/receipts/terminal.json')
    startup = read(z, 'imported/receipts/startup.json')
    plan = read(z, 'sealed/PREPARED.json')
    assert terminal == manifest['terminal']
    assert terminal['passed'] is True
    assert accepted['job_id'] == terminal['job_id'] == row['job_id']
    assert accepted['plan_sha256'] == terminal['plan_sha256'] == sha(z.read('sealed/PREPARED.json'))
    assert startup['inventory_sha256'] == sha(z.read('sealed/inventory.json'))
    assert f"Job <{row['job_id']}>" in manifest['scheduler'] and 'Status <DONE>' in manifest['scheduler']
    assert 'User <yding4>' in manifest['scheduler'] and accepted['job_name'] in manifest['scheduler']
    assert sha(z.read(row['receipt_path'])) == row['receipt_sha256']
    result = read(z, row['receipt_path'])
    assert result['status'] == 'passed' and result['cuda_available'] is True
    assert row['source_sha256'] == result['source']['source_sha256'] == plan['source_sha256']['current']
    assert package(z, 'sealed/source/src/clipp1d/') == row['source_sha256']
    assert result['helpers'] == row['helpers']
    for name, digest in result['artifacts'].items():
        assert sha(z.read('imported/results/'+name)) == digest
    candidates = [r for r in result['trials'] if r['surrogate_policy'] == 'coordinate_backtracking_v1']
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate['status'] == 'passed' and candidate['starts_attempted'] == 99
    assert candidate['full_path']['search_status'] == 'complete'
    assert candidate['full_path']['final_export_and_publication_qualified'] is True
    assert result['comparisons'] == [row['comparisons']]
    assert row['comparisons']['exact_labels'] and row['comparisons']['refit_max_difference'] == 0
    assert row['comparisons']['score_difference'] == 0
    if 'literal_replay' in result:
        replay = result['literal_replay']
        assert replay['problem_unchanged'] and not replay['fitted_qualified']
        assert not replay['compiled']['independently_qualified']
        assert not replay['eager']['independently_qualified']
    exposed = ROOT/'receipts'/row['attempt']
    assert (exposed/'experiment.json').read_bytes() == z.read(row['receipt_path'])
    assert (exposed/'terminal.json').read_bytes() == z.read('imported/receipts/terminal.json')


def verify_cohort(z, entry):
    imported = read(z, 'IMPORT_MANIFEST.json')
    for name, digest in imported['files'].items():
        assert sha(z.read('readback/'+name)) == digest
    inventory = read(z, 'sealed/inventory.json')
    omitted = entry['omitted_sealed_members']
    assert set(omitted) == {n for n in inventory if n == 'payload.zip' or '__pycache__' in Path(n).parts}
    assert all(inventory[n] == digest for n, digest in omitted.items())
    for name, digest in inventory.items():
        if name not in omitted:
            assert sha(z.read('sealed/'+name)) == digest
    for name in ('worker.py', 'common.py', 'source/benchmarks/cohort_staging.py', 'PREPARED.json', 'inventory.json'):
        assert z.read('sealed/'+name) == z.read('readback/'+name)
    assert sha(z.read('sealed/cases.json')) == imported['cases_sha256']
    case = next(c for c in read(z, 'sealed/cases.json') if c['key'] == '001783')
    assert case == imported['case']
    plan = read(z, 'readback/PREPARED.json')
    assert plan['imported_cases'] == 112 and len(read(z, 'sealed/cases.json')) == 5456
    assert package(z, 'sealed/source/src/clipp1d/') == plan['source_sha256']
    accepted = read(z, 'readback/receipts/001783/accepted.json')
    startup = read(z, 'readback/receipts/001783/startup.json')
    terminal = read(z, 'readback/receipts/001783/terminal.json')
    assert accepted['job_id'] == startup['job_id'] == terminal['job_id'] == '77339914'
    assert accepted['plan_sha256'] == sha(z.read('readback/PREPARED.json'))
    assert startup['runtime']['source']['source_sha256'] == plan['source_sha256']
    assert terminal['status'] == 'validated_complete'
    assert terminal['validated_sha256'] == sha(z.read('readback/results/001783/validated.json'))
    validated = read(z, 'readback/results/001783/validated.json')
    assert validated['source_sha256'] == plan['source_sha256']
    original = z.read('readback/results/001783/input/'+case['input_filename'])
    assert sha(original) == case['input_sha256'] == validated['input_sha256']
    assert case['input_filename'] == case['tumor_id']+'.tsv'
    assert not any(line.startswith(b'##tumor_id') for line in original.splitlines())
    for name, digest in validated['output_sha256'].items():
        assert sha(z.read('readback/results/001783/output/'+name)) == digest
    mutations = list(csv.DictReader(io.StringIO(z.read('readback/results/001783/output/mutation_clusters.tsv').decode()), delimiter='\t'))
    assert len(mutations) == case['retained_mutations'] == 200
    assert all((r['tumor_id'], r['sample_id']) == (case['tumor_id'], case['sample_id']) for r in mutations)
    ids = sorted(r['mutation_id'] for r in mutations)
    assert sha(json.dumps(ids).encode()) == case['retained_ids_sha256']
    # The frozen worker is included for inspection of fit/refit/validation routing.
    worker = z.read('readback/worker.py').decode()
    assert 'stage_case_input(task,case,bundle)' in worker
    assert "fit(staged_input_path(task,case),task/'output'" in worker
    assert '_validate_result(result, read_tumor(staged_input_path(output.parent, case)), records)' in worker


def main():
    for name, expected in json.loads((ROOT/'SHA256.json').read_text()).items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha((ROOT/name).read_bytes()) == expected
    archives = json.loads((ROOT/'ARCHIVES.json').read_text())
    rows = {r['attempt']+'.zip': r for r in json.loads((ROOT/'QUALIFICATION.json').read_text())['stages']}
    assert len(rows) == 6
    for name, entry in archives.items():
        path = ROOT/name
        assert path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256']
        with ZipFile(path) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            if name in rows:
                verify_experiment(z, rows[name])
            else:
                assert name == 'cohort-staging-canary.zip'
                verify_cohort(z, entry)
        print(name+': verified')
    print('Six CUDA experiment receipts and the actual cohort staging canary verified; no numerical rerun.')


if __name__ == '__main__':
    main()
