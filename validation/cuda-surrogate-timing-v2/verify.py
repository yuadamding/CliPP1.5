"""Verify the additional timing evidence, including incomplete scalar searches."""
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    for name, digest in json.loads((ROOT/'SHA256.json').read_text()).items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha((ROOT/name).read_bytes()) == digest
    coverage = {sha(p.read_bytes()) for p in (ROOT.parent/'cuda-surrogate-review-v4/receipts').glob('*/experiment.json')}
    for entry in json.loads((ROOT/'INDEX.json').read_text())['attempts']:
        path = ROOT/entry['archive']
        assert path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256']
        with ZipFile(path) as z:
            def read(name):
                return json.loads(z.read(name))
            names = z.namelist()
            assert len(names) == len(set(names))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
            for name, value in read('IMPORT_MANIFEST.json')['files'].items():
                body = z.read('imported/'+name)
                assert sha(body) == value['sha256'] and len(body) == value['bytes']
            for name, digest in read('sealed/inventory.json').items():
                assert sha(z.read('sealed/'+name)) == digest
            accepted, terminal = (read('imported/receipts/'+n+'.json') for n in ('accepted', 'terminal'))
            assert accepted['job_id'] == terminal['job_id'] == entry['job_id']
            assert accepted['plan_sha256'] == terminal['plan_sha256'] == sha(z.read('sealed/PREPARED.json'))
            assert terminal == read('IMPORT_MANIFEST.json')['terminal'] and terminal['passed']
            scheduler = read('IMPORT_MANIFEST.json')['scheduler']
            assert f"Job <{entry['job_id']}>" in scheduler and 'Status <DONE>' in scheduler
            r = read(entry['receipt'])
            assert (ROOT/(entry['attempt']+'.json')).read_bytes() == z.read(entry['receipt'])
            assert r['schema'] == 'clipp1d.cuda.surrogate_timing.v1' and r['status'] == 'passed'
            assert r['cuda_available'] and not r['diagnostic_observer'] and not r['candidate_tracer']
            assert r['coverage']['sha256'] in coverage and r['coverage']['sha256'] == entry['coverage_sha256']
            for name, digest in r['artifacts'].items():
                assert sha(z.read('imported/results/'+name)) == digest
            for key, name in [('timing_driver', 'time_surrogate_cuda.py'), ('surrogate_driver', 'qualify_surrogate_cuda.py')]:
                assert r['helpers'][key] == sha(z.read('sealed/source/benchmarks/'+name))
            assert len(r['trials']) == 8 and len(r['comparisons']) == 4
            for trial in r['trials']:
                assert trial['status'] in ('passed', 'incomplete')
                assert trial['starts_attempted'] == trial['starts_returned'] == 99
                assert trial['starts_raised'] == 0
                assert 0 <= trial['starts_qualified'] <= 99
                if trial['publication_qualified']:
                    assert trial['status'] == 'passed' and trial['starts_qualified'] == 99
                else:
                    assert trial['status'] == 'incomplete'
                if trial['surrogate_policy'] == 'coordinate_backtracking_v1':
                    assert trial['publication_qualified']
            pairs = [p for p in r['comparisons'] if not p['warmup'] and p['complete_pair']]
            summary = r['timing_summary']
            assert summary == entry['timing_summary'] and summary['complete_warm_pairs'] == len(pairs)
            for key, field in [('median_paired_speedup', 'speedup_control_over_candidate'),
                               ('candidate_median_seconds', 'candidate_seconds'),
                               ('control_median_seconds', 'control_seconds')]:
                assert summary[key] == (median(p[field] for p in pairs) if pairs else None)
            assert all(p['speedup_control_over_candidate'] is None for p in r['comparisons'] if not p['complete_pair'])
        print(entry['attempt']+': verified')
    print('Evidence integrity and timing arithmetic verified; no numerical rerun.')


if __name__ == '__main__':
    main()
