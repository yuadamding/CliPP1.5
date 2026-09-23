"""Validate retained timing receipts and archives without running code from them."""
import hashlib
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(z, name):
    return json.loads(z.read(name))


def main():
    for name, expected in json.loads((ROOT/'SHA256.json').read_text()).items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha((ROOT/name).read_bytes()) == expected
    index = json.loads((ROOT/'INDEX.json').read_text())
    for entry in index['attempts']:
        path = ROOT/entry['archive']
        assert path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256']
        with ZipFile(path) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            manifest = read(z, 'IMPORT_MANIFEST.json')
            for name, value in manifest['files'].items():
                data = z.read('imported/'+name)
                assert sha(data) == value['sha256'] and len(data) == value['bytes']
            for name, digest in read(z, 'sealed/inventory.json').items():
                assert sha(z.read('sealed/'+name)) == digest
            accepted = read(z, 'imported/receipts/accepted.json')
            terminal = read(z, 'imported/receipts/terminal.json')
            startup = read(z, 'imported/receipts/startup.json')
            assert accepted['job_id'] == terminal['job_id'] == entry['job_id']
            assert accepted['plan_sha256'] == terminal['plan_sha256'] == sha(z.read('sealed/PREPARED.json'))
            assert terminal == manifest['terminal']
            assert startup['inventory_sha256'] == sha(z.read('sealed/inventory.json'))
            assert f"Job <{entry['job_id']}>" in manifest['scheduler']
            assert 'Status <DONE>' in manifest['scheduler'] or 'Status <EXIT>' in manifest['scheduler']
            receipt = read(z, entry['receipt'])
            assert (ROOT/(entry['attempt']+'.json')).read_bytes() == z.read(entry['receipt'])
            assert receipt['schema'] == 'clipp1d.cuda.surrogate_timing.v1'
            assert receipt['status'] == entry['status'] and receipt['cuda_available'] is True
            assert receipt['diagnostic_observer'] is False and receipt['candidate_tracer'] is False
            for name, digest in receipt['artifacts'].items():
                assert sha(z.read('imported/results/'+name)) == digest
            for key, name in [('timing_driver', 'time_surrogate_cuda.py'), ('surrogate_driver', 'qualify_surrogate_cuda.py')]:
                assert receipt['helpers'][key] == sha(z.read('sealed/source/benchmarks/'+name))
            prior = ROOT.parent/'cuda-surrogate-review-v4/receipts/below64-a/experiment.json'
            assert sha(prior.read_bytes()) == receipt['coverage']['sha256']
            if entry['status'] == 'passed':
                assert terminal['passed'] is True and len(receipt['trials']) == 8
                for trial in receipt['trials']:
                    assert trial['status'] == 'passed' and trial['publication_qualified'] is True
                    assert trial['starts_attempted'] == trial['starts_returned'] == trial['starts_qualified'] == 99
                    assert trial['starts_raised'] == 0
                pairs = [p for p in receipt['comparisons'] if not p['warmup'] and p['complete_pair']]
                assert len(pairs) == 3
                assert receipt['timing_summary'] == entry['timing_summary']
                assert receipt['timing_summary']['median_paired_speedup'] == median(p['speedup_control_over_candidate'] for p in pairs)
                assert all(p['exact_labels'] and p['refit_max_difference'] == 0 and p['score_difference'] == 0 for p in pairs)
            else:
                assert receipt['error'] == 'Public timing phases are missing, negative, or nonfinite'
        print(entry['attempt']+': verified '+entry['status'])
    owner = json.loads((ROOT/'controller-started.json').read_text())
    for name, digest in owner['helpers'].items():
        assert sha((ROOT/'operations'/name).read_bytes()) == digest
    print('Timing evidence verified; frozen controller snapshot is not live status.')


if __name__ == '__main__':
    main()
