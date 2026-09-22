"""Historical constrained integration replay against the frozen 2026-09-21 prototype.

This is not a qualification of current unconstrained inference. A pinned original
package and fresh output directory are required; importing this module runs no fits.
"""
import argparse
import json
from pathlib import Path
import sys


SOURCE_SHA256 = '6f2b1b35672b9067ebac404774ed3c4905c74d4c59a24c4fc49f8745e8adea8d'


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-source', type=Path, required=True,
                        help='Pinned historical source directory containing clipp1d/')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--study', type=Path, default=repo/'results/clipp2-transfer-study-20260921-v1')
    parser.add_argument('--panels', type=Path, nargs='+', default=[
        repo/'results/simclone-matched-20260921-v2', repo/'results/simclone-additional-four-20260921-v1'])
    args = parser.parse_args()
    from benchmark_chain import configure_execution
    controls = configure_execution(cpu_count=1, threads=1)
    sys.path.insert(0, str(args.package_source.resolve()))
    from clipp1d import api
    from clipp1d.report import write_json
    source = api.source_provenance()
    if (Path(api.__file__).resolve().parent.parent != args.package_source.resolve() or
            source['source_sha256'] != SOURCE_SHA256):
        raise ValueError('Historical prototype replay requires its original pinned constrained package')
    args.outdir.mkdir(parents=True)
    records = []
    for parent in args.panels:
        for case in json.loads((parent/'plan.json').read_text())['cases']:
            name = case['tumor_id']
            expected = json.loads((args.study/'best-seed-known-results'/f'{name}.json').read_text())
            result = api.fit(parent/name/'input.tsv', args.outdir/name)
            variant = expected['variants']['ward_and_boundary']
            candidate = variant.get('result', variant)
            if 'cuts' not in candidate:
                candidate = variant['refit']
            assert list(result.partition) == candidate['cuts'], name
            assert abs(result.selection_score-candidate['score']) < 1e-8, name
            assert result.candidate_provenance['raw_reference']['refit_score'] >= result.selection_score-1e-8
            assert result.provenance['input_sha256'] == expected['input_sha256']
            records.append(dict(case_id=name, score=result.selection_score,
                                direct=result.selected_lambda is None,
                                seconds=result.provenance['elapsed_seconds']))
            print(records[-1], flush=True)
    write_json(args.outdir/'qualification.json', dict(status='passed', cases=records,
               controls=controls, source=source, scope='Historical constrained prototype replay'))


if __name__ == '__main__':
    main()
