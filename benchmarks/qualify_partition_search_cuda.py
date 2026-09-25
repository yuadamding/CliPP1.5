"""Allocated-CUDA qualification of the separate partition-estimation contract.

Paired complete default-path fits use identical inputs. This small engineering
panel does not replace the predeclared held-out scientific panel or the frozen
193-case partition replay. Never run on a shared, unallocated login GPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from clipp1d.api import source_provenance
from clipp1d.cuda_api import fit, require_cuda, PARTITION_SCHEMA
from clipp1d.io import SCHEMA_COLUMNS


def write_case(path, alt, major):
    rows = []
    for i, (count, cn) in enumerate(zip(alt, major)):
        row = dict(zip(SCHEMA_COLUMNS, [f'm{i:03}', 's1', count, 100-count, 1, 1., 2,
                                       f'seg{i}', 'cn1', 1., cn, 1]))
        rows.append('\t'.join(str(row[key]) for key in SCHEMA_COLUMNS))
    path.write_text('\t'.join(SCHEMA_COLUMNS)+'\n'+'\n'.join(rows)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = require_cuda(args.device)
    args.outdir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records = []
    fixtures = [('diploid', [20, 48, 21, 46], [1]*4),
                ('mixture', [20, 30, 21, 32, 48, 25], [1, 2, 1, 2, 1, 3]),
                ('clonal', [49, 50, 51], [1]*3)]
    try:
        for name, alt, major in fixtures:
            path = args.outdir/(name+'.tsv')
            write_case(path, alt, major)
            baseline = fit(path, args.outdir/(name+'-baseline'), device=str(device))
            extended = fit(path, args.outdir/(name+'-partition'), device=str(device), partition_search=True)
            generic = fit(path, args.outdir/(name+'-generic'), device=str(device), partition_search=True,
                          generic_partition_grouping=True)
            for variant in (extended, generic):
                assert baseline.graph_sha256 == variant.graph_sha256
                assert baseline.selected_lambda == variant.selected_lambda
                assert baseline.search_status == variant.search_status
                assert np.array_equal(baseline.cluster_labels, variant.cluster_labels)
                for field in ('raw_phi', 'refitted_phi'):
                    np.testing.assert_allclose(getattr(baseline, field), getattr(variant, field), rtol=0., atol=1e-9)
                assert abs(baseline.selection_score - variant.selection_score) < 1e-7
                assert variant.partition_estimate.score <= baseline.selection_score
                assert variant.provenance['compiled_inference'] is True
            output = json.loads((args.outdir/(name+'-partition')/'run.json').read_bytes())
            assert output['schema'] == PARTITION_SCHEMA and len(output['table_sha256']) == 5
            for filename, expected in output['table_sha256'].items():
                assert hashlib.sha256((args.outdir/(name+'-partition')/filename).read_bytes()).hexdigest() == expected
            records.append(dict(case=name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                baseline_score=baseline.selection_score, partition_score=extended.partition_estimate.score,
                                generic_score=generic.partition_estimate.score,
                                family=extended.partition_estimate.provenance['candidate_family'],
                                baseline_seconds=baseline.operation_metrics['elapsed_seconds'],
                                partition_seconds=extended.operation_metrics['elapsed_seconds'],
                                generic_seconds=generic.operation_metrics['elapsed_seconds'],
                                partition_search_status=extended.partition_estimate.search_status,
                                peak_allocated_bytes=extended.provenance['peak_allocated_bytes']))
        receipt = dict(status='passed', paired_complete_path_cases=len(records), records=records)
    except Exception as error:
        receipt = dict(status='failed', error=repr(error), records=records)
        raise
    finally:
        receipt.update(created_utc=datetime.now(timezone.utc).isoformat(), source=source_provenance(),
                       device=str(device), gpu_name=torch.cuda.get_device_name(device),
                       scope='allocated compiled CUDA engineering qualification; not cohort accuracy or cross-device equivalence')
        with (args.outdir/'QUALIFICATION.json').open('x') as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
