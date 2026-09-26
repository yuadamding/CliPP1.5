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
from clipp1d.cuda.refinement import PartitionSearchPolicy
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
            current = fit(path, args.outdir/(name+'-birth-off'), device=str(device), partition_search=True,
                          partition_policy=PartitionSearchPolicy(birth_mode='off'))
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
            assert extended.partition_estimate.score <= current.partition_estimate.score
            variants = {}
            for mode, seeds in [('off', 3), ('single_cluster', 3), ('any_cluster', 3)]:
                label = f'{mode}-seeds{seeds}'
                variant = fit(path, args.outdir/(name+'-'+label), device=str(device), partition_search=True,
                              partition_policy=PartitionSearchPolicy(birth_mode=mode, seed_bank_size=seeds))
                assert variant.graph_sha256 == baseline.graph_sha256
                assert variant.selected_lambda == baseline.selected_lambda
                assert variant.search_status == baseline.search_status
                np.testing.assert_array_equal(variant.cluster_labels, baseline.cluster_labels)
                np.testing.assert_allclose(variant.raw_phi, baseline.raw_phi, atol=1e-9, rtol=0.)
                assert variant.partition_estimate.score <= current.partition_estimate.score
                variants[label] = dict(score=variant.partition_estimate.score,
                    seconds=variant.operation_metrics['elapsed_seconds'],
                    peak_allocated_bytes=variant.provenance['peak_allocated_bytes'])
            output = json.loads((args.outdir/(name+'-partition')/'run.json').read_bytes())
            assert output['schema'] == PARTITION_SCHEMA and len(output['table_sha256']) == 5
            for filename, expected in output['table_sha256'].items():
                assert hashlib.sha256((args.outdir/(name+'-partition')/filename).read_bytes()).hexdigest() == expected
            records.append(dict(case=name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                baseline_score=baseline.selection_score, partition_score=extended.partition_estimate.score,
                                generic_score=generic.partition_estimate.score,
                                family=extended.partition_estimate.provenance['candidate_family'],
                                previous_partition_score=current.partition_estimate.score,
                                independent_variants=variants,
                                baseline_seconds=baseline.operation_metrics['elapsed_seconds'],
                                partition_seconds=extended.operation_metrics['elapsed_seconds'],
                                generic_seconds=generic.operation_metrics['elapsed_seconds'],
                                partition_search_status=extended.partition_estimate.search_status,
                                peak_allocated_bytes=extended.provenance['peak_allocated_bytes']))
        # Exercise publication of a birth-derived result whose independent raw
        # reference is genuinely fused, not just a component-level replay.
        from clipp1d.io import read_tumor
        from clipp1d.model import compile_model
        from clipp1d.cuda.model import TensorModel
        from clipp1d.cuda.selection import fit_tensor_model
        from clipp1d.cuda_api import _export, _publish
        path = args.outdir/'birth-publication.tsv'
        write_case(path, [12, 13, 38, 39], [1]*4)
        data = read_tumor(path)
        host = compile_model(data)
        host = host.subset(np.argsort(host.mutation_ids, kind='stable'))
        model = TensorModel.from_host(host, device, compiled=True)
        d = fit_tensor_model(model, lambda_values=[1000.], partition_search=PartitionSearchPolicy())
        result = _export(d, source_provenance())
        destination = args.outdir/'birth-publication'
        destination.mkdir()
        _publish(result, data, destination, d.records)
        assert result.partition_estimate.provenance['proposal']['ancestry'][0]['operation'] == 'birth'
        assert result.partition_estimate.score < result.selection_score
        records.append(dict(case='birth-publication-restricted-penalty',
                            score=result.partition_estimate.score, raw_partition_score=result.selection_score,
                            scope='compiled birth ancestry/publication on certified raw reference; not default-path timing'))
        receipt = dict(status='passed', paired_complete_path_cases=len(fixtures),
                       restricted_penalty_birth_publications=1, records=records)
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
