"""Paired B/P/D cost: fresh processes, isolated cold caches, three warm repeats."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_perturbation import (  # noqa: E402
    _publish_arm, additional_search, deterministic_starts, penalty_neighborhood,
    pilot_proposals, prepare_model, sha, synchronized_time, write_json,
)
from clipp1d.api import source_provenance  # noqa: E402
from clipp1d.cuda.refinement import PartitionSearchPolicy  # noqa: E402
from clipp1d.cuda.selection import fit_tensor_model  # noqa: E402
from clipp1d.cuda_api import require_cuda  # noqa: E402


def measure_once(input_file, tumor_id, outdir, arm, *, device='cuda:0'):
    device = require_cuda(device)
    torch.set_num_threads(1)
    outdir.mkdir()
    # Initialize the CUDA context before touching allocator statistics. Keep
    # the reset before model upload so its allocations remain in the peak.
    begin = synchronized_time(device)
    torch.cuda.reset_peak_memory_stats(device)
    data, model = prepare_model(input_file, device)
    source = source_provenance()
    source.update(input_sha256=data.input_sha256, max_major_cn=4, backend='cuda')
    with torch.no_grad():
        baseline = fit_tensor_model(model, partition_search=PartitionSearchPolicy())
        _publish_arm(baseline, source, data, outdir/'B')
        if arm in ('P', 'D'):
            if arm == 'P':
                starts, pilots = pilot_proposals(model, baseline.pilot.phi, tumor_id, outdir/'pilots')
                coverage = pilots['status']
            else:
                starts, coverage = deterministic_starts(model), 'complete'
            penalties = penalty_neighborhood(baseline.records, float(baseline.lambda_value))
            extra = additional_search(baseline, starts, penalties, outdir/'search', arm, pilot_coverage=coverage)
            _publish_arm(replace(baseline, partition_estimate=extra), source, data, outdir/arm)
        record = dict(arm=arm, synchronized_fit_seconds=synchronized_time(device)-begin,
            peak_device_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_device_reserved_bytes=torch.cuda.max_memory_reserved(device),
            input_sha256=sha(input_file), source=source,
            gpu=str(torch.cuda.get_device_properties(device)))
    write_json(outdir/'MEASUREMENT.json', record)


def measure_case(input_file, tumor_id, outdir):
    # This coordinator never initializes CUDA. Each sequential child owns the
    # GPU alone and exits. Unique per-arm caches prevent free compilation in P.
    outdir.mkdir(parents=True, exist_ok=False)
    records = []
    for arm in ('B', 'P', 'D'):
        (outdir/f'cache-{arm}').mkdir()
    for repeat in range(4):
        arms = ['B', 'P', 'D']
        offset = repeat % 3
        arms = arms[offset:]+arms[:offset]
        for arm in arms:
            target = outdir/f'{repeat}-{arm}'
            cache = outdir/f'cache-{arm}'
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
                       TRITON_CACHE_DIR=str(cache/'triton'), CUDA_CACHE_PATH=str(cache/'cuda'))
            argv = [sys.executable, '-B', __file__, '--input-file', str(input_file),
                    '--tumor-id', tumor_id, '--outdir', str(target), '--once', arm]
            begin = perf_counter()
            subprocess.run(argv, env=env, check=True)
            elapsed = perf_counter()-begin
            record = json.loads((target/'MEASUREMENT.json').read_text())
            record.update(repeat=repeat, temperature='cold' if repeat == 0 else 'warm',
                          order=arms, seconds=elapsed)
            records.append(record)
            print(arm, repeat, elapsed, flush=True)
    write_json(outdir/'TIMING.json', dict(input_sha256=sha(input_file), records=records,
        scope='full process wall time including startup, input preparation, B, attempted auxiliary work, qualification and publication',
        warm_scope='same per-arm on-disk compiler caches; fresh process, tensors, graph and fit each repetition',
        cold_scope='first process with absent per-fixture/per-arm Inductor, Triton and CUDA cache paths'))


def aggregate(manifest_path, results, outdir):
    manifest = json.loads(manifest_path.read_text())
    ratios, sources = [], set()
    for case in manifest['cases']:
        record = json.loads((results/case['case_id']/'TIMING.json').read_text())
        assert record['input_sha256'] == case['input_sha256']
        measurements = {(row['repeat'], row['arm']): row for row in record['records']}
        assert len(measurements) == 12
        assert {(r, a) for r in range(4) for a in ('B', 'P', 'D')} == set(measurements)
        for repeat in range(4):
            base = measurements[repeat, 'B']
            sources.add(base['source']['source_sha256'])
            for arm in ('P', 'D'):
                row = measurements[repeat, arm]
                assert row['gpu'] == base['gpu']
                assert row['source']['source_sha256'] == base['source']['source_sha256']
                ratios.append(dict(case_id=case['case_id'], n=case['n'], repeat=repeat, arm=arm,
                    temperature='cold' if repeat == 0 else 'warm',
                    runtime=row['seconds']/base['seconds'],
                    allocated_memory=row['peak_device_allocated_bytes']/base['peak_device_allocated_bytes'],
                    reserved_memory=row['peak_device_reserved_bytes']/base['peak_device_reserved_bytes']))
    def summary(selected):
        return dict(median_total_runtime_ratio=float(np.median([r['runtime'] for r in selected])),
                    p95_total_runtime_ratio=float(np.quantile([r['runtime'] for r in selected], .95)),
                    maximum_peak_memory_ratio=max(r['allocated_memory'] for r in selected))
    primary = [r for r in ratios if r['arm'] == 'P']
    assert len(sources) == 1, 'Mixed numerical source identities in performance results'
    output = dict(summary(primary), comparisons=ratios,
        source_sha256=next(iter(sources)),
        per_arm={arm: {temperature: summary([r for r in ratios if r['arm'] == arm and
                    (temperature == 'all' or r['temperature'] == temperature)])
            for temperature in ('all', 'cold', 'warm')} for arm in ('P', 'D')},
        gate_scope='all 36 prespecified fixture/repetition pairs; peak memory uses allocated CUDA bytes',
        manifest_sha256=sha(manifest_path))
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir/'PERFORMANCE.json', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path)
    parser.add_argument('--tumor-id')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--once', choices=['B', 'P', 'D'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    if args.manifest:
        aggregate(args.manifest, args.results, args.outdir)
    elif args.once:
        measure_once(args.input_file, args.tumor_id, args.outdir, args.once)
    else:
        measure_case(args.input_file, args.tumor_id, args.outdir)


if __name__ == '__main__':
    main()
