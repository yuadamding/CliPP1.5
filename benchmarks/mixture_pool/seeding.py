"""Generate a missing complete-graph seed without changing the mixture policy.

This stage uses the published CUDA fit and partition-search policy. It never
reads truth, loosens memory admission, substitutes another method, or overwrites
an existing seed. Prepared manifests are bound to the original frozen task.
"""
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

from common import read, sha, unit, write


def request(root, task):
    path = root/'payload'/task['manifest']
    if path.is_symlink() or sha(path) != task['manifest_sha256']:
        raise ValueError('Changed seed-stage request')
    data = read(path)
    if (data['schema'] != 'clipp1d.mixture_seed_request.v1' or
            set(data) != {'schema', 'case_id', 'input_path', 'input_sha256', 'policy',
                          'partition_policy', 'cuda_policy', 'source_sha256'} or
            data['case_id'] != task['case_id'] or
            sha(data['input_path']) != data['input_sha256']):
        raise ValueError('Invalid seed-stage input or scientific policy binding')
    return data


def check_seed(path, task):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    ids = [r['mutation_id'] for r in rows]
    if (len(ids) != task['retained_mutations'] or len(set(ids)) != len(ids) or
            hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest() != task['retained_ids_sha256']):
        raise ValueError('Seed does not cover the original retained population')
    centers = {}
    for r in rows:
        phi, mult = float(r['partition_ccf']), float(r['multiplicity_call'])
        label = int(r['cluster_label'])
        if (not math.isfinite(phi) or not 0 <= phi <= 1 or not math.isfinite(mult) or
                mult != int(mult) or not 1 <= mult <= 4):
            raise ValueError('Invalid seed estimates')
        if label in centers and centers[label] != phi:
            raise ValueError('Seed cluster has inconsistent centers')
        centers[label] = phi
    if set(centers) != set(range(len(centers))) or abs(centers[0]-1) != min(abs(c-1) for c in centers.values()):
        raise ValueError('Noncanonical seed cluster labels')
    return rows


def prepared_manifest(root, task, output):
    data = request(root, task)
    proof = read(output/'SEED_COMPLETE.json')
    manifest = output/'INFERENCE.json'
    if (proof['request_sha256'] != task['manifest_sha256'] or
            proof['input_sha256'] != data['input_sha256'] or
            proof['source_sha256'] != data['source_sha256'] or
            proof['manifest_sha256'] != sha(manifest)):
        raise ValueError('Changed generated seed authority')
    baseline = output/'baseline'
    if sha(baseline/'run.json') != proof['run_sha256']:
        raise ValueError('Changed baseline publication')
    run = read(baseline/'run.json')
    provenance = run['provenance']
    if proof.get('managed_memory_admission') != provenance.get('managed_memory_admission'):
        raise ValueError('Changed managed memory execution authority')
    if (run['schema'] != 'clipp1d.cuda.run.v5' or run['status'] != 'success' or
            provenance['input_sha256'] != data['input_sha256'] or
            provenance['source_sha256'] != data['source_sha256'] or
            provenance['backend'] != 'cuda' or provenance['dtype'] != 'float64' or
            provenance['compiled_inference'] is not True or provenance['cpu_numeric_fallback'] is not False or
            provenance['clonal_constraint'] is not False or provenance['max_major_cn'] != 4 or
            provenance['policy'] != data['cuda_policy'] or
            provenance['partition_search'] != data['partition_policy']):
        raise ValueError('Different complete-graph seed estimator')
    required = {'mutation_clusters.tsv', 'mutation_multiplicity.tsv', 'cluster_centers.tsv',
                'partition_mutation_clusters.tsv', 'partition_cluster_centers.tsv'}
    if set(run['table_sha256']) != required:
        raise ValueError('Incomplete baseline bundle')
    for name, digest in run['table_sha256'].items():
        if sha(baseline/name) != digest:
            raise ValueError('Changed baseline output')
    seed = output/'seed.tsv'
    if sha(seed) != proof['seed_sha256']:
        raise ValueError('Changed generated seed')
    rows = check_seed(seed, task)
    with (baseline/'partition_mutation_clusters.tsv').open() as stream:
        original = list(csv.DictReader(stream, delimiter='\t'))
    original = [r for r in original if r['status'] == 'retained']
    columns = ['mutation_id', 'cluster_label', 'partition_ccf', 'multiplicity_call']
    if [{k: r[k] for k in columns} for r in original] != rows:
        raise ValueError('Generated seed differs from the published partition')
    expected = dict(schema='clipp1d.mixture_experiment_inputs.v1', policy=data['policy'], cases=[dict(
        case_id=data['case_id'], input_path=data['input_path'], input_sha256=data['input_sha256'],
        seed_path=str(seed), seed_sha256=proof['seed_sha256'])])
    if read(manifest) != expected:
        raise ValueError('Generated inference manifest changes the frozen task')
    return manifest


def generate(root, task, output):
    data = request(root, task)
    from clipp1d.api import source_provenance
    from clipp1d.cuda.policy import CudaPolicy
    from clipp1d.cuda.refinement import PartitionSearchPolicy
    from clipp1d.cuda_api import fit

    if source_provenance()['source_sha256'] != data['source_sha256'] or asdict(CudaPolicy()) != data['cuda_policy']:
        raise ValueError('Different loaded seed-stage source or policy')
    fit(data['input_path'], output/'baseline', device='cuda:0', max_major_cn=4,
        partition_search=True, generic_partition_grouping=False,
        partition_policy=PartitionSearchPolicy(**data['partition_policy']))
    with (output/'baseline/partition_mutation_clusters.tsv').open() as stream:
        source = list(csv.DictReader(stream, delimiter='\t'))
    columns = ['mutation_id', 'cluster_label', 'partition_ccf', 'multiplicity_call']
    seed = output/'seed.tsv'
    with seed.open('x') as stream:
        writer = csv.DictWriter(stream, columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in source if r['status'] == 'retained')
    check_seed(seed, task)
    manifest = output/'INFERENCE.json'
    write(manifest, dict(schema='clipp1d.mixture_experiment_inputs.v1', policy=data['policy'], cases=[dict(
        case_id=data['case_id'], input_path=data['input_path'], input_sha256=data['input_sha256'],
        seed_path=str(seed), seed_sha256=sha(seed))]))
    managed = read(output/'baseline/run.json')['provenance'].get('managed_memory_admission')
    write(output/'SEED_COMPLETE.json', dict(request_sha256=task['manifest_sha256'],
          source_sha256=data['source_sha256'], input_sha256=data['input_sha256'],
          seed_sha256=sha(seed), run_sha256=sha(output/'baseline/run.json'), manifest_sha256=sha(manifest),
          truth_used=False, cpu_numeric_fallback=False, managed_memory_admission=managed))
    prepared_manifest(root, task, output)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(sys.argv[1]).resolve()
    generate(root, unit(root, sys.argv[2]), Path(sys.argv[3]).resolve())
