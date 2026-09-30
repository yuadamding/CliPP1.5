"""No-clobber receipts for the separately identified soft-mixture experiment."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

GPU_MODELS = {'NVIDIA L40': 'NVIDIAL40', 'NVIDIA A40': 'NVIDIAA40'}


def qualified_gpu_model(qroot, qualification):
    """Bind follow-on jobs to the actual device that passed qualification."""
    allocation = [s for s in qualification['stages'] if s['stage'] == 'allocation']
    path = qroot/'output/allocation.log'
    if (len(allocation) != 1 or allocation[0]['returncode'] != 0 or
            sha(path) != allocation[0]['log_sha256']):
        raise ValueError('Unbound qualification hardware evidence')
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.startswith('{')]
    if len(records) != 1 or records[0].get('gpu') not in GPU_MODELS:
        raise ValueError('Unsupported or ambiguous qualification GPU')
    return GPU_MODELS[records[0]['gpu']]


def admitted_gpu_model(root):
    admission = read(root/'receipts/qualification-admitted.json')
    if admission['gpu_model'] not in GPU_MODELS.values():
        raise ValueError('Unsupported qualified GPU model')
    return admission['gpu_model']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name+'.writing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def command(argv, timeout=60):
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        return dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        return dict(code=None, error=repr(exc), stdout='', stderr='')


def identity(pid):
    proc = Path('/proc')/str(pid)
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], cmdline_sha256=sha(proc/'cmdline'),
                host=os.uname().nodename)


def plan(root):
    if root.resolve() != root or root.stat().st_uid != os.getuid() or os.getuid() != 307469:
        raise ValueError('Unexpected remote root or owner')
    p = read(root/'payload/RUN_PLAN.json')
    old = p['schema'] == 'clipp1d.experimental_shared_pool.v1' and p['max_submitted'] == 22
    full = (p['schema'] == 'clipp1d.experimental_full_cohorts.v1' and p['max_submitted'] == 15
            and {f: v['workers'] for f, v in p['pools'].items()} == {'a100': 10, 'h100': 4})
    managed = (p['schema'] == 'clipp1d.experimental_managed_seeds.v1' and
               p['max_submitted'] == 15 and p['host_memory_gb'] == 384 and
               p['seed_memory_backend'] == 'cuda_managed' and p['pools'] == {})
    if p['remote_root'] != str(root) or not (old or full or managed):
        raise ValueError('Unexpected study identity or admission cap')
    return p


def verify(root, *, full=False):
    payload = root/'payload'
    inventory = read(payload/'INVENTORY.json')
    for name, expected in inventory.items():
        if not full and name.startswith(('inputs/', 'manifests/', 'expected/')):
            continue
        path = payload/name
        if path.is_symlink() or not path.resolve().is_relative_to(payload) or sha(path) != expected:
            raise ValueError('Changed payload: '+name)


def unit(root, key):
    values = [u for u in read(root/'payload/UNITS.json') if u['key'] == key]
    if len(values) != 1:
        raise ValueError('Unknown or duplicate task')
    return values[0]


def validate_outputs(root, task, output_base=None):
    """Read back the complete, single-case experimental output and ancestry."""
    output_base = root/'outputs' if output_base is None else output_base
    payload = root/'payload'
    if task.get('requires_seed'):
        from seeding import prepared_manifest
        manifest_path = prepared_manifest(root, task, output_base/task['key'])
        manifest_sha256 = sha(manifest_path)
    else:
        manifest_path = payload/task['manifest']
        manifest_sha256 = task['manifest_sha256']
        if sha(manifest_path) != manifest_sha256:
            raise ValueError('Changed single-case manifest')
    manifest = read(manifest_path)
    expected_case, = manifest['cases']
    for kind in ('input', 'seed'):
        if sha(expected_case[kind+'_path']) != expected_case[kind+'_sha256']:
            raise ValueError('Changed case input')
    authority = {}
    inventory = read(payload/'INVENTORY.json')
    numerical = {name.removeprefix('source/'): digest for name, digest in inventory.items()
                 if name.startswith('source/src/') and name.endswith('.py')}
    if not numerical:
        raise ValueError('Missing frozen numerical inventory')
    with Path(expected_case['seed_path']).open() as stream:
        seed_rows = list(csv.DictReader(stream, delimiter='\t'))
    seed_ids = {r['mutation_id'] for r in seed_rows}
    if len(seed_rows) != len(seed_ids) or len(seed_ids) != task['retained_mutations']:
        raise ValueError('Invalid seed population')
    for mode in ('mixture', 'guarded'):
        folder = output_base/task['key']/mode
        binding, complete = read(folder/'BINDING.json'), read(folder/'COMPLETE.json')
        if (binding['policy'] != manifest['policy'] or
                binding['manifest_sha256'] != manifest_sha256 or
                not complete['source_unchanged'] or not complete['manifest_unchanged'] or complete['failures']):
            raise ValueError('Invalid completion binding')
        result, = complete['results']
        if result['case_id'] != task['case_id'] or Path(result['output']) != folder/'00000':
            raise ValueError('Wrong completed case identity')
        receipt_path = folder/'00000/EXPERIMENT.json'
        receipt = read(receipt_path)
        if receipt['inputs'] != expected_case or receipt['policy'] != manifest['policy']:
            raise ValueError('Output identity mismatch')
        helpers = ['run_mixture_experiment.py']
        if mode == 'guarded':
            helpers += ['mixture_structural_guard.py', 'apply_mixture_structural_guard.py']
        expected_source = dict(numerical)
        for name in helpers:
            expected_source['benchmarks/'+name] = inventory['source/benchmarks/'+name]
        if binding['source_inventory'] != expected_source:
            raise ValueError('Incomplete or different source inventory')
        for name, digest in expected_source.items():
            if sha(payload/'source'/name) != digest:
                raise ValueError('Different output source')
        required = {'mixture_mutation_clusters.tsv', 'mixture_cluster_centers.tsv'}
        if set(receipt['output_sha256']) != required or receipt['case_id'] != task['case_id']:
            raise ValueError('Incomplete output contract')
        for name, digest in receipt['output_sha256'].items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts or sha(receipt_path.parent/path) != digest:
                raise ValueError('Changed output table')
        with (receipt_path.parent/'mixture_mutation_clusters.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        if len(rows) != len(seed_ids) or {r['mutation_id'] for r in rows} != seed_ids:
            raise ValueError('Output lost or duplicated retained mutations')
        groups = {}
        for row in rows:
            phi = float(row['mixture_ccf'])
            label = int(row['cluster_label'])
            dosage = float(row['multiplicity'])
            if (not math.isfinite(phi) or not 0 <= phi <= 1 or not math.isfinite(dosage) or
                    dosage != int(dosage) or not 1 <= dosage <= 4):
                raise ValueError('Invalid output CCF or multiplicity')
            groups.setdefault(label, []).append((row['mutation_id'], phi))
        if set(groups) != set(range(len(groups))):
            raise ValueError('Noncanonical public labels')
        for members in groups.values():
            if len({phi for _, phi in members}) != 1:
                raise ValueError('One public cluster has different CCFs')
        clonal = min(groups, key=lambda k: (abs(groups[k][0][1]-1), min(mid for mid, _ in groups[k])))
        if clonal != 0:
            raise ValueError('Closest-to-one occupied cluster is not label zero')
        with (receipt_path.parent/'mixture_cluster_centers.tsv').open() as stream:
            centers = list(csv.DictReader(stream, delimiter='\t'))
        if len(centers) != len(groups) or {int(r['cluster_label']) for r in centers} != set(groups):
            raise ValueError('Center population differs from labels')
        for center in centers:
            members = groups[int(center['cluster_label'])]
            if float(center['mixture_ccf']) != members[0][1] or int(center['n_mutations']) != len(members):
                raise ValueError('Center table differs from mutation output')
        if mode == 'mixture':
            if binding['execution'] != 'compiled_cuda' or not receipt['device'].startswith('cuda'):
                raise ValueError('CUDA execution is required')
        else:
            if (binding['selector_execution'] != 'compiled_cuda' or
                    receipt['selector_execution'] != 'compiled_cuda' or
                    receipt['selection_policy'] != binding['selection_policy'] or
                    Path(receipt['parent_receipt']) != output_base/task['key']/'mixture/00000/EXPERIMENT.json' or
                    sha(receipt['parent_receipt']) != receipt['parent_receipt_sha256']):
                raise ValueError('Invalid selector execution or ancestry')
        authority[mode] = dict(binding_sha256=sha(folder/'BINDING.json'),
            complete_sha256=sha(folder/'COMPLETE.json'), receipt_sha256=sha(receipt_path),
            output_sha256=receipt['output_sha256'])
    return authority
