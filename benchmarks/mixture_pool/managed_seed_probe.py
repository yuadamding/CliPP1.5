"""Fresh-process raw parity and real above-VRAM allocation qualification."""
from pathlib import Path
import csv
import math
import sys
import time

from common import plan, read, unit, write
from run_managed_seed import binding
from seeding import request


def compare(base, key):
    reference = base/'standard-baseline'
    actual = base/'outputs'/key/'baseline'
    a, b = read(reference/'run.json'), read(actual/'run.json')
    if (a['status'] != b['status'] or a['status'] != 'success' or
            a['graph_sha256'] != b['graph_sha256'] or a['search_status'] != b['search_status'] or
            a['provenance']['source_sha256'] != b['provenance']['source_sha256'] or
            a['provenance'].get('managed_memory_admission') is not None or
            b['provenance']['managed_memory_admission']['cpu_numeric_fallback'] is not False):
        raise ValueError('Managed storage changed the raw fit identity')
    fields = ('selected_lambda', 'raw_objective', 'selection_score')
    delta = {key: abs(a[key]-b[key]) for key in fields}
    if any(not math.isfinite(v) or v > 1e-6 for v in delta.values()):
        raise ValueError('Managed storage changed raw fit objectives')
    maximum = 0.
    exact = {'tumor_id','sample_id','mutation_id','status','cluster_label','designated_clonal',
             'multiplicity_call','raw_multiplicity_call','refitted_multiplicity_call','n_mutations'}
    for name in a['table_sha256']:
        def rows(folder):
            with (folder/name).open() as stream:
                return list(csv.DictReader(stream, delimiter='\t'))
        left, right = rows(reference), rows(actual)
        if len(left) != len(right):
            raise ValueError('Managed storage changed output population')
        for x, y in zip(left, right):
            if set(x) != set(y):
                raise ValueError('Managed storage changed output schema')
            for column in x:
                if x[column] == y[column]:
                    continue
                if column in exact:
                    raise ValueError('Managed storage changed labels or multiplicity')
                d = abs(float(x[column])-float(y[column]))
                if not math.isfinite(d) or d > 1e-9:
                    raise ValueError('Managed storage changed published estimates')
                maximum = max(maximum, d)
    write(base/'RAW_PARITY.json', dict(status='passed', objective_deltas=delta,
          maximum_table_numeric_delta=maximum, graph_identity_equal=True,
          labels_multiplicity_schema_status_equal=True, independent_fresh_processes=True))


def main():
    import torch
    from clipp1d.cuda.managed_memory import workspace, native_stats
    from clipp1d.cuda.kernels import StructuralCompileBank
    from clipp1d.cuda.refinement import PartitionSearchPolicy
    from clipp1d.cuda_api import fit

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root, base = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    mode = sys.argv[3]
    p = plan(root)
    task = unit(root, p['seed_stage_canary_key'])
    data = request(root, task)
    if mode == 'standard':
        fit(data['input_path'], base/'standard-baseline', device='cuda:0', max_major_cn=4,
            partition_search=True, partition_policy=PartitionSearchPolicy(**data['partition_policy']))
    elif mode == 'compare':
        compare(base, task['key'])
    elif mode == 'capacity':
        allocator = binding(root)
        physical = torch.cuda.get_device_properties(0).total_memory
        elements = (physical//8)*5//4
        began = time.monotonic()
        with workspace(allocator, 384_000_000_000) as state:
            values = torch.full((elements,), 2., device='cuda:0', dtype=torch.float64)
            if state['native'].clipp_is_managed(values.data_ptr()) != 1:
                raise ValueError('Above-VRAM tensor is not managed')
            kernel = StructuralCompileBank(lambda x: x*2.+1.)
            result = kernel(values)
            if not (float(result.min()) == 5. and float(result.max()) == 5. and
                    float(result.sum()) == 5.*elements):
                raise ValueError('Above-VRAM CUDA computation failed')
            torch.cuda.synchronize()
            statistics = native_stats(state['native'])
            if statistics['peak_bytes'] <= physical or statistics['release_error']:
                raise ValueError('Managed allocation did not exceed VRAM cleanly')
            write(base/'MANAGED_CAPACITY.json', dict(status='passed', physical_vram_bytes=physical,
                tensor_bytes=elements*8, logical_managed_allocation=statistics,
                elapsed_seconds=time.monotonic()-began, cuda_compiled=True,
                cpu_numeric_fallback=False, complete_large_fit_qualified=False))
    else:
        raise ValueError('Unknown managed qualification stage')


if __name__ == '__main__':
    main()
