"""Bound managed-storage seed stage; inference stays on PyTorch CUDA."""
from pathlib import Path
import os
import sys

from common import plan, read, sha, unit, write


def binding(root):
    p = plan(root)
    if p.get('seed_memory_backend') != 'cuda_managed':
        raise ValueError('Managed storage was not requested by this frozen plan')
    built = read(root/'native/BINDING.json')
    receipt = read(root/'receipts/NATIVE_PUBLISHED.json')
    if (sha(root/'native/BINDING.json') != receipt['binding_sha256'] or
            receipt['plan_sha256'] != sha(root/'payload/RUN_PLAN.json')):
        raise ValueError('Different native allocator binding')
    job = os.environ.get('LSB_JOBID')
    matches = [read(a) for a in (root/'receipts').glob('*/accepted.json') if read(a)['job_id'] == job]
    if len(matches) != 1 or matches[0]['memory_gb'] != 384 or matches[0]['cpu'] != 1:
        raise ValueError('Managed storage requires the exact 384-GB host reservation')
    return built


def main():
    import torch
    from clipp1d.cuda.managed_memory import workspace, native_stats
    from seeding import generate

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(sys.argv[1]).resolve()
    task = unit(root, sys.argv[2])
    output = Path(sys.argv[3]).resolve()
    allocator = binding(root)
    # Decimal GB is deliberately conservative relative to an LSF GiB limit.
    with workspace(allocator, 384_000_000_000) as state:
        generate(root, task, output)
        write(output/'MANAGED_MEMORY.json', dict(admission=state['admission'],
              logical_managed_allocation=native_stats(state['native']),
              allocation_is_not_resident_vram=True, cpu_numeric_fallback=False))


if __name__ == '__main__':
    main()
