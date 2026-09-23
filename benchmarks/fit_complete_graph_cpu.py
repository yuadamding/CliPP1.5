"""Explicit CPU benchmark adapter for the current complete-graph tensor pipeline.

This executes the same float64 likelihood, pilots, graph, scalar backtracking,
planned search, refit, score and admission gates with eager PyTorch CPU kernels.
It is never a fallback from the public CUDA entry point. CPU execution and the
adapter hash remain explicit in the shared run-receipt format.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

from clipp1d.api import source_provenance
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.selection import fit_tensor_model
from clipp1d.cuda_api import (
    _export, _json, _model_hash, _prepared_result, _publish, _validate_result,
)
from clipp1d.io import read_tumor
from clipp1d.model import compile_model
from clipp1d.policy import Policy


def available_memory():
    """Host/cgroup headroom; admission is conservative, not an allocation promise."""
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024
    maximum, current = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if maximum.is_file() and current.is_file() and maximum.read_text().strip() != 'max':
        available = min(available, max(0, int(maximum.read_text())-int(current.read_text())))
    return available


def workspace_bytes(nodes):
    return 32 * 8 * nodes**2 + 64 * 4096 * 8 * 4


@torch.no_grad()
def fit(input_file, outdir=None, *, max_major_cn=4, memory_budget_bytes, expected_source_sha256):
    """Explicit one-thread CPU fit, with caller-owned memory/concurrency admission."""
    if (isinstance(memory_budget_bytes, bool) or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0):
        raise ValueError('A positive per-worker CPU memory budget is required')
    if torch.get_num_threads() != 1:
        raise ValueError('CPU benchmark workers must set torch intra-op threads to one')
    start = perf_counter()
    destination = None if outdir is None else Path(outdir)
    if destination is not None:
        if destination.is_symlink():
            raise FileExistsError('Output directory must not be a symlink')
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError('Output directory must be empty; preserve prior evidence')
    source = source_provenance()
    source.update(backend='cpu', requested_device='cpu', device='cpu', cpu_numeric_fallback=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  cpu_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpu_threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  precision_scope='Explicit PyTorch CPU float64 complete-graph evaluation; not CUDA qualification',
                  compiled_kernel_policy='eager PyTorch CPU; explicitly requested, no CUDA fallback')
    try:
        if source['source_sha256'] != expected_source_sha256:
            raise ValueError('CPU numerical package differs from the frozen source')
        input_policy, policy = Policy(max_major_cn=max_major_cn), CudaPolicy()
        data = read_tumor(input_file, input_policy)
        canonical = compile_model(data, input_policy)
        canonical = canonical.subset(np.argsort(np.asarray(canonical.mutation_ids), kind='stable'))
        estimated = workspace_bytes(len(canonical))
        free = available_memory()
        if estimated > min(memory_budget_bytes, int(policy.memory_fraction * free)):
            raise MemoryError(f'CPU complete graph needs approximately {estimated} bytes; '
                              f'worker budget {memory_budget_bytes}, available {free}')
        source.update(input_sha256=data.input_sha256, max_major_cn=max_major_cn,
                      policy=asdict(policy), clonal_constraint=False,
                      clonal_label_rule='nearest_to_one_l2_v1', torch_version=torch.__version__,
                      estimated_workspace_bytes=estimated, memory_budget_bytes=memory_budget_bytes,
                      host_available_bytes_at_start=free)
        source['model_sha256'] = _model_hash(canonical.mutation_ids, canonical.eps,
            {key: getattr(canonical, key) for key in ('alt', 'ref', 'lower', 'upper', 'slope', 'log_prior', 'valid')})
        phases = dict(input_preparation_seconds=perf_counter()-start)
        began = perf_counter()
        model = TensorModel.from_host(canonical, 'cpu', compiled=False)
        assert model.device.type == 'cpu' and not model.kernels.compiled
        phases['device_upload_and_compile_seconds'] = perf_counter()-began
        fitted = fit_tensor_model(model, policy)
        for key in ('pilot_seconds', 'graph_build_seconds', 'path_seconds', 'refit_seconds', 'stage_integrity_seconds'):
            phases[key] = fitted.timings[key]
        source.update(numerical_stages=fitted.timings, phase_seconds=phases,
                      process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                      compilation_timing_scope='No compilation; host-to-CPU tensor construction recorded separately')
        result = _export(fitted, source, wall_started=start)
        assert result.provenance['execution_scope'] == 'cpu_numerical_reference'
        assert result.provenance['compiled_inference'] is False
        if destination is not None:
            return _publish(result, data, destination, fitted.records, wall_started=start)
        began = perf_counter()
        _validate_result(result, data, fitted.records)
        result = _prepared_result(result, began, start, tables_written=False)
        return replace(result, operation_metrics=dict(elapsed_seconds=perf_counter()-start,
                       completed_utc=datetime.now(timezone.utc).isoformat(),
                       scope='Complete in-memory CPU evaluation; no durable output requested'))
    except Exception as error:
        if destination is not None and not (destination/'run.json').exists():
            _json(destination/'run.json',dict(schema='clipp1d.cpu.failure.v1', status='failure',
                  search_status='not_completed', provenance=source,
                  error_type=type(error).__name__, error=str(error)))
        raise
