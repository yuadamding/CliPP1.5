"""Source-bound QP attribution, shared by daaf50a and the current checkout.

Run each checkout in a separate fresh process with this identical script. Latency
samples have no profiler or stage hooks. Separate profiler calls measure kernel
dispatches and instrumented substage attribution; their wall times are never
mixed into the latency distribution. Numeric execution requires CUDA float64.
"""

import argparse
import ast
from bisect import bisect_right
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import sys
import textwrap
from time import perf_counter
import traceback

import numpy as np
from scipy.special import logsumexp
import torch

from clipp1d.api import source_provenance
from clipp1d.cuda import qp
from clipp1d.cuda.kernels import Kernels, differences
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda_api import require_cuda


SCHEMA = "clipp1d.cuda.qp_attribution.v2"
STAGES = ("admm_update", "equality_proposal", "dual_flow_repair", "certificate")
PREFIX = "clipp1d.qp.stage."
ARRAYS = ("h", "target", "lower", "upper", "caps", "start")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def literal_caps(pilot, penalty):
    """Host fixture preparation; the same literal matrix feeds both checkouts."""
    gaps = np.diff(np.sort(pilot))
    gaps = gaps[gaps > 0]
    floor = max(1e-8, .1 * np.median(gaps)) if gaps.size else 1e-8
    raw = 1 / np.maximum(np.abs(pilot[:, None] - pilot[None, :]), floor)
    np.fill_diagonal(raw, 0.)
    return raw / (raw.sum() / (len(pilot) * (len(pilot) - 1))) * penalty


def fixture(name):
    """Deterministic literal QPs; no fitted pilot or source-dependent solver."""
    if name in ("resource64", "resource256"):
        n = int(name.removeprefix("resource"))
        i = np.arange(n, dtype=np.float64)
        target = .15 + .35 * (i % 3) + .003 * np.sin(i)
        h = 80 + 40 * (i % 7)
        lower, upper = np.full(n, 1e-6), np.ones(n)
        start, caps = target.copy(), literal_caps(target, .002)
        recipe = "qualify_cuda resource QP recipe; NumPy float64 sin and literal host matrix shared between sources"
        extra = {}
    elif name == "mixed64":
        n, eps = 64, 1e-6
        i = np.arange(n)
        alt = np.array([16., 28., 37., 19., 40., 54.])[i % 6]
        depth = 100. + 3 * (i % 7)
        ref = depth - alt
        counts = np.array([1, 2, 3, 4, 2, 3])[i % 6]
        scale = np.array([.4, .25, .15, .12, .35, .42])[i % 6]
        valid = np.arange(1, 5) <= counts[:, None]
        slope = scale[:, None] * np.where(valid, np.arange(1, 5), 0)
        prior = np.where(valid, -np.log(counts[:, None]), -np.inf)
        lower = np.full(n, eps)
        upper = np.minimum(1., (1 - eps) / (scale * counts))
        start = np.minimum(upper, np.maximum(lower, np.linspace(.2, .85, n)))
        mass = slope * start[:, None]
        p = np.clip(mass, eps, 1 - eps)
        joint = alt[:, None] * np.log(p) + ref[:, None] * np.log1p(-p) + prior
        post = np.exp(joint - logsumexp(joint, axis=-1, keepdims=True))
        moving = np.where((mass > eps) & (mass < 1 - eps), slope, 0.)
        gradient = -(post * moving * (alt[:, None] / p - ref[:, None] / (1 - p))).sum(-1)
        h = np.maximum(1., (post * moving ** 2 *
                            (alt[:, None] / p ** 2 + ref[:, None] / (1 - p) ** 2)).sum(-1))
        target = start - gradient / h
        caps = literal_caps(start, .05)
        recipe = "mixed6 support/slopes extended to64 heterogeneous depths; literal interior probe surrogate, not a qualified pilot or full fit"
        extra = dict(alt=alt.tolist(), ref=ref.tolist(), support_counts=counts.tolist(),
                     slopes=slope.tolist(), probe=start.tolist())
    else:
        raise ValueError(f"Unknown QP fixture: {name}")
    return dict(name=name, recipe=recipe, derivation=extra,
                arrays={key: np.asarray(value, dtype=np.float64) for key, value in
                        zip(ARRAYS, (h, target, lower, upper, caps, start))})


def fixture_identity(problem):
    values, combined = {}, hashlib.sha256()
    for name in ARRAYS:
        value = np.ascontiguousarray(problem["arrays"][name])
        descriptor = dict(dtype=value.dtype.str, shape=list(value.shape),
                          sha256=hashlib.sha256(value.tobytes()).hexdigest())
        combined.update(json.dumps([name, descriptor], sort_keys=True).encode())
        values[name] = descriptor
    return dict(name=problem["name"], recipe=problem["recipe"], arrays=values,
                input_sha256=combined.hexdigest())


def upload(problem, device):
    return {key: torch.tensor(value.copy(), dtype=torch.float64, device=device)
            for key, value in problem["arrays"].items()}


def solve(solver, arrays, kernels):
    return solver(*(arrays[key] for key in ARRAYS[:-1]), kernels,
                  CudaPolicy(), start=arrays["start"])


def kernel_counts(kernels):
    return {name: row['calls'] for name, row in kernels.compilation_diagnostics().items()}


def counter_delta(after, before):
    return {key: value - before.get(key, 0) for key, value in after.items()}


def qualify(result, arrays, snapshots, reference_kernels):
    """Fresh original-QP certificate, outside every measured solve interval."""
    for name in ARRAYS:
        if not torch.equal(arrays[name], snapshots[name]):
            raise AssertionError(f"QP input was modified: {name}")
    if any(value.dtype != torch.float64 or value.device != arrays["h"].device
           for value in (result.x, result.dual)):
        raise AssertionError("QP result left the input float64 device")
    stats = reference_kernels.gap_kkt(result.x, result.dual,
                                     *(arrays[key] for key in ARRAYS[:-1]))
    policy = CudaPolicy()
    if not (result.qualified and bool(torch.isfinite(stats).all())
            and bool(stats[0] <= policy.inner_atol + policy.inner_rtol * stats[1])
            and bool(stats[2] <= policy.inner_kkt_tol)):
        raise AssertionError(f"Original QP certificate failed: {stats.detach().cpu().tolist()}")
    h, target, lower, upper, caps = (arrays[key] for key in ARRAYS[:-1])
    reference = target.clamp(lower, upper)
    delta = result.x - reference
    objective = (.5 * h * delta.square() + h * (reference - target) * delta).sum()
    objective += .5 * (caps * (differences(result.x).abs() - differences(reference).abs())).sum()
    if not bool(torch.isfinite(objective)):
        raise AssertionError("Original shifted QP objective is nonfinite")
    return dict(qualified=True, gap=float(stats[0]), gap_scale=float(stats[1]), kkt=float(stats[2]),
                allowed_gap=float(policy.inner_atol + policy.inner_rtol * stats[1]),
                objective=float(objective), admm_iterations=int(result.iterations),
                polish_iterations=int(result.polish_iterations), x=result.x.detach().cpu().tolist(),
                minimum_curvature=float(h.min()),
                certificate_scope="independently recomputed eager kernel on original device/problem; no inherited certificate")


class Stages:
    """Disjoint benchmark ranges; no event/synchronize timer in any iteration."""

    def __init__(self):
        self.active = None
        self.counts = Counter()

    @contextmanager
    def stage(self, name, operation):
        if name not in STAGES:
            raise ValueError("Unknown attribution stage")
        self.counts[operation] += 1
        if self.active is not None:
            raise RuntimeError("Overlapping QP attribution ranges would double count work")
        self.active = name
        try:
            with torch.profiler.record_function(PREFIX + name):
                yield
        finally:
            self.active = None

    def wrap(self, function, stage, name):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with self.stage(stage, name):
                return function(*args, **kwargs)
        return wrapped


def instrument_admm(solver, stages):
    """Wrap original AST statement ranges, preserving every numeric statement.

    This is benchmark instrumentation only. Unknown loop/checkpoint structure is
    rejected rather than silently measuring a partial ADMM update.
    """
    source = textwrap.dedent(inspect.getsource(solver))
    tree = ast.parse(source)
    function = tree.body[0]
    loops = [node for node in function.body if isinstance(node, ast.For)
             and "inner_max_iterations" in ast.unparse(node.iter)]
    if len(loops) != 1:
        raise ValueError("Unknown QP ADMM loop; attribution requires a reviewed boundary")
    loop = loops[0]
    checkpoints = [index for index, node in enumerate(loop.body)
                   if isinstance(node, ast.If) and "check_every" in ast.unparse(node.test)]
    if len(checkpoints) != 2 or checkpoints[0] == 0:
        raise ValueError("Unknown QP checkpoint boundaries; refusing partial attribution")
    first, balancing = checkpoints
    def wrap_statements(body, operation):
        call = ast.Call(ast.Attribute(ast.Name("_qp_attribution_stages", ast.Load()), "stage", ast.Load()),
                        [ast.Constant("admm_update"), ast.Constant(operation)], [])
        return ast.With([ast.withitem(call)], body)
    balancing_node = loop.body[balancing]
    if balancing_node.orelse or balancing != len(loop.body) - 1:
        raise ValueError("Unknown residual-balancing tail; refusing partial attribution")
    balancing_node.body = [wrap_statements(balancing_node.body, "residual_balance")]
    loop.body = ([wrap_statements(loop.body[:first], "admm_update")]
                 + loop.body[first:balancing] + [balancing_node])
    function.decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(solver.__globals__, _qp_attribution_stages=stages)
    exec(compile(tree, "<benchmark-only QP attribution>", "exec"), namespace)
    return namespace[function.name]


@contextmanager
def attribution(kernels):
    stages, restored = Stages(), []
    def hook(owner, name, stage):
        original = getattr(owner, name)
        setattr(owner, name, stages.wrap(original, stage, name))
        restored.append((owner, name, original))
    try:
        hook(qp, "polish_quadratic", "equality_proposal")
        hook(qp, "quadratic_value", "certificate")
        hook(kernels, "gap_kkt", "certificate")
        if hasattr(qp, "prepare_flow_polish") and hasattr(qp, "flow_polish_step"):
            hook(qp, "prepare_flow_polish", "equality_proposal")
            hook(qp, "flow_polish_step", "dual_flow_repair")
            stages.flow_boundary = "prepared proposal geometry; dual-dependent flow step only"
        else:
            hook(qp, "_polish_dual", "dual_flow_repair")
            stages.flow_boundary = "legacy step includes rebuilding fixed proposal geometry each call"
        yield instrument_admm(qp.solve_qp, stages), stages
    finally:
        for owner, name, original in reversed(restored):
            setattr(owner, name, original)


def profile_summary(profile, stage_counts=None):
    """Count actual Kineto CUDA work, excluding GPU annotation range mirrors.

    PyTorch 2.9 attaches GPU user annotations to CPU ``kernels`` lists; therefore
    FunctionEvent.device_time_total also includes their span/launch gaps. Neither
    that recursive total nor CUDA annotation spans are device-work measurements.
    Real device work is assigned through its linked CPU correlation and the
    enclosing disjoint CPU stage. Duplicate CPU IDs must agree on the identical
    stage instance (or all be outside stages); no arbitrary owner is selected.
    Uncorrelated work stays explicitly unattributed.
    """
    raw = list(profile.profiler.kineto_results.events())
    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    stages = {name: dict(calls=0, host_range_seconds=0., device_kernel_seconds=0.) for name in STAGES}
    cpu_ranges, owners = [], {}
    for event in raw:
        if event.device_type() != cpu:
            continue
        if event.linked_correlation_id() == 0:
            key = event.correlation_id()
            if key > 0:
                owners.setdefault(key, []).append(event)
        if event.name().startswith(PREFIX):
            name = event.name().removeprefix(PREFIX)
            if name not in stages or event.start_thread_id() != event.end_thread_id():
                raise AssertionError("Unknown or asynchronous CPU attribution stage")
            start, end = event.start_ns(), event.end_ns()
            if end < start:
                raise AssertionError("Negative CPU attribution interval")
            cpu_ranges.append(event)
            stages[name]['calls'] += 1
            stages[name]['host_range_seconds'] += (end - start) / 1e9
    ranges_by_thread = {}
    for stage in cpu_ranges:
        ranges_by_thread.setdefault(stage.start_thread_id(), []).append(stage)
    stage_starts = {}
    for thread, ranges in ranges_by_thread.items():
        ranges.sort(key=lambda event: event.start_ns())
        if any(left.end_ns() > right.start_ns() for left, right in zip(ranges, ranges[1:])):
            raise AssertionError("Overlapping CPU attribution ranges")
        stage_starts[thread] = [event.start_ns() for event in ranges]
    def enclosing_stage(owner):
        thread = owner.start_thread_id()
        index = bisect_right(stage_starts.get(thread, []), owner.start_ns()) - 1
        if index >= 0 and thread == owner.end_thread_id():
            stage = ranges_by_thread[thread][index]
            if owner.end_ns() <= stage.end_ns():
                return thread, index
        return None

    device_events, kernels, discarded, correlated, uncorrelated = 0, 0, 0, 0, 0
    agreed_duplicate_owners = 0
    device_seconds = 0.
    for event in raw:
        if event.device_type() != cuda:
            continue
        if event.is_user_annotation() or event.name().startswith(PREFIX):
            discarded += 1
            continue
        seconds = (event.end_ns() - event.start_ns()) / 1e9
        if seconds < 0:
            raise AssertionError("Negative CUDA work duration")
        device_events += 1
        kernels += int(not event.name().lower().startswith(('memcpy', 'memset')))
        device_seconds += seconds
        candidates = owners.get(event.linked_correlation_id(), [])
        if not candidates:
            uncorrelated += 1
            continue
        correlated += 1
        ownership = {enclosing_stage(owner) for owner in candidates}
        if len(ownership) != 1:
            detail = dict(cuda_event=event.name(), linked_correlation_id=event.linked_correlation_id(),
                          candidate_count=len(candidates), candidates=[dict(
                              name=owner.name(), thread=owner.start_thread_id(),
                              start_ns=owner.start_ns(), end_ns=owner.end_ns(),
                              stage=enclosing_stage(owner)) for owner in candidates[:8]])
            raise AssertionError("Ambiguous CUDA work stage ownership: " + json.dumps(detail, sort_keys=True))
        agreed_duplicate_owners += int(len(candidates) > 1)
        owner_stage = ownership.pop()
        if owner_stage is not None:
            thread, index = owner_stage
            name = ranges_by_thread[thread][index].name().removeprefix(PREFIX)
            stages[name]['device_kernel_seconds'] += seconds
    attributed = sum(value['device_kernel_seconds'] for value in stages.values())
    if attributed > device_seconds + max(1e-8, device_seconds * 1e-6):
        raise AssertionError("Overlapping or inconsistent profiler device attribution")
    if stage_counts is not None:
        expected_calls = sum(stage_counts.values())
        if sum(row['calls'] for row in stages.values()) != expected_calls:
            raise AssertionError("CPU annotation coverage differs from instrumented operation counts")
    averages = {event.key: event.count for event in profile.key_averages()}
    return dict(stages=stages, stage_operation_counts=dict(stage_counts or {}),
                host_stage_seconds={name + '_seconds': row['host_range_seconds'] for name, row in stages.items()},
                device_stage_seconds={name + '_seconds': row['device_kernel_seconds'] for name, row in stages.items()},
                kernel_dispatches=kernels, device_events=device_events,
                discarded_cuda_user_annotations=discarded,
                duplicate_cpu_correlation_ids=sum(len(rows) > 1 for rows in owners.values()),
                cuda_work_with_unanimous_duplicate_owners=agreed_duplicate_owners,
                correlated_cuda_work_events=correlated, uncorrelated_cuda_work_events=uncorrelated,
                device_kernel_and_copy_seconds=device_seconds,
                unattributed_device_seconds=max(0., device_seconds - attributed),
                host_scalar_read_proxies={name: averages.get(name, 0) for name in
                                          ("aten::item", "aten::_local_scalar_dense", "aten::is_nonzero")},
                device_accounting='raw Kineto CUDA work only; user annotations excluded; linked CPU correlation assigns disjoint stage ownership',
                scope="disjoint CPU annotation wall spans and summed real CUDA kernel/copy durations; neither is uninstrumented latency")


def measured_profile(call, device, *, trace=None, stage_counts=None):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    started = perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        result = call()
        synchronize(device)
    seconds = perf_counter() - started
    detail = profile_summary(prof, stage_counts)
    host_ranges = sum(row['host_range_seconds'] for row in detail['stages'].values())
    if host_ranges > seconds + max(1e-8, seconds * 1e-6):
        raise AssertionError('Stage host intervals overlap or exceed profiled wall scope')
    detail.update(instrumented_wall_seconds=seconds,
                  unattributed_host_wall_seconds=max(0., seconds - host_ranges),
                  unattributed_scope='input checks, initialization, host admission/control, residual operations outside named ranges, and profiler/boundary synchronization overhead',
                  numerical_execution="CUDA float64" if device.type == "cuda" else "CPU reference test only")
    if device.type == "cuda" and detail['kernel_dispatches'] == 0:
        raise RuntimeError("CUDA profiler returned no kernel events; dispatch attribution is unqualified")
    if trace is not None:
        prof.export_chrome_trace(str(trace))
        detail.update(trace=str(trace), trace_sha256=digest(trace))
    return result, detail


def qualify_stage_coverage(detail, fitted):
    """Reconcile actual operations; an already certified dual can need no repair.

    This validates accounting only. CPU callers remain CPU reference tests;
    production admission separately requires real CUDA dispatch and an original
    QP certificate before invoking this helper.
    """
    operations = {
        'admm_update': ('admm_update', 'residual_balance'),
        'equality_proposal': ('polish_quadratic', 'prepare_flow_polish'),
        'dual_flow_repair': ('flow_polish_step', '_polish_dual'),
        'certificate': ('gap_kkt', 'quadratic_value'),
    }
    counts = detail['stage_operation_counts']
    known = {name for names in operations.values() for name in names}
    if set(counts) - known or any(type(value) is not int or value < 0 for value in counts.values()):
        raise AssertionError('Invalid attribution stage operation counts')
    for stage, names in operations.items():
        if detail['stages'][stage]['calls'] != sum(counts.get(name, 0) for name in names):
            raise AssertionError('Attribution stage operation coverage differs')
    if counts.get('admm_update', 0) != fitted.iterations:
        raise AssertionError('Attribution did not cover every ADMM iteration')
    flow_steps = sum(counts.get(name, 0) for name in operations['dual_flow_repair'])
    if flow_steps != fitted.polish_iterations:
        raise AssertionError('Attribution did not cover every flow repair step')
    certificate = detail['certificate']
    if not (fitted.qualified and certificate['qualified']
            and certificate['admm_iterations'] == fitted.iterations
            and certificate['polish_iterations'] == fitted.polish_iterations
            and counts.get('gap_kkt', 0) > 0):
        raise AssertionError('Missing original QP certificate coverage')
    if any(detail['stages'][stage]['calls'] == 0 for stage in STAGES if stage != 'dual_flow_repair'):
        raise AssertionError('The fixed QP fixture did not exercise required non-repair stages')
    if flow_steps == 0:
        row = detail['stages']['dual_flow_repair']
        if row['host_range_seconds'] != 0 or row['device_kernel_seconds'] != 0:
            raise AssertionError('Zero flow repair has nonzero attributed work')
    return dict(qualified=True, flow_repair_steps=flow_steps,
                zero_flow_repair=flow_steps == 0,
                scope='Actual operation/range counts and independent certificate; no positive repair count is required')


def compare_receipts(baseline, current):
    """Reject incomparable or unresolved measurements before reporting ratios."""
    for receipt in (baseline, current):
        if (receipt.get('schema') != SCHEMA or receipt.get('status') != 'passed'
                or receipt.get('numerical_execution') != 'CUDA float64'):
            raise ValueError("Comparisons require completed CUDA-only receipts")
    for key in ('script_sha256', 'policy', 'controls', 'environment'):
        if baseline[key] != current[key]:
            raise ValueError(f"Unmatched QP comparison {key}")
    left = {case['fixture']['name']: case for case in baseline['fixtures']}
    right = {case['fixture']['name']: case for case in current['fixtures']}
    planned = baseline['controls']['fixtures']
    if (not planned or len(set(planned)) != len(planned) or set(left) != set(right)
            or set(left) != set(planned) or len(baseline['fixtures']) != len(left)
            or len(current['fixtures']) != len(right)):
        raise ValueError("Unmatched QP fixture coverage")
    comparisons = []
    for name in sorted(left):
        a, b = left[name], right[name]
        if a['fixture']['input_sha256'] != b['fixture']['input_sha256']:
            raise ValueError("Literal QP input mismatch")
        for case in (a, b):
            if len(case['latency_samples']) != baseline['controls']['repeats']:
                raise ValueError("Incomplete uninstrumented latency samples")
            if len(case['warmups']) != baseline['controls']['warmups']:
                raise ValueError("Incomplete QP warmup coverage")
            rows = case['latency_samples'] + case['warmups']
            rows += [case['dispatch_profile'], case['attribution']]
            policy = baseline['policy']
            for row in rows:
                c = row['certificate']
                finite = all(np.isfinite(c[key]) for key in
                             ('gap', 'gap_scale', 'kkt', 'objective', 'minimum_curvature'))
                if not (c['qualified'] and finite and c['gap'] >= 0 and c['gap_scale'] >= 0
                        and c['minimum_curvature'] > 0
                        and c['gap'] <= policy['inner_atol'] + policy['inner_rtol'] * c['gap_scale']
                        and 0 <= c['kkt'] <= policy['inner_kkt_tol']):
                    raise ValueError("Unqualified QP comparison sample")
                if not c['x'] or np.asarray(c['x']).ndim != 1 or not np.isfinite(c['x']).all():
                    raise ValueError('Invalid QP comparison state')
            if any(not np.isfinite(row['seconds']) or row['seconds'] <= 0
                   for row in case['latency_samples']):
                raise ValueError("Invalid uninstrumented QP latency")
            if any(case[key]['kernel_dispatches'] <= 0 for key in ('dispatch_profile', 'attribution')):
                raise ValueError("Missing CUDA dispatch coverage")
        ca, cb = a['latency_samples'][0]['certificate'], b['latency_samples'][0]['certificate']
        if len(ca['x']) != len(cb['x']):
            raise ValueError('QP comparison dimensions differ')
        bound = np.sqrt(2 * ca['gap'] / ca['minimum_curvature']) + np.sqrt(2 * cb['gap'] / cb['minimum_curvature'])
        error = float(np.linalg.norm(np.asarray(ca['x']) - np.asarray(cb['x'])))
        roundoff = 256 * np.finfo(float).eps * (1 + np.linalg.norm(ca['x']) + np.linalg.norm(cb['x']))
        if error > bound + roundoff:
            raise ValueError("Qualified fixed-QP solutions exceed their strong-convexity gap bound")
        objective_error = abs(ca['objective'] - cb['objective'])
        objective_bound = ca['gap'] + cb['gap'] + 256 * np.finfo(float).eps * (
            1 + abs(ca['objective']) + abs(cb['objective']))
        if objective_error > objective_bound:
            raise ValueError('Fixed-QP objective difference exceeds the combined certificate gap')
        aa = statistics.median(row['seconds'] for row in a['latency_samples'])
        bb = statistics.median(row['seconds'] for row in b['latency_samples'])
        comparisons.append(dict(name=name, baseline_median_seconds=aa, current_median_seconds=bb,
                                baseline_over_current_latency=aa / bb,
                                baseline_kernel_dispatches=a['dispatch_profile']['kernel_dispatches'],
                                current_kernel_dispatches=b['dispatch_profile']['kernel_dispatches'],
                                solution_l2_difference=error, combined_gap_distance_bound=float(bound),
                                objective_absolute_difference=objective_error,
                                combined_objective_gap_bound=objective_bound,
                                baseline_attribution=a['attribution'], current_attribution=b['attribution']))
    return dict(schema=SCHEMA + '.comparison', status='passed', comparisons=comparisons,
                baseline_source_sha256=baseline['source']['source_sha256'],
                current_source_sha256=current['source']['source_sha256'],
                scope='matched synthetic cold-start QPs with warmed compiler shapes; separate profiler calls include overhead')


@torch.no_grad()
def run(args, receipt, artifacts, record):
    device = require_cuda(args.device)
    cc = os.environ.get('CC')
    if not cc or shutil.which(cc) is None:
        raise RuntimeError('Set CC to an available compiler before CUDA attribution')
    props = torch.cuda.get_device_properties(device)
    receipt['environment'] = dict(gpu=props.name, capability=list(torch.cuda.get_device_capability(device)),
                                  device_uuid=str(getattr(props, 'uuid', 'unavailable')),
                                  torch=torch.__version__, cuda_runtime=torch.version.cuda,
                                  compiler=shutil.which(cc), python=platform.python_version())
    for name in args.fixtures:
        problem = fixture(name)
        identity = fixture_identity(problem)
        input_file = artifacts / f'{name}.inputs.json'
        write_json(input_file, dict(identity=identity, derivation=problem['derivation'],
                                   arrays={key: value.tolist() for key, value in problem['arrays'].items()}))
        record('fixture_started', identity)
        arrays = upload(problem, device)
        snapshots = {key: value.clone() for key, value in arrays.items()}
        kernels, reference = Kernels(device, compiled=True), Kernels(device, compiled=False)
        if not kernels.compiled:
            raise RuntimeError('Missing compiled CUDA admission')
        warmups = []
        for _ in range(args.warmups):
            synchronize(device)
            started = perf_counter()
            fitted = solve(qp.solve_qp, arrays, kernels)
            synchronize(device)
            seconds = perf_counter() - started
            warmups.append(dict(seconds=seconds, certificate=qualify(fitted, arrays, snapshots, reference)))
        case = dict(fixture=identity, input_file_sha256=digest(input_file), warmups=warmups,
                    warmup_scope='includes first-shape compilation; excluded from latency samples', latency_samples=[])
        for repeat in range(args.repeats):
            before = kernel_counts(kernels)
            torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            started = perf_counter()
            fitted = solve(qp.solve_qp, arrays, kernels)
            synchronize(device)
            seconds = perf_counter() - started
            sample = dict(repeat=repeat, seconds=seconds,
                          compiled_entry_point_calls=counter_delta(kernel_counts(kernels), before),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                          certificate=qualify(fitted, arrays, snapshots, reference))
            case['latency_samples'].append(sample)
            record('latency_qualified', dict(name=name, **sample))
        trace = artifacts / f'{name}.dispatch.trace.json' if args.traces else None
        fitted, case['dispatch_profile'] = measured_profile(
            lambda: solve(qp.solve_qp, arrays, kernels), device, trace=trace)
        case['dispatch_profile']['certificate'] = qualify(fitted, arrays, snapshots, reference)
        with attribution(kernels) as (instrumented_solver, stages):
            trace = artifacts / f'{name}.attribution.trace.json' if args.traces else None
            fitted, case['attribution'] = measured_profile(
                lambda: solve(instrumented_solver, arrays, kernels), device,
                trace=trace, stage_counts=stages.counts)
            case['attribution']['flow_boundary'] = stages.flow_boundary
        case['attribution']['certificate'] = qualify(fitted, arrays, snapshots, reference)
        case['attribution']['stage_coverage'] = qualify_stage_coverage(case['attribution'], fitted)
        all_calls = warmups + case['latency_samples'] + [case['dispatch_profile'], case['attribution']]
        case['work_totals'] = dict(
            qp_calls=len(all_calls), independent_original_certificates=len(all_calls),
            admm_iterations=sum(row['certificate']['admm_iterations'] for row in all_calls),
            polish_iterations=sum(row['certificate']['polish_iterations'] for row in all_calls),
            scope='all warmup, uninstrumented latency and separate profiler QP calls in this fixture')
        case['compilation_statistics'] = kernels.compilation_diagnostics()
        write_json(artifacts / f'{name}.measurements.json', case)
        receipt['fixtures'].append(case)
        record('fixture_qualified', dict(name=name, input_sha256=identity['input_sha256'],
                                         measurements_sha256=digest(artifacts / f'{name}.measurements.json')))
    if (source_provenance()['source_sha256'] != receipt['source']['source_sha256']
            or digest(__file__) != receipt['script_sha256']):
        raise RuntimeError('Source or attribution script changed during execution')
    receipt['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fixtures', nargs='+', choices=('resource64', 'resource256', 'mixed64'),
                        default=['resource64', 'resource256', 'mixed64'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--traces', action='store_true')
    args = parser.parse_args()
    if min(args.repeats, args.warmups, args.timeout_seconds) < 1 or len(set(args.fixtures)) != len(args.fixtures):
        parser.error('positive counts and unique fixture names required')
    out = args.out.resolve()
    artifacts, events = out.with_suffix('.artifacts'), out.with_suffix('.events.jsonl')
    if any(path.exists() for path in (out, artifacts, events)):
        raise FileExistsError('Existing attribution evidence is never overwritten')
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    receipt = dict(schema=SCHEMA, status='running', label=args.label, source=source_provenance(),
                   script_sha256=digest(__file__), policy=asdict(CudaPolicy()), started_utc=utc_now(),
                   numerical_execution='CUDA float64', fixtures=[], command=sys.argv,
                   interpreter=sys.executable, lsf_job_id=os.environ.get('LSB_JOBID'),
                   controls=dict(fixtures=args.fixtures, repeats=args.repeats, warmups=args.warmups,
                                 initialization='literal cold primal start, zero dual for every call',
                                 latency_scope='uninstrumented warmed solve with synchronization at boundaries; excludes independent certificate',
                                 memory_scope='uninstrumented solve only; input/snapshot and compiled caches present at reset',
                                 profiling_scope='separate profiler-only dispatch and disjoint annotated attribution; instrumented totals include overhead'))
    write_json(artifacts / 'source-and-plan.json', receipt)
    began = perf_counter()
    code = 0
    with events.open('x', encoding='utf-8') as stream:
        def record(kind, detail):
            value = dict(kind=kind, utc=utc_now(), elapsed_seconds=perf_counter() - began, **detail)
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)
        def interrupted(signum, frame):
            raise TimeoutError(f'QP attribution interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(args.timeout_seconds)
        try:
            run(args, receipt, artifacts, record)
        except BaseException as error:
            receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc())
            record('failure', dict(error_type=type(error).__name__, error=str(error)))
            code = 1
        finally:
            signal.alarm(0)
            receipt.update(finished_utc=utc_now(), elapsed_seconds=perf_counter() - began,
                           events_sha256=digest(events))
            write_json(out, receipt)
    return code


if __name__ == '__main__':
    sys.exit(main())
