"""Source-bound CUDA scalar profiling, usable against 371003f and the current API.

The CPU profiler records host dispatch and scalar-read proxies. Fitted numeric
tensors and compiled kernels remain on CUDA; this is not CPU inference profiling.
Run baseline and current sources in separate fresh processes with this same script.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

from clipp1d.api import source_provenance
from clipp1d.cuda_api import require_cuda
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.scalar import pilot
from clipp1d.types import CountModel


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def fixture(name):
    if name == "analytical32":
        i = np.arange(32)
        alt = (np.asarray((100, 250, 400))[i % 3] + (i // 3) % 41).astype(float)
        depth, counts, scales = 1000, np.ones(32, dtype=int), np.full(32, 0.5)
    elif name == "mixed6":
        alt = np.array([16.0, 28.0, 37.0, 19.0, 40.0, 54.0])
        depth, counts = 100, np.array([1, 2, 3, 4, 2, 3])
        scales = np.array([0.4, 0.25, 0.15, 0.12, 0.35, 0.42])
    else:
        raise ValueError(name)
    support = np.arange(1, int(counts.max()) + 1)
    valid = support <= counts[:, None]
    return CountModel(tuple(f"m{i:04}" for i in range(alt.size)), alt, depth - alt,
                      np.full(alt.size, 1e-6), np.minimum(1.0, (1 - 1e-6) / (scales * counts)),
                      scales[:, None] * np.where(valid, support, 0),
                      np.where(valid, -np.log(counts[:, None]), -np.inf), valid, 1e-6)


def identity(host):
    arrays = {}
    aggregate = hashlib.sha256(json.dumps(host.mutation_ids).encode())
    for key in ("alt", "ref", "slope", "log_prior", "lower", "upper", "valid"):
        value = np.asarray(getattr(host, key))
        arrays[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                           sha256=hashlib.sha256(value.tobytes()).hexdigest())
        aggregate.update(key.encode())
        aggregate.update(value.tobytes())
    return dict(model_sha256=aggregate.hexdigest(), arrays=arrays,
                mutation_ids=list(host.mutation_ids), eps=host.eps)


def counters(model):
    return {name: dict(getattr(model, name, {})) for name in
            ("integrity_counters", "scalar_work_counters")}


def difference(after, before):
    return {kind: {key: value - before.get(kind, {}).get(key, 0) for key, value in values.items()}
            for kind, values in after.items()}


def scalar_result(result, model):
    p = CudaPolicy()
    if not bool(result.qualified.all() & torch.isfinite(result.phi).all()
                & torch.isfinite(result.loss).all() & torch.isfinite(result.gap).all()
                & (result.phi >= model.lower).all() & (result.phi <= model.upper).all()
                & (result.gap >= 0).all()
                & (result.gap <= p.scalar_atol + p.scalar_rtol * result.loss.abs()).all()):
        raise AssertionError("Scalar result failed unchanged production qualification")
    return {name: getattr(result, name).cpu().tolist() for name in
            ("phi", "loss", "lower_bound", "gap", "qualified")}


@torch.no_grad()
def run(args, receipt, artifacts, record):
    device = require_cuda(args.device)
    cc = os.environ.get("CC")
    if not cc or shutil.which(cc) is None:
        raise RuntimeError("Set CC to an available compiler before profiling")
    receipt.update(gpu=torch.cuda.get_device_name(device), device=str(device), cc=cc,
                   cuda_runtime=torch.version.cuda, capability=list(torch.cuda.get_device_capability(device)))
    for name in ("analytical32", "mixed6"):
        host = fixture(name)
        record("fixture_started", dict(name=name, fixture=identity(host)))
        model = TensorModel.from_host(host, device, compiled=True)
        if not model.kernels.compiled:
            raise AssertionError("Compiled CUDA admission missing")
        torch.cuda.synchronize(device)
        began = perf_counter()
        warm = pilot(model)
        torch.cuda.synchronize(device)
        warm_seconds = perf_counter() - began
        warm_result = scalar_result(warm, model)
        record("warmup_qualified", dict(name=name, seconds=warm_seconds, result=warm_result))
        before = counters(model)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        began = perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=False, profile_memory=False) as profile:
            fitted = pilot(model)
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - began
        after = counters(model)
        trace = artifacts / f"{name}.trace.json"
        profile.export_chrome_trace(str(trace))
        result = scalar_result(fitted, model)
        counts = {event.key: dict(count=event.count, self_cpu_time_us=event.self_cpu_time_total,
                                  cpu_time_total_us=event.cpu_time_total)
                  for event in profile.key_averages()}
        profile_file = artifacts / f"{name}.counts.json"
        write(profile_file, counts)
        if (fitted.phi.device != device or fitted.loss.device != device
                or fitted.phi.dtype != torch.float64):
            raise AssertionError("Scalar fitted state left the CUDA float64 boundary")
        np.testing.assert_allclose(result["phi"], warm_result["phi"], atol=0, rtol=0)
        record("profile_qualified", dict(
            name=name, seconds=elapsed, timing_scope="warmed scalar call plus CPU profiler overhead",
            result=result, scalar_read_proxy_counts={key: counts.get(key, {}).get("count", 0)
                for key in ("aten::_local_scalar_dense", "aten::item", "aten::is_nonzero")},
            proxy_scope="host scalar-read operations; not a direct CUDA synchronization duration or CPU inference measure",
            counters_before=before, counters_after=after, counter_deltas=difference(after, before),
            optional_counters_available=any(before.values()), trace=str(trace), trace_sha256=digest(trace),
            counts_file=str(profile_file), counts_sha256=digest(profile_file),
            compilation_statistics=model.kernels.compilation_diagnostics(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    if (source_provenance()["source_sha256"] != receipt["source"]["source_sha256"]
            or digest(__file__) != receipt["script_sha256"]):
        raise RuntimeError("Source or common profiler script changed during execution")
    receipt["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout-seconds", default=600, type=int)
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("timeout-seconds must be positive")
    out = args.out.resolve()
    artifacts = out.with_suffix(".artifacts")
    events = out.with_suffix(".events.jsonl")
    if any(path.exists() for path in (out, artifacts, events)):
        raise FileExistsError("Existing profiler evidence is never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    receipt = dict(schema="clipp1d.cuda.scalar_profile.v1", status="running", started_utc=utc_now(),
                   source=source_provenance(), script_sha256=digest(__file__), policy=asdict(CudaPolicy()),
                   python=platform.python_version(), interpreter=sys.executable, torch=torch.__version__,
                   command=sys.argv, lsf_job_id=os.environ.get("LSB_JOBID"), cases=[],
                   numerical_execution="CUDA float64", profiler_activities=["CPU host dispatch"])
    write(artifacts / "source-and-plan.json", receipt)
    began = perf_counter()
    code = 0
    with events.open("x", encoding="utf-8") as handle:
        def record(kind, detail):
            event = dict(kind=kind, utc=utc_now(), elapsed_seconds=perf_counter() - began, **detail)
            receipt["cases"].append(event)
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)

        def interrupted(signum, frame):
            raise TimeoutError(f"Scalar CUDA profiler interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(args.timeout_seconds)
        try:
            run(args, receipt, artifacts, record)
        except BaseException as error:
            receipt.update(status="failed", error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc())
            record("failure", dict(error_type=type(error).__name__, error=str(error)))
            code = 1
        finally:
            signal.alarm(0)
            receipt.update(finished_utc=utc_now(), elapsed_seconds=perf_counter() - began,
                           events_sha256=digest(events))
            write(out, receipt)
    return code


if __name__ == "__main__":
    sys.exit(main())
