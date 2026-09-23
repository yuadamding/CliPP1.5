"""Bounded CPU reference diagnosis only; never CUDA acceptance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback

import torch
from clipp1d.api import source_provenance
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda import selection

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qualification_reference", ROOT / "benchmarks/qualify_cuda.py")
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)
torch.set_num_threads(1)
host = qualifier.scaling_fixture(256)
source = source_provenance()
started = perf_counter()
index = 0


def save(name, data):
    qualifier.write_json(OUT / name, data)


def event(kind, **values):
    row = dict(kind=kind, elapsed_seconds=perf_counter() - started, **values)
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(qualifier.clean(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(qualifier.clean(row)), flush=True)


original_fit = selection.fit_lambda


def traced_fit(*args, **kwargs):
    global index
    current = index
    index += 1
    penalty = float(args[3])
    begin = perf_counter()
    event("lambda_started", index=current, lambda_value=penalty)
    result = original_fit(*args, **kwargs)
    event("lambda_finished", index=current, lambda_value=penalty,
          seconds=perf_counter() - begin, qualified=result.qualified,
          complete=result.diagnostics.get("search_complete"), objective=float(result.objective),
          diagnostics=result.diagnostics)
    return result


def stop(signum, frame):
    raise TimeoutError("120-second numerical diagnostic limit reached")


save("source-and-plan.json", dict(source=source, fixture=qualifier.host_identity(host),
     scope=__doc__, numerical_seconds_limit=120, interpreter=sys.executable,
     torch=torch.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
     driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
selection.fit_lambda = traced_fit
signal.signal(signal.SIGALRM, stop)
signal.alarm(120)
try:
    model = TensorModel.from_host(host, "cpu", False)
    result = selection.fit_tensor_model(model)
    signal.alarm(0)
    save("path.json", result.records)
    save("terminal.json", dict(status="complete", scope=__doc__,
         elapsed_seconds=perf_counter() - started, search_status=result.search_status,
         lambda_value=float(result.lambda_value), score=float(result.refit.score),
         raw_x=result.raw.x.tolist(), refit_phi=result.refit.phi.tolist(),
         timings=result.timings, source_unchanged=source == source_provenance()))
except BaseException as error:
    signal.alarm(0)
    save("terminal.json", dict(status="limited" if isinstance(error, TimeoutError) else "failed",
         scope=__doc__, error=repr(error), traceback=traceback.format_exc(),
         elapsed_seconds=perf_counter() - started, source_unchanged=source == source_provenance()))
    raise
