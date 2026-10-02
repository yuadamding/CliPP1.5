"""Public fitting API with isolated work, stage timing and verified publication."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import uuid

import numpy as np
import pandas as pd
import psutil

from .config import FitConfig
from .kernel import _prepare_chain, _run_kernel
from .native import load_native, sha256
from .selection import run_model_selection
from .subsampling import run_clipp_sub
from .versions import IDENTITIES, NUMERICS

_FIT_LOCK = threading.Lock()  # Native environment flags are process-global.


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _Profile:
    def __init__(self):
        self.stages = {}
        self.peak = 0
        self.stop = threading.Event()
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while True:
            rss = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, rss)
            if self.stop.wait(0.02):
                return

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = time.perf_counter() - start


@contextmanager
def _backend_environment(device):
    previous = {key: os.environ.get(key) for key in ("CLIPP_FORCE_CPU", "CLIPP_REQUIRE_CUDA")}
    os.environ["CLIPP_FORCE_CPU"] = "1" if device == "cpu" else "0"
    os.environ["CLIPP_REQUIRE_CUDA"] = "1" if device == "cuda" else "0"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _choose_backend(library, requested, count):
    if requested == "cpu" or (requested == "auto" and count <= 1000):
        return "cpu"
    library.CliPPWarmupCUDA.argtypes = []
    library.CliPPWarmupCUDA.restype = ctypes.c_int
    status = library.CliPPWarmupCUDA()
    if status == 0:
        return "cuda"
    if status == 2 and requested == "auto":
        return "cpu"
    if status == 2:
        raise RuntimeError("CUDA requested but this build/device is unavailable; CPU fallback is forbidden")
    raise RuntimeError(f"CUDA warmup failed with status {status}; refusing CPU fallback")


def fit(snv_input, cn_input, purity_input, output, *, config=None):
    """Fit three legacy-format inputs into a *new* external output directory.

    Returns a summary after independent verification. Existing output paths are
    never reused. Failures retain an isolated attempt and logs, without a success
    marker. Concurrent callers in one process are serialized; use processes for
    independent fits. Fitting never reads simulation truth.
    """
    config = config or FitConfig()
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig")
    with _FIT_LOCK:
        return _fit(snv_input, cn_input, purity_input, output, config)


def _fit(snv_input, cn_input, purity_input, output, config):
    started, started_utc = time.perf_counter(), utc_now()
    requested = config.resolved_device()
    out = Path(output).absolute()
    package = Path(__file__).resolve().parent
    if out.resolve().is_relative_to(package):
        raise ValueError("Output must be outside the installed package")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new run directory: {out}")
    sources = {
        name: Path(path).resolve(strict=True)
        for name, path in (("snv", snv_input), ("cna", cn_input), ("purity", purity_input))
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name("." + out.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    work = out.with_name("." + out.name + ".inprogress." + uuid.uuid4().hex)
    profile = _Profile()
    try:
        os.write(
            lock_fd, json.dumps({"pid": os.getpid(), "output": str(out), "started_utc": started_utc}).encode()
        )
        work.mkdir()
        profile.thread.start()
        with profile.stage("native_identity"):
            library, native = load_native()
        copied = work / "inputs"
        copied.mkdir()
        inputs = {}
        with profile.stage("input_snapshot"):
            for name, source in sources.items():
                digest = sha256(source)
                destination = copied / (name + ".txt")
                shutil.copyfile(source, destination)
                if sha256(destination) != digest or sha256(source) != digest:
                    raise RuntimeError(f"Input changed while copying: {source}")
                inputs[name] = {
                    "original_path": str(source),
                    "path": str(destination.relative_to(work)),
                    "sha256": digest,
                }
        pre, raw, final = (
            work / name for name in ("preprocess_result", "preliminary_result", "final_result")
        )
        with profile.stage("preprocessing"):
            rscript = shutil.which("Rscript")
            if rscript is None:
                raise RuntimeError("Rscript is required; install R and its data.table package")
            with (
                (work / "preprocess.stdout.log").open("w") as stdout,
                (work / "preprocess.stderr.log").open("w") as stderr,
            ):
                process = subprocess.run(
                    [
                        rscript,
                        str(package / "resources/preprocess.R"),
                        *(str(copied / (name + ".txt")) for name in ("snv", "cna", "purity")),
                        config.sample_id,
                        str(pre),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if process.returncode:
                detail = (work / "preprocess.stderr.log").read_text().strip()
                raise ValueError(f"Preprocessing failed: {detail}")
            retained = pd.read_csv(
                pre / "retained.tsv", sep="\t", dtype={"mutation_id": str}, keep_default_na=False
            )
            capacities = config.capacities(len(retained))
        with profile.stage("backend_warmup"):
            actual_device = _choose_backend(library, requested, len(retained))
            cuda_info = None
            if actual_device == "cuda":
                library.CliPPCUDADeviceInfo.argtypes = []
                library.CliPPCUDADeviceInfo.restype = ctypes.c_char_p
                metadata = library.CliPPCUDADeviceInfo()
                if metadata is None:
                    raise RuntimeError("CUDA device metadata unavailable after warmup")
                cuda_info = json.loads(metadata.decode())
        with profile.stage("initialization"):
            prepared = _prepare_chain(pre, raw)
        with _backend_environment(actual_device), profile.stage("native_candidates"):
            if config.subsample_size is None:
                (r, n, major, total, purity), pilot, order = prepared
                _run_kernel(
                    r[order], n[order], major[order], total[order], purity, pilot[order], raw, capacities
                )
            else:
                run_clipp_sub(
                    pre,
                    raw,
                    capacities,
                    config.subsample_size,
                    config.replicates,
                    config.window_size,
                    config.overlap,
                    config.seed,
                    prepared,
                )
        with profile.stage("refit_and_selection"):
            selected_k = run_model_selection(
                pre,
                raw,
                final,
                capacities,
                reps=config.replicates if config.subsample_size is not None else None,
            )
        ledger = pd.read_csv(pre / "input_ledger.tsv", sep="\t")
        replicas = []
        for indices_file in sorted(raw.glob("sample_indices_rep*.txt")):
            rep = int(indices_file.stem.split("rep")[-1])
            indices = np.loadtxt(indices_file, dtype=int, ndmin=1)
            replicas.append(
                {
                    "replicate": rep,
                    "seed": config.seed + rep,
                    "indices_file": str(indices_file.relative_to(work)),
                    "mutation_ids": retained.iloc[indices].mutation_id.tolist(),
                }
            )
        manifest = {
            **IDENTITIES,
            "status": "fit_pending_verification",
            "started_utc": started_utc,
            "source_commit": native["source_commit"],
            "native": native,
            "inputs": inputs,
            "config": asdict(config),
            "numerics": NUMERICS,
            "capacities": capacities,
            "backend_requested": config.device,
            "backend_resolved": requested,
            "backend_actual": actual_device,
            "num_input_rows": len(ledger),
            "num_retained": len(retained),
            "num_excluded": int((ledger.status == "excluded").sum()),
            "selected_k": selected_k,
            "canonical_input_sha256": sha256(pre / "retained.tsv"),
            "mutation_identity_sha256": sha256(pre / "input_ledger.tsv"),
            "chain_order_sha256": sha256(raw / "chain_order.txt"),
            "initializer": json.loads((raw / "chain_initialization.json").read_text()),
            "subsamples": replicas,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cuda": cuda_info,
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "threads": {
                    name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
                "dependencies": {
                    name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "psutil")
                },
                "rscript": rscript,
                "r_version": subprocess.check_output(
                    [rscript, "--version"], stderr=subprocess.STDOUT, text=True
                ).strip(),
                "data_table_version": subprocess.check_output(
                    [rscript, "-e", 'cat(as.character(packageVersion("data.table")))'], text=True
                ).strip(),
            },
            "artifacts": {
                str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob("*")) if p.is_file()
            },
        }
        write_json(work / "manifest.json", manifest)
        from .verify import verify_run

        with profile.stage("independent_verification"):
            verified = verify_run(work, require_complete=False)
        manifest["verification"] = verified
        manifest["status"] = "verified_fit"
        manifest["stage_seconds"] = profile.stages.copy()
        write_json(work / "manifest.json", manifest)
        with profile.stage("publication"):
            if out.exists() or out.is_symlink():
                raise FileExistsError(f"Output appeared during fitting: {out}")
            os.rename(work, out)
            work = out
        # COMPLETE is the final public record. Its absence always means incomplete.
        completion = {
            "status": "complete",
            "finished_utc": utc_now(),
            "manifest_sha256": sha256(out / "manifest.json"),
            "stage_seconds": profile.stages,
            "end_to_end_seconds": time.perf_counter() - started,
            "peak_process_tree_rss_bytes_sampled": profile.peak,
            "memory_sampling_interval_seconds": 0.02,
            "timing_scope": "API entry through verified directory publication; excludes final completion-record write",
            "gpu_peak_memory_bytes": None,
        }
        write_json(out / "COMPLETE.json", completion)
        return {"output": str(out), **verified, **completion}
    except BaseException as error:
        if work.is_dir() and not (work / "COMPLETE.json").exists():
            write_json(
                work / "FAILURE.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "type": type(error).__name__,
                    "finished_utc": utc_now(),
                    "stage_seconds": profile.stages,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        raise
    finally:
        profile.stop.set()
        if profile.thread.is_alive():
            profile.thread.join(timeout=1)
        os.close(lock_fd)
        lock.unlink()
