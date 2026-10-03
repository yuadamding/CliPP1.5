"""Public fitting API with isolated work, stage timing and verified publication."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import tempfile
import time
import uuid

import numpy as np

from ._io import write_json
from ._candidate_store import CandidateStore
from .config import FitConfig
from .environment import fingerprint
from .preprocessing import preprocess
from .kernel import _prepare_chain, _run_kernel
from .native import load_native, sha256
from .selection import run_model_selection
from .subsampling import run_clipp_sub
from .versions import IDENTITIES, NUMERICS

_FIT_LOCK = threading.Lock()  # Native environment flags are process-global.


def _progress(record):
    print(
        "CLIPP_PROGRESS " + json.dumps({"utc": utc_now(), **record}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class _Profile:
    def __init__(self):
        self.stages = {}

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
    config = FitConfig() if config is None else config
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
    scratch = Path(tempfile.gettempdir()) / ("clipp-candidates-" + uuid.uuid4().hex + ".sqlite")
    profile = _Profile()
    candidate_store = None
    try:
        os.write(
            lock_fd,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "output": str(out),
                    "started_utc": started_utc,
                    "candidate_evidence": str(scratch),
                }
            ).encode(),
        )
        os.fsync(lock_fd)
        work.mkdir()
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
            canonical = preprocess(
                *(copied / (name + ".txt") for name in ("snv", "cna", "purity")), output=pre
            )
            retained = canonical.retained
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
            prepared = _prepare_chain(canonical, raw)
        with _backend_environment(actual_device), profile.stage("native_candidates"):
            if config.subsample_size is None:
                (r, n, major, total, purity), pilot, order = prepared
                _run_kernel(
                    r[order],
                    n[order],
                    major[order],
                    total[order],
                    purity,
                    pilot[order],
                    raw,
                    capacities,
                    library=library,
                )
            else:
                run_clipp_sub(
                    prepared,
                    raw,
                    capacities,
                    config.subsample_size,
                    config.replicates,
                    config.window_size,
                    config.overlap,
                    config.seed,
                    library=library,
                )
        candidate_store = CandidateStore(scratch)
        with profile.stage("refit_and_selection"):
            selection = run_model_selection(
                canonical,
                prepared[2],
                raw,
                final,
                capacities,
                reps=config.replicates if config.subsample_size is not None else None,
                candidate_store=candidate_store,
                progress=_progress,
            )
        ledger = canonical.ledger
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
            "selected_k": selection["selected_k"],
            "fits": selection["fits"],
            "purity": canonical.purity,
            "selected_outputs": {
                "mutations": "final_result/mutations.tsv",
                "clusters": "final_result/clusters.tsv",
            },
            "canonical_input_sha256": sha256(pre / "retained.tsv"),
            "mutation_identity_sha256": sha256(pre / "input_ledger.tsv"),
            "chain_order_sha256": sha256(raw / "chain_order.txt"),
            "initializer": json.loads((raw / "chain_initialization.json").read_text()),
            "subsamples": replicas,
            "environment": {**fingerprint(), "cuda": cuda_info},
            "artifacts": {
                str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob("*")) if p.is_file()
            },
        }
        write_json(work / "manifest.json", manifest)
        from .verify import verify_run

        with profile.stage("independent_verification"):
            verified = verify_run(
                work,
                require_complete=False,
                _candidate_evidence=(selection["selection"], selection["candidates"]),
                _progress=_progress,
            )
        manifest["selection_resources"] = {
            "candidate_records": len(candidate_store),
            "peak_scratch_bytes": candidate_store.peak_scratch_bytes,
            "storage": "transient_sqlite_bounded_cache",
            "scratch_measurement_scope": (
                "observed owned SQLite database/journal/sidecar logical file sizes; "
                "excludes SQLite temporary files, page cache, process memory and unsampled peaks"
            ),
            **selection.get("telemetry", {}),
        }
        candidate_store.close()
        candidate_store = None
        del selection  # Evidence is removed only after independent verification.
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
            "peak_process_tree_rss_bytes_sampled": None,
            "memory_measurement": "unmeasured; use an external process-tree monitor for sampled RSS",
            "memory_sampling_interval_seconds": None,
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
                    "candidate_evidence": str(scratch) if scratch.exists() else None,
                },
            )
        raise
    finally:
        pending_error = sys.exception()
        try:
            if candidate_store is not None:
                candidate_store.close(remove=False)
        except Exception as cleanup_error:
            if pending_error is None:
                raise
            pending_error.add_note("Candidate scratch close also failed: " + str(cleanup_error))
        finally:
            os.close(lock_fd)
            lock.unlink()
