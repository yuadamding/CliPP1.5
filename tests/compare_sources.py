"""Exact, separate-process compaction comparison; generated evidence stays external.

Build each checkout first, with the same supported dependency stack, then run::

    python tests/compare_sources.py --baseline /path/baseline --candidate /path/candidate \
        --output /external/new-evidence --device cpu

Use the baseline path for both sources to qualify the harness before refactoring.
CUDA must be requested explicitly and reported by both runs; no fallback is accepted.
This captures transient evidence without modifying the fitter or bypassing verification.
Logical progress calls are observed before the wall-clock throttle, which is left intact.
The measurements include instrumentation overhead, and are not performance benchmarks.
"""

import argparse
from contextlib import ExitStack
import filecmp
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
from unittest.mock import patch


# Only measured time/storage fields are removed. Cache capacities and work counters
# remain compared, as do every candidate field, eligibility flag and raw diagnostic.
_MEASURED = {
    "elapsed_seconds",
    "scratch_bytes",
    "peak_scratch_bytes",
    "conditional_refit_seconds",
    "weight_solver_seconds",
    "weight_optimization_seconds",
    "likelihood_column_assembly_seconds",
    "boundary_dynamic_programming_seconds",
    "conditional_center_refit_seconds",
    "grid_preparation_seconds",
    "grid_evaluation_seconds",
    "scalar_optimization_seconds",
}
_BUILD_IDENTITY = {"source_commit", "source_hashes", "native_build_id", "library_sha256"}
_THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def _default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def _json(value):
    # Float repr preserves exact same-stack differences (and signed zero). NaNs
    # in failed-candidate records are preserved rather than replaced with null.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def _write(path, value):
    path.write_text(_json(value) + "\n")


def _logical(value):
    if isinstance(value, dict):
        return {key: _logical(item) for key, item in value.items() if key not in _MEASURED}
    if isinstance(value, list):
        return [_logical(item) for item in value]
    return value


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_path(root):
    source = root / "src" if (root / "src/clipp").is_dir() else root
    if not (source / "clipp/__init__.py").is_file():
        raise ValueError(f"Not a CliPP source/package root: {root}")
    return source


def _fixtures(baseline, output):
    output.mkdir()
    for name in ("snv", "cna", "purity"):
        shutil.copyfile(baseline / "sample" / f"sample.{name}.txt", output / f"sample.{name}.txt")
    rng = random.Random(1729)
    snv = ["chromosome_index\tposition\talt_count\tref_count"]
    cn = ["chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn"]
    for index in range(36):
        major, minor = ((1, 1), (2, 0), (4, 2), (403, 3))[index % 4]
        depth = 80 + index
        cp = (0.12, 0.37, 0.7)[index % 3]
        multiplicity = 1 + index % major
        probability = cp * multiplicity / (2 * (1 - 0.8) + 0.8 * (major + minor))
        alt = sum(rng.random() < probability for _ in range(depth))
        if index == 0:
            alt = 0
        elif index == 1:
            alt = depth
        position = 10 * index + 1
        snv.append(f"1\t{position}\t{alt}\t{depth - alt}")
        cn.append(f"1\t{position}\t{position}\t{major}\t{minor}\t{major + minor}")
    (output / "high_cn.snv.txt").write_text("\n".join(snv) + "\n")
    (output / "high_cn.cna.txt").write_text("\n".join(cn) + "\n")
    (output / "high_cn.purity.txt").write_text("0.8\n")
    cases = [
        {"name": "sample", "input": "sample", "config": {"max_clusters": 10}},
        {"name": "high_cn", "input": "high_cn", "config": {"max_clusters": 4}},
        {
            "name": "subsample",
            "input": "sample",
            "config": {
                "max_clusters": 3,
                "subsample_size": 80,
                "replicates": 2,
                "seed": 1729,
            },
        },
    ]
    _write(output / "cases.json", cases)
    return output / "cases.json"


def _capture_case(api, verifier, progress_class, selection_module, config, inputs, output):
    output.mkdir()
    original_verify = verifier.verify_run
    original_update = progress_class.update
    original_progress = api._progress
    original_polish = selection_module.polish_chain_partition
    evidence_count = 0
    with ExitStack() as stack:
        logical_events = stack.enter_context((output / "progress.jsonl").open("w"))
        histories = stack.enter_context((output / "refinement.jsonl").open("w"))

        def event(value):
            logical_events.write(_json(_logical(value)) + "\n")

        def progress(record):
            # Throttled events vary with wall time. Their full logical stream is
            # recorded below, before the untouched production throttle runs.
            if record.get("stage") != "numerical_progress":
                event(record)
            original_progress(record)

        def update(self, name, *, force=False, **fields):
            event(
                {
                    "stage": "numerical_progress",
                    **self.fields,
                    **self.counters,
                    "event": name,
                    "force": force,
                    **fields,
                }
            )
            return original_update(self, name, force=force, **fields)

        def polish(*args, **kwargs):
            result = original_polish(*args, **kwargs)
            histories.write(_json(result) + "\n")
            return result

        def verify(*args, **kwargs):
            nonlocal evidence_count
            evidence = kwargs.get("_candidate_evidence")
            if evidence is not None:
                evidence_count += 1
                attempts, candidates = evidence
                _write(
                    output / "selection.json",
                    {
                        "columns": attempts.columns.tolist(),
                        "records": attempts.to_dict("records"),
                    },
                )
                with (output / "candidates.jsonl").open("w") as stream:
                    for row in candidates.iter_rows():
                        stream.write(_json(row) + "\n")
            return original_verify(*args, **kwargs)

        stack.enter_context(patch.object(api, "_progress", progress))
        stack.enter_context(patch.object(progress_class, "update", update))
        stack.enter_context(patch.object(selection_module, "polish_chain_partition", polish))
        stack.enter_context(patch.object(verifier, "verify_run", verify))
        started = time.perf_counter()
        summary = api.fit(*inputs, output / "run", config=config)
        elapsed = time.perf_counter() - started
    if evidence_count != 1:
        raise AssertionError(f"Expected one complete transient evidence bank, saw {evidence_count}")
    independent = original_verify(output / "run")
    if independent["backend"] != config.device or summary["backend"] != config.device:
        raise AssertionError("Requested backend was not used")
    manifest = json.loads((output / "run/manifest.json").read_text())
    scientific = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "started_utc",
            "source_commit",
            "native",
            "environment",
            "stage_seconds",
        }
    }
    for record in scientific["inputs"].values():
        record.pop("original_path")
    _write(output / "manifest-scientific.json", _logical(scientific))
    _write(output / "verified.json", independent)
    # Read every capacity/replicate through the public reader too.
    from clipp.output import load_result

    result = load_result(output / "run")
    with (output / "partitions.jsonl").open("w") as stream:
        for record in manifest["fits"]:
            partition = result.partition(record["requested_k"], replicate=record["replicate"])
            stream.write(_json(partition) + "\n")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    _write(
        output / "measurement.json",
        {
            "instrumented_wall_seconds": elapsed,
            "process_lifetime_maxrss": usage.ru_maxrss,
            "maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB",
            "scope": "instrumented child; RSS includes earlier cases; not an isolated fit benchmark",
            "complete": summary,
        },
    )
    return manifest


def _worker(args):
    source = _source_path(args.worker)
    sys.path.insert(0, str(source))
    import clipp.api as api
    import clipp.selection as selection
    import clipp.verify as verifier
    from clipp._progress import NumericalProgress
    from clipp.config import FitConfig
    from clipp.environment import fingerprint
    from clipp.native import load_native

    if Path(api.__file__).resolve().parent != (source / "clipp").resolve():
        raise AssertionError("Wrong checkout imported")
    _, native = load_native()  # Enforces all normal source/native identity checks.
    environment = fingerprint()
    versions = environment["dependencies"]
    if sys.version_info < (3, 12) or versions["scipy"].split(".")[:2] != ["1", "18"]:
        raise RuntimeError("Differential qualification requires Python >=3.12 and SciPy >=1.18,<1.19")
    args.output.mkdir()
    _write(
        args.output / "identity.json",
        {
            "source": str(args.worker),
            "python_executable": sys.executable,
            "environment": environment,
            "native": native,
        },
    )
    cases = json.loads(args.inputs.read_text())
    for case in cases:
        inputs = [args.inputs.parent / f"{case['input']}.{name}.txt" for name in ("snv", "cna", "purity")]
        config = FitConfig(sample_id=case["name"], device=args.device, **case["config"])
        manifest = _capture_case(
            api, verifier, NumericalProgress, selection, config, inputs, args.output / case["name"]
        )
        # Include the actual CUDA device/runtime identity; CPU records null.
        _write(args.output / case["name"] / "backend.json", manifest["environment"].get("cuda"))


def _first_difference(left, right):
    with left.open("rb") as lhs, right.open("rb") as rhs:
        for number, (a, b) in enumerate(itertools.zip_longest(lhs, rhs), 1):
            if a != b:
                return {"line": number, "baseline": repr(a)[:600], "candidate": repr(b)[:600]}
    return None


def _compare(output, inputs):
    baseline, candidate = output / "baseline", output / "candidate"
    identity = [json.loads((root / "identity.json").read_text()) for root in (baseline, candidate)]
    differences = []
    for key in ("environment", "native"):
        lhs, rhs = (item[key] for item in identity)
        if key == "native":
            lhs, rhs = ({k: v for k, v in value.items() if k not in _BUILD_IDENTITY} for value in (lhs, rhs))
        if _json(lhs) != _json(rhs):
            differences.append(
                {"identity": key, "reason": "numerical stack or compiler configuration differs"}
            )
    compared = []
    for case in json.loads(inputs.read_text()):
        name = case["name"]
        files = {
            "candidates.jsonl",
            "selection.json",
            "manifest-scientific.json",
            "verified.json",
            "partitions.jsonl",
            "progress.jsonl",
            "refinement.jsonl",
            "backend.json",
        }
        manifests = [
            json.loads((root / name / "run/manifest.json").read_text()) for root in (baseline, candidate)
        ]
        artifacts = [set(manifest["artifacts"]) for manifest in manifests]
        if artifacts[0] != artifacts[1]:
            differences.append({"case": name, "reason": "artifact inventories differ"})
        files.update("run/" + artifact for artifact in artifacts[0] | artifacts[1])
        for relative in sorted(files):
            lhs, rhs = (root / name / relative for root in (baseline, candidate))
            if not lhs.is_file() or not rhs.is_file():
                differences.append({"case": name, "file": relative, "reason": "missing"})
            elif not filecmp.cmp(lhs, rhs, shallow=False):
                differences.append({"case": name, "file": relative, **_first_difference(lhs, rhs)})
            else:
                compared.append({"case": name, "file": relative, "sha256": _hash(lhs)})
    report = {
        "status": "passed" if not differences else "failed",
        "device": manifests[0]["backend_actual"],
        "comparison": "exact byte equality; numeric tolerances are not used",
        "excluded_measured_fields": sorted(_MEASURED),
        "excluded_build_identity_fields": sorted(_BUILD_IDENTITY),
        "manifest_exclusions": [
            "started_utc",
            "source_commit",
            "native",
            "environment",
            "stage_seconds",
            "inputs.*.original_path",
        ],
        "instrumentation": "complete candidate bank and pre-throttle logical progress; normal verifier preserved",
        "differences": differences,
        "compared_files": compared,
    }
    _write(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inputs", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        _worker(args)
        return
    if args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required")
    sources = {name: getattr(args, name).resolve(strict=True) for name in ("baseline", "candidate")}
    for source in sources.values():
        _source_path(source)
        if args.output.is_relative_to(source):
            parser.error("generated evidence must be outside both source trees")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = _fixtures(sources["baseline"], args.output / "inputs")
    env = {**os.environ, **_THREADS, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("CLIPP_FORCE_CPU", "CLIPP_REQUIRE_CUDA", "PYTHONPATH"):
        env.pop(key, None)
    for name, root in sources.items():
        command = [
            getattr(args, name + "_python"),
            str(Path(__file__).resolve()),
            "--worker",
            str(root),
            "--inputs",
            str(inputs),
            "--output",
            str(args.output / name),
            "--device",
            args.device,
        ]
        with (args.output / f"{name}.log").open("w") as log:
            completed = subprocess.run(
                command, cwd=args.output, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if completed.returncode:
            raise RuntimeError(
                f"{name} failed (exit {completed.returncode}); see {args.output / (name + '.log')}"
            )
    report = _compare(args.output, inputs)
    print(
        f"{report['status']}: {len(report['compared_files'])} exact files; {args.output / 'comparison.json'}"
    )
    if report["differences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
