"""Replay captured production QPs and deterministic wide-range numerical checks.

Example from a fresh checkout with the package dependencies installed::

    python benchmarks/replay_robustness.py --outdir results/robustness

Optionally pass an archived ``df44e6a`` package directory with
``--baseline-package /path/to/archive/clipp1d``. Its complete source fingerprint
must match the source recorded in the fixture manifest. Output must be fresh.
These are numerical component checks, not full-fit throughput measurements.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d import api, solver  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(package, expected_source):
    package = package.resolve()
    init = package / "__init__.py"
    if not init.is_file():
        raise ValueError("--baseline-package must point to an archived clipp1d package directory")
    name = "_clipp1d_robustness_baseline"
    spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    baseline_api = importlib.import_module(f"{name}.api")
    baseline_solver = importlib.import_module(f"{name}.solver")
    source = baseline_api.source_provenance()
    if (source["source_sha256"] != expected_source["source_sha256"] or
            source["source_files"] != expected_source["source_files"]):
        raise ValueError("Baseline package source does not match the captured fixture source")
    return baseline_api, baseline_solver, source


def validate_fixtures(directory):
    manifest_path = directory / "capture.json"
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if not cases or len({case["file"] for case in cases}) != len(cases):
        raise ValueError("Fixture manifest must contain distinct captured cases")
    for case in cases:
        filename = case["file"]
        if Path(filename).name != filename or not filename.endswith(".npz"):
            raise ValueError("Invalid fixture filename in capture manifest")
        if sha256(directory / filename) != case["sha256"]:
            raise ValueError(f"Fixture hash does not match capture manifest: {filename}")
    return manifest, sha256(manifest_path)


def finite(value):
    return float(value) if np.isfinite(value) else None


def result_record(fit):
    return dict(qualified=bool(fit.qualified), gap=finite(fit.gap),
                gap_scale=finite(fit.gap_scale), kkt_residual=finite(fit.kkt_residual), work=fit.work)


def summarize_counts(attempted, qualified, failures):
    return dict(attempted=attempted, qualified=qualified, failed=attempted - qualified,
                failure_counts=dict(failures))


def run(outdir, fixtures, baseline_package=None):
    if outdir.exists():
        raise FileExistsError("Output directory must be fresh; existing evidence is never overwritten")
    manifest, manifest_sha256 = validate_fixtures(fixtures)
    arms = {"revised": (api, solver, api.source_provenance())}
    if baseline_package is not None:
        arms["baseline"] = load_baseline(baseline_package, manifest["source"])
    outdir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = dict(
        schema="clipp1d.robustness_replay.v1", started_utc=datetime.now(timezone.utc).isoformat(),
        component_scope="captured quadratic surrogates and synthetic fixed-QP/profile checks",
        script_sha256=sha256(Path(__file__)), fixture_manifest_sha256=manifest_sha256,
        captured_source=manifest["source"], revised_source=arms["revised"][2],
        baseline_source=arms["baseline"][2] if "baseline" in arms else None,
        production_cases=[], production_summary={}, stress_cases=1000, stress_seed=349,
        stress_nodes=100, stress_curvature_log10_range=[-2, 9], stress_caps_log10_range=[-2, 8],
        stress_fixed_rule="odd case indices fix the middle node to one", stress_summary={},
        profiling_cases=80, profiling_seed=882, profiling_nodes=30, profile_summary={})

    qualified = Counter()
    failures = {name: Counter() for name in arms}
    for case in manifest["cases"]:
        with np.load(fixtures / case["file"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ("h", "target", "lower", "upper", "caps", "start")}
        record = dict(file=case["file"], sha256=case["sha256"], captured_context=case)
        for name, (_, numerical, _) in arms.items():
            fit = numerical.solve_quadratic(**arrays)
            record[name] = result_record(fit)
            qualified[name] += int(fit.qualified)
            if not fit.qualified:
                failures[name][fit.work.get("failure_reason", "certificate_gate")] += 1
        report["production_cases"].append(record)
    report["production_summary"] = {
        name: summarize_counts(len(manifest["cases"]), qualified[name], failures[name]) for name in arms}

    rng = np.random.default_rng(349)
    qualified = Counter()
    failures = {name: Counter() for name in arms}
    max_kkt = {name: 0. for name in arms}
    for seed in range(1000):
        n = 100
        h, target = 10**rng.uniform(-2, 9, n), rng.normal(.5, 2, n)
        lower, upper = np.zeros(n), np.ones(n)
        caps = 10**rng.uniform(-2, 8, n - 1)
        if seed % 2:
            lower[n // 2] = upper[n // 2] = 1
        for name, (_, numerical, _) in arms.items():
            fit = numerical.solve_quadratic(h, target, lower, upper, caps, lower)
            qualified[name] += int(fit.qualified)
            max_kkt[name] = max(max_kkt[name], float(fit.kkt_residual))
            if not fit.qualified:
                failures[name][fit.work.get("failure_reason", "certificate_gate")] += 1
    report["stress_summary"] = {
        name: {**summarize_counts(1000, qualified[name], failures[name]),
               "maximum_kkt_residual": finite(max_kkt[name])} for name in arms}

    rng = np.random.default_rng(882)
    qualified = Counter()
    failures = {name: Counter() for name in arms}
    for _ in range(80):
        n = 30
        h, target = 10**rng.uniform(-2, 9, n), rng.normal(.5, 2, n)
        lower, upper = np.zeros(n), np.where(rng.random(n) < .7, 1., .7)
        upper[-1] = 1
        caps = 10**rng.uniform(-2, 8, n - 1)
        for name, (_, numerical, _) in arms.items():
            try:
                profile = numerical.profile_quadratic_witnesses(h, target, lower, upper, caps)
            except ArithmeticError as exc:
                failures[name][str(exc)] += 1
            else:
                qualified[name] += int(profile.qualified)
                if not profile.qualified:
                    failures[name]["qualification_gate"] += 1
    report["profile_summary"] = {
        name: summarize_counts(80, qualified[name], failures[name]) for name in arms}
    report["source_unchanged"] = {
        name: provenance.source_provenance()["source_sha256"] == source["source_sha256"]
        for name, (provenance, _, source) in arms.items()}
    report["elapsed_seconds"] = perf_counter() - started
    passed = (all(report["source_unchanged"].values()) and
              all(report[field]["revised"]["failed"] == 0 for field in
                  ("production_summary", "stress_summary", "profile_summary")))
    report["status"] = "qualified" if passed else "unresolved"
    with (outdir / "replay.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "production_summary", "stress_summary", "profile_summary",
                       "source_unchanged", "elapsed_seconds")}, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parent / "fixtures" / "df44e6a_failed_qps")
    args = parser.parse_args()
    try:
        return run(args.outdir, args.fixtures, args.baseline_package)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
