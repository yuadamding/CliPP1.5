"""Source-bound component parity and paired timings against an archived package.

Run with ``--baseline-package /path/to/ff5b5c3/clipp1d --outdir fresh-directory``.
The fixed inventory contains 200 moderate and 80 wide-range common QPs, 3,000
signed interval scans, 200 anchored likelihood audits, and paired timings at
100, 1,000, and 4,000 mutations.
This component experiment does not measure end-to-end fitting throughput.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

# Establish thread controls before importing either package or NumPy/SciPy.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
for thread_variable in THREAD_VARIABLES:
    os.environ[thread_variable] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d import api, solver  # noqa: E402
from clipp1d.intervals import interval_descent  # noqa: E402
from clipp1d.policy import Policy  # noqa: E402
from clipp1d.types import CountModel  # noqa: E402


def load_baseline(directory, expected_hash):
    directory = directory.resolve()
    if not (directory / "__init__.py").is_file():
        raise ValueError("--baseline-package must point to an archived clipp1d package directory")
    name = "_clipp1d_reuse_reference"
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    baseline_api = importlib.import_module(f"{name}.api")
    baseline_solver = importlib.import_module(f"{name}.solver")
    source = baseline_api.source_provenance()
    if expected_hash is not None and source["source_sha256"] != expected_hash:
        raise ValueError("Baseline package source does not match --baseline-sha256")
    return baseline_api, baseline_solver, source


def array_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def quadratic_cases():
    for scenario, count, seed in (("moderate", 200, 721), ("wide", 80, 882)):
        rng = np.random.default_rng(seed)
        for index in range(count):
            n = 2 + index % 29
            h = (10**rng.uniform(-2, 9, n) if scenario == "wide" else
                 np.exp(rng.uniform(-2, 6, n)))
            target = rng.normal(.5, 2, n)
            lower, upper = np.zeros(n), np.where(rng.random(n) < .7, 1., .7)
            upper[-1] = 1
            caps = (10**rng.uniform(-2, 8, n - 1) if scenario == "wide" else
                    np.exp(rng.uniform(-3, 5, n - 1)))
            if scenario == "moderate" and index % 5 == 0:
                lower[0] = upper[0] = .3
            yield scenario, index, (h, target, lower, upper, caps)


def interval_cases():
    special = np.array([0., -0., 1., -1., 2.**53, -2.**53, 2.**-53, -2.**-53])
    for adversarial in (False, True):
        rng = np.random.default_rng(62441 if adversarial else 234)
        for index in range(1500):
            n = 64 + index % 129
            x = np.repeat(rng.choice([.1, .5, .9], (n + 3) // 4), 4)[:n]
            lower, upper = np.where(rng.random(n) < .1, x, 0.), np.where(rng.random(n) < .1, x, 1.)
            if adversarial:
                left, right = rng.choice(special, (2, n))
                caps, tolerance = abs(rng.choice(special, n - 1)), [0., 2.**-53, 2e-5, .5][index % 4]
            else:
                left, right = rng.normal(0, 3, (2, n))
                caps, tolerance = rng.uniform(0, 5, n - 1), [0., 2e-5, .1][index % 3]
            yield adversarial, index, (x, left, right, lower, upper, caps, tolerance)


def same_interval(first, second):
    return (first == second and (first is None or
            (first[3].hex(), first[4].hex()) == (second[3].hex(), second[4].hex())))


def audit_cases():
    """One model/x context shared across its first and last occupied-one anchors."""
    rng = np.random.default_rng(8821)
    eps = 1e-6
    scenarios = ("smooth", "exact_upper_clipping", "upper_plateau", "lower_clipping")
    for index in range(100):
        n, scenario = 6 + index % 67, scenarios[index % 4]
        alt, ref = rng.integers(1, 100, (2, n)).astype(float)
        lower, upper = np.full(n, eps), np.where(rng.random(n) < .2, .8, 1.)
        upper[[0, -1]] = 1
        slopes = np.broadcast_to(np.array([.5, 2.]), (n, 2)).copy()
        if scenario == "smooth":
            slopes = rng.uniform(.2, .8, (n, 2))
            x = rng.uniform(.1, .7, n)
        elif scenario == "exact_upper_clipping":
            x = np.full(n, (1 - eps) / 2)
        elif scenario == "upper_plateau":
            x = np.full(n, .8)
        else:
            x = np.where(rng.random(n) < .5, eps / .5, eps)
        x = np.clip(x, lower, upper)
        x[[0, -1]] = 1
        priors = rng.uniform(.1, 1, (n, 2))
        priors /= priors.sum(axis=1)[:, None]
        caps = np.exp(rng.uniform(-3, 4, n - 1))
        dual = caps * rng.uniform(-1, 1, n - 1)
        if index % 20 == 0:
            alt.fill(0)
            ref.fill(0)
            caps.fill(0)
            dual.fill(0)
        model = CountModel(tuple(f"m{i}" for i in range(n)), alt, ref, lower, upper,
                           slopes, np.log(priors), np.ones_like(slopes, dtype=bool), eps)
        yield index, scenario, model, x, dual, caps


def compare_audits(baseline):
    records = []
    policy = Policy()
    for index, scenario, model, x, dual, caps in audit_cases():
        context = solver.prepare_audit(model, x)
        original_left, original_right = baseline.one_sided_derivatives(model, x)
        derivatives_equal = (np.array_equal(original_left, context.left) and
                             np.array_equal(original_right, context.right))
        model_arrays = tuple(getattr(model, name) for name in
                             ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid"))
        for anchor in (0, len(x) - 1):
            lower, upper = model.lower.copy(), model.upper.copy()
            lower[anchor] = upper[anchor] = 1
            force = index % 3 == 0
            first_stationarity = baseline.stationarity(model, x, dual, lower, upper, caps)
            second_stationarity = solver.stationarity(model, x, dual, lower, upper, caps, context=context)
            first_kink, first_restart = baseline._kink_check(model, x, caps, lower, upper, policy,
                                                           force_intervals=force)
            second_kink, second_restart = solver._kink_check(model, x, caps, lower, upper, policy,
                                                            force_intervals=force, context=context)
            first_direction = baseline.interval_descent(x, original_left, original_right, lower, upper,
                                                        caps, policy.stationarity_tol)
            second_direction = interval_descent(x, context.left, context.right, lower, upper,
                                                caps, policy.stationarity_tol)
            first_qualified = bool(first_stationarity[1] and first_stationarity[0] <= policy.stationarity_tol and
                                   first_kink and first_restart is None)
            second_qualified = bool(second_stationarity[1] and second_stationarity[0] <= policy.stationarity_tol and
                                    second_kink and second_restart is None)
            records.append(dict(
                model_index=index, scenario=scenario, mutations=len(x), anchor=anchor, force_intervals=force,
                input_sha256=array_hash((*model_arrays, x, dual, caps, lower, upper)),
                same_one_sided_derivatives=derivatives_equal,
                same_stationarity=(first_stationarity == second_stationarity and
                                   first_stationarity[0].hex() == second_stationarity[0].hex()),
                same_kink_decision=first_kink == second_kink,
                same_restart=(first_restart is None and second_restart is None) or
                             (first_restart is not None and second_restart is not None and
                              np.array_equal(first_restart, second_restart)),
                same_interval=same_interval(first_direction, second_direction),
                same_qualification=first_qualified == second_qualified,
                baseline_qualified=first_qualified, revised_qualified=second_qualified,
                restart_proposed=second_restart is not None))
    return records


def profile_comparison(first, second, upper):
    eligible = upper == 1
    return dict(
        baseline_qualified=bool(first.qualified), revised_qualified=bool(second.qualified),
        same_witness=first.witness == second.witness,
        same_witness_values=bool(np.array_equal(first.relative_objectives, second.relative_objectives)),
        same_qualification=first.qualified == second.qualified and first.fit.qualified == second.fit.qualified,
        witness=int(second.witness),
        witness_value_max_error=float(np.max(abs(first.relative_objectives[eligible] - second.relative_objectives[eligible]))),
        primal_max_error=float(np.max(abs(first.fit.x - second.fit.x))),
        dual_max_error=float(np.max(abs(first.fit.dual - second.fit.dual))) if first.fit.dual.size else 0.,
        primal_close=bool(np.allclose(first.fit.x, second.fit.x, atol=2e-12, rtol=3e-12)),
        dual_close=bool(np.allclose(first.fit.dual, second.fit.dual, atol=1e-7, rtol=3e-12)))


def paired_times(functions, args, repeats):
    # One unmeasured warmup per arm, then alternate order across paired repeats.
    for function in functions.values():
        function(*args)
    samples, results = {name: [] for name in functions}, {}
    for index in range(repeats):
        order = ("baseline", "revised") if index % 2 == 0 else ("revised", "baseline")
        for name in order:
            started = perf_counter()
            results[name] = functions[name](*args)
            samples[name].append(perf_counter() - started)
    summary = {name: dict(seconds=times, median_seconds=float(np.median(times)),
                          minimum_seconds=min(times), maximum_seconds=max(times))
               for name, times in samples.items()}
    summary["median_speed_ratio"] = summary["baseline"]["median_seconds"] / summary["revised"]["median_seconds"]
    return summary, results


def run(args):
    if args.outdir.exists():
        raise FileExistsError("Output directory must be fresh; existing evidence is never overwritten")
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("This controlled benchmark requires Linux process CPU affinity")
    allowed = sorted(os.sched_getaffinity(0))
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        raise ValueError("Requested CPU is outside the process's allowed affinity")
    os.sched_setaffinity(0, {cpu})
    baseline_api, baseline, baseline_source = load_baseline(args.baseline_package, args.baseline_sha256)
    source = api.source_provenance()
    args.outdir.mkdir(parents=True, exist_ok=False)
    report = dict(schema="clipp1d.reuse_comparison.v1", started_utc=datetime.now(timezone.utc).isoformat(),
                  component_scope="fixed common-QP reconstruction and signed interval scan; no full fits",
                  source=source, baseline_source=baseline_source,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), initially_allowed_cpus=allowed,
                  thread_environment={name: os.environ[name] for name in THREAD_VARIABLES},
                  warmups_per_arm=1, alternating_paired_repeats=args.repeats,
                  profile_inventory=dict(moderate_count=200, moderate_seed=721, wide_count=80, wide_seed=882),
                  interval_inventory=dict(random_count=1500, random_seed=234, adversarial_count=1500,
                                          adversarial_seed=62441, equality="full tuple plus float.hex diagnostics"),
                  audit_inventory=dict(models=100, decisions=200, seed=8821, anchors="first and last occupied ones"),
                  quadratic_cases=[], interval_mismatches=[], component_timings=[])
    for scenario, index, arrays in quadratic_cases():
        first = baseline.profile_quadratic_witnesses(*arrays)
        second = solver.profile_quadratic_witnesses(*arrays)
        report["quadratic_cases"].append(dict(scenario=scenario, index=index, input_sha256=array_hash(arrays),
                                               **profile_comparison(first, second, arrays[3])))
    interval_digest = hashlib.sha256()
    for adversarial, index, arrays in interval_cases():
        interval_digest.update(array_hash(arrays[:-1]).encode())
        interval_digest.update(float(arrays[-1]).hex().encode())
        first, second = baseline.interval_descent(*arrays), interval_descent(*arrays)
        if not same_interval(first, second):
            report["interval_mismatches"].append(dict(adversarial=adversarial, index=index,
                                                      baseline=first, revised=second))
    report["interval_inputs_sha256"] = interval_digest.hexdigest()
    report["audit_cases"] = compare_audits(baseline)
    for n in (100, 1000, 4000):
        rng = np.random.default_rng(541 + n)
        h, target = rng.uniform(.2, 12, n), rng.normal(.5, 1, n)
        lower, upper, caps = np.zeros(n), np.ones(n), rng.uniform(.1, 5, n - 1)
        arrays = h, target, lower, upper, caps
        times, results = paired_times({"baseline": baseline.profile_quadratic_witnesses,
                                      "revised": solver.profile_quadratic_witnesses}, arrays, args.repeats)
        report["component_timings"].append(dict(component="selected_profile_reconstruction", mutations=n,
                                                input_sha256=array_hash(arrays), **times,
                                                parity=profile_comparison(results["baseline"], results["revised"], upper)))
        for scenario in ("long_fused", "singletons", "mixed_runs"):
            x = np.full(n, .5) if scenario == "long_fused" else np.linspace(.1, .9, n)
            if scenario == "mixed_runs":
                x = np.repeat(np.linspace(.1, .9, (n + 15) // 16), 16)[:n]
            left, right = rng.normal(size=(2, n))
            edge_caps = rng.uniform(0, 1, n - 1)
            arrays = x, left, right, lower, upper, edge_caps, 2e-5
            times, results = paired_times({"baseline": baseline.interval_descent,
                                          "revised": interval_descent}, arrays, args.repeats)
            report["component_timings"].append(dict(component="signed_interval_scan", scenario=scenario,
                                                    mutations=n, input_sha256=array_hash(arrays[:-1]),
                                                    tolerance=2e-5, **times,
                                                    exact_parity=same_interval(results["baseline"], results["revised"])))
    report["source_unchanged"] = source["source_sha256"] == api.source_provenance()["source_sha256"]
    report["baseline_source_unchanged"] = baseline_source["source_sha256"] == baseline_api.source_provenance()["source_sha256"]
    profiles = report["quadratic_cases"] + [row["parity"] for row in report["component_timings"] if "parity" in row]
    passed = (report["source_unchanged"] and report["baseline_source_unchanged"] and
              not report["interval_mismatches"] and
              all(row["same_witness"] and row["same_witness_values"] and row["same_qualification"] and row["primal_close"] and
                  row["dual_close"] for row in profiles) and
              all(all(row[key] for key in ("same_one_sided_derivatives", "same_stationarity", "same_kink_decision",
                                          "same_restart", "same_interval", "same_qualification"))
                  for row in report["audit_cases"]) and
              all(row.get("exact_parity", True) for row in report["component_timings"]))
    report["status"] = "equivalent" if passed else "unresolved"
    with (args.outdir / "comparison.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=report["status"], source_sha256=source["source_sha256"],
                          quadratic_cases=len(report["quadratic_cases"]),
                          baseline_qualified=sum(row["baseline_qualified"] for row in report["quadratic_cases"]),
                          revised_qualified=sum(row["revised_qualified"] for row in report["quadratic_cases"]),
                          interval_cases=3000, interval_mismatches=len(report["interval_mismatches"]),
                          audit_cases=len(report["audit_cases"]),
                          audit_mismatches=sum(not all(value for key, value in row.items() if key.startswith("same_"))
                                               for row in report["audit_cases"]),
                          timings=[{key: value for key, value in row.items() if key in
                                    ("component", "scenario", "mutations", "median_speed_ratio")}
                                   for row in report["component_timings"]]), indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", required=True, type=Path)
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("At least three alternating paired repeats are required")
    try:
        return run(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
