"""Controlled graph kernels, qualified QPs, and fixed-surrogate witness profiles.

The complete-graph arm is a NumPy attribution reference, not CliPP2's production
solver. CPU affinity and thread limits do not imply exclusive host resources.
"""

import argparse
import ast
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter
import tracemalloc

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
REGIMES = {
    "weak_uniform": (.02, False),
    "moderate_heterogeneous": (.5, True),
    "strong_heterogeneous": (5., True),
}


def configure_execution(cpu_count=1, cpus=None, threads=1):
    """Set controls before importing numerical libraries, using allowed CPUs only."""
    if "numpy" in sys.modules:
        raise RuntimeError("Resource controls must be set before NumPy is imported")
    if cpu_count < 1 or threads < 1:
        raise ValueError("CPU and thread counts must be positive")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    selected = None
    if allowed is not None:
        selected = sorted(set(cpus)) if cpus is not None else allowed[:cpu_count]
        if not selected or not set(selected) <= set(allowed):
            raise ValueError("Requested CPUs must be a nonempty subset of allowed CPUs")
        if cpus is None and cpu_count > len(allowed):
            raise ValueError("Requested CPU count exceeds the allowed affinity")
        os.sched_setaffinity(0, selected)
        selected = sorted(os.sched_getaffinity(0))
    elif cpus is not None:
        raise ValueError("Explicit affinity is unsupported on this platform")
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    return {
        "allowed_cpus_before": allowed,
        "selected_cpus": selected,
        "requested_cpu_count": cpu_count,
        "thread_environment": {name: os.environ[name] for name in THREAD_ENVIRONMENT},
        "controls_set_before_numpy_import": True,
        "affinity_supported": allowed is not None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus_reported": os.cpu_count(),
        "load_average_at_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "resource_scope": "Process affinity and numerical thread limits; shared host, no exclusivity claim",
    }



def require_historical_cpu_package(package):
    """Reject CUDA-only source before a historical CPU fit runner creates work.

    This checks the selected package, never switches implementations. Frozen
    historical packages remain runnable under their original CPU contract.
    """
    package = Path(package).resolve()
    path = package / "api.py"
    if not path.is_file():
        raise ValueError(f"Historical CPU runner requires a frozen package with api.py: {package}")
    tree = ast.parse(path.read_text(), filename=str(path))
    exported = {alias.asname or alias.name for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    exported.update(node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    cuda_entry = any(isinstance(node, ast.ImportFrom) and node.module == "cuda_api"
                     for node in ast.walk(tree))
    if cuda_entry or not {"fit", "compute_pilot", "build_chain"} <= exported:
        raise ValueError("Historical CPU runner cannot launch the current CUDA-only complete-graph "
                         "package. Use a pinned historical chain package for this runner; "
                         "current inference requires the CUDA entry point. No CPU fallback is available.")


def load_numerics():
    # The CLI calls configure_execution first. Importing this file in tests has no
    # affinity/thread side effects and does not itself import numerical modules.
    global np, source_provenance, difference, adjoint, Policy, write_json
    global profile_quadratic_witnesses, solve_quadratic
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from clipp1d.api import source_provenance
    from clipp1d.chain import adjoint, difference
    from clipp1d.policy import Policy
    from clipp1d.report import write_json
    from clipp1d.solver import profile_quadratic_witnesses, solve_quadratic


def timing_summary(samples):
    values = np.asarray(samples, dtype=float)
    return dict(samples_seconds=values.tolist(), repetitions=len(values),
                median_seconds=float(np.median(values)), minimum_seconds=float(np.min(values)),
                maximum_seconds=float(np.max(values)), p10_seconds=float(np.percentile(values, 10)),
                p90_seconds=float(np.percentile(values, 90)),
                population_standard_deviation_seconds=float(np.std(values)))


def array_sha256(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(str(a.shape).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def peak_rss_bytes():
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


def current_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def measured_memory(call):
    """Separate untimed allocation pass; existing input arrays are excluded."""
    gc.collect()
    before_peak, before_rss = peak_rss_bytes(), current_rss_bytes()
    if tracemalloc.is_tracing():
        raise RuntimeError("Memory phase requires ownership of a fresh tracemalloc session")
    tracemalloc.start()
    started = perf_counter()
    try:
        result = call()
        live, peak = tracemalloc.get_traced_memory()
        elapsed = perf_counter() - started
    finally:
        tracemalloc.stop()
    after_peak, after_rss = peak_rss_bytes(), current_rss_bytes()
    return result, dict(
        phase="separate additional solve; excluded from timing samples",
        phase_seconds=elapsed,
        traced_new_peak_bytes=peak,
        traced_new_live_bytes_at_return=live,
        process_current_rss_bytes_before=before_rss,
        process_current_rss_bytes_after=after_rss,
        process_lifetime_peak_rss_bytes_before=before_peak,
        process_lifetime_peak_rss_bytes_after=after_peak,
        process_high_water_increase_bytes=after_peak - before_peak,
        scope=("Traced new allocations include Python heaps/knot dictionaries, reconstruction, "
               "audit working arrays and returned solution. Preexisting inputs and graph arrays "
               "are excluded and reported separately. Native allocations not registered with "
               "tracemalloc may be absent. RSS is a whole-process lifetime high-water bound, "
               "not an isolated solver peak; zero high-water increase does not mean zero allocation."),
    )


def operators(n, complete):
    if not complete:
        return difference, adjoint, np.arange(n - 1), np.arange(1, n), 4.
    left, right = np.triu_indices(n, 1)

    def diff(x):
        return x[right] - x[left]

    def adj(q):
        return np.bincount(right, weights=q, minlength=n) - np.bincount(left, weights=q, minlength=n)

    return diff, adj, left, right, float(n)


def reference_certificate(x, q, h, target, lower, upper, caps, ops):
    diff, adj, left, right, _ = ops
    free = lower < upper
    a, jumps = adj(q), diff(x)
    z = np.clip(target[free] - a[free] / h[free], lower[free], upper[free])
    delta = x[free] - z
    unconstrained = target[free] - a[free] / h[free]
    normal_gradient = np.where(unconstrained < lower[free], h[free] * (lower[free] - target[free]) + a[free],
                               np.where(unconstrained > upper[free], h[free] * (upper[free] - target[free]) + a[free], 0.))
    normal = normal_gradient * delta
    edge = caps * abs(jumps) - q * jumps
    margin = 128 * np.finfo(float).eps * (1 + abs(h[free] * (z - target[free])) + abs(a[free])) * abs(delta)
    if np.any(normal < -margin) or np.any(edge < -128 * np.finfo(float).eps * (1 + caps) * abs(jumps)):
        return np.inf, 0., np.inf
    gap = float(np.sum(.5 * h[free] * delta**2) + np.sum(np.maximum(normal, 0)) + np.sum(np.maximum(edge, 0)))
    r = np.clip(target[free], lower[free], upper[free])
    d = x[free] - r
    scale = float(np.sum(np.maximum(.5 * h[free] * d**2 + h[free] * (r - target[free]) * d, 0)) +
                  np.dot(caps[free[left] | free[right]], abs(jumps[free[left] | free[right]])))
    g = np.zeros(len(x))
    g[free] = h[free] * (x[free] - target[free])
    residual = g + a
    residual = np.where(x == lower, np.minimum(residual, 0), residual)
    residual = np.where(x == upper, np.maximum(residual, 0), residual)
    residual[~free] = 0
    kkt = max(float(np.max(abs(residual) / (1 + abs(g) + abs(a)))),
              float(np.max(abs(q - np.clip(q + jumps, -caps, caps)) / (1 + caps))))
    return gap, scale, kkt


def first_order(h, target, lower, upper, caps, ops, seconds, policy):
    diff, adj, _, _, norm_squared = ops
    free = lower < upper
    mu = np.median(h[free])
    tau, sigma = .49 / mu, .49 * mu * 4 / norm_squared
    x, q = np.clip(target, lower, upper), np.zeros(len(caps))
    extrapolated = x.copy()
    started = perf_counter()
    qualified = False
    for iteration in range(1, policy.inner_max_iterations + 1):
        q = np.clip(q + sigma * diff(extrapolated), -caps, caps)
        previous = x
        x = lower.copy()
        x[free] = np.clip((previous[free] - tau * adj(q)[free] + tau * h[free] * target[free]) /
                          (1 + tau * h[free]), lower[free], upper[free])
        extrapolated = 2 * x - previous
        if iteration % 10 == 0:
            gap, scale, kkt = reference_certificate(x, q, h, target, lower, upper, caps, ops)
            qualified = bool(np.isfinite(gap) and gap <= policy.inner_atol + policy.inner_rtol * scale and kkt <= policy.inner_kkt_tol)
            if qualified or perf_counter() - started >= seconds:
                break
    gap, scale, kkt = reference_certificate(x, q, h, target, lower, upper, caps, ops)
    qualified = bool(np.isfinite(gap) and gap <= policy.inner_atol + policy.inner_rtol * scale and kkt <= policy.inner_kkt_tol)
    elapsed = perf_counter() - started
    status = "qualified" if qualified else ("time_limit" if elapsed >= seconds else "iteration_limit")
    return x, dict(seconds=elapsed, qualified=qualified, gap=gap, gap_scale=scale,
                   kkt_residual=kkt, iterations=iteration, edges=len(caps), status=status,
                   requested_seconds_limit=seconds)


def kernel(n, complete, repetitions=5):
    ops = operators(n, complete)
    diff, adj, left, _, _ = ops
    rng = np.random.default_rng(17)
    x, initial_q = rng.normal(size=n), np.zeros(len(left))
    operations_per_sample = max(1, min(1000, 100_000 // max(len(left), 1)))
    adj(np.clip(initial_q + .1 * diff(x), -.5, .5))  # One untimed warmup.
    samples = []
    for _ in range(repetitions):
        q = initial_q.copy()
        started = perf_counter()
        for _ in range(operations_per_sample):
            q = np.clip(initial_q + .1 * diff(x), -.5, .5)
            adj(q)
        samples.append((perf_counter() - started) / operations_per_sample)
    return dict(mutations=n, topology="complete" if complete else "chain", edges=len(initial_q),
                fp64_edge_array_bytes=initial_q.nbytes, edge_index_array_bytes=ops[2].nbytes + ops[3].nbytes,
                **timing_summary(samples), operations_per_sample=operations_per_sample,
                scope="Fixed-input diff/clip/adjoint operation; graph construction and initial q copy excluded; operator allocations included")


def penalty_caps(n, ops, regime, strength_mode):
    strength, heterogeneous = REGIMES[regime]
    left, right = ops[2:4]
    shape = .25 + 1.5 * ((17 * left + 31 * right) % 13) / 12 if heterogeneous else np.ones(len(left))
    shape = shape / np.mean(shape)
    caps = strength * shape
    if strength_mode == "total_matched":
        caps *= (n - 1) / len(caps)
    elif strength_mode != "per_edge":
        raise ValueError("Unknown strength mode")
    return caps


def relative_objective(x, h, target, lower, upper, caps, ops):
    reference = np.clip(target, lower, upper)
    delta = x - reference
    return float(np.sum(.5 * h * delta**2 + h * (reference - target) * delta) + np.dot(caps, abs(ops[0](x))))


def qp_case(n, seconds, regime="moderate_heterogeneous", strength_mode="per_edge", repetitions=3, memory=True):
    rng = np.random.default_rng(71)
    h, target = rng.uniform(.5, 5, n), rng.normal(.5, .3, n)
    lower, upper = np.zeros(n), np.ones(n)
    lower[n // 2] = upper[n // 2] = 1
    rows, chain_solution, chain_qualified = [], None, False
    for arm, complete in (("complete_common_first_order", True), ("chain_common_first_order", False), ("chain_direct_tv", False)):
        ops = operators(n, complete)
        caps = penalty_caps(n, ops, regime, strength_mode)

        def call():
            if arm != "chain_direct_tv":
                return first_order(h, target, lower, upper, caps, ops, seconds, Policy())
            started = perf_counter()
            fit = solve_quadratic(h, target, lower, upper, caps, lower)
            return fit.x, dict(seconds=perf_counter() - started, qualified=fit.qualified,
                               gap=fit.gap, gap_scale=fit.gap_scale, kkt_residual=fit.kkt_residual,
                               passes=fit.iterations, work=fit.work, edges=len(caps),
                               status="qualified" if fit.qualified else "numerically_unresolved")

        samples = []
        for _ in range(repetitions):
            x, metric = call()
            metric["relative_objective"] = relative_objective(x, h, target, lower, upper, caps, ops)
            samples.append(metric)
        memory_record = None
        if memory:
            (_, memory_metric), memory_record = measured_memory(call)
            memory_record["solve_qualified"] = memory_metric["qualified"]
            memory_record["solve_status"] = memory_metric["status"]
        timing = timing_summary([s["seconds"] for s in samples])
        row = dict(arm=arm, seconds=timing["median_seconds"], timing=timing, samples=samples,
                   qualified=all(s["qualified"] for s in samples),
                   qualified_samples=sum(s["qualified"] for s in samples),
                   unqualified_samples=sum(not s["qualified"] for s in samples),
                   edges=len(caps), caps_sum=float(np.sum(caps)), caps_min=float(np.min(caps)),
                   caps_max=float(np.max(caps)), caps_sha256=array_sha256(caps),
                   surrogate_sha256=array_sha256(h, target, lower, upper, caps),
                   existing_input_array_bytes=sum(a.nbytes for a in (h, target, lower, upper, caps)),
                   benchmark_operator_index_array_bytes=ops[2].nbytes + ops[3].nbytes,
                   measured_solver_memory=memory_record)
        if arm == "chain_common_first_order":
            chain_solution, chain_qualified = x.copy(), samples[-1]["qualified"]
        if arm == "chain_direct_tv":
            row["comparison_to_last_chain_first_order"] = dict(
                both_qualified=bool(samples[-1]["qualified"] and chain_qualified),
                max_coordinate_difference=float(np.max(abs(x - chain_solution))),
                relative_objective_difference=abs(samples[-1]["relative_objective"] -
                    relative_objective(chain_solution, h, target, lower, upper, caps, ops)))
        rows.append(row)
    return dict(schema="clipp1d.quadratic_benchmark.v2", mutations=n, regime=regime,
                strength_mode=strength_mode, nominal_mean_per_edge_cap=REGIMES[regime][0],
                unary_box_sha256=array_sha256(h, target, lower, upper), arms=rows,
                strength_scope=("Equal mean per-edge cap; complete total cap is M/2 times chain total cap"
                    if strength_mode == "per_edge" else
                    "Equal summed caps in both graphs; topology and individual edge strengths differ"),
                scope="Same unaries/boxes/FP64/CPU and qualification thresholds; different graph objectives, not equal optimization difficulty")


def profile_case(n, repetitions=3, seconds=30.):
    rng = np.random.default_rng(92)
    h, target, caps = rng.uniform(.5, 5, n), rng.normal(.5, .3, n), rng.uniform(.1, 2, n - 1)
    lower, upper = np.zeros(n), np.ones(n)
    reference = np.clip(target, lower, upper)
    samples = []
    for _ in range(repetitions):
        started = perf_counter()
        shared = profile_quadratic_witnesses(h, target, lower, upper, caps)
        shared_seconds = perf_counter() - started
        started = perf_counter()
        error, failures, completed = 0., 0, 0
        for k in range(n):
            branch_lower = lower.copy()
            branch_lower[k] = 1
            result = solve_quadratic(h, target, branch_lower, upper, caps, lower)
            failures += int(not result.qualified)
            delta = result.x - reference
            value = np.sum(.5 * h * delta**2 + h * (reference - target) * delta) + np.dot(caps, abs(np.diff(result.x)))
            error = max(error, abs(value - shared.relative_objectives[k]))
            completed += 1
            if perf_counter() - started >= seconds:
                break
        samples.append(dict(shared_seconds=shared_seconds, independent_seconds=perf_counter() - started,
                            shared_qualified=shared.qualified, independent_unqualified=failures,
                            independent_completed=completed, independent_total=n,
                            independent_status="complete" if completed == n else "time_limit",
                            maximum_profile_value_error=error, selected_witness=shared.witness))
    return dict(schema="clipp1d.profile_benchmark.v2", mutations=n,
                surrogate_sha256=shared.surrogate_sha256, samples=samples,
                shared_timing=timing_summary([s["shared_seconds"] for s in samples]),
                independent_timing=timing_summary([s["independent_seconds"] for s in samples]),
                independent_seconds_limit_per_sample=seconds,
                all_witnesses_qualified=all(s["shared_qualified"] and s["independent_completed"] == n and
                                            not s["independent_unqualified"] for s in samples),
                scope="One fixed common quadratic only; partial enumeration times are not complete speedup evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--sizes", nargs="*", type=int, default=[100, 1000])
    parser.add_argument("--kernel-sizes", nargs="*", type=int, default=[500, 1000, 2000])
    parser.add_argument("--profile-sizes", nargs="*", type=int, default=[100])
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--strength-modes", nargs="+", choices=["per_edge", "total_matched"], default=["per_edge", "total_matched"])
    parser.add_argument("--first-order-seconds", type=float, default=2.)
    parser.add_argument("--profile-seconds", type=float, default=30.)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--kernel-repetitions", type=int, default=5)
    parser.add_argument("--cpu-count", type=int, default=1)
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-memory", action="store_true", help="Skip the separate extra traced solver pass")
    args = parser.parse_args()
    if any(n < 2 for n in args.sizes + args.kernel_sizes + args.profile_sizes):
        parser.error("sizes must be >=2")
    if min(args.first_order_seconds, args.profile_seconds, args.repetitions, args.kernel_repetitions) <= 0:
        parser.error("time limits and repetition counts must be positive")
    try:
        execution = configure_execution(args.cpu_count, args.cpus, args.threads)
    except ValueError as exc:
        parser.error(str(exc))
    load_numerics()
    args.outdir.mkdir(parents=True, exist_ok=False)
    setup = dict(schema="clipp1d.chain_benchmark_setup.v2", provenance=source_provenance(),
                 policy=asdict(Policy()), execution=execution,
                 parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 timing_scope="Tracemalloc disabled during timing samples; memory measured in separate additional solves")
    write_json(args.outdir / "setup.json", setup)
    for n in args.kernel_sizes:
        record = dict(schema="clipp1d.kernel.v2", arms=[kernel(n, True, args.kernel_repetitions), kernel(n, False, args.kernel_repetitions)])
        write_json(args.outdir / f"kernel-{n}.json", record)
        print(f"kernel M={n}: timing samples saved", flush=True)
    for n in args.sizes:
        for regime in args.regimes:
            for mode in args.strength_modes:
                record = qp_case(n, args.first_order_seconds, regime, mode, args.repetitions, not args.no_memory)
                write_json(args.outdir / f"quadratic-{n}-{regime}-{mode}.json", record)
                print(json.dumps(dict(mutations=n, regime=regime, strength_mode=mode,
                      arms=[dict(arm=a["arm"], seconds=a["seconds"], qualified=a["qualified"]) for a in record["arms"]])), flush=True)
    for n in args.profile_sizes:
        record = profile_case(n, args.repetitions, args.profile_seconds)
        write_json(args.outdir / f"profile-{n}.json", record)
        print(f"profile M={n}: all witnesses qualified={record['all_witnesses_qualified']}", flush=True)


if __name__ == "__main__":
    main()
