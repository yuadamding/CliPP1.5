"""Separate topology kernels, three QP arms, and common-surrogate witness profiling.

Complete-graph code here is an attribution reference, not CliPP2's production
solver. No result from this script is an end-to-end CliPP2 speedup measurement.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d.api import source_provenance  # noqa: E402
from clipp1d.chain import adjoint, difference  # noqa: E402
from clipp1d.policy import Policy  # noqa: E402
from clipp1d.report import write_json  # noqa: E402
from clipp1d.solver import profile_quadratic_witnesses, solve_quadratic  # noqa: E402


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
    normal = (h[free] * (z - target[free]) + a[free]) * delta
    edge = caps * abs(jumps) - q * jumps
    # The certificate is for feasible iterates; this arm's proximal updates ensure it.
    margin = 128 * np.finfo(float).eps * (1 + abs(h[free] * (z - target[free])) + abs(a[free])) * abs(delta)
    if np.any(normal < -margin) or np.any(edge < -128 * np.finfo(float).eps * (1 + caps) * abs(jumps)):
        return np.inf, 0., np.inf
    gap = float(np.sum(.5 * h[free] * delta**2) + np.sum(np.maximum(normal, 0)) + np.sum(np.maximum(edge, 0)))
    r = np.clip(target[free], lower[free], upper[free])
    d = x[free] - r
    scale = float(np.sum(.5 * h[free] * d**2 + h[free] * (r - target[free]) * d) +
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
    return x, dict(seconds=perf_counter() - started, qualified=qualified, gap=gap, gap_scale=scale,
                   kkt_residual=kkt, iterations=iteration, edges=len(caps))


def kernel(n, complete):
    ops = operators(n, complete)
    diff, adj, left, _, _ = ops
    rng = np.random.default_rng(17)
    x, q = rng.normal(size=n), np.zeros(len(left))
    repetitions = 10 if complete else 1000
    samples = []
    for _ in range(7):
        started = perf_counter()
        for _ in range(repetitions):
            q = np.clip(q + .1 * diff(x), -.5, .5)
            adj(q)
        samples.append((perf_counter() - started) / repetitions)
    return dict(mutations=n, topology="complete" if complete else "chain", edges=len(q),
                fp64_edge_array_bytes=q.nbytes, median_seconds=float(np.median(samples)), batches=7,
                scope="graph construction excluded; temporary operator allocations included")


def qp_case(n, seconds):
    rng = np.random.default_rng(71)
    h, target = rng.uniform(.5, 5, n), rng.normal(.5, .3, n)
    lower, upper = np.zeros(n), np.ones(n)
    lower[n // 2] = upper[n // 2] = 1
    rows = []
    chain_solution = None
    for complete in (True, False):
        ops = operators(n, complete)
        caps = np.full(len(ops[2]), .5)
        x, metrics = first_order(h, target, lower, upper, caps, ops, seconds, Policy())
        rows.append(dict(arm="complete_common_first_order" if complete else "chain_common_first_order", **metrics))
        if not complete:
            chain_solution = x
    started = perf_counter()
    direct = solve_quadratic(h, target, lower, upper, np.full(n - 1, .5), lower)
    rows.append(dict(arm="chain_direct_tv", seconds=perf_counter() - started, qualified=direct.qualified,
                     gap=direct.gap, gap_scale=direct.gap_scale, kkt_residual=direct.kkt_residual,
                     passes=direct.iterations, work=direct.work, edges=n - 1,
                     max_ccf_difference_from_first_order=float(np.max(abs(direct.x - chain_solution)))))
    digest = hashlib.sha256()
    for a in (h, target, lower, upper):
        digest.update(a.tobytes())
    return dict(mutations=n, unary_box_sha256=digest.hexdigest(), arms=rows,
                scope="same unaries/boxes/FP64/CPU; topology changes the TV objective; common tolerances")


def profile_case(n):
    rng = np.random.default_rng(92)
    h, target, caps = rng.uniform(.5, 5, n), rng.normal(.5, .3, n), rng.uniform(.1, 2, n - 1)
    lower, upper = np.zeros(n), np.ones(n)
    started = perf_counter()
    shared = profile_quadratic_witnesses(h, target, lower, upper, caps)
    shared_seconds = perf_counter() - started
    reference = np.clip(target, lower, upper)
    started = perf_counter()
    error = 0.
    failures = 0
    for k in range(n):
        branch_lower = lower.copy()
        branch_lower[k] = 1
        result = solve_quadratic(h, target, branch_lower, upper, caps, lower)
        failures += int(not result.qualified)
        delta = result.x - reference
        value = np.sum(.5 * h * delta**2 + h * (reference - target) * delta) + np.dot(caps, abs(np.diff(result.x)))
        error = max(error, abs(value - shared.relative_objectives[k]))
    return dict(mutations=n, surrogate_sha256=shared.surrogate_sha256,
                shared_seconds=shared_seconds, independent_seconds=perf_counter() - started,
                shared_qualified=shared.qualified, independent_unqualified=failures,
                maximum_profile_value_error=error, selected_witness=shared.witness,
                scope="one fixed common quadratic only; not a nonconvex full-fit speedup")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[500, 1000, 2000, 4000])
    parser.add_argument("--first-order-seconds", type=float, default=10.)
    args = parser.parse_args()
    if any(n < 2 for n in args.sizes + args.kernel_sizes) or args.first_order_seconds <= 0:
        parser.error("sizes must be >=2 and time limit must be positive")
    args.outdir.mkdir(parents=True, exist_ok=False)
    setup = dict(provenance=source_provenance(), policy=asdict(Policy()),
                 benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.outdir / "setup.json", setup)
    for n in args.kernel_sizes:
        record = dict(schema="clipp1d.kernel.v1", arms=[kernel(n, True), kernel(n, False)])
        write_json(args.outdir / f"kernel-{n}.json", record)
        print(f"kernel M={n}: measured", flush=True)
    for n in args.sizes:
        record = qp_case(n, args.first_order_seconds)
        write_json(args.outdir / f"quadratic-{n}.json", record)
        print(json.dumps(record), flush=True)
        record = profile_case(n)
        write_json(args.outdir / f"profile-{n}.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
