"""Instrument an unchanged frozen package to diagnose outer-limit trajectories.

Instrumentation wraps function boundaries; it does not trace Python lines, alter
numeric arguments, raise budgets, change the 1e-3 restart cap, or replace results.
Timings are diagnostic and include wrapper overhead, not controlled speed claims.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from benchmark_chain import configure_execution, require_historical_cpu_package


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class Timers:
    """Nested exclusive timings keep posterior work separate from its caller."""

    def __init__(self):
        self.stack = []
        self.values = {}

    @contextmanager
    def measure(self, name):
        frame = [perf_counter(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            value = self.values.setdefault(name, dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
            value["calls"] += 1
            value["inclusive_seconds"] += elapsed
            value["exclusive_seconds"] += elapsed - frame[1]


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def worker(args):
    require_historical_cpu_package(args.frozen_package)
    execution = configure_execution(cpu_count=1, threads=1)
    global np
    import numpy as np
    frozen = args.frozen_package.resolve()
    sys.path.insert(0, str(frozen.parent))
    from clipp1d import api, model as likelihood, selection, solver
    raw_module = sys.modules[selection.fit_fixed_lambda.__module__]
    start_name = "solve_unconstrained" if hasattr(raw_module, "solve_unconstrained") else "solve_profiled"
    from clipp1d.policy import Policy
    if Path(solver.__file__).resolve().parent != frozen:
        raise RuntimeError("Diagnosis must import the requested frozen package")
    policy = Policy()
    if policy.outer_max_iterations != 150:
        raise ValueError("Expected unchanged outer budget of 150")
    provenance = api.source_provenance()
    metadata = dict(schema="clipp1d.outer_progress_diagnosis.v1", frozen_package=str(frozen),
                    expected_commit=args.source_commit, source=provenance,
                    input_path=str(args.input_file.resolve()), input_sha256=file_hash(args.input_file),
                    harness_sha256=file_hash(__file__), policy=asdict(policy), execution=execution,
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    instrumentation="Function wrappers, unchanged arrays/results; timings include wrapper overhead",
                    step_policy="Original interval backtracking starts at min(1e-3, feasible room)")
    write_json(args.outdir / "setup.json", metadata)
    originals = {name: getattr(solver, name) for name in
                 (start_name, "profile_quadratic_witnesses", "stationarity", "_kink_check",
                  "evaluate", "one_sided_derivatives", "interval_descent", "local_interval_delta",
                  "clipping_breakpoints")}
    original_raw = selection.fit_fixed_lambda
    state = dict(current=None, penalty_index=-1, start_index=0, audit_depth=0, kink=None)
    totals = Timers()
    starts = []

    def emit(event):
        with (args.outdir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(clean(event), sort_keys=True, allow_nan=False) + "\n")

    @contextmanager
    def audit_time(name):
        current = state["current"]
        if current is None:
            yield
            return
        with totals.measure(name), current["timers"].measure(name):
            yield

    def evaluated(*values, **kwargs):
        if state["audit_depth"]:
            with audit_time("likelihood_and_posterior"):
                return originals["evaluate"](*values, **kwargs)
        return originals["evaluate"](*values, **kwargs)

    def derivatives(*values, **kwargs):
        if state["audit_depth"]:
            with audit_time("one_sided_derivative_assembly"):
                return originals["one_sided_derivatives"](*values, **kwargs)
        return originals["one_sided_derivatives"](*values, **kwargs)

    def scan(*values, **kwargs):
        with audit_time("signed_interval_scan"):
            result = originals["interval_descent"](*values, **kwargs)
        if state["kink"] is not None:
            state["kink"]["direction"] = None if result is None else list(result)
        return result

    def local_delta(*values, **kwargs):
        with audit_time("finite_proposal_evaluation"):
            result = originals["local_interval_delta"](*values, **kwargs)
        if state["kink"] is not None:
            state["kink"]["proposal_count"] += 1
            state["kink"]["last_proposal_delta"] = float(result)
        return result

    def breakpoints(*values, **kwargs):
        if state["audit_depth"]:
            with audit_time("finite_breakpoint_setup"):
                return originals["clipping_breakpoints"](*values, **kwargs)
        return originals["clipping_breakpoints"](*values, **kwargs)

    def stationarity(*values, **kwargs):
        caller = inspect.currentframe().f_back.f_locals
        state["audit_depth"] += 1
        try:
            with audit_time("stationarity_other"):
                result = originals["stationarity"](*values, **kwargs)
        finally:
            state["audit_depth"] -= 1
        current = state["current"]
        if current is not None and "outer" in caller:
            point = dict(outer=int(caller["outer"]) + 1, objective=float(caller["current"]),
                         witness=None if caller["witness_index"] is None else int(caller["witness_index"]),
                         curvature_scale=float(caller["curvature_scale"]),
                         accepted_steps=int(caller["accepted"]), backtracks=int(caller["backtracks"]),
                         stationarity=float(result[0]), feasible=bool(result[1]),
                         max_abs_surrogate_step=float(np.max(np.abs(caller.get("step", 0.)))))
            # The final audit repeats the last accepted state unless a final
            # restart occurred; retain that changed state rather than discarding it.
            if not current["outer"] or point != current["outer"][-1]:
                current["outer"].append(point)
        return result

    def kink(model, x, caps, lower, upper, policy, **kwargs):
        caller = inspect.currentframe().f_back.f_locals
        current = state["current"]
        frame = dict(direction=None, proposal_count=0, last_proposal_delta=None)
        previous = state["kink"]
        state["kink"] = frame
        state["audit_depth"] += 1
        try:
            with audit_time("kink_other"):
                result = originals["_kink_check"](model, x, caps, lower, upper, policy, **kwargs)
        finally:
            state["audit_depth"] -= 1
            state["kink"] = previous
        if current is not None:
            current["audit_calls"] += 1
            current["audit_forced_calls"] += int(kwargs.get("force_intervals", False))
            current["local_proposal_count"] += frame["proposal_count"]
            if result[1] is not None:
                trial = result[1]
                changed = np.flatnonzero(trial != x)
                first, last = int(changed[0]), int(changed[-1]) + 1
                displacement = trial[changed] - x[changed]
                event = dict(outer=int(caller.get("outer", -1)) + 1,
                             anchor=int(np.flatnonzero(lower == upper)[0]) if np.any(lower == upper) else None,
                             forced=bool(kwargs.get("force_intervals", False)),
                             first=first, stop=last, changed_nodes=len(changed),
                             min_abs_step=float(np.min(np.abs(displacement))),
                             max_abs_step=float(np.max(np.abs(displacement))),
                             sign=int(np.sign(displacement[0])), direction=frame["direction"],
                             proposal_count=frame["proposal_count"],
                             objective_delta=frame["last_proposal_delta"],
                             objective_before=float(caller.get("current", np.nan)),
                             curvature_scale_before_reset=float(caller.get("curvature_scale", np.nan)))
                current["restarts"].append(event)
        return result

    def profile(h, target, lower, upper, caps, *values, **kwargs):
        caller = inspect.currentframe().f_back.f_locals
        begun = perf_counter()
        result = originals["profile_quadratic_witnesses"](h, target, lower, upper, caps, *values, **kwargs)
        current = state["current"]
        if current is not None:
            current["profiles"].append(dict(
                outer=int(caller["outer"]) + 1, attempt=int(caller["attempt"]) + 1,
                curvature_scale=float(caller["trial_scale"]), h_min=float(np.min(h)), h_max=float(np.max(h)),
                objective_before=float(caller["current"]), previous_witness=int(caller["witness_index"]),
                selected_witness=int(result.witness), qualified=bool(result.qualified),
                inner_gap=float(result.fit.gap), inner_kkt=float(result.fit.kkt_residual),
                seconds=perf_counter() - begun,
                max_abs_proposed_step=float(np.max(np.abs(result.fit.x - caller["x"])))))
        return result

    def start(model, chain, penalty, initial, *values, **kwargs):
        state["start_index"] += 1
        current = dict(penalty_index=state["penalty_index"], start_index=state["start_index"],
                       lambda_value=float(penalty), initial_sha256=array_hash(initial),
                       initial_witnesses=np.flatnonzero(initial == 1).tolist(),
                       outer=[], restarts=[], profiles=[], timers=Timers(),
                       audit_calls=0, audit_forced_calls=0, local_proposal_count=0)
        state["current"] = current
        begun = perf_counter()
        result = originals[start_name](model, chain, penalty, initial, *values, **kwargs)
        current["wall_seconds"] = perf_counter() - begun
        state["current"] = None
        current["timings"] = current.pop("timers").values
        current.update(qualified=result.qualified, diagnostics=result.diagnostics,
                       final_objective=result.objective, final_sha256=array_hash(result.x),
                       final_witness=None if result.witness is None else int(result.witness),
                       restart_count=len(current["restarts"]))
        if not result.qualified:
            fixture = args.outdir / f"unresolved-p{state['penalty_index']:02}-s{state['start_index']:02}.npz"
            with fixture.open("xb") as handle:
                np.savez_compressed(handle, initial=initial, final=result.x, final_dual=result.dual,
                                    chain_order=chain.order, chain_weights=chain.weights)
            current.update(fixture=fixture.name, fixture_sha256=file_hash(fixture))
        detail = args.outdir / f"start-p{state['penalty_index']:02}-s{state['start_index']:02}.json"
        write_json(detail, current)
        summary = {key: value for key, value in current.items() if key not in ("outer", "restarts", "profiles")}
        summary["detail"] = detail.name
        starts.append(summary)
        emit(dict(stage="start_complete", **summary))
        print(f"penalty {state['penalty_index']}, start {state['start_index']}: "
              f"{result.diagnostics['status']}, restarts={current['restart_count']}", flush=True)
        return result

    def raw(*values, **kwargs):
        state["penalty_index"] += 1
        state["start_index"] = 0
        emit(dict(stage="penalty_start", penalty_index=state["penalty_index"], lambda_value=float(values[3])))
        return original_raw(*values, **kwargs)

    solver.evaluate = likelihood.evaluate = evaluated
    solver.one_sided_derivatives = derivatives
    solver.interval_descent, solver.local_interval_delta = scan, local_delta
    solver.clipping_breakpoints = breakpoints
    solver.stationarity, solver._kink_check = stationarity, kink
    solver.profile_quadratic_witnesses = profile
    setattr(raw_module, start_name, start)
    selection.fit_fixed_lambda = raw
    started = perf_counter()
    result = api.fit(args.input_file, args.outdir / "fit")
    report = dict(**metadata, wall_seconds=perf_counter() - started,
                  search_status=result.search_status, selected_lambda=result.selected_lambda,
                  selected_score=result.selection_score, selected_raw_objective=result.raw_objective,
                  timings=totals.values, starts=starts,
                  unresolved_starts=[row for row in starts if not row["qualified"]],
                  likelihood_timing_scope="Includes posterior, logs and derivatives within evaluate; caller exclusive times avoid double-counting",
                  finite_search_scope="finite_proposal_evaluation plus finite_breakpoint_setup; other finite-search control belongs to kink_other")
    write_json(args.outdir / "diagnosis.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-package", type=Path, required=True)
    parser.add_argument("--source-commit", help="Optional expected revision annotation; actual source hashes are recorded")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=360.)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_historical_cpu_package(args.frozen_package)
    if args.worker:
        worker(args)
        return
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    args.outdir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--frozen-package", str(args.frozen_package.resolve()),
               "--input-file", str(args.input_file.resolve()),
               "--outdir", str(args.outdir.resolve())]
    if args.source_commit is not None:
        command.extend(["--source-commit", args.source_commit])
    began = perf_counter()
    with (args.outdir / "worker.log").open("x") as handle:
        try:
            run = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                 timeout=args.timeout_seconds, check=False)
            outcome = dict(status="completed" if run.returncode == 0 else "failed", returncode=run.returncode)
        except subprocess.TimeoutExpired:
            outcome = dict(status="timeout", returncode=None)
    with (args.outdir / "controller.json").open("x") as handle:
        json.dump(dict(**outcome, wall_seconds=perf_counter() - began, timeout_seconds=args.timeout_seconds), handle, indent=2)
        handle.write("\n")
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    main()
