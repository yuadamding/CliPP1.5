"""Controlled fresh-process full paths with timeouts and bounded failure fixtures."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_chain import configure_execution, require_historical_cpu_package  # noqa: E402


def load_numerics(package_source=None):
    global np, api, clonal, selection, solver, Policy, simulate, reference_enumeration
    import numpy as np
    source_root = (Path(package_source) if package_source is not None else
                   Path(__file__).resolve().parents[1] / "src").resolve()
    if not (source_root / "clipp1d" / "api.py").is_file():
        raise ValueError("--package-source must contain clipp1d/api.py")
    sys.path.insert(0, str(source_root))
    from clipp1d import api, clonal, selection, solver
    if Path(api.__file__).resolve().parent.parent != source_root:
        raise RuntimeError("Loaded package does not match --package-source; use a fresh process")
    from clipp1d.policy import Policy
    from simulate import simulate
    import reference_enumeration


def runtime_precision():
    # Baseline packages may predate precision receipts. Measure the process
    # directly, without changing or importing an overlay into their source.
    formats = {}
    for name, dtype in (("float64", np.float64), ("longdouble", np.longdouble)):
        info = np.finfo(dtype)
        formats[name] = dict(storage_bits=np.dtype(dtype).itemsize * 8, nmant=int(info.nmant),
                             significand_bits=int(info.nmant + 1), exponent_bits=int(info.iexp),
                             minexp=int(info.minexp), maxexp=int(info.maxexp), eps=str(info.eps))
    return dict(platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, scipy=api.scipy.__version__,
                floating_point_formats=formats,
                scope="Actual runtime formats; these observations do not qualify other platforms")


ADDITIVE_START_METRICS = (
    "outer_iterations", "accepted_steps", "backtracks", "restart_witness_switches",
    "audit_context_count", "audit_anchor_count", "audit_scan_count",
    "audit_context_seconds", "audit_scan_seconds", "audit_finite_search_seconds",
    "interval_restart_count", "interval_restart_length_sum",
    "interval_restart_objective_decrease", "surrogate_objective_decrease", "objective_decrease",
)


def accumulate_start_metrics(totals, diagnostics):
    """Add disjoint start work, retaining extrema only over actual restarts."""
    for key in ADDITIVE_START_METRICS:
        if key in diagnostics:
            totals[key] = totals.get(key, 0) + diagnostics[key]
    if diagnostics.get("interval_restart_count", 0):
        for suffix, choose in (("min", min), ("max", max)):
            key = f"interval_restart_length_{suffix}"
            if key in diagnostics:
                totals[key] = choose(totals[key], diagnostics[key]) if key in totals else diagnostics[key]
    if "largest_curvature_scale" in diagnostics:
        totals["largest_curvature_scale"] = max(totals.get("largest_curvature_scale", 0),
                                               diagnostics["largest_curvature_scale"])


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _clean(value.tolist()) if value.ndim == 0 else [_clean(v) for v in value]
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        converted = float(value)  # np.longdouble.item() can remain a NumPy scalar.
        return converted if np.isfinite(converted) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(_clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for name, value in arrays.items():
        value = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class FailureCapture:
    """Bounded, source-bound actual failed calls; never alter numerical results."""

    names = ("h", "target", "lower", "upper", "caps")

    def __init__(self, directory, provenance, context, limit=8, *, finalizer_instrumented=True):
        if not 0 <= limit <= 8:
            raise ValueError("Capture limit must be between zero and eight")
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.provenance, self.context, self.limit = provenance, context, limit
        self.count = self.failed_calls = self.duplicates = self.over_limit = 0
        self.reasons, self.seen = {}, set()
        self.on_capture = None
        self.on_progress = None
        self.last_progress = perf_counter()
        self.profile_boxes = None
        self.quadratic_calls = self.profile_calls = self.finalization_calls = 0
        self.finalizer_instrumented = finalizer_instrumented
        self.profile_nested_quadratic_calls = 0
        self.profile_nested_quadratic_seconds = 0.
        self.completed = {"quadratic": 0, "profile": 0, "finalization": 0}
        self.qualified = {"quadratic": 0, "profile": 0, "finalization": 0}
        self.completed_seconds = {"quadratic": 0., "profile": 0., "finalization": 0.}
        write_json(self.directory / "setup.json", dict(
            schema="clipp1d.failed_surrogate_capture.v1", provenance=provenance,
            maximum_fixtures=limit, array_names=self.names,
            scope="Actual failed production/reference calls; original profile boxes and frozen selected-witness boxes are distinguished",
            publication="Only completed NPZ files listed in manifest.jsonl are fixtures; an interrupted unlisted file is incomplete"))

    def snapshot(self):
        return dict(failed_calls=self.failed_calls, captured=self.count,
                    duplicates_not_recaptured=self.duplicates, over_limit=self.over_limit,
                    limit=self.limit, failure_reason_counts=dict(self.reasons),
                    quadratic_calls=self.quadratic_calls, profile_calls=self.profile_calls,
                    finalization_calls=self.finalization_calls)

    def work_snapshot(self):
        return dict(
            quadratic_calls_started=self.quadratic_calls,
            profile_calls_started=self.profile_calls,
            finalization_calls_started=self.finalization_calls,
            quadratic_calls_completed=self.completed["quadratic"],
            profile_calls_completed=self.completed["profile"],
            finalization_calls_completed=self.completed["finalization"],
            quadratic_calls_qualified=self.qualified["quadratic"],
            profile_calls_qualified=self.qualified["profile"],
            finalization_calls_qualified=self.qualified["finalization"],
            finalizer_instrumented=self.finalizer_instrumented,
            profile_nested_quadratic_calls_completed=self.profile_nested_quadratic_calls,
            profile_nested_quadratic_completed_seconds=self.profile_nested_quadratic_seconds,
            quadratic_completed_seconds=self.completed_seconds["quadratic"],
            profile_completed_seconds=self.completed_seconds["profile"],
            finalization_completed_seconds=self.completed_seconds["finalization"],
            surrogate_completed_seconds=(self.completed_seconds["quadratic"] +
                                         self.completed_seconds["profile"] -
                                         self.profile_nested_quadratic_seconds),
            scope=("Returned surrogate calls, including calls within unfinished nonlinear starts. "
                   "Current profiled reconstruction bypasses solve_quadratic; legacy profiles call it. "
                   "surrogate_completed_seconds sums quadratic and profile time, subtracting quadratic time "
                   "nested inside profiles. Finalization is nested inside either path; its time must not be added. "
                   "Started minus completed can include an in-progress or raised call."))

    def completed_call(self, kind, started, qualified, *, nested_in_profile=False):
        finished = perf_counter()
        self.completed[kind] += 1
        self.qualified[kind] += int(qualified)
        self.completed_seconds[kind] += finished - started
        if kind == "quadratic" and nested_in_profile:
            self.profile_nested_quadratic_calls += 1
            self.profile_nested_quadratic_seconds += finished - started
        if self.on_progress is not None and finished - self.last_progress >= 1.:
            self.last_progress = finished
            self.on_progress(kind)

    def capture(self, values, reason, box_scope, diagnostics=None, **context):
        self.failed_calls += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        if self.count >= self.limit:
            self.over_limit += 1
            return
        arrays = {name: np.asarray(value) for name, value in zip(self.names, values)}
        digest = arrays_digest(arrays)
        key = (box_scope, reason, digest)
        if key in self.seen:
            self.duplicates += 1
            return
        self.seen.add(key)
        path = self.directory / f"failed-surrogate-{self.count + 1:03}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        record = dict(
            schema="clipp1d.failed_surrogate.v1", fixture=path.name,
            fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            arrays_sha256=digest,
            array_shapes={k: list(v.shape) for k, v in arrays.items()},
            array_dtypes={k: str(v.dtype) for k, v in arrays.items()},
            source_sha256=self.provenance["source_sha256"],
            failure_reason=reason, box_scope=box_scope,
            context={**self.context, "quadratic_call": self.quadratic_calls,
                     "profile_call": self.profile_calls, "finalization_call": self.finalization_calls,
                     **context},
            diagnostics=diagnostics or {},
        )
        with (self.directory / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_clean(record), sort_keys=True, allow_nan=False) + "\n")
        self.count += 1
        if self.on_capture is not None:
            self.on_capture(record)

    def wrap_quadratic(self, call):
        def measured(h, target, lower, upper, caps, *args, **kwargs):
            self.quadratic_calls += 1
            started = perf_counter()
            finalizations_before = self.finalization_calls
            nested_in_profile = self.profile_boxes is not None
            result = call(h, target, lower, upper, caps, *args, **kwargs)
            self.completed_call("quadratic", started, result.qualified, nested_in_profile=nested_in_profile)
            # Common finalization captures certificate failures. Only a direct
            # reconstruction failure can return before reaching that wrapper.
            if not result.qualified and self.finalization_calls == finalizations_before:
                self.capture_quadratic_result((h, target, lower, upper, caps), result)
            return result
        return measured

    def capture_quadratic_result(self, values, result):
        reason = result.work.get("failure_reason", "unspecified_quadratic_failure")
        context = {}
        if self.profile_boxes is not None:
            original_lower, original_upper = self.profile_boxes
            lower, upper = values[2:4]
            changed = np.flatnonzero((lower != original_lower) | (upper != original_upper))
            context["selected_witness"] = int(changed[0]) if len(changed) == 1 else None
            context["modified_box_coordinates"] = int(len(changed))
        self.capture(values, reason,
                     "selected_witness_frozen_boxes" if self.profile_boxes is not None else
                     "original_unconstrained_boxes" if self.context.get("clonal_constraint") is False else "fixed_witness_branch_boxes",
                     dict(gap=result.gap, gap_scale=result.gap_scale,
                          kkt_residual=result.kkt_residual, work=result.work), **context)

    def wrap_finalizer(self, call):
        def measured(h, target, lower, upper, caps, *args, **kwargs):
            self.finalization_calls += 1
            started = perf_counter()
            result = call(h, target, lower, upper, caps, *args, **kwargs)
            self.completed_call("finalization", started, result.qualified)
            if not result.qualified:
                self.capture_quadratic_result((h, target, lower, upper, caps), result)
            return result
        return measured

    def wrap_profile(self, call):
        def measured(h, target, lower, upper, caps, *args, **kwargs):
            self.profile_calls += 1
            started = perf_counter()
            previous_boxes = self.profile_boxes
            self.profile_boxes = (lower, upper)
            try:
                result = call(h, target, lower, upper, caps, *args, **kwargs)
            except ArithmeticError as exc:
                self.capture((h, target, lower, upper, caps), str(exc), "original_profile_boxes",
                             dict(stage="witness_profile", exception_type=type(exc).__name__))
                raise
            finally:
                self.profile_boxes = previous_boxes
            self.completed_call("profile", started, result.qualified)
            # Selected-QP failures were already captured with the actual frozen
            # boxes. Capture original boxes separately only for the profile gate.
            if not result.qualified and result.fit.qualified:
                self.capture((h, target, lower, upper, caps), "witness_profile_value_gate",
                             "original_profile_boxes", result.diagnostics,
                             selected_witness=int(result.witness), surrogate_sha256=result.surrogate_sha256)
            return result
        return measured


def worker(outdir, size, seed, scenario, execution, reference=False, capture_limit=8):
    require_historical_cpu_package(Path(api.__file__).resolve().parent)
    source, truth = simulate(outdir / "input", size, seed, ambiguous=scenario == "mixture")
    started = perf_counter()
    totals = dict(starts_completed=0, starts_failed=0, starts_seconds=0., inner_seconds=0.,
                  inner_iterations=0, kink_seconds=0., scalar_fits_performed=0,
                  singleton_pilots_reused=0, scalar_evaluations=0,
                  profile_calls=0, surrogate_witnesses_profiled=0, inner_certificates_reused=0,
                  witness_switches=0)
    failures, failure_statuses = {}, {}
    last_snapshot = started
    provenance = api.source_provenance()
    raw_module = sys.modules[selection.fit_fixed_lambda.__module__]
    free_search = hasattr(raw_module, "solve_unconstrained")
    start_name = "solve_unconstrained" if free_search else "solve_profiled"
    policy_name = ("independent_witness_enumeration_reference" if reference else
                   "unconstrained_chain_multistart_v1" if free_search else "common_surrogate_multistart_v1")
    context = dict(search_policy=policy_name, clonal_constraint=not free_search or reference,
                   mutations=size, scenario=scenario, seed=seed,
                   input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    capture = FailureCapture(outdir / "failed_surrogates", provenance, context, capture_limit,
                             finalizer_instrumented=hasattr(solver, "finalize_quadratic"))

    def emit(stage, **fields):
        event = dict(stage=stage, elapsed_seconds=perf_counter() - started,
                     peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     totals=dict(totals), start_failure_counts=dict(failures),
                     start_failure_status_counts=dict(failure_statuses),
                     failure_capture=capture.snapshot(), surrogate_work=capture.work_snapshot(), **fields)
        with (outdir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(_clean(event), sort_keys=True, allow_nan=False) + "\n")

    capture.on_capture = lambda record: emit("surrogate_failure_fixture", fixture=record)
    capture.on_progress = lambda kind: emit("surrogate_progress", completed_call=kind,
                                            active_start_context=dict(context))
    emit("start", provenance=provenance, policy=asdict(Policy()), execution=execution,
         runtime_precision=runtime_precision(), package_source=str(Path(api.__file__).resolve().parent.parent),
         completed_start_timing_scope=("Audit context, scan and finite-search timings are separate subphases. "
                                      "kink_seconds includes scan/finite search and must not be added to them. "
                                      "Objective decreases sum separate nonlinear trajectories, not one fit's decrease."),
         search_policy=policy_name, reference_enumeration=reference,
         started_utc=datetime.now(timezone.utc).isoformat(), mutations=size, scenario=scenario,
         seed=seed, input_sha256=context["input_sha256"],
         truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
         benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         simulator_sha256=hashlib.sha256(Path(simulate.__code__.co_filename).read_bytes()).hexdigest(),
         reference_sha256=hashlib.sha256(Path(reference_enumeration.__file__).read_bytes()).hexdigest() if reference else None)
    if reference and free_search:
        emit("fit_failed", error_type="ValueError",
             message="Witness enumeration requires a pinned historical constrained package; it cannot replace unconstrained production inference.")
        return
    original_pilot, original_chain = api.compute_pilot, api.build_chain
    original_raw, original_refit = selection.fit_fixed_lambda, selection.refit_partition
    original_start = solver.solve_branch if reference else getattr(raw_module, start_name)
    original_quadratic, original_profile = solver.solve_quadratic, solver.profile_quadratic_witnesses
    original_finalizer = getattr(solver, "finalize_quadratic", None)
    raw_backend = reference_enumeration.fit_fixed_lambda if reference else original_raw

    # Instrument stage boundaries without changing policy, starts, or numerical calls.
    def pilot_call(model, policy):
        begin = perf_counter()
        emit("pilot_start")
        result = original_pilot(model, policy)
        emit("pilot_complete", seconds=perf_counter() - begin,
             evaluations=sum(r.evaluations for r in result.scalar_results),
             bound_evaluations=sum(r.bound_evaluations for r in result.scalar_results),
             exact_fits=sum(r.method == "same_slope_single_candidate_exact" for r in result.scalar_results),
             model_array_bytes=sum(getattr(model, n).nbytes for n in
                                   ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid")))
        return result

    def chain_call(*args):
        chain = original_chain(*args)
        context["chain_sha256"] = chain.fingerprint
        emit("chain_complete", edges=chain.weights.size,
             graph_array_bytes=chain.order.nbytes + chain.inverse_order.nbytes + chain.weights.nbytes,
             graph_array_scope="order, inverse order, and weights only; excludes solver working state",
             chain_sha256=chain.fingerprint)
        return chain

    def start_call(*args, **kwargs):
        nonlocal last_snapshot
        begin = perf_counter()
        context["start_index"] = totals["starts_completed"] + 1
        context["fixed_witness"] = int(args[3]) if reference else None
        start = args[4] if reference else args[3]
        context["start_sha256"] = arrays_digest({"primal": np.asarray(start)})
        result = original_start(*args, **kwargs)
        totals["starts_seconds"] += perf_counter() - begin
        totals["starts_completed"] += 1
        totals["starts_failed"] += int(not result.qualified)
        diagnostics = result.diagnostics
        totals["inner_seconds"] += diagnostics.get("inner_solve_seconds", 0.)
        totals["inner_iterations"] += diagnostics.get("inner_iterations", 0)
        totals["kink_seconds"] += diagnostics.get("kink_check_seconds", 0.)
        for key in ("profile_calls", "surrogate_witnesses_profiled", "inner_certificates_reused", "witness_switches"):
            totals[key] += diagnostics.get(key, 0)
        accumulate_start_metrics(totals, diagnostics)
        if not result.qualified:
            reason = diagnostics.get("failure_reason") or diagnostics.get("last_inner_work", {}).get("failure_reason") or diagnostics["status"]
            failures[reason] = failures.get(reason, 0) + 1
            status = diagnostics["status"]
            failure_statuses[status] = failure_statuses.get(status, 0) + 1
        if perf_counter() - last_snapshot >= 1:
            emit("start_progress", lambda_value=float(args[2]))
            last_snapshot = perf_counter()
        return result

    def raw_call(*args, **kwargs):
        begin = perf_counter()
        context["lambda_value"] = float(args[3])
        emit("raw_start", lambda_value=context["lambda_value"])
        try:
            result = raw_backend(*args, **kwargs)
        except Exception as exc:
            emit("raw_unresolved", lambda_value=context["lambda_value"], seconds=perf_counter() - begin,
                 error_type=type(exc).__name__, message=str(exc), diagnostics=getattr(exc, "diagnostics", {}))
            raise
        emit("raw_complete", lambda_value=context["lambda_value"], seconds=perf_counter() - begin,
             diagnostics=result.diagnostics, objective=result.objective, witness=result.witness)
        return result

    def refit_call(*args, **kwargs):
        begin = perf_counter()
        emit("refit_start", blocks=len(args[1]) - 1)
        try:
            result = original_refit(*args, **kwargs)
        except Exception as exc:
            emit("refit_unresolved", seconds=perf_counter() - begin,
                 error_type=type(exc).__name__, message=str(exc), diagnostics=getattr(exc, "diagnostics", {}))
            raise
        for key in ("scalar_fits_performed", "singleton_pilots_reused", "scalar_evaluations"):
            totals[key] += getattr(result, key)
        emit("refit_complete", seconds=perf_counter() - begin, blocks=len(result.centers), gap=result.gap)
        return result

    api.compute_pilot, api.build_chain = pilot_call, chain_call
    selection.fit_fixed_lambda, selection.refit_partition = raw_call, refit_call
    if reference:
        solver.solve_branch = start_call
    else:
        setattr(raw_module, start_name, start_call)
    solver.solve_quadratic = capture.wrap_quadratic(original_quadratic)
    solver.profile_quadratic_witnesses = capture.wrap_profile(original_profile)
    if original_finalizer is not None:
        solver.finalize_quadratic = capture.wrap_finalizer(original_finalizer)
    try:
        result = api.fit(source, outdir / "fit")
    except Exception as exc:
        emit("fit_failed", error_type=type(exc).__name__, message=str(exc))
        return
    finally:
        api.compute_pilot, api.build_chain = original_pilot, original_chain
        selection.fit_fixed_lambda, selection.refit_partition = original_raw, original_refit
        if reference:
            solver.solve_branch = original_start
        else:
            setattr(raw_module, start_name, original_start)
        solver.solve_quadratic, solver.profile_quadratic_witnesses = original_quadratic, original_profile
        if original_finalizer is not None:
            solver.finalize_quadratic = original_finalizer
    emit("fit_complete", search_status=result.search_status, selected_clusters=len(result.cluster_centers),
         selection_score=result.selection_score, search=result.search_diagnostics)


def summarize(directory, size, scenario, elapsed, timed_out, returncode):
    events = []
    progress = directory / "progress.jsonl"
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a hard timeout can interrupt only the final append
    by_stage = {e["stage"]: e for e in events}
    initial, last = (events[0], events[-1]) if events else ({}, {})
    finished = by_stage.get("fit_complete", {})
    raw_done = [e for e in events if e["stage"] in ("raw_complete", "raw_unresolved")]
    raw_starts = sum(e["stage"] == "raw_start" for e in events)
    incomplete = sum(not e.get("diagnostics", {}).get("search_complete", e.get("diagnostics", {}).get("witness_search_complete", False)) for e in raw_done)
    record = dict(schema="clipp1d.scaling.case.v4", mutations=size, scenario=scenario,
                  status="timeout" if timed_out else ("success" if finished else "failure"),
                  search_status=finished.get("search_status", "not_completed"),
                  search_policy=initial.get("search_policy"),
                  wall_seconds=elapsed, returncode=returncode, setup=initial,
                  last_stage=last.get("stage"), partial_totals=last.get("totals", {}),
                  partial_totals_scope="Completed nonlinear starts/refits only; surrogate_work also covers returned inner calls in interrupted starts",
                  surrogate_work=last.get("surrogate_work", {}),
                  start_failure_counts=last.get("start_failure_counts", {}),
                  start_failure_status_counts=last.get("start_failure_status_counts", {}),
                  failure_capture=last.get("failure_capture", {}),
                  failure_capture_manifest="failed_surrogates/manifest.jsonl" if (directory / "failed_surrogates/manifest.jsonl").exists() else None,
                  peak_process_rss_kib=max((e["peak_process_rss_kib"] for e in events), default=None),
                  rss_scope="observed through last completed stage; lower bound on timeout",
                  raw_penalties_started=raw_starts, raw_penalties_finished=len(raw_done),
                  raw_penalties_interrupted=raw_starts - len(raw_done),
                  raw_penalties_unresolved=sum(e["stage"] == "raw_unresolved" for e in raw_done),
                  raw_penalties_incomplete_search=incomplete,
                  refits_unresolved=sum(e["stage"] == "refit_unresolved" for e in events),
                  pilot_seconds=by_stage.get("pilot_complete", {}).get("seconds"),
                  solver_completed_seconds=sum(e["seconds"] for e in raw_done),
                  refit_completed_seconds=sum(e["seconds"] for e in events if e["stage"] in ("refit_complete", "refit_unresolved")),
                  chain=by_stage.get("chain_complete"), completed_fit=finished or None)
    write_json(directory / "benchmark.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--scenarios", nargs="+", choices=("easy", "mixture"), default=["easy", "mixture"])
    parser.add_argument("--timeout-seconds", type=float, default=120.)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reference-enumeration", action="store_true", help="Historical constrained witness comparison; requires a pinned constrained --package-source")
    parser.add_argument("--max-failure-captures", type=int, default=8)
    parser.add_argument("--package-source", type=Path,
                        help="Directory containing clipp1d/; load unchanged source in each fresh worker")
    parser.add_argument("--cpu-count", type=int, default=1)
    parser.add_argument("--cpus", nargs="+", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or any(n < 1 for n in args.sizes) or (args.worker is not None and args.worker < 1):
        parser.error("sizes and timeout must be positive")
    if not 0 <= args.max_failure_captures <= 8:
        parser.error("failure captures must be between zero and eight")
    try:
        execution = configure_execution(args.cpu_count, args.cpus, args.threads)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        require_historical_cpu_package((args.package_source or Path(__file__).resolve().parents[1] / "src") / "clipp1d")
        load_numerics(args.package_source)
    except ValueError as exc:
        parser.error(str(exc))
    if args.worker is not None:
        worker(args.outdir, args.worker, args.seed, args.scenarios[0], execution,
               args.reference_enumeration, args.max_failure_captures)
        return
    args.outdir.mkdir(parents=True, exist_ok=False)
    write_json(args.outdir / "setup.json", dict(
        schema="clipp1d.scaling.setup.v4", execution=execution,
        runtime_precision=runtime_precision(), package_source=str(Path(api.__file__).resolve().parent.parent),
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        provenance=api.source_provenance(), benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    records = []
    for size in args.sizes:
        for scenario in args.scenarios:
            directory = args.outdir / f"{scenario}-n{size}"
            directory.mkdir()
            started = perf_counter()
            timed_out = False
            command = [sys.executable, __file__, "--worker", str(size), "--seed", str(args.seed),
                       "--scenarios", scenario, "--outdir", str(directory),
                       "--timeout-seconds", str(args.timeout_seconds), "--cpu-count", str(args.cpu_count),
                       "--threads", str(args.threads), "--max-failure-captures", str(args.max_failure_captures)]
            if args.cpus is not None:
                command.extend(["--cpus", *map(str, args.cpus)])
            if args.reference_enumeration:
                command.append("--reference-enumeration")
            if args.package_source is not None:
                command.extend(["--package-source", str(args.package_source.resolve())])
            with (directory / "worker.log").open("x") as log:
                try:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=args.timeout_seconds, check=False)
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    timed_out, returncode = True, None
            record = summarize(directory, size, scenario, perf_counter() - started, timed_out, returncode)
            records.append(record)
            print(f"{scenario} M={size}: {record['status']}, search={record['search_status']}, "
                  f"raw penalties finished={record['raw_penalties_finished']}/{record['raw_penalties_started']}", flush=True)
    write_json(args.outdir / "scaling.json", dict(schema="clipp1d.scaling.v4", results=records,
               timeout_seconds=args.timeout_seconds,
               qualification="synthetic CPU; timeouts retain partial evidence, never qualified full fits"))


if __name__ == "__main__":
    main()
