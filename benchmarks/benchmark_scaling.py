"""Fresh-process CPU scaling with bounded runtime and durable partial coverage."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clipp1d import api, clonal, selection  # noqa: E402
from clipp1d.policy import Policy  # noqa: E402
from clipp1d.report import write_json  # noqa: E402
from simulate import simulate  # noqa: E402


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def worker(outdir, size, seed, scenario):
    source, truth = simulate(outdir / "input", size, seed, ambiguous=scenario == "mixture")
    started = perf_counter()
    totals = dict(starts_completed=0, starts_failed=0, starts_seconds=0., inner_seconds=0.,
                  inner_iterations=0, kink_seconds=0., scalar_fits_performed=0,
                  singleton_pilots_reused=0, scalar_evaluations=0)
    failures = {}
    last_snapshot = started

    def emit(stage, **fields):
        event = dict(stage=stage, elapsed_seconds=perf_counter() - started,
                     peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     totals=dict(totals), start_failure_counts=dict(failures), **fields)
        with (outdir / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(_clean(event), sort_keys=True, allow_nan=False) + "\n")

    emit("start", provenance=api.source_provenance(), policy=asdict(Policy()),
         started_utc=datetime.now(timezone.utc).isoformat(), mutations=size, scenario=scenario,
         seed=seed, input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
         benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         simulator_sha256=hashlib.sha256(Path(simulate.__code__.co_filename).read_bytes()).hexdigest())
    original_pilot, original_chain = api.compute_pilot, api.build_chain
    original_raw, original_refit, original_branch = selection.fit_fixed_lambda, selection.refit_partition, clonal.solve_branch

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
        emit("chain_complete", edges=chain.weights.size,
             graph_array_bytes=chain.order.nbytes + chain.inverse_order.nbytes + chain.weights.nbytes,
             chain_sha256=chain.fingerprint)
        return chain

    def branch_call(*args, **kwargs):
        nonlocal last_snapshot
        begin = perf_counter()
        result = original_branch(*args, **kwargs)
        totals["starts_seconds"] += perf_counter() - begin
        totals["starts_completed"] += 1
        totals["starts_failed"] += int(not result.qualified)
        totals["inner_seconds"] += result.diagnostics["inner_solve_seconds"]
        totals["inner_iterations"] += result.diagnostics["inner_iterations"]
        totals["kink_seconds"] += result.diagnostics["kink_check_seconds"]
        if not result.qualified:
            reason = result.diagnostics["status"]
            failures[reason] = failures.get(reason, 0) + 1
        if perf_counter() - last_snapshot >= 1:
            emit("branch_progress", lambda_value=float(args[2]))
            last_snapshot = perf_counter()
        return result

    def raw_call(*args, **kwargs):
        begin = perf_counter()
        emit("raw_start", lambda_value=float(args[3]))
        try:
            result = original_raw(*args, **kwargs)
        except Exception as exc:
            emit("raw_unresolved", lambda_value=float(args[3]), seconds=perf_counter() - begin,
                 error_type=type(exc).__name__, message=str(exc), diagnostics=getattr(exc, "diagnostics", {}))
            raise
        emit("raw_complete", lambda_value=float(args[3]), seconds=perf_counter() - begin,
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
    selection.fit_fixed_lambda, selection.refit_partition, clonal.solve_branch = raw_call, refit_call, branch_call
    try:
        result = api.fit(source, outdir / "fit")
    except Exception as exc:
        emit("fit_failed", error_type=type(exc).__name__, message=str(exc))
        return
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
    record = dict(schema="clipp1d.scaling.case.v2", mutations=size, scenario=scenario,
                  status="timeout" if timed_out else ("success" if finished else "failure"),
                  search_status=finished.get("search_status", "not_completed"),
                  wall_seconds=elapsed, returncode=returncode, setup=initial,
                  last_stage=last.get("stage"), partial_totals=last.get("totals", {}),
                  start_failure_counts=last.get("start_failure_counts", {}),
                  peak_process_rss_kib=max((e["peak_process_rss_kib"] for e in events), default=None),
                  rss_scope="observed through last completed stage; lower bound on timeout",
                  raw_penalties_started=raw_starts, raw_penalties_finished=len(raw_done),
                  raw_penalties_interrupted=raw_starts - len(raw_done),
                  raw_penalties_unresolved=sum(e["stage"] == "raw_unresolved" for e in raw_done),
                  raw_penalties_incomplete_witness_search=sum(not e.get("diagnostics", {}).get("witness_search_complete", False) for e in raw_done),
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
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or any(n < 1 for n in args.sizes):
        parser.error("sizes and timeout must be positive")
    if args.worker is not None:
        worker(args.outdir, args.worker, args.seed, args.scenarios[0])
        return
    if args.outdir.exists():
        raise FileExistsError("Use a new benchmark directory")
    args.outdir.mkdir(parents=True)
    records = []
    for size in args.sizes:
        for scenario in args.scenarios:
            directory = args.outdir / f"{scenario}-n{size}"
            directory.mkdir()
            started = perf_counter()
            timed_out = False
            with (directory / "worker.log").open("x") as log:
                try:
                    proc = subprocess.run([sys.executable, __file__, "--worker", str(size), "--seed", str(args.seed),
                                           "--scenarios", scenario, "--outdir", str(directory)],
                                          stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout_seconds, check=False)
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    timed_out, returncode = True, None
            record = summarize(directory, size, scenario, perf_counter() - started, timed_out, returncode)
            records.append(record)
            print(f"{scenario} M={size}: {record['status']}, search={record['search_status']}, "
                  f"raw penalties finished={record['raw_penalties_finished']}/{record['raw_penalties_started']}", flush=True)
    write_json(args.outdir / "scaling.json", dict(schema="clipp1d.scaling.v2", results=records,
               timeout_seconds=args.timeout_seconds,
               qualification="synthetic CPU; timeouts retain partial evidence, never qualified full fits"))


if __name__ == "__main__":
    main()
