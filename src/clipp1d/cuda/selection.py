"""GPU graph-path candidates, membership refits, and unchanged allocation score.

The legacy score-selected raw state remains primary. Optional partition search
streams qualified starts and returns an independent estimator; it never changes
the raw continuation, the path-extension decision, or the baseline winner.
"""
from dataclasses import dataclass
from time import perf_counter
import math
import torch
from .graph import build_graph, penalty_reference
from .scalar import pilot
from .partition import LastPartitionCache, QualifiedPilot, refit
from .solver import PrimalWarmState, fit_lambda
from .policy import CudaPolicy, QualificationError


@dataclass
class DeviceFit:
    model: object
    graph: object
    pilot: object
    raw: object
    refit: object
    lambda_value: torch.Tensor
    search_status: str
    records: list
    timings: dict
    policy: CudaPolicy = CudaPolicy()
    partition_estimate: object = None


@torch.no_grad()
def fit_tensor_model(model, policy=CudaPolicy(), *, lambda_values=None, partition_search=None):
    """Low-level CPU execution serves numerical tests, never public fallback."""
    boundary_started = _synchronized_time(model)
    with model.validated_stage():
        result = _fit_tensor_model(model, policy, lambda_values=lambda_values, partition_search=partition_search)
    boundary_finished = _synchronized_time(model)
    # Include entry/exit integrity reconciliation in the numerical wall scope.
    # It is reported separately from the four inner numerical phases.
    result.timings["stage_integrity_seconds"] = (
        boundary_finished - boundary_started - result.timings["numerical_wall_seconds"])
    result.timings["numerical_wall_seconds"] = boundary_finished - boundary_started
    return result


def _synchronized_time(model):
    # Phase boundaries report completed device work, not asynchronous dispatch.
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return perf_counter()


def _fit_tensor_model(model, policy, *, lambda_values, partition_search=None):
    started = _synchronized_time(model)
    pilots = pilot(model, policy)
    pilot_reuse = QualifiedPilot(model, pilots, policy)
    pilot_finished = _synchronized_time(model)
    pilot_seconds = pilot_finished - started
    graph = build_graph(pilots.phi, mutation_ids=model.mutation_ids)
    curvature = model.terms(pilots.phi)[2]
    reference = penalty_reference(model, graph, pilots.phi, curvature)
    graph_finished = _synchronized_time(model)
    graph_build_seconds = graph_finished - pilot_finished
    cache = LastPartitionCache(model, policy)
    proposals = None
    if partition_search is not None:
        from .partition_search import PartitionSearch
        from .refinement import PartitionSearchPolicy
        if not isinstance(partition_search, PartitionSearchPolicy):
            raise ValueError("partition_search requires an explicit PartitionSearchPolicy")
        proposals = PartitionSearch(model, graph, pilot_reuse, policy, partition_search)
    refit_seconds = 0.
    if lambda_values is None:
        path = [reference.new_tensor(0.)]
        if model.n > 1:
            path += [torch.ldexp(reference, torch.tensor(k, device=reference.device))
                     for k in range(policy.path_min_exponent, policy.path_max_exponent + 1)]
            if any(not bool(torch.isfinite(v) & (v > 0)) for v in path[1:]):
                raise ValueError("Logarithmic path penalties must be finite and strictly positive")
    else:
        path = [reference.new_tensor(v) for v in lambda_values]
    if not path or any(v.ndim != 0 or not bool(torch.isfinite(v) & (v >= 0)) for v in path):
        raise ValueError("Penalty path must contain finite nonnegative scalars")
    records, best, best_key, previous = [], None, None, None
    index = extensions = 0
    while index < len(path):
        lam = path[index]
        record = dict(lambda_value=float(lam), raw_status="not_attempted", refit_status="not_attempted")
        begin = perf_counter()
        try:
            if proposals is not None:
                proposals.begin_lambda(index, lam)
            if proposals is not None and partition_search.all_starts:
                raw = fit_lambda(model, graph, pilots, lam, previous, policy, on_qualified=proposals.observe)
            else:
                raw = fit_lambda(model, graph, pilots, lam, previous, policy)
                if proposals is not None:
                    proposals.observe(raw, "raw_winner")
            if not raw.qualified or not bool(torch.isfinite(raw.objective)):
                raise QualificationError("Unqualified path state cannot enter selection", **raw.diagnostics)
            record.update(raw.diagnostics)
            record.update(raw_status="qualified", raw_objective=float(raw.objective))
            previous = PrimalWarmState(raw.x, graph.identity)
            refit_started = _synchronized_time(model)
            counters_before = cache.refits_computed, cache.refits_reused, cache.singleton_pilots_reused
            try:
                secondary = refit(model, raw.x, policy, pilot_reuse=pilot_reuse, cache=cache)
            finally:
                duration = _synchronized_time(model) - refit_started
                refit_seconds += duration
                record.update(refit_seconds=duration,
                              refits_computed=cache.refits_computed - counters_before[0],
                              refits_reused=cache.refits_reused - counters_before[1],
                              singleton_pilots_reused=cache.singleton_pilots_reused - counters_before[2])
            score = float(secondary.score)
            refit_gap = float(secondary.gap)
            if not math.isfinite(score) or not math.isfinite(refit_gap) or refit_gap < 0:
                raise QualificationError("Refit score and gap must be finite; gap must be nonnegative")
            clusters = secondary.centers.numel()
            record.update(refit_status="qualified", score=score,
                          refit_gap=refit_gap, clusters=clusters)
            # Same deterministic score, occupied-K, lambda tie order as CPU.
            key = score, clusters, float(lam)
            if best is None or key < best_key:
                best, best_key = (lam, raw, secondary), key
        except QualificationError as error:
            if record["raw_status"] != "qualified":
                record["raw_status"] = "unresolved"
            else:
                record["refit_status"] = "unresolved"
            record.update(error=str(error), failure_diagnostics=error.diagnostics, search_complete=False)
            for name in ("qp_seconds", "audit_seconds", "qp_calls", "qp_admm_iterations", "audit_calls", "qp_dual_warm_starts",
                         "qp_dual_warm_resets", "qp_polish_iterations"):
                if name in error.diagnostics:
                    record[name] = error.diagnostics[name]
        record["seconds"] = _synchronized_time(model) - begin
        records.append(record)
        index += 1
        if (lambda_values is None and index == len(path) and model.n > 1 and best is not None
                and bool(best[0] == path[-1]) and extensions < policy.path_extensions):
            next_lambda = path[-1] * 2.
            if not bool(torch.isfinite(next_lambda)):
                raise QualificationError("Penalty extension overflowed", path=records)
            path.append(next_lambda)
            extensions += 1
    if best is None:
        raise QualificationError("No qualified complete-graph path candidate", path=records)
    selected_at_boundary = model.n > 1 and bool(best[0] == path[-1])
    extension_limit_reached = (lambda_values is None and selected_at_boundary and
                               extensions >= policy.path_extensions)
    complete = all(r["raw_status"] == r["refit_status"] == "qualified" and r.get("search_complete", False)
                   for r in records)
    partition_estimate = None if proposals is None else proposals.finish(best[1], best[2], best[0])
    finished = _synchronized_time(model)
    proposal_seconds = 0. if partition_estimate is None else partition_estimate.seconds
    return DeviceFit(model, graph, pilots, best[1], best[2], best[0],
                     "complete" if complete else "incomplete", records,
                     dict(pilot_seconds=pilot_seconds, graph_build_seconds=graph_build_seconds,
                          path_seconds=finished - graph_finished - refit_seconds - proposal_seconds,
                          partition_search_seconds=proposal_seconds,
                          refit_seconds=refit_seconds, numerical_wall_seconds=finished - started,
                          phase_timing_scope="nonoverlapping synchronized numerical stages; excludes final publication",
                          refits_computed=cache.refits_computed, refits_reused=cache.refits_reused,
                          singleton_pilots_reused=cache.singleton_pilots_reused,
                          qp_seconds=sum(r.get("qp_seconds", 0.) for r in records),
                          audit_seconds=sum(r.get("audit_seconds", 0.) for r in records),
                          inner_timing_scope="QP and audit times are subsets of path_seconds",
                          qp_calls=sum(r.get("qp_calls", 0) for r in records),
                          qp_admm_iterations=sum(r.get("qp_admm_iterations", 0) for r in records),
                          qp_work_scope="all attempted QPs across all starts and planned penalties, including unresolved attempts",
                          audit_calls=sum(r.get("audit_calls", 0) for r in records),
                          qp_dual_warm_starts=sum(r.get("qp_dual_warm_starts", 0) for r in records),
                          qp_dual_warm_resets=sum(r.get("qp_dual_warm_resets", 0) for r in records),
                          qp_polish_iterations=sum(r.get("qp_polish_iterations", 0) for r in records),
                          lambda_reference=float(reference), extensions=extensions,
                          candidate_family="qualified_complete_graph_path",
                          raw_unresolved_penalties=sum(r["raw_status"] != "qualified" for r in records),
                          refit_unresolved_penalties=sum(r["refit_status"] == "unresolved" for r in records),
                          selected_at_upper_boundary=selected_at_boundary,
                          extension_limit_reached=extension_limit_reached,
                          path_truncated=extension_limit_reached,
                          planned_path_complete=complete,
                          completion_scope="qualified resolution of the bounded planned path; not all penalties"),
                     policy, partition_estimate)
