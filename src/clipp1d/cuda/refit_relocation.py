"""Bounded existing-group relocations scored after qualified scalar refits.

The caller supplies an immutable parent partition. A full scan considers every
mutation/destination pair, including moves blocked by the old destination center.
This is a local neighborhood, not a global partition or raw-solver certificate.
"""
import math

import torch

from .ancestry import refit_identity
from .partition import partition_score, refit_labels
from .policy import CudaPolicy, QualificationError
from .scalar import Problems, solve_scalar


ALGORITHM = "refit_aware_existing_group_relocation_v1"


def neighborhood_size(n, k):
    if type(n) is not int or type(k) is not int or not 1 <= k <= n:
        raise ValueError("Require positive N and occupied K <= N")
    return n * (k - 1)


def _feasible(model, nodes):
    return nodes.numel() == 0 or bool(model.lower[nodes].max() <= model.upper[nodes].min())


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def scan_refit_relocations(model, initial, policy=CudaPolicy(), *, max_candidates,
                           batch_candidates=16, minimum_decrease=1e-8):
    """Return a qualified improving child or the original parent and exact coverage.

    A scan that exceeds the remaining global candidate budget does no candidate
    work. Every accepted child is independently refitted and validated. Its own
    neighborhood is unscanned until the caller runs a fresh pass on that child.
    """
    from ..cuda_api import _validate_refit

    if type(max_candidates) is not int or max_candidates < 0:
        raise ValueError("max_candidates must be a nonnegative integer")
    if type(batch_candidates) is not int or batch_candidates < 1:
        raise ValueError("batch_candidates must be a positive integer")
    if (isinstance(minimum_decrease, bool) or not math.isfinite(minimum_decrease)
            or minimum_decrease <= 0):
        raise ValueError("minimum_decrease must be finite and positive")
    _validate_refit(model, initial, policy)
    parent = refit_identity(initial)
    k = int(initial.centers.numel())
    planned = neighborhood_size(model.n, k)
    record = dict(algorithm=ALGORITHM, parent=parent, planned_candidates=planned,
        evaluated_candidates=0, infeasible_candidates=0, unresolved_candidates=[],
        accepted=False, scan_complete=False, selected_endpoint_complete=False,
        coverage_scope="all_mutations_to_existing_groups", best_proposal=None,
        scalar_batches=0, scalar_groups=0, independent_child_refits=0)
    if planned > max_candidates:
        record.update(status="candidate_budget_exhausted", remaining_candidates=planned)
        return initial, record
    margin = max(minimum_decrease,
                 128 * torch.finfo(torch.float64).eps * (1 + abs(float(initial.score))))
    labels = initial.labels.tolist()
    groups = [torch.nonzero(initial.labels == i, as_tuple=True)[0] for i in range(k)]
    pending, best = [], None

    def unresolved(node, source, destination, **detail):
        record["unresolved_candidates"].append(dict(node=node, source=source,
                                                    destination=destination, **detail))

    def process_batch():
        nonlocal best
        if not pending:
            return
        rows, lengths, slots = [], [], []
        for _, _, _, source_nodes, destination_nodes in pending:
            indices = []
            for nodes in (source_nodes, destination_nodes):
                if nodes.numel():
                    indices.append(len(lengths))
                    lengths.append(nodes.numel())
                    rows.append(nodes)
            slots.append(indices)
        problems = Problems(model.subset(torch.cat(rows)),
                            torch.tensor(lengths, dtype=torch.long, device=model.device))
        record["scalar_batches"] += 1
        record["scalar_groups"] += len(lengths)
        try:
            solved = solve_scalar(problems, policy)
        except QualificationError as error:
            for node, source, destination, _, _ in pending:
                record["evaluated_candidates"] += 1
                unresolved(node, source, destination, status="scalar_batch_unresolved",
                           error=str(error), diagnostics=error.diagnostics)
            pending.clear()
            return
        for (node, source, destination, _, _), indices in zip(pending, slots, strict=True):
            record["evaluated_candidates"] += 1
            if not bool(solved.qualified[indices].all()):
                unresolved(node, source, destination, status="changed_group_refit_unresolved",
                    group_lengths=[lengths[i] for i in indices],
                    scalar_groups=[dict(qualified=bool(solved.qualified[i]),
                        loss=_finite(solved.loss[i]), lower_bound=_finite(solved.lower_bound[i]),
                        gap=_finite(solved.gap[i]),
                        allowed_gap=_finite(policy.scalar_atol + policy.scalar_rtol * solved.loss[i].abs()))
                        for i in indices], subdivisions=solved.subdivisions)
                continue
            loss = (initial.loss - initial.scalar.loss[source] - initial.scalar.loss[destination]
                    + solved.loss[indices].sum())
            gap = (initial.gap - initial.scalar.gap[source] - initial.scalar.gap[destination]
                   + solved.gap[indices].sum()).clamp_min(0)
            sizes = initial.sizes.clone()
            sizes[source] -= 1
            sizes[destination] += 1
            score = partition_score(loss, sizes[sizes > 0])
            candidate = dict(node=node, source=source, destination=destination,
                score=float(score), score_lower=float(score - 2 * gap), score_upper=float(score),
                refit_gap=float(gap), source_deleted=bool(sizes[source] == 0))
            if (candidate["score_upper"] < parent["score_lower"] - margin
                    and (best is None or (candidate["score"], node, destination)
                         < (best["score"], best["node"], best["destination"]))):
                best = candidate
        pending.clear()

    with model.validated_stage():
        for node in range(model.n):
            source = labels[node]
            source_nodes = groups[source][groups[source] != node]
            for destination in range(k):
                if source == destination:
                    continue
                destination_nodes = torch.cat((groups[destination], initial.labels.new_tensor([node])))
                if not _feasible(model, source_nodes) or not _feasible(model, destination_nodes):
                    record["infeasible_candidates"] += 1
                    continue
                pending.append((node, source, destination, source_nodes, destination_nodes))
                if len(pending) == batch_candidates:
                    process_batch()
        process_batch()
    charged = record["evaluated_candidates"] + record["infeasible_candidates"]
    if charged != planned:
        raise QualificationError("Refit relocation candidate accounting differs")
    record.update(scan_complete=True, remaining_candidates=0, best_proposal=best)
    if best is None:
        complete = not record["unresolved_candidates"]
        record.update(status="fixed_point" if complete else "unresolved",
                      selected_endpoint_complete=complete)
        return initial, record
    moved = initial.labels.clone()
    moved[best["node"]] = best["destination"]
    record["independent_child_refits"] = 1
    try:
        fitted = refit_labels(model, moved, policy)
        _validate_refit(model, fitted, policy)
    except QualificationError as error:
        record.update(status="unresolved", independent_refit_error=str(error),
                      failure_diagnostics=error.diagnostics)
        return initial, record
    child = refit_identity(fitted)
    discrepancy = abs(child["score"] - best["score"])
    allowed = 2 * (child["refit_gap"] + best["refit_gap"]) + margin
    if discrepancy > allowed or not child["score_upper"] < parent["score_lower"] - margin:
        record.update(status="unresolved", independent_child=child,
            independent_refit_error="Independent child failed score reconciliation or certified decrease")
        return initial, record
    record.update(status="accepted", accepted=True, child=child,
        comparison_margin=margin, score_reconciliation_difference=discrepancy,
        score_reconciliation_allowed=allowed,
        certified_improvement_margin=parent["score_lower"] - child["score_upper"] - margin)
    return fitted, record
