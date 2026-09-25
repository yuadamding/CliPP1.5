"""Truth-free partition proposals using the unchanged CUDA likelihood and score.

Parallel likelihood/move evaluation, sequential deterministic acceptance. No
proposal changes a graph, raw continuation state, or raw solver certificate.
"""
from dataclasses import dataclass
import torch
from .partition import canonical_labels, partition_score, refit_labels
from .policy import CudaPolicy, QualificationError


@dataclass(frozen=True)
class PartitionSearchPolicy:
    policy_id: str = "explicit_partition_search_v1"
    all_starts: bool = True
    separate_exact_one: bool = True  # Retained baseline; False is a distinct grouping ablation.
    max_rounds: int = 4
    max_moves: int = 10000  # Per fixed-center sweep, reported if exhausted.
    cost_block_size: int = 128
    minimum_decrease: float = 1e-8

    def __post_init__(self):
        for name in ("all_starts", "separate_exact_one"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("max_rounds", "max_moves", "cost_block_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.minimum_decrease < float("inf"):
            raise ValueError("minimum_decrease must be finite and positive")


def center_costs(model, centers, block_size=128):
    """Use the existing loss-only kernel in bounded row/proposal blocks."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("Cost block size must be a positive integer")
    if (centers.ndim != 1 or centers.numel() == 0 or centers.dtype != torch.float64 or
            centers.device != model.device or not bool(torch.isfinite(centers).all())):
        raise ValueError("Centers must be a finite float64 vector on the model device")
    costs = centers.new_empty((model.n, centers.numel()))
    with model.validated_stage():
        for start in range(0, model.n, block_size):
            stop = min(start + block_size, model.n)
            for first in range(0, centers.numel(), block_size):
                last = min(first + block_size, centers.numel())
                points = centers[first:last].expand(stop - start, -1)
                costs[start:stop, first:last] = model.kernels.loss_only(
                    model.alt[start:stop], model.ref[start:stop], model.slope[start:stop],
                    model.log_prior[start:stop], points, model.eps)
    if not bool(torch.isfinite(costs).all()):
        raise QualificationError("Partition cost matrix is nonfinite")
    return costs


def move_deltas(costs, labels, sizes, feasible):
    """Exact current-score differences; destination/self/empty masks included."""
    n = labels.numel()
    rows = torch.arange(n, device=labels.device)
    occupied = sizes > 0
    k = occupied.sum().to(torch.float64)
    source_size = sizes[labels].to(torch.float64)
    destination_size = sizes.to(torch.float64)
    delta = 2 * (costs - costs[rows, labels, None])
    delta += 1.4 * torch.log(source_size[:, None] / (destination_size[None, :] + 1))
    # B(K-1)-B(K), added only when the source singleton disappears. For K=1
    # all moves are masked; clamp keeps unused intermediate arithmetic finite.
    correction = -torch.log(k.new_tensor(float(n))) - 1.4 * torch.log(
        (n + k - 1) / (k * (k - 1).clamp_min(1)))
    delta += torch.where(source_size[:, None] == 1, correction, 0.)
    allowed = feasible & occupied[None, :] & (sizes[labels, None] > 0)
    allowed[rows, labels] = False
    return torch.where(allowed, delta, float("inf"))


@dataclass
class Reassignment:
    labels: torch.Tensor
    centers: torch.Tensor
    loss: torch.Tensor
    score: torch.Tensor
    moves: list
    status: str


@torch.no_grad()
def reassign_fixed_centers(model, labels, centers, search_policy=PartitionSearchPolicy()):
    """One deterministic sweep to a one-move fixed point or a declared budget.

    Input centers follow canonical first-node membership order. Ties choose the
    first canonical mutation, then the sweep's original center order. No stale
    count batches are accepted. Membership tensors and centers are never edited
    in place in the caller's candidate.
    """
    with model.validated_stage():
        y, _, sizes = canonical_labels(labels)
        if y.shape != model.lower.shape or y.device != model.device or centers.numel() != sizes.numel():
            raise ValueError("Memberships and canonical centers must match the model")
        centers = centers.clone()
        costs = center_costs(model, centers, search_policy.cost_block_size)
        feasible = (centers[None, :] >= model.lower[:, None]) & (centers[None, :] <= model.upper[:, None])
        rows = torch.arange(model.n, device=model.device)
        if not bool(feasible[rows, y].all()):
            raise QualificationError("Initial partition is outside original bounds")
        loss = costs[rows, y].sum()
        score = partition_score(loss, sizes)
        moves = []
        status = "fixed_point"
        for _ in range(search_policy.max_moves):
            if int((sizes > 0).sum()) == 1:
                break
            delta = move_deltas(costs, y, sizes, feasible)
            flat = int(delta.argmin())
            node, destination = divmod(flat, centers.numel())
            predicted = delta[node, destination]
            margin = max(search_policy.minimum_decrease,
                         128 * torch.finfo(torch.float64).eps * (1 + abs(float(score))))
            if not bool(predicted < -margin):
                break
            source = int(y[node])
            proposal = y.clone()
            proposal[node] = destination
            counts = sizes.clone()
            counts[source] -= 1
            counts[destination] += 1
            new_loss = costs[rows, proposal].sum()
            new_score = partition_score(new_loss, counts[counts > 0])
            difference = new_score - score
            check_margin = 512 * torch.finfo(torch.float64).eps * (1 + score.abs() + new_score.abs())
            if not bool((difference - predicted).abs() <= check_margin):
                raise QualificationError("Incremental move disagrees with the complete partition score")
            if not bool(new_score < score - margin):
                # A rounded/tiny move is not an accepted optimization step.
                status = "roundoff_stop"
                break
            moves.append(dict(node=node, source=source, destination=destination,
                              score_before=float(score), score_after=float(new_score)))
            y, sizes, loss, score = proposal, counts, new_loss, new_score
        else:
            status = "move_budget_exhausted"
        renamed, order, lengths = canonical_labels(y)
        first_nodes = order[lengths.cumsum(0) - lengths]
        return Reassignment(renamed, centers[y[first_nodes]], loss, score, moves, status)


def refine_memberships(model, initial, policy=CudaPolicy(), search_policy=PartitionSearchPolicy()):
    """Alternate exact-score membership moves and independently qualified refits.

    Original candidates remain available when a proposal is unresolved or a
    budget ends. The returned status is separate from raw-path completeness.
    """
    best = initial
    records = []
    for iteration in range(search_policy.max_rounds):
        moved = reassign_fixed_centers(model, best.labels, best.centers, search_policy)
        record = dict(round=iteration, fixed_center_status=moved.status, moves=moved.moves,
                      seed_score=float(best.score), fixed_center_score=float(moved.score))
        if not moved.moves:
            record.update(status=moved.status, refit_status="not_needed")
            records.append(record)
            return best, records, moved.status
        try:
            fitted = refit_labels(model, moved.labels, policy, incumbent_centers=moved.centers)
        except QualificationError as error:
            record.update(status="unresolved", refit_status="unresolved", error=str(error),
                          failure_diagnostics=error.diagnostics)
            records.append(record)
            return best, records, "unresolved"
        if not bool(fitted.score <= moved.score +
                    256 * torch.finfo(torch.float64).eps * (1 + moved.score.abs())):
            raise QualificationError("Qualified refit lost its feasible incumbent score")
        if not bool(fitted.score < best.score - search_policy.minimum_decrease):
            record.update(status="no_qualified_improvement", refit_status="qualified", score=float(fitted.score))
            records.append(record)
            return best, records, "roundoff_stop"
        best = fitted
        record.update(status="accepted", refit_status="qualified", score=float(fitted.score),
                      refit_gap=float(fitted.gap), clusters=fitted.centers.numel())
        records.append(record)
    return best, records, "round_budget_exhausted"
