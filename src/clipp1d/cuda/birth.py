"""Whole-group split proposals under the unchanged global partition score.

The default grid reproduces the frozen K=1 discovery procedure. Conditional
splits are a separate mode: exact over memberships for a fixed center pair,
never a certificate of global optimality over centers or partitions.
"""
from dataclasses import dataclass
import torch
from .partition import canonical_labels, refit_labels
from .policy import CudaPolicy, QualificationError
from .refinement import PartitionSearchPolicy, center_costs


def split_complexity(n_total, k, n_parent, child_size):
    """Tumor-wide K -> K+1 complexity increment, including allocation mass."""
    t = child_size.to(torch.float64)
    if not (1 <= k <= n_total and 2 <= n_parent <= n_total - k + 1):
        raise ValueError("Invalid global split dimensions")
    if not bool(((t >= 1) & (t < n_parent) & (t == t.round())).all()):
        raise ValueError("Both split children must be occupied")
    n = t.new_tensor(float(n_parent))
    return (t.new_tensor(float(n_total)).log() +
            1.4 * t.new_tensor((n_total + k) / (k * (k + 1))).log() +
            1.4 * (torch.lgamma(n + 1) - torch.lgamma(t + 1) - torch.lgamma(n - t + 1)))


def conditional_split(costs, feasible, *, n_total, k, parent_loss):
    """Optimal nonempty binary assignment conditional on two feasible centers.

Returns local child labels and global score increment, or None for an
infeasible pair. Stable likelihood-difference sorting breaks ties by node ID.
Forced assignments are included before scanning every flexible prefix size.
"""
    if (costs.ndim != 2 or costs.shape[1] != 2 or feasible.shape != costs.shape or
            feasible.dtype != torch.bool or feasible.device != costs.device or
            costs.dtype != torch.float64 or not bool(torch.isfinite(costs).all())):
        raise ValueError("Expected finite float64 two-center costs and feasibility mask")
    n = costs.shape[0]
    if n < 2 or not bool(feasible.any(1).all()):
        return None
    only_v = ~feasible[:, 0]
    only_u = ~feasible[:, 1]
    flexible = torch.nonzero(feasible.all(1)).flatten()
    differences = costs[flexible, 1] - costs[flexible, 0]
    order = torch.argsort(differences, stable=True)
    prefix = torch.cat((costs.new_zeros(1), differences[order].cumsum(0)))
    counts = torch.arange(prefix.numel(), device=costs.device) + only_v.sum()
    valid = (counts > 0) & (counts < n)
    if not bool(valid.any()):
        return None
    base = costs[only_v, 1].sum() + costs[only_u, 0].sum() + costs[flexible, 0].sum()
    slots = torch.nonzero(valid).flatten()
    deltas = (2 * (base + prefix[valid] - parent_loss) +
              split_complexity(n_total, k, n, counts[valid]))
    best = int(deltas.argmin())
    labels = only_v.long()
    labels[flexible[order[:int(slots[best])]]] = 1
    return labels, deltas[best]


@dataclass
class BirthCandidate:
    refit: object
    proposal: dict


@torch.no_grad()
def birth_candidates(model, parent, policy=CudaPolicy(), search_policy=PartitionSearchPolicy()):
    """Stream independently qualified full partitions, never singleton moves.

Initial-center scores never screen refitted candidates. A candidate may be
worse than its parent and still seed a bounded later refinement/generation.
No truth, raw multiplicity calls or pilot-order contiguity enter this search.
"""
    k = parent.centers.numel()
    if (search_policy.birth_mode == "off" or
            (search_policy.birth_mode == "single_cluster" and k != 1)):
        return
    centers = torch.linspace(.05, 1., 20, device=model.device, dtype=torch.float64)
    costs = center_costs(model, centers, search_policy.cost_block_size)
    feasible = (centers[None, :] >= model.lower[:, None]) & (centers[None, :] <= model.upper[:, None])
    # Deterministic bounded parent order, no exemption for designated clonal.
    parents = [j for j in range(k) if int(parent.sizes[j]) > 1]
    for group in parents[:search_policy.birth_max_parents]:
        nodes = torch.nonzero(parent.labels == group).flatten()
        seen = []  # At most 19 memberships for this parent.
        for low in range(19):
            pair_cost = costs[nodes][:, [low, 19]]
            allowed = feasible[nodes][:, [low, 19]]
            proposal = dict(algorithm=("grid_k1_birth_v1" if search_policy.birth_mode == "single_cluster"
                                       else "conditional_global_split_v1"),
                            parent_group=group, center_pair=[float(centers[low]), 1.],
                            tumor_n=model.n, occupied_k=k, parent_size=nodes.numel())
            if not bool(allowed.any(1).all()):
                yield None, dict(proposal, status="infeasible_center_pair")
                continue
            if search_policy.birth_mode == "single_cluster":
                local = torch.where(allowed, pair_cost, float("inf")).argmin(1)
                if local.unique().numel() != 2:
                    continue
            else:
                split = conditional_split(pair_cost, allowed, n_total=model.n, k=k,
                                          parent_loss=parent.scalar.loss[group])
                if split is None:
                    continue
                local, delta = split
                proposal["conditional_fixed_center_score_delta"] = float(delta)
            labels = parent.labels.clone()
            labels[nodes[local == 1]] = k
            labels = canonical_labels(labels)[0]
            if any(torch.equal(labels[nodes], prior) for prior in seen):
                continue
            seen.append(labels[nodes].clone())
            try:
                fitted = refit_labels(model, labels, policy)
            except QualificationError as error:
                yield None, dict(proposal, status="unresolved", error=str(error),
                                 failure_diagnostics=error.diagnostics)
                continue
            # Refit the complete tumor, so K>1 proposals never use a local score.
            proposal.update(status="qualified", child_sizes=torch.bincount(local).tolist())
            yield BirthCandidate(fitted, proposal), None
    if len(parents) > search_policy.birth_max_parents:
        yield None, dict(status="parent_budget_exhausted", proposed_parents=search_policy.birth_max_parents,
                         eligible_parents=len(parents))
