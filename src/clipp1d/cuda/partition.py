"""GPU memberships and independent, unconstrained secondary likelihood refits."""
from dataclasses import dataclass
import torch
from .scalar import Problems, ScalarBatch, solve_scalar
from .policy import CudaPolicy, QualificationError
from .kernels import differences, adjoint


def grouping(x, tolerance, *, separate_clonal=True):
    """Bounded-diameter groups, canonically labeled by their minimum node index.

    Current-value order only determines tolerance runs. An overwide connected
    tolerance run becomes singletons, preserving the CUDA overlay's conservative
    parallel rule. Exact-one nodes never group with merely near-one nodes.
    """
    if x.ndim != 1 or x.numel() == 0 or not 0 <= tolerance < float("inf"):
        raise ValueError("Grouping requires a nonempty vector and finite nonnegative tolerance")
    if x.dtype != torch.float64 or not bool(torch.isfinite(x).all()):
        raise ValueError("Grouping requires finite float64 CCFs")
    n = x.numel()
    value_order = torch.argsort(x, stable=True)
    values = x[value_order]
    idx = torch.arange(n, device=x.device)
    cuts = torch.ones(n, device=x.device, dtype=torch.bool)
    cuts[1:] = values[1:] - values[:-1] > tolerance
    if separate_clonal:
        cuts[1:] |= (values[1:] == 1) != (values[:-1] == 1)
    start = torch.cummax(torch.where(cuts, idx, 0), 0).values
    stops = torch.ones_like(cuts)
    stops[:-1] = cuts[1:]
    end = torch.cummin(torch.where(stops, idx, n - 1).flip(0), 0).values.flip(0)
    cuts |= values[end] - values[start] > tolerance
    value_labels = cuts.long().cumsum(0) - 1
    first_node = torch.full((n,), n, device=x.device, dtype=torch.long)
    first_node.scatter_reduce_(0, value_labels, value_order, reduce="amin")
    block_order = torch.argsort(first_node, stable=True)
    canonical = torch.empty_like(block_order)
    canonical[block_order] = idx
    labels = torch.empty_like(value_labels)
    labels[value_order] = canonical[value_labels]
    order = torch.argsort(labels, stable=True)
    sorted_labels = labels[order]
    counts = torch.zeros(n, dtype=torch.long, device=x.device)
    counts.scatter_add_(0, labels, torch.ones_like(labels))
    return order, sorted_labels, labels, counts


def polish_quadratic(x, q, h, target, lower, upper, caps, tolerance):
    """A numerical equality proposal only; caller must re-audit the original QP."""
    order, _, labels, lengths = grouping(x, tolerance, separate_clonal=False)
    same = labels[:, None] == labels[None, :]
    external = adjoint(torch.where(same, 0., caps * differences(x).sign()))

    def reduce(v, op="sum"):
        return torch.segment_reduce(v[order], op, lengths=lengths)

    total_h = reduce(h)
    safe_target = torch.where(lower < upper, target, lower)
    center = reduce(h * safe_target - external) / total_h.clamp_min(torch.finfo(x.dtype).tiny)
    lo, hi = reduce(lower, "max"), reduce(upper, "min")
    center = torch.maximum(lo, torch.minimum(hi, center))
    candidate = center[labels]
    return torch.where(torch.isfinite(candidate) & (lo <= hi)[labels], candidate, x)


@dataclass
class Refit:
    labels: torch.Tensor
    centers: torch.Tensor
    phi: torch.Tensor
    sizes: torch.Tensor
    clonal: int  # Closest-to-one designation only; no center is fixed at one.
    loss: torch.Tensor
    gap: torch.Tensor
    score: torch.Tensor
    scalar: ScalarBatch


def partition_score(loss, sizes):
    if (sizes.ndim != 1 or sizes.numel() == 0 or
            not bool(torch.isfinite(sizes).all() & (sizes > 0).all() &
                     (sizes == sizes.round()).all() & torch.isfinite(loss))):
        raise ValueError("Score requires finite loss and occupied integer memberships")
    k = sizes.numel()
    n = sizes.sum().to(torch.float64)
    s = sizes.to(torch.float64)
    kt = n.new_tensor(float(k))
    mass = torch.lgamma(kt) - torch.lgamma(n + kt) + torch.lgamma(s + 1).sum() + torch.lgamma(kt + 1)
    score = 2 * loss + kt * n.log() - 1.4 * mass
    if not bool(torch.isfinite(score)):
        raise QualificationError("Partition score is nonfinite")
    return score


def refit(model, x, policy=CudaPolicy()):
    """Refit arbitrary memberships in original boxes; designate clonal post hoc."""
    if x.shape != model.lower.shape or x.device != model.device:
        raise ValueError("Refit CCF vector must match the model")
    model.validate()
    order, _, labels, lengths = grouping(x, policy.fusion_tol)
    k = int((lengths > 0).sum())  # One output-shape/control decision per candidate.
    lengths = lengths[:k]
    ordered = model.subset(order)
    problems = Problems(ordered, lengths)
    scalar = solve_scalar(problems, policy)
    valid = (scalar.qualified.all() & torch.isfinite(scalar.phi).all() &
             torch.isfinite(scalar.loss).all() & torch.isfinite(scalar.lower_bound).all() &
             torch.isfinite(scalar.gap).all() & (scalar.gap >= 0).all() &
             (scalar.phi >= problems.lower).all() & (scalar.phi <= problems.upper).all() &
             (scalar.gap <= policy.scalar_atol + policy.scalar_rtol * scalar.loss.abs()).all())
    if not bool(valid):
        raise QualificationError("Secondary membership refit unresolved", maximum_gap=float(scalar.gap.max()))
    centers = scalar.phi.clone()
    clonal = int((centers - 1.).abs().argmin())
    loss, gap = scalar.loss.sum(), scalar.gap.sum()
    if not bool(torch.isfinite(loss) & torch.isfinite(gap)):
        raise QualificationError("Aggregate membership loss or scalar gap is nonfinite")
    return Refit(labels, centers, centers[labels], lengths, clonal, loss, gap,
                 partition_score(loss, lengths), scalar)
