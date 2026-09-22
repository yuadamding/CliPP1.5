"""Frozen complete graph in a symmetric-weight/skew-state matrix layout."""

from dataclasses import dataclass
import torch
from .kernels import differences, adjoint


@dataclass(frozen=True)
class CompleteGraph:
    weights: torch.Tensor
    pilot: torch.Tensor
    gap_floor: torch.Tensor
    normalization: torch.Tensor
    mutation_ids: tuple[str, ...] = ()
    weight_rule: str = "all_pairs_inverse_gap_mean_one_adjacent_floor_v1"

    def __post_init__(self):
        values = (self.weights, self.pilot, self.gap_floor, self.normalization)
        n = self.pilot.numel()
        if (
            n == 0
            or self.pilot.ndim != 1
            or self.weights.shape != (n, n)
            or self.gap_floor.ndim != 0
            or self.normalization.ndim != 0
            or any(v.dtype != torch.float64 or v.device != self.pilot.device for v in values)
        ):
            raise ValueError(
                "Complete graph tensors must have canonical float64 shapes and one device"
            )
        diagonal = torch.eye(n, device=self.pilot.device, dtype=torch.bool)
        valid = (
            torch.stack([torch.isfinite(v).all() for v in values]).all()
            & (self.weights == self.weights.T).all()
            & (self.weights.diagonal() == 0).all()
            & ((self.weights > 0) | diagonal).all()
            & (self.gap_floor > 0)
            & (self.normalization > 0)
        )
        if not bool(valid):
            raise ValueError(
                "Complete graph requires finite positive all-pairs weights and a zero diagonal"
            )
        object.__setattr__(
            self,
            "_versions",
            tuple(
                (id(v), v._version)
                for v in (self.weights, self.pilot, self.gap_floor, self.normalization)
            ),
        )
        # Tensor data remain mutable even inside a frozen dataclass. A private
        # value snapshot catches .data writes, which bypass PyTorch's _version.
        object.__setattr__(
            self,
            "_snapshots",
            tuple(
                v.detach().clone()
                for v in (self.weights, self.pilot, self.gap_floor, self.normalization)
            ),
        )
        object.__setattr__(self, "identity", object())

    @property
    def n(self):
        return self.weights.shape[0]

    @property
    def edges(self):
        return self.n * (self.n - 1) // 2

    def validate(self):
        now = tuple(
            (id(v), v._version)
            for v in (self.weights, self.pilot, self.gap_floor, self.normalization)
        )
        if now != self._versions:
            raise ValueError("Frozen graph or pilot was modified")
        values = (self.weights, self.pilot, self.gap_floor, self.normalization)
        same = torch.stack(
            [(value == frozen).all() for value, frozen in zip(values, self._snapshots)]
        ).all()
        if not bool(same):
            raise ValueError("Frozen graph or pilot was modified")


def build_graph(pilot, mutation_ids=()):
    n = pilot.numel()
    if pilot.ndim != 1 or n == 0 or pilot.dtype != torch.float64:
        raise ValueError("Graph requires a nonempty float64 pilot")
    if not bool(torch.isfinite(pilot).all()):
        raise ValueError("Graph requires finite pilot values")
    mutation_ids = tuple(mutation_ids)
    if mutation_ids and (
        len(mutation_ids) != n
        or any(not isinstance(mid, str) for mid in mutation_ids)
        or len(set(mutation_ids)) != n
        or tuple(sorted(mutation_ids)) != mutation_ids
    ):
        raise ValueError("Graph IDs must be unique strings in canonical mutation-ID order")
    sorted_p = torch.sort(pilot).values
    gaps = sorted_p[1:] - sorted_p[:-1]
    # Fixed-size masked sorting avoids a dynamic CUDA nonzero()/boolean gather.
    positive = torch.sort(torch.where(gaps > 0, gaps, float("inf"))).values
    count = (gaps > 0).sum()
    if n > 1:
        lo = ((count - 1).clamp_min(0) // 2).reshape(1)
        hi = (count // 2).clamp_max(n - 2).reshape(1)
        median = (positive.gather(0, lo)[0] + positive.gather(0, hi)[0]) * 0.5
        floor = torch.where(count > 0, median * 0.1, pilot.new_tensor(1e-8)).clamp_min(1e-8)
    else:
        floor = pilot.new_tensor(1e-8)
    raw = differences(pilot).abs().clamp_min(floor).reciprocal()
    raw.fill_diagonal_(0.0)
    norm = raw.sum() / max(1, n * (n - 1))
    norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    return CompleteGraph(raw / norm, pilot.clone(), floor, norm, mutation_ids)


def penalty_reference(model, graph, pilot, curvature):
    graph.validate()
    if model.n == 1:
        return pilot.new_tensor(0.0)
    h = curvature.clamp_min(1.0)
    beta = (h * pilot).sum() / h.sum()
    beta = beta.clamp(model.lower.max(), model.upper.min())
    g = h * (beta - pilot)
    balance = -g.sum()
    active_upper = (beta == model.upper.min()) & (balance > 0)
    active_lower = (beta == model.lower.max()) & (balance < 0)
    active = torch.where(
        active_upper,
        model.upper == beta,
        torch.where(active_lower, model.lower == beta, torch.ones_like(g, dtype=torch.bool)),
    )
    g = g + active * balance / active.sum()
    q = -differences(g) / model.n
    safe_w = torch.where(graph.weights > 0, graph.weights, 1.0)
    ref = (q.abs() / safe_w).max()
    return torch.where(ref > 1e-12, ref, ref.new_tensor(1e-3))


__all__ = ["CompleteGraph", "build_graph", "penalty_reference", "differences", "adjoint"]
