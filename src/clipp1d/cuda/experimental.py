"""Opt-in, isolated single-start experiments; never alters production continuation."""
from dataclasses import dataclass
from copy import deepcopy
import torch

from .audit import audit_raw
from .partition import _Snapshot
from .policy import CudaPolicy, QualificationError
from .solver import objective, solve_start


@dataclass(frozen=True)
class ExtraStartResult:
    """A result can be consumed only under the model/graph that actually solved it."""
    model: object
    graph: object
    penalty: torch.Tensor
    policy: CudaPolicy
    raw: object

    def __post_init__(self):
        object.__setattr__(self, "penalty", self.penalty.detach().clone())
        object.__setattr__(self, "_snapshot", _Snapshot(self._tensors()))
        object.__setattr__(self, "_diagnostics", deepcopy(self.raw.diagnostics))
        object.__setattr__(self, "_qualified", self.raw.qualified)

    def _tensors(self):
        return self.penalty, self.raw.x, self.raw.dual, self.raw.objective

    def validate(self, model, graph, penalty, policy):
        if model is not self.model or graph is not self.graph or policy != self.policy:
            raise ValueError("Extra-start result is bound to its exact model, graph and policy")
        model.validate(full=True)
        graph.validate(full=True)
        if not torch.equal(penalty, self.penalty):
            raise ValueError("Extra-start result is bound to its absolute penalty")
        self._snapshot.validate(self._tensors(), "Extra-start result")
        if self.raw.diagnostics != self._diagnostics or self.raw.qualified != self._qualified:
            raise ValueError("Extra-start qualification was modified")


@torch.no_grad()
def solve_extra_start(model, graph, penalty, start, policy=CudaPolicy()):
    """Fresh original-objective solve and audit, with no dual or certificate reuse.

    Only a feasible coordinate tensor is accepted as a start. Callers may obtain
    it from another likelihood, but that likelihood's pilots, caches and raw
    certificates cannot enter this function. Failure is explicit and caller-owned.
    """
    model.validate(full=True)
    graph.validate(full=True)
    if model.mutation_ids != graph.mutation_ids or model.n != graph.n or model.eps != policy.eps:
        raise ValueError("Experimental model/graph identities or clipping differ")
    if (not isinstance(penalty, torch.Tensor) or penalty.ndim != 0 or
            penalty.dtype != torch.float64 or penalty.device != model.device or
            not bool(torch.isfinite(penalty) & (penalty > 0))):
        raise ValueError("Extra starts require a finite positive absolute penalty")
    if (not isinstance(start, torch.Tensor) or start.shape != model.lower.shape or
            start.dtype != torch.float64 or start.device != model.device or
            not bool(torch.isfinite(start).all() & (start >= model.lower).all() &
                     (start <= model.upper).all())):
        raise ValueError("Extra start must be a feasible float64 coordinate vector")
    raw = solve_start(model, graph, penalty, start.detach().clone(), policy)
    if raw.qualified:
        caps = graph.weights * penalty
        audit = audit_raw(model, raw.x, raw.dual, caps, None, policy)
        observed = objective(model, raw.x, caps)
        margin = 128 * torch.finfo(torch.float64).eps * (1 + observed.abs())
        if not bool(audit.qualified and torch.isfinite(observed) and
                    (raw.objective-observed).abs() <= margin):
            raise QualificationError("Extra start failed independent original-objective audit")
    return ExtraStartResult(model, graph, penalty, policy, raw)
