"""A single upload of canonical float64 likelihood inputs; no fitted CPU state."""
from dataclasses import dataclass
import math
import torch
from .kernels import Kernels


@dataclass(frozen=True)
class TensorModel:
    mutation_ids: tuple[str, ...]
    alt: torch.Tensor
    ref: torch.Tensor
    slope: torch.Tensor
    log_prior: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    eps: float
    kernels: Kernels

    def __post_init__(self):
        arrays = (self.alt, self.ref, self.slope, self.log_prior, self.lower, self.upper)
        if any(v.dtype != torch.float64 or v.device != self.alt.device for v in arrays):
            raise ValueError("Source model tensors must share float64 dtype and device")
        n = self.alt.numel()
        if (n == 0 or any(v.shape != (n,) for v in (self.alt, self.ref, self.lower, self.upper))
                or self.slope.ndim != 2 or self.slope.shape[0] != n or self.slope.shape[1] == 0
                or self.log_prior.shape != self.slope.shape):
            raise ValueError("Invalid source likelihood tensor dimensions")
        if self.mutation_ids and (len(self.mutation_ids) != n or
                                  len(set(self.mutation_ids)) != n or
                                  any(not isinstance(x, str) for x in self.mutation_ids)):
            raise ValueError("Source mutation IDs must be unique strings matching the rows")
        if not math.isfinite(self.eps) or not 0 < self.eps < .5:
            raise ValueError("Invalid likelihood probability clipping bounds")
        valid = torch.isfinite(self.log_prior)
        finite = (torch.isfinite(self.alt).all() & torch.isfinite(self.ref).all() &
                  torch.isfinite(self.slope).all() & torch.isfinite(self.lower).all() &
                  torch.isfinite(self.upper).all())
        domain = ((self.alt >= 0).all() & (self.ref >= 0).all() &
                  (self.alt + self.ref > 0).all() & (self.slope >= 0).all() &
                  (self.lower <= self.upper).all() & valid.any(-1).all() &
                  (valid | torch.isneginf(self.log_prior)).all() &
                  ((self.slope > 0) | ~valid).all())
        if not bool(finite & domain):
            raise ValueError("Invalid source counts, supports, or original feasibility bounds")
        object.__setattr__(self, "_versions", tuple((id(v), v._version) for v in arrays))
        object.__setattr__(self, "_snapshots", tuple(v.detach().clone() for v in arrays))

    def validate(self):
        arrays = (self.alt, self.ref, self.slope, self.log_prior, self.lower, self.upper)
        if tuple((id(v), v._version) for v in arrays) != self._versions:
            raise ValueError("Canonical likelihood tensors were modified")
        if not bool(torch.stack([(value == frozen).all()
                                 for value, frozen in zip(arrays, self._snapshots)]).all()):
            raise ValueError("Canonical likelihood tensors were modified")

    @classmethod
    def from_host(cls, model, device="cuda:0", compiled=True):
        device = torch.device(device)
        arrays = {name: torch.tensor(getattr(model, name).copy(), dtype=torch.float64, device=device)
                  for name in ("alt", "ref", "slope", "log_prior", "lower", "upper")}
        return cls(tuple(model.mutation_ids), eps=model.eps, kernels=Kernels(device, compiled), **arrays)

    @property
    def n(self):
        return self.alt.numel()

    @property
    def device(self):
        return self.alt.device

    def subset(self, idx):
        # IDs stay in host metadata. Numeric indexing remains entirely on device.
        self.validate()
        return TensorModel((), self.alt[idx], self.ref[idx], self.slope[idx], self.log_prior[idx],
                           self.lower[idx], self.upper[idx], self.eps, self.kernels)

    def terms(self, x):
        self.validate()
        if x.shape != (self.n,) or x.dtype != torch.float64 or x.device != self.device:
            raise ValueError("Likelihood values must be a float64 vector on the model device")
        out = self.kernels.likelihood(self.alt, self.ref, self.slope, self.log_prior,
                                      x[:, None], self.eps)
        return tuple(v[:, 0] for v in out)
