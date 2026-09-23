"""A single upload of canonical float64 likelihood inputs; no fitted CPU state."""
from dataclasses import dataclass
from contextlib import contextmanager
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
        object.__setattr__(self, "_versions", self._metadata())
        object.__setattr__(self, "_snapshots", tuple(v.detach().clone() for v in arrays))
        object.__setattr__(self, "_stage_depth", 0)
        object.__setattr__(self, "integrity_counters", dict(full_checks=0, metadata_checks=0))
        object.__setattr__(self, "scalar_work_counters", dict(
            scalar_rows_evaluated=0, scalar_proposals_evaluated=0, loss_only_calls=0,
            loss_gradient_calls=0, full_terms_calls=0, analytical_groups=0,
            general_groups=0, golden_wells=0, golden_padded_group_slots=0))

    def _metadata(self):
        arrays = (self.alt, self.ref, self.slope, self.log_prior, self.lower, self.upper)
        return (self.mutation_ids, self.eps, id(self.kernels),
                tuple((id(v), v._version, v.shape, v.stride(), v.dtype, v.device, v.data_ptr())
                      for v in arrays))

    def validate(self, *, full=None):
        """Check metadata cheaply inside an owned stage, otherwise reconcile values.

        ``full=True`` always checks immutable snapshots, including writes through
        ``Tensor.data`` that bypass the version counter. Stage entry and exit do
        this unconditionally; no fitted stage result escapes before reconciliation.
        """
        self.integrity_counters["metadata_checks"] += 1
        if self._metadata() != self._versions:
            raise ValueError("Canonical likelihood tensors were modified")
        if full is None:
            full = self._stage_depth == 0
        if full:
            self.integrity_counters["full_checks"] += 1
            arrays = (self.alt, self.ref, self.slope, self.log_prior, self.lower, self.upper)
            if not bool(torch.stack([(value == frozen).all()
                                     for value, frozen in zip(arrays, self._snapshots)]).all()):
                raise ValueError("Canonical likelihood tensors were modified")

    @contextmanager
    def validated_stage(self):
        outermost = self._stage_depth == 0
        self.validate(full=outermost)
        object.__setattr__(self, "_stage_depth", self._stage_depth + 1)
        try:
            yield self
        finally:
            object.__setattr__(self, "_stage_depth", self._stage_depth - 1)
            self.validate(full=outermost)

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
        result = TensorModel((), self.alt[idx], self.ref[idx], self.slope[idx], self.log_prior[idx],
                             self.lower[idx], self.upper[idx], self.eps, self.kernels)
        object.__setattr__(result, "integrity_counters", self.integrity_counters)
        object.__setattr__(result, "scalar_work_counters", self.scalar_work_counters)
        return result

    def _validate_point(self, x):
        self.validate()
        if x.shape != (self.n,) or x.dtype != torch.float64 or x.device != self.device:
            raise ValueError("Likelihood values must be a float64 vector on the model device")

    def loss(self, x):
        self._validate_point(x)
        self.scalar_work_counters["loss_only_calls"] += 1
        return self.kernels.loss_only(self.alt, self.ref, self.slope, self.log_prior,
                                      x[:, None], self.eps)[:, 0]

    def loss_gradient(self, x):
        self._validate_point(x)
        self.scalar_work_counters["loss_gradient_calls"] += 1
        out = self.kernels.loss_gradient(self.alt, self.ref, self.slope, self.log_prior,
                                         x[:, None], self.eps)
        return tuple(value[:, 0] for value in out)

    def terms(self, x):
        self._validate_point(x)
        self.scalar_work_counters["full_terms_calls"] += 1
        out = self.kernels.likelihood(self.alt, self.ref, self.slope, self.log_prior,
                                      x[:, None], self.eps)
        return tuple(v[:, 0] for v in out)
