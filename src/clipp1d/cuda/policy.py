"""One FP64 complete-graph policy; changes from the chain are explicit."""
from dataclasses import dataclass
import math
from ..types import NumericalQualificationError


@dataclass(frozen=True)
class CudaPolicy:
    policy_id: str = "clipp1d_complete_cuda_unconstrained_v2"
    weight_rule: str = "all_pairs_inverse_gap_mean_one_adjacent_floor_v1"
    eps: float = 1e-6
    scalar_atol: float = 1e-7
    scalar_rtol: float = 1e-10
    scalar_max_intervals: int = 4096
    scalar_batch_size: int = 64
    inner_atol: float = 1e-10
    inner_rtol: float = 1e-11
    inner_kkt_tol: float = 1e-7
    inner_max_iterations: int = 20000
    outer_max_iterations: int = 150
    max_backtracks: int = 24
    stationarity_tol: float = 2e-5
    fusion_tol: float = 2e-5
    path_min_exponent: int = -12
    path_max_exponent: int = 12
    path_extensions: int = 3
    check_every: int = 16
    memory_fraction: float = 0.7

    def __post_init__(self):
        if not (0 < self.eps < .5 and 0 < self.memory_fraction < 1):
            raise ValueError("Invalid probability or memory policy")
        for key in ("scalar_atol", "scalar_rtol", "inner_atol", "inner_rtol",
                    "inner_kkt_tol", "stationarity_tol", "fusion_tol"):
            if not 0 < getattr(self, key) < float("inf"):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("scalar_max_intervals", "scalar_batch_size", "inner_max_iterations",
                    "outer_max_iterations", "max_backtracks", "check_every"):
            val = getattr(self, key)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("path_min_exponent", "path_max_exponent", "path_extensions"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if not math.isfinite(self.memory_fraction) or not math.isfinite(self.eps):
            raise ValueError("Probability and memory policies must be finite")
        if self.path_min_exponent > self.path_max_exponent or self.path_extensions < 0:
            raise ValueError("Invalid penalty path")


class QualificationError(NumericalQualificationError):
    """A numerical failure is unresolved, never an approximate success."""

    def __init__(self, message, **diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics
