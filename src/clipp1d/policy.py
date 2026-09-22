"""One versioned inference policy; numerical controls are internal."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    policy_id: str = "clipp1d_chain_v4"
    weight_rule: str = "adaptive_adjacent_gap_v1"
    score_rule: str = "clipp2_compatible_partition_score_v1"
    max_major_cn: int = 4
    eps: float = 1e-6
    scalar_atol: float = 1e-7
    scalar_rtol: float = 1e-10
    scalar_max_intervals: int = 4096
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

    def __post_init__(self):
        if isinstance(self.max_major_cn, bool) or not isinstance(self.max_major_cn, int):
            raise ValueError("max_major_cn must be a positive integer")
        if self.max_major_cn < 1:
            raise ValueError("max_major_cn must be a positive integer")
