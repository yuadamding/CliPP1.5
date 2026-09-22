"""Small records; scientific arrays are immutable byte-backed float64 arrays."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def readonly(value, dtype=None):
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


class FitError(ValueError):
    def __init__(self, message, **diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class InputError(FitError):
    pass


class NoEligibleMutationsError(FitError):
    pass


class ClonalConstraintInfeasibleError(FitError):
    pass


class NumericalQualificationError(FitError):
    pass


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    alt: int | None
    ref: int | None
    observed: bool
    normal_cn: float
    states: tuple[tuple[float, int, int], ...]
    exclusion: str | None


@dataclass(frozen=True)
class TumorInput:
    tumor_id: str
    sample_id: str
    purity: float
    mutations: tuple[Mutation, ...]
    input_sha256: str

    @property
    def retained(self):
        return tuple(m for m in self.mutations if m.exclusion is None)


@dataclass(frozen=True)
class CountModel:
    mutation_ids: tuple[str, ...]
    alt: np.ndarray
    ref: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    slope: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    eps: float

    def __post_init__(self):
        for name in ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid"):
            object.__setattr__(self, name, readonly(getattr(self, name),
                                                  bool if name == "valid" else np.float64))

    def __len__(self):
        return len(self.mutation_ids)

    def subset(self, indices):
        idx = np.asarray(indices, dtype=int).reshape(-1)
        return CountModel(tuple(self.mutation_ids[i] for i in idx),
                          *(getattr(self, n)[idx] for n in
                            ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid")),
                          self.eps)


@dataclass(frozen=True)
class LikelihoodTerms:
    loss: np.ndarray
    gradient: np.ndarray | None
    curvature: np.ndarray | None
    posterior: np.ndarray


@dataclass(frozen=True)
class ScalarResult:
    argmin: float
    attained_loss: float
    lower_bound: float
    optimality_gap: float
    qualified: bool
    alternatives: tuple[float, ...] = ()
    intervals: int = 0
    evaluations: int = 0
    bound_evaluations: int = 0
    method: str = "interval_search"


@dataclass(frozen=True)
class PilotResult:
    phi: np.ndarray
    lower_bounds: np.ndarray
    losses: np.ndarray
    curvature: np.ndarray
    alternative_phi: np.ndarray
    gaps: np.ndarray
    scalar_results: tuple[ScalarResult, ...] = ()
    scalar_keys: tuple[str, ...] = ()
    mutation_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("phi", "lower_bounds", "losses", "curvature", "alternative_phi", "gaps"):
            object.__setattr__(self, name, readonly(getattr(self, name), np.float64))


@dataclass(frozen=True)
class FrozenChain:
    order: np.ndarray
    inverse_order: np.ndarray
    weights: np.ndarray
    gap_floor: float
    fingerprint: str

    def __post_init__(self):
        for name in ("order", "inverse_order", "weights"):
            object.__setattr__(self, name, readonly(getattr(self, name)))


@dataclass
class InnerFit:
    x: np.ndarray
    dual: np.ndarray
    gap: float
    qualified: bool
    iterations: int
    kkt_residual: float = float("inf")
    gap_scale: float = 0.0
    algorithm: str = "primal_dual_reference"
    work: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuadraticWitnessProfile:
    """Witness values for one common quadratic, not different likelihood iterates."""
    witness: int
    fit: InnerFit
    relative_objectives: np.ndarray
    objective_offset: float
    qualified: bool
    surrogate_sha256: str
    diagnostics: dict[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "relative_objectives", readonly(self.relative_objectives, np.float64))


@dataclass
class RawFit:
    # x and dual are always in frozen chain order, never input order.
    x: np.ndarray
    dual: np.ndarray
    objective: float
    witness: int
    qualified: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PrimalWarmState:
    """Chain-bound primal continuation for the direct production backend."""
    x: np.ndarray
    chain_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, "x", readonly(self.x, np.float64))

    def validate(self, chain):
        if (self.chain_fingerprint != chain.fingerprint or self.x.shape != chain.order.shape or
                not np.all(np.isfinite(self.x))):
            raise ValueError("Warm state must match the frozen chain and contain a finite primal")


@dataclass(frozen=True)
class WarmState:
    """One previous primal/dual state in chain order, with no witness bounds."""
    x: np.ndarray
    dual: np.ndarray
    chain_fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, "x", readonly(self.x, np.float64))
        object.__setattr__(self, "dual", readonly(self.dual, np.float64))

    def validate(self, chain):
        if (self.chain_fingerprint != chain.fingerprint or self.x.shape != chain.order.shape or
                self.dual.shape != chain.weights.shape or not np.all(np.isfinite(self.x)) or
                not np.all(np.isfinite(self.dual))):
            raise ValueError("Warm state must match the frozen chain and contain finite vectors")


@dataclass(frozen=True)
class PartitionRefit:
    cuts: tuple[int, ...]
    centers: np.ndarray
    designated_clonal_block: int
    loss: float
    gap: float
    score: float
    score_components: dict[str, float]
    scalar_fits_performed: int = 0
    singleton_pilots_reused: int = 0
    scalar_evaluations: int = 0


@dataclass
class FitResult:
    input_identifiers: dict[str, Any]
    retained_mask: np.ndarray
    exclusion_reasons: tuple[str | None, ...]
    pilot: PilotResult
    frozen_chain: FrozenChain
    selected_lambda: float | None
    raw_phi: np.ndarray  # Independent qualified fusion reference, not a direct winner.
    raw_diagnostics: dict[str, Any]
    partition: tuple[int, ...]
    refitted_phi: np.ndarray
    cluster_centers: np.ndarray
    cluster_labels: np.ndarray
    designated_clonal_block: int
    multiplicity_calls: np.ndarray
    selection_score: float
    score_components: dict[str, float]
    provenance: dict[str, Any] = field(default_factory=dict)
    search_diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_objective: float = float("nan")
    raw_witness_index: int = -1
    raw_witness_mutation_id: str = ""
    search_status: str = "incomplete"
    candidate_provenance: dict[str, Any] = field(default_factory=dict)
    raw_reference_lambda: float | None = None
