"""Small versioned CN-first generator for current-estimator validation.

This is a new, independently labelled fixture generator, not a reconstruction of
unavailable truth for the legacy sample or the historical 4K cohort generator.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

GENERATOR_VERSION = "autosomal_cnfirst_binomial_pcg64_v1"


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 1701
    mutations: int = 60
    purity: float = 0.8
    centers: tuple = (1.0, 0.55, 0.2)
    proportions: tuple = (0.4, 0.4, 0.2)
    cn_states: tuple = ((1, 1), (2, 0), (2, 2), (4, 1), (6, 2))
    mean_depth: float = 200.0
    multiplicity_policy: str = "uniform"
    overdispersion: float = 0.0
    reported_purity: float | None = None
    minimum_alt: int = 0

    def validate(self):
        if (
            type(self.seed) is not int
            or self.seed < 0
            or type(self.mutations) is not int
            or self.mutations < 1
        ):
            raise ValueError("Require nonnegative integer seed and positive mutation count")
        centers, weights = np.array(self.centers), np.array(self.proportions)
        if (
            centers.ndim != 1
            or not len(centers)
            or len(centers) != len(weights)
            or not np.isfinite(centers).all()
            or np.any((centers < 0) | (centers > 1))
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("Invalid centers/proportions")
        if not 0 < self.purity <= 1 or (
            self.reported_purity is not None and not 0 < self.reported_purity <= 1
        ):
            raise ValueError("Invalid purity")
        if not np.isfinite(self.mean_depth) or self.mean_depth <= 0 or not 0 <= self.overdispersion < 1:
            raise ValueError("Invalid depth/overdispersion")
        if type(self.minimum_alt) is not int or self.minimum_alt < 0:
            raise ValueError("minimum_alt must be nonnegative integer")
        if self.multiplicity_policy not in {"uniform", "one"}:
            raise ValueError("Unknown multiplicity policy")
        states = np.array(self.cn_states)
        if (
            states.ndim != 2
            or states.shape[1] != 2
            or not len(states)
            or not np.isfinite(states).all()
            or np.any(states != np.floor(states))
            or np.any(states[:, 0] < 1)
            or np.any(states[:, 1] < 0)
            or np.any(states[:, 0] < states[:, 1])
        ):
            raise ValueError("Invalid major/minor CN states")


def generate(directory, config=None):
    """Create input and separate truth files in a new directory; never overwrite."""
    config = config or SimulationConfig()
    config.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.Generator(np.random.PCG64(config.seed))
    count = config.mutations
    labels = rng.choice(len(config.centers), count, p=config.proportions)
    cn = np.array(config.cn_states, dtype=int)[rng.integers(len(config.cn_states), size=count)]
    major, minor = cn.T
    multiplicity = (
        rng.integers(1, major + 1) if config.multiplicity_policy == "uniform" else np.ones(count, dtype=int)
    )
    ccf = np.array(config.centers)[labels]
    p = config.purity * ccf * multiplicity / (2 * (1 - config.purity) + config.purity * (major + minor))
    if config.overdispersion:
        concentration = 1 / config.overdispersion - 1
        interior = (p > 0) & (p < 1)
        p[interior] = rng.beta(p[interior] * concentration, (1 - p[interior]) * concentration)
    depth = np.maximum(1, rng.poisson(config.mean_depth, count))
    alt = rng.binomial(depth, p)
    positions = 1000 + 10 * np.arange(count)
    keep = alt >= config.minimum_alt
    if not keep.any():
        raise ValueError("Ascertainment excluded every mutation")
    ids = np.array([f"m{i + 1:05d}" for i in range(count)])
    pd.DataFrame(
        {
            "mutation_id": ids[keep],
            "chromosome_index": 1,
            "position": positions[keep],
            "ref_count": (depth - alt)[keep],
            "alt_count": alt[keep],
        }
    ).to_csv(root / "snv.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "chromosome_index": 1,
            "start_position": positions,
            "end_position": positions,
            "major_cn": major,
            "minor_cn": minor,
            "total_cn": major + minor,
        }
    ).to_csv(root / "cna.tsv", sep="\t", index=False)
    (root / "purity.txt").write_text(
        f"{config.purity if config.reported_purity is None else config.reported_purity:.17g}\n"
    )
    truth = pd.DataFrame(
        {
            "mutation_id": ids,
            "cluster": labels,
            "ccf": ccf,
            "cp": ccf * config.purity,
            "multiplicity": multiplicity,
            "major_cn": major,
            "minor_cn": minor,
            "ascertained": keep,
        }
    )
    truth.to_csv(root / "truth.tsv", sep="\t", index=False, float_format="%.17g")
    from .native import sha256

    receipt = {
        "generator_version": GENERATOR_VERSION,
        "config": asdict(config),
        "rng": "numpy.PCG64",
        "numpy_version": np.__version__,
        "requested_mutations": count,
        "input_mutations": int(keep.sum()),
        "files": {p.name: sha256(p) for p in sorted(root.iterdir())},
    }
    (root / "simulation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
