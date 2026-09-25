"""CliPPSim4K generation with CN-first, uniform conditional multiplicity.

Install ``clipp1d[simulation]`` for pandas and tqdm. See docs/SIMULATION.md
for the source identity, generation contract, and reproducibility checks.
"""

from .generate_clippsim4k import ClusterDesign, SimulationConfig, generate_cohort

__all__ = ["ClusterDesign", "SimulationConfig", "generate_cohort"]
