#!/home/yding1995/miniforge3/envs/ml1/bin/python
"""Generate raw CliPPSim cohort.

The default design contains exactly 4,000 valid cases distributed as evenly
as possible across the 27 sample-level design cells:

    3 depths x 3 purities x 3 CNA rates

Each cell receives 148 samples and four reproducibly selected cells receive
one additional sample.

Each case contains cna.txt, snv.txt, purity.txt, and a consolidated truth.txt.
For each sample, raw true sMF is drawn from Uniform[0, 0.8). Values below 0.1
are set to zero and produce K = 1. Otherwise, K is drawn from the feasible
values in {2, 3, 4}, subject to at least 20 SNVs per cluster. The root-level
generation_summary.tsv records the raw, thresholded, and realized sMF values,
total SNV count, eligible cluster counts, and realized cluster sizes.
Major and minor copy numbers are generated before mutation multiplicity.
Conditional on major CN, multiplicity is uniform on the integers 1..major CN,
including at balanced amplified loci. This revises the original generator's
mutation-bearing-allele rule while preserving its copy-number distribution.
The maf.txt files found in the analyzed cohort were created by a later
PhylogicNDT input-preparation step and are intentionally not generated here.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DEPTHS = (100, 200, 500)
DEFAULT_SUBCLONAL_CLUSTERS = (2, 3, 4)
DEFAULT_PURITIES = (0.4, 0.6, 0.9)
DEFAULT_CNA_RATES = (0.1, 0.4, 0.7)
DEFAULT_SAMPLE_COUNT = 4000
DEFAULT_SEED = 20260730
MIN_TOTAL_SNVS = 200
MAX_TOTAL_SNVS = 800
MIN_CLUSTER_SNVS = 20
MIN_SUBCLONAL_MUTATION_FRACTION = 0.0
MAX_SUBCLONAL_MUTATION_FRACTION = 0.8
CLONAL_SMF_THRESHOLD = 0.1
MIN_CLUSTER_CCF = 0.2
MAX_CLUSTER_CCF = 1.0
MIN_CLUSTER_CCF_SEPARATION = 0.2
NORMAL_TOTAL_CN = 2
MODEL_VERSION = "cn-first-uniform-multiplicity-v1"
PARENT_GENERATOR_SHA256 = "6fbc988cc335a562d5e8a2b42715c61e2de4d7774d6230977c5b30403744ca3d"
LEGACY_SOURCE = (
    "/home/yding1995/.vscode-server/data/User/History/"
    "1fcb286d/GN9d.ipynb"
)
EXPECTED_FILES = (
    "cna.txt",
    "snv.txt",
    "purity.txt",
    "truth.txt",
)


@dataclass(frozen=True)
class SimulationConfig:
    depths: tuple[int, ...]
    subclonal_clusters: tuple[int, ...]
    purities: tuple[float, ...]
    cna_rates: tuple[float, ...]
    sample_count: int
    seed: int

    @property
    def case_count(self) -> int:
        return self.sample_count


@dataclass(frozen=True)
class ClusterDesign:
    number_of_clusters: int
    total_snvs: int
    raw_sampled_true_smf: float
    sampled_true_smf: float
    realized_true_smf: float
    cluster_sizes: tuple[int, ...]
    eligible_cluster_counts: tuple[int, ...]


def parse_csv(value: str, converter: type[int] | type[float]) -> tuple:
    """Parse a comma-separated command-line value."""
    try:
        parsed = tuple(converter(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated value: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def validate_config(config: SimulationConfig) -> None:
    if any(depth <= 0 for depth in config.depths):
        raise ValueError("all depths must be positive")
    if any(
        cluster_count < 2 or cluster_count > 4
        for cluster_count in config.subclonal_clusters
    ):
        raise ValueError("subclonal cluster counts must be between 2 and 4")
    if len(set(config.subclonal_clusters)) != len(config.subclonal_clusters):
        raise ValueError("subclonal cluster counts must be unique")
    if any(purity <= 0 or purity > 1 for purity in config.purities):
        raise ValueError("purities must be in (0, 1]")
    if any(rate < 0 or rate > 1 for rate in config.cna_rates):
        raise ValueError("CNA rates must be in [0, 1]")
    if config.sample_count <= 0:
        raise ValueError("sample count must be positive")


def format_number(value: int | float) -> str:
    """Match the compact number formatting in the original case IDs."""
    return format(value, "g")


def make_case_id(
    depth: int,
    cluster_count: int,
    purity: float,
    cna_rate: float,
    replicate: int,
) -> str:
    return "_".join(
        (
            str(depth),
            str(cluster_count),
            format_number(purity),
            format_number(cna_rate),
            f"rep{replicate}",
        )
    )


def sample_cluster_ccfs(
    rng: np.random.RandomState,
    cluster_count: int,
) -> np.ndarray:
    """Set cluster 0 to CCF 1 and draw well-separated subclonal CCFs."""
    if cluster_count == 1:
        return np.array([MAX_CLUSTER_CCF], dtype=float)

    while True:
        cluster_ccfs = np.empty(cluster_count, dtype=float)
        cluster_ccfs[0] = MAX_CLUSTER_CCF
        cluster_ccfs[1:] = rng.uniform(
            MIN_CLUSTER_CCF,
            MAX_CLUSTER_CCF,
            size=cluster_count - 1,
        )
        sorted_ccfs = np.sort(cluster_ccfs)
        if np.min(np.diff(sorted_ccfs)) >= MIN_CLUSTER_CCF_SEPARATION:
            return cluster_ccfs


def sample_cluster_design(
    rng: np.random.RandomState,
    subclonal_cluster_counts: tuple[int, ...],
) -> ClusterDesign:
    """Sample sMF first, then derive K without fixing the K = 1 count.

    Raw sMF is Uniform[0, 0.8). Values below 0.1 are thresholded to zero and
    make a fully clonal K = 1 sample. For nonzero sMF, K is drawn uniformly
    from the requested values that can support at least 20 SNVs in every
    subclonal cluster. The remaining mutations are partitioned as evenly as
    possible among the K - 1 subclones.
    """
    total_snvs = int(rng.randint(MIN_TOTAL_SNVS, MAX_TOTAL_SNVS + 1))
    raw_sampled_true_smf = float(
        rng.uniform(
            MIN_SUBCLONAL_MUTATION_FRACTION,
            MAX_SUBCLONAL_MUTATION_FRACTION,
        )
    )

    if raw_sampled_true_smf < CLONAL_SMF_THRESHOLD:
        return ClusterDesign(
            number_of_clusters=1,
            total_snvs=total_snvs,
            raw_sampled_true_smf=raw_sampled_true_smf,
            sampled_true_smf=0.0,
            realized_true_smf=0.0,
            cluster_sizes=(total_snvs,),
            eligible_cluster_counts=(1,),
        )

    sampled_true_smf = raw_sampled_true_smf
    subclonal_snvs = int(np.floor(sampled_true_smf * total_snvs))
    clonal_snvs = total_snvs - subclonal_snvs
    eligible_cluster_counts = tuple(
        cluster_count
        for cluster_count in subclonal_cluster_counts
        if (
            clonal_snvs >= MIN_CLUSTER_SNVS
            and subclonal_snvs
            >= MIN_CLUSTER_SNVS * (cluster_count - 1)
        )
    )
    if not eligible_cluster_counts:
        raise RuntimeError(
            "no feasible subclonal K for "
            f"total_snvs={total_snvs}, sampled_true_smf={sampled_true_smf}"
        )
    cluster_count = int(rng.choice(eligible_cluster_counts))

    subclone_count = cluster_count - 1
    subclone_size, remainder = divmod(subclonal_snvs, subclone_count)
    subclonal_cluster_sizes = np.full(
        subclone_count,
        subclone_size,
        dtype=int,
    )
    if remainder:
        extra_snv_clusters = rng.choice(
            subclone_count,
            size=remainder,
            replace=False,
        )
        subclonal_cluster_sizes[extra_snv_clusters] += 1
    cluster_sizes = np.concatenate(
        (np.array([clonal_snvs], dtype=int), subclonal_cluster_sizes)
    )
    if np.any(cluster_sizes < MIN_CLUSTER_SNVS):
        raise RuntimeError("derived cluster contains fewer than 20 SNVs")

    return ClusterDesign(
        number_of_clusters=cluster_count,
        total_snvs=total_snvs,
        raw_sampled_true_smf=raw_sampled_true_smf,
        sampled_true_smf=sampled_true_smf,
        realized_true_smf=subclonal_snvs / total_snvs,
        cluster_sizes=tuple(int(value) for value in cluster_sizes),
        eligible_cluster_counts=eligible_cluster_counts,
    )


def sample_copy_numbers(
    rng: np.random.RandomState,
    mutation_count: int,
    cna_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample major/minor CN with the original joint CN distribution.

    The two allele draws have no mutation-bearing designation. Sorting the
    original 1..4 and 0..4 draws preserves their joint major/minor distribution;
    multiplicity is sampled separately after these copy numbers are fixed.
    """
    major_cn = np.ones(mutation_count, dtype=int)
    minor_cn = np.ones(mutation_count, dtype=int)
    if cna_rate > 0:
        cna_mask = rng.binomial(1, cna_rate, size=mutation_count).astype(bool)
        affected_count = int(cna_mask.sum())
        allele_a_cn = rng.randint(1, 5, size=affected_count)
        allele_b_cn = rng.randint(0, 5, size=affected_count)
        major_cn[cna_mask] = np.maximum(allele_a_cn, allele_b_cn)
        minor_cn[cna_mask] = np.minimum(allele_a_cn, allele_b_cn)
    return major_cn, minor_cn


def sample_multiplicity(
    rng: np.random.RandomState,
    major_cn: np.ndarray,
) -> np.ndarray:
    """Draw uniformly from 1..major CN, inclusive, for each mutation."""
    return rng.randint(1, major_cn + 1)


def simulate_case(
    rng: np.random.RandomState,
    output_dir: Path,
    depth: int,
    purity: float,
    cna_rate: float,
    replicate: int,
    subclonal_cluster_counts: tuple[int, ...],
) -> tuple[str, ClusterDesign]:
    """Simulate and write one tumor using the CliPPSim4K design model."""
    design = sample_cluster_design(rng, subclonal_cluster_counts)
    cluster_count = design.number_of_clusters
    case_id = make_case_id(
        depth,
        cluster_count,
        purity,
        cna_rate,
        replicate,
    )
    case_dir = output_dir / case_id
    cluster_sizes = np.asarray(design.cluster_sizes, dtype=int)
    mutation_count = design.total_snvs
    cluster_ccfs = sample_cluster_ccfs(rng, cluster_count)

    cluster_ids = np.repeat(np.arange(cluster_count, dtype=int), cluster_sizes)
    mutation_ccfs = np.repeat(cluster_ccfs, cluster_sizes)
    mutation_cellular_prevalence = purity * mutation_ccfs

    major_cn, minor_cn = sample_copy_numbers(rng, mutation_count, cna_rate)
    total_cn = major_cn + minor_cn
    multiplicity = sample_multiplicity(rng, major_cn)

    coverage = rng.poisson(depth, size=mutation_count)
    vaf = (
        mutation_cellular_prevalence
        * multiplicity
        / (NORMAL_TOTAL_CN * (1 - purity) + purity * total_cn)
    )
    alt_count = rng.binomial(coverage, vaf)
    ref_count = coverage - alt_count

    mutation_index = np.arange(mutation_count)
    cna = pd.DataFrame(
        {
            "chromosome_index": np.ones(mutation_count, dtype=int),
            "start_position": 3 * mutation_index + 1,
            "end_position": 3 * mutation_index + 3,
            "major_cn": major_cn,
            "minor_cn": minor_cn,
            "total_cn": total_cn,
        }
    )
    snv = pd.DataFrame(
        {
            "chromosome_index": np.ones(mutation_count, dtype=int),
            "position": 3 * mutation_index + 2,
            "alt_count": alt_count,
            "ref_count": ref_count,
        }
    )
    truth = pd.DataFrame(
        {
            "chromosome_index": np.ones(mutation_count, dtype=int),
            "position": 3 * mutation_index + 2,
            "cluster_id": cluster_ids,
            "ccf": mutation_ccfs,
            "multiplicity": multiplicity,
        }
    )

    case_dir.mkdir(parents=True, exist_ok=False)
    cna.to_csv(case_dir / "cna.txt", sep="\t", index=False)
    snv.to_csv(case_dir / "snv.txt", sep="\t", index=False)
    (case_dir / "purity.txt").write_text(str(purity), encoding="utf-8")
    truth.to_csv(case_dir / "truth.txt", sep="\t", index=False)
    return case_id, design


def prepare_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}. "
            "Choose a new directory to avoid overwriting an existing cohort."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(output_dir: Path, config: SimulationConfig) -> None:
    design_cell_count = len(config.depths) * len(config.purities) * len(
        config.cna_rates
    )
    minimum_cell_size, extra_sample_cells = divmod(
        config.sample_count,
        design_cell_count,
    )
    manifest = {
        "generator": Path(__file__).name,
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": (
            "emergent-K CliPPSim4K with CN-first uniform mutation multiplicity"
        ),
        "lineage": {
            "parent_generator": "generate_clippsim4k.py",
            "parent_generator_sha256": PARENT_GENERATOR_SHA256,
            "legacy_notebook": LEGACY_SOURCE,
            "consolidated_legacy_script": "generate_legacy_clippsim4k.py",
            "sole_current_generator": Path(__file__).name,
            "legacy_mechanics_retained": [
                "per-mutation CNA Bernoulli sampling",
                "joint major/minor tumor copy-number distribution",
                "Poisson total-read sampling",
                "legacy VAF equation",
                "Binomial variant-read sampling",
                "SNV and CNA genomic-coordinate construction",
                "tab-delimited cna.txt, snv.txt, and purity.txt outputs",
            ],
            "revised_components": [
                "exact 4000-sample cohort size",
                "read depths 100, 200, and 500",
                "raw true sMF sampled from Uniform[0, 0.8)",
                "raw true sMF below 0.1 set to zero and assigned K = 1",
                "total SNVs sampled from DiscreteUniform[200, 800]",
                "remaining SNVs evenly partitioned among subclones",
                "cluster 0 fixed as the clonal CCF-1 population",
                "consolidated truth.txt with coordinates and multiplicity",
                "major/minor CN generated before mutation multiplicity",
                "multiplicity uniform on integers 1..major CN at every locus",
            ],
        },
        "config": asdict(config),
        "case_count": config.case_count,
        "case_files": list(EXPECTED_FILES),
        "cohort_summary_file": "generation_summary.tsv",
        "rng": "numpy.random.RandomState (MT19937)",
        "distributions": {
            "design_cell_allocation": (
                f"{config.sample_count} total samples allocated as evenly as "
                f"possible across {design_cell_count} read-depth/purity/CNA "
                f"cells: {minimum_cell_size} samples per cell plus one sample "
                f"in {extra_sample_cells} reproducibly selected cells"
            ),
            "proposed_total_snvs": (
                f"DiscreteUniform[{MIN_TOTAL_SNVS}, {MAX_TOTAL_SNVS}]"
            ),
            "proposed_true_subclonal_mutation_fraction": (
                f"raw sMF ~ Uniform["
                f"{MIN_SUBCLONAL_MUTATION_FRACTION}, "
                f"{MAX_SUBCLONAL_MUTATION_FRACTION})"
            ),
            "clonal_threshold": (
                f"raw sMF < {CLONAL_SMF_THRESHOLD} is set to 0 and assigned "
                "K = 1"
            ),
            "nonclonal_cluster_count": (
                "for thresholded sMF > 0, K is sampled uniformly from the "
                f"feasible values in {list(config.subclonal_clusters)}; "
                "feasibility requires at least 20 SNVs in every cluster"
            ),
            "subclonal_cluster_allocation": (
                "after assigning the clonal mutations to cluster 0, all "
                "remaining SNVs are partitioned as evenly as possible across "
                "the K - 1 subclonal clusters; their sizes differ by at most "
                "one SNV, and any remainder is assigned to randomly selected "
                "subclonal clusters"
            ),
            "minimum_snvs_per_retained_cluster": MIN_CLUSTER_SNVS,
            "cluster_feasibility_policy": (
                "select K only from values compatible with the sampled total "
                "SNV count, sampled sMF, and the minimum cluster-size rule; "
                "samples and sMF draws are not rejected"
            ),
            "cluster_ccf_beta": (
                f"cluster 0 = {MAX_CLUSTER_CCF}; remaining clusters are sampled "
                f"from Uniform[{MIN_CLUSTER_CCF}, {MAX_CLUSTER_CCF}) conditional "
                "on the minimum pairwise separation"
            ),
            "minimum_pairwise_ccf_separation": MIN_CLUSTER_CCF_SEPARATION,
            "cellular_prevalence_phi": "purity * cluster_ccf_beta",
            "cna_selection": "independent Bernoulli(cna_rate) per mutation",
            "tumor_copy_numbers": (
                "unselected loci: major_cn = minor_cn = 1; selected loci: "
                "independent A ~ DiscreteUniform[1, 4], B ~ DiscreteUniform[0, 4], "
                "then major_cn = max(A, B), minor_cn = min(A, B)"
            ),
            "mutation_multiplicity": (
                "DiscreteUniform[1, major_cn], conditional on previously "
                "sampled CN, including when major_cn == minor_cn"
            ),
            "total_reads_n": "Poisson(mean_depth)",
            "variant_reads_r": "Binomial(total_reads_n, vaf_theta)",
        },
        "normal_total_cn": NORMAL_TOTAL_CN,
        "truth": {
            "file": "truth.txt",
            "columns": {
                "chromosome_index": "SNV chromosome index",
                "position": "SNV genomic position",
                "cluster_id": "true cluster assignment",
                "ccf": "true cluster CCF beta",
                "multiplicity": "true variant-allele copy number b^V",
            },
            "alignment": "joined to snv.txt by chromosome_index and position",
        },
        "vaf_formula": (
            "cellular_prevalence_phi * variant_copy_number / "
            "((1 - purity) * normal_total_cn + purity * tumor_total_cn)"
        ),
        "notes": [
            "truth.txt consolidates cluster ID, beta (CCF), and multiplicity.",
            "Tumor total CN is major_cn + minor_cn and is not changed by multiplicity.",
            "This model revision changes seeded outputs relative to the parent generator.",
            (
                "generation_summary.tsv records raw Uniform sMF, thresholded "
                "sampled sMF, and realized sMF. For nonclonal samples, the "
                "small thresholded-to-realized difference is due only to "
                "flooring to an integer number of subclonal SNVs."
            ),
            (
                f"Raw sMF below {CLONAL_SMF_THRESHOLD} is set to 0, producing "
                "a K = 1 sample; the number of K = 1 samples is not fixed."
            ),
            "Cellular prevalence phi equals purity times beta.",
            "maf.txt is a downstream PhylogicNDT input and is not generated here.",
        ],
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def allocate_design_cell_sizes(
    config: SimulationConfig,
    cell_count: int,
) -> np.ndarray:
    """Allocate the exact cohort size as evenly as possible across cells."""
    minimum_cell_size, extra_sample_cells = divmod(config.sample_count, cell_count)
    cell_sizes = np.full(cell_count, minimum_cell_size, dtype=int)
    if extra_sample_cells:
        allocation_rng = np.random.RandomState(config.seed ^ 0x5A17C0DE)
        selected_cells = allocation_rng.choice(
            cell_count,
            size=extra_sample_cells,
            replace=False,
        )
        cell_sizes[selected_cells] += 1
    return cell_sizes


def parse_integer_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list stored in the cohort summary."""
    return tuple(int(item) for item in value.split(","))


def validate_generation_summary(
    summary: pd.DataFrame,
    config: SimulationConfig,
) -> None:
    """Verify the requested emergent-K and mutation-allocation design."""
    if len(summary) != config.sample_count:
        raise RuntimeError(
            f"generated {len(summary)} samples; expected {config.sample_count}"
        )
    if summary["sample"].duplicated().any():
        raise RuntimeError("generated sample identifiers are not unique")

    raw_smf = summary["raw_sampled_true_smf"].to_numpy(dtype=float)
    sampled_smf = summary["sampled_true_smf"].to_numpy(dtype=float)
    realized_smf = summary["realized_true_smf"].to_numpy(dtype=float)
    cluster_counts = summary["number_of_clusters"].to_numpy(dtype=int)
    total_snvs = summary["total_snvs"].to_numpy(dtype=int)
    clonal_samples = raw_smf < CLONAL_SMF_THRESHOLD

    if np.any(
        (raw_smf < MIN_SUBCLONAL_MUTATION_FRACTION)
        | (raw_smf >= MAX_SUBCLONAL_MUTATION_FRACTION)
    ):
        raise RuntimeError("raw true sMF values fall outside Uniform[0, 0.8)")
    if not np.array_equal(cluster_counts == 1, clonal_samples):
        raise RuntimeError("K = 1 does not match raw true sMF < 0.1")
    if np.any(sampled_smf[clonal_samples] != 0):
        raise RuntimeError("thresholded K = 1 samples do not have true sMF 0")
    if not np.array_equal(
        sampled_smf[~clonal_samples],
        raw_smf[~clonal_samples],
    ):
        raise RuntimeError("nonclonal sampled sMF differs from the Uniform draw")
    if np.any((total_snvs < MIN_TOTAL_SNVS) | (total_snvs > MAX_TOTAL_SNVS)):
        raise RuntimeError("total SNV counts fall outside [200, 800]")

    allowed_nonclonal_counts = set(config.subclonal_clusters)
    for row in summary.itertuples(index=False):
        cluster_sizes = parse_integer_list(row.cluster_sizes)
        if len(cluster_sizes) != row.number_of_clusters:
            raise RuntimeError(
                f"{row.sample}: cluster-size count does not equal K"
            )
        if sum(cluster_sizes) != row.total_snvs:
            raise RuntimeError(
                f"{row.sample}: cluster sizes do not sum to total SNVs"
            )
        if min(cluster_sizes) < MIN_CLUSTER_SNVS:
            raise RuntimeError(
                f"{row.sample}: a retained cluster contains fewer than 20 SNVs"
            )

        if row.number_of_clusters == 1:
            if cluster_sizes != (row.total_snvs,) or row.realized_true_smf != 0:
                raise RuntimeError(f"{row.sample}: invalid fully clonal design")
            continue

        if row.number_of_clusters not in allowed_nonclonal_counts:
            raise RuntimeError(f"{row.sample}: invalid nonclonal K")
        expected_subclonal_snvs = int(
            np.floor(row.sampled_true_smf * row.total_snvs)
        )
        subclonal_sizes = cluster_sizes[1:]
        if sum(subclonal_sizes) != expected_subclonal_snvs:
            raise RuntimeError(
                f"{row.sample}: subclonal SNVs do not match sampled true sMF"
            )
        if max(subclonal_sizes) - min(subclonal_sizes) > 1:
            raise RuntimeError(
                f"{row.sample}: subclonal SNVs are not evenly partitioned"
            )
        expected_realized_smf = expected_subclonal_snvs / row.total_snvs
        if not np.isclose(
            row.realized_true_smf,
            expected_realized_smf,
            rtol=0,
            atol=np.finfo(float).eps,
        ):
            raise RuntimeError(f"{row.sample}: realized true sMF is inconsistent")

    expected_cells = pd.MultiIndex.from_product(
        (config.depths, config.purities, config.cna_rates),
        names=("read_depth", "purity", "cna_rate"),
    )
    design_counts = (
        summary.groupby(
            ["read_depth", "purity", "cna_rate"],
            observed=True,
        )
        .size()
        .reindex(expected_cells, fill_value=0)
        .to_numpy(dtype=int)
    )
    if design_counts.max() - design_counts.min() > 1:
        raise RuntimeError(
            "samples are not allocated evenly across depth/purity/CNA cells"
        )

    if not np.allclose(
        realized_smf[clonal_samples],
        0,
        rtol=0,
        atol=0,
    ):
        raise RuntimeError("K = 1 realized true sMF is not zero")


def generate_cohort(output_dir: Path, config: SimulationConfig) -> None:
    validate_config(config)
    prepare_output_directory(output_dir)
    rng = np.random.RandomState(config.seed)

    combinations: Sequence[tuple[int, float, float]] = tuple(
        itertools.product(
            config.depths,
            config.purities,
            config.cna_rates,
        )
    )
    design_cell_sizes = allocate_design_cell_sizes(config, len(combinations))
    if int(design_cell_sizes.sum()) != config.case_count:
        raise RuntimeError("design-cell allocation does not match sample count")
    summary_rows: list[dict[str, object]] = []
    case_schedule = (
        (depth, purity, cna_rate, int(design_cell_size), replicate)
        for (depth, purity, cna_rate), design_cell_size in zip(
            combinations,
            design_cell_sizes,
            strict=True,
        )
        for replicate in range(int(design_cell_size))
    )
    for depth, purity, cna_rate, design_cell_size, replicate in tqdm(
        case_schedule,
        total=config.case_count,
        desc="Generating revised CliPPSim4K",
        unit="sample",
        dynamic_ncols=True,
    ):
        case_id, design = simulate_case(
            rng,
            output_dir,
            depth,
            purity,
            cna_rate,
            replicate,
            config.subclonal_clusters,
        )
        summary_rows.append(
            {
                "sample": case_id,
                "read_depth": depth,
                "number_of_clusters": design.number_of_clusters,
                "purity": purity,
                "cna_rate": cna_rate,
                "replicate": replicate,
                "design_cell_size": design_cell_size,
                "raw_sampled_true_smf": design.raw_sampled_true_smf,
                "sampled_true_smf": design.sampled_true_smf,
                "realized_true_smf": design.realized_true_smf,
                "total_snvs": design.total_snvs,
                "eligible_cluster_counts": ",".join(
                    str(value)
                    for value in design.eligible_cluster_counts
                ),
                "cluster_sizes": ",".join(
                    str(value) for value in design.cluster_sizes
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    validate_generation_summary(summary, config)
    summary.to_csv(
        output_dir / "generation_summary.tsv",
        sep="\t",
        index=False,
    )
    write_manifest(output_dir, config)
    print(f"Generated {len(summary_rows)} cases in {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the revised raw 4,000-case CliPPSim cohort."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CliPPSim4K_generated"),
        help="new or empty output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--depths",
        type=lambda value: parse_csv(value, int),
        default=DEFAULT_DEPTHS,
        help="comma-separated mean read depths (default: 100,200,500)",
    )
    parser.add_argument(
        "--subclonal-clusters",
        type=lambda value: parse_csv(value, int),
        default=DEFAULT_SUBCLONAL_CLUSTERS,
        help=(
            "comma-separated candidate K values when thresholded sMF > 0 "
            "(default: 2,3,4)"
        ),
    )
    parser.add_argument(
        "--purities",
        type=lambda value: parse_csv(value, float),
        default=DEFAULT_PURITIES,
        help="comma-separated tumor purities (default: 0.4,0.6,0.9)",
    )
    parser.add_argument(
        "--cna-rates",
        type=lambda value: parse_csv(value, float),
        default=DEFAULT_CNA_RATES,
        help="comma-separated per-mutation CNA rates (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=(
            "exact total number of samples allocated as evenly as possible "
            "across depth/purity/CNA combinations (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="NumPy random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the resolved design without writing files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimulationConfig(
        depths=tuple(args.depths),
        subclonal_clusters=tuple(args.subclonal_clusters),
        purities=tuple(args.purities),
        cna_rates=tuple(args.cna_rates),
        sample_count=args.sample_count,
        seed=args.seed,
    )
    validate_config(config)

    if args.dry_run:
        print(json.dumps({"output_dir": str(args.output_dir), **asdict(config)}, indent=2))
        print(f"case_count: {config.case_count}")
        return

    generate_cohort(args.output_dir, config)


if __name__ == "__main__":
    main()
