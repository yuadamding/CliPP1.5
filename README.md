# CliPP1.5

CliPP1.5 clusters mutations from **one bulk tumor sample** using a binomial
likelihood with latent multiplicity and one fixed chain of mutations. It fits
at most 10 occupied blocks. No cluster is constrained to CCF=1.

This repository now packages the conditional fixed-chain estimator from
`15d397b`, with explicit input validation, native build identity, numerical
regression tests and independently verifiable results. It is not the historical
complete-graph/SCAD pipeline. Current engineering checks are described in
[validation](docs/validation.md); they do not establish a cohort accuracy ranking.

## Install and run

Requirements: Python 3.10+, a C++17 compiler, R with `data.table`, and the Python
dependencies declared in `pyproject.toml`. Linux CPU is the tested development
platform. CUDA builds and the container recipe are **experimental/unqualified**
for this revision; CPU test passes do not qualify them.

From a complete checkout (including `sample/`):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Install R/data.table through your system/environment if absent; then:
Rscript -e 'stopifnot(requireNamespace("data.table", quietly=TRUE))'
CLIPP_USE_CUDA=0 python -m pip install .
clipp doctor
clipp fit sample/sample.snv.txt sample/sample.cna.txt sample/sample.purity.txt \
  --device cpu --sample-id example --output /tmp/clipp-example
clipp verify /tmp/clipp-example
```

Use a **new output path** for every attempt; `/tmp/clipp-example` must not exist.
Outputs never go into the installed package. The legacy example contains 487
retained SNVs and has no established biological truth. The frozen CPU regression
selects K=3, q=3, with BIC-form score approximately 4550.464737. Small
cross-platform floating-point differences are covered by stated test tolerances.

For a single capacity, add `--clusters 3`. Otherwise `--max-clusters 10` compares
capacities 1 through min(10, retained N). K is a capacity; q is the actual number
of occupied chain blocks. K=3 can produce fewer than three blocks.

```python
from clipp import fit
from clipp.config import FitConfig

result = fit("snv.tsv", "cna.tsv", "purity.txt", "new-result",
             config=FitConfig(sample_id="tumor-1", device="cpu"))
```

## Inputs and results

Inputs use autosomes 1–22 (`chr1` and `1` normalize identically), 1-based positions,
and **inclusive, nonoverlapping CNA intervals** in a common assembly. The normal
copy number is two. Sex chromosomes, ambiguous overlaps and duplicate loci are
rejected. Invalid read counts/zero depth or missing CNA coverage are explicitly
excluded and reconciled in `preprocess_result/input_ledger.tsv`. See the
[input contract](docs/input.md) before adapting data from another tool.

A run is complete only when `COMPLETE.json` exists **and `clipp verify` passes**.
`final_result/Best_K/` contains assignments, cluster centers and conditional
multiplicity posterior summaries. The manifest records inputs, source/build
identity, configuration, environment, chain, seeds and checksums. Completion
records full API wall time and sampled process-tree peak RSS. See
[outputs and certificates](docs/output.md).

`cellular_prevalence` is CP; `cancer_cell_fraction` is CP / purity. Label 0 has the
highest CP; it is **not evidence of clonality**. `mixture_weight` is fitted for the
observed-likelihood score, while `assignment_proportion` counts assigned SNVs.
Neither is automatically a tumor-cell clone abundance.

## Scientific scope

Every integer multiplicity from 1 through major CN receives equal prior mass.
The population pilot freezes a chain once, followed by distance-to-set penalty
proposals, conditional block refits, adjacent coarsenings and ordered boundary
polishing. Selection uses observed-mixture likelihood at those conditionally
fitted centers with `(2q-1) log(N)` complexity. This is a **BIC-form score**, not
ordinary BIC at an unrestricted mixture maximum-likelihood solution.

The CPU likelihood loop is serial. CUDA accelerates likelihood evaluations while
chain optimization stays on the host. Subsampling still initializes and scores
the full retained dataset. No total linear-time, fixed-memory, globally optimal,
calibrated-uncertainty or current-cohort accuracy claim is made.

- [Normative model and algorithm](docs/model.md)
- [Configuration, reproducibility and profiling](docs/reproducibility.md)
- [Limitations and failure interpretation](docs/limitations.md)
- [Migration from older pipelines](docs/migration.md)
- [Development, tests and releases](CONTRIBUTING.md)
- [Simulation and protected benchmark protocol](benchmarks/README.md)
- [Implementation checklist disposition](docs/checklist.md)

## Attribution and citation

Preserve the repository's [AGPLv3 license](LICENSE). Legacy implementation
attributions include Kaixian Yu, Yujie Jiang, Shuangxi Ji and Yuxin Tang; see
[legacy notes](legacy/README.md). Cite this software's exact commit/build and
[CITATION.cff](CITATION.cff) separately from any historical CliPP paper. Earlier
CliPP accuracy claims do not automatically apply to this fixed-chain estimator.
