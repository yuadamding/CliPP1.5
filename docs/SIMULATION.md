# CliPPSim4K simulation

The package derives from the requested generator and now implements the
user-requested **CN-first multiplicity** revision:

- Original: `/data/CliPP_Sim/07222026/generate_clippsim4k.py`
- Packaged: [`src/clipp1d/simulation/generate_clippsim4k.py`](../src/clipp1d/simulation/generate_clippsim4k.py)
- Original source SHA-256 (the parent of this revision):
  `6fbc988cc335a562d5e8a2b42715c61e2de4d7774d6230977c5b30403744ca3d`

The manifest identifies the current model as
`cn-first-uniform-multiplicity-v1` and records the parent source hash. Major and
minor CN are sampled first, followed by multiplicity uniformly on 1..major CN.
The conditional joint CN distribution, depth/purity/cluster design, VAF formula
and file schemas are retained. The current CNA defaults are 0.1/0.4/0.7 (the
original used 0/0.1/0.2). The additional random draws change seeded outputs and can
also change later tumors' sampled cluster designs. Existing cohorts retain their
original model identity. The external original script is preserved.

The thin package and command entry points call the revised generator. The
original absolute shebang and historical provenance strings are retained; use
the module or installed command to select your own Python interpreter.

## Use

Install the simulation extra in the chosen environment (NumPy is already a core
dependency; the extra supplies pandas and tqdm):

```bash
python -m pip install -e '.[simulation]'
python -m clipp1d.simulation --dry-run
python -m clipp1d.simulation --output-dir results/CliPPSim4K_generated
```

The installed `clipp1d-simulate` command accepts the same options. From an
uninstalled checkout with the dependencies available:

```bash
PYTHONPATH=src python -m clipp1d.simulation --output-dir results/simulation-smoke --sample-count 31
```

Generation runs on CPU and does not import the fitting backend or require a GPU.
It refuses a nonempty output directory. A dry run creates no files. Use `--help`
for `--depths`, `--purities`, `--cna-rates`, `--subclonal-clusters`, `--sample-count`
and `--seed`. Despite its historical name, `--subclonal-clusters` specifies
candidate **total** cluster counts K for nonclonal tumors.

The Python API also exposes `SimulationConfig`, `ClusterDesign` and
`generate_cohort`:

```python
from pathlib import Path
from clipp1d.simulation import SimulationConfig, generate_cohort

config = SimulationConfig(
    depths=(100, 200, 500),
    subclonal_clusters=(2, 3, 4),
    purities=(0.4, 0.6, 0.9),
    cna_rates=(0.1, 0.4, 0.7),
    sample_count=4000,
    seed=20260730,
)
generate_cohort(Path("results/CliPPSim4K_generated"), config)
```

## Design and outputs

The default cohort has exactly 4,000 tumors across 27 depth/purity/CNA design
cells: depths 100/200/500, purities 0.4/0.6/0.9 and CNA rates 0.1/0.4/0.7. Each cell
gets 148 tumors; four reproducibly selected cells get one extra. The default seed
is 20260730, using NumPy `RandomState` (MT19937).

Each tumor has 200–800 SNVs and at least 20 SNVs per cluster. Raw sMF is uniform
on [0, 0.8); values below 0.1 become zero and yield K=1. Otherwise K is sampled
from feasible values in {2, 3, 4}. The clonal truth cluster is label 0 with CCF 1;
subclonal CCFs are drawn from [0.2, 1) subject to a minimum pairwise separation
of 0.2. The subclonal count is `floor(sMF * N)` and is divided as evenly as
possible among the K−1 subclones.

For each mutation, copy number and multiplicity are generated in this order:

1. Start with major CN = minor CN = 1. Select the mutation for CNA sampling with
   probability `cna_rate`.
2. At selected loci, draw two independent allele copy numbers,
   `A ~ DiscreteUniform[1, 4]` and `B ~ DiscreteUniform[0, 4]`, then set
   `major_cn = max(A, B)` and `minor_cn = min(A, B)`. These draws preserve the
   original joint distribution of major/minor CN. They carry no mutation-bearing
   designation. Selection can still produce a 1/1 locus.
3. With CN fixed, independently draw integer multiplicity uniformly from
   `{1, ..., major_cn}`. This applies at every locus, including balanced
   amplification and LOH. It does not depend on CCF or cluster identity.
4. Keep total tumor CN equal to `major_cn + minor_cn`; multiplicity does not
   redefine either allele's copy number.

Examples: CN 1/1 gives multiplicity 1; CN 2/2 gives 1 or 2, equally likely;
CN 4/0 and CN 4/4 both give 1, 2, 3 or 4, equally likely.

The generator retains normal total CN 2, Poisson total depth, and binomial
variant counts with

```text
VAF = purity * CCF * multiplicity / (2 * (1 - purity) + purity * tumor_total_CN)
```

Every tumor directory contains `cna.txt`, `snv.txt`, `purity.txt` and `truth.txt`.
The cohort root contains `generation_summary.tsv` and `generation_manifest.json`.
Truth is joined to SNVs by chromosome and position. These use the original raw
simulation file schemas; conversion to the twelve-column fitting input is a separate
step. `maf.txt` is also a downstream input-preparation artifact. The historical
small fixture generator in `benchmarks/simulate.py` has a different design.

The simulator's clonal truth population does not impose a fitting constraint:
CliPP1.5 fitting remains unconstrained, with its closest-to-one refitted cluster
labeled 0.

## Reproducibility verification

[`tests/fixtures/clippsim4k_original_reference.json`](../tests/fixtures/clippsim4k_original_reference.json)
is retained as historical evidence from executing the **external original script**,
with 31 tumors,
seed 20260730 and all 27 default design cells. It records the source hash,
dependency versions, hashes of all 124 per-tumor files and the cohort summary,
and the complete manifest except its generation timestamp. It covers K=1, 2, 3
and 4 and uneven allocation across design cells. Its file hashes describe the
original allele-based multiplicity model and are not current-model expectations.

[`tests/test_simulation.py`](../tests/test_simulation.py) verifies the analytic
joint CN probabilities, conditional multiplicity support and uniform frequencies,
diploid behavior, and balanced-amplification/LOH cases. A controlled-CN simulation
checks that saved CN stays fixed and that the binomial read probabilities use the
sampled multiplicity and the original VAF formula.

Two 31-tumor panels, using the historical and current CNA-rate grids, check all
27 design cells per panel and byte-identical outputs between the revised Python
API and module CLI. Only `generated_at_utc` is omitted from
the manifest comparison. Tests also validate the new model identity and parent
provenance, CN/multiplicity bounds, default CLI settings, a write-free dry run,
and rejection of a nonempty destination. The external original path is not
required to run the tests.

```bash
python -m pytest -q tests/test_simulation.py
```

Byte-identical reproducibility requires the same model, configuration, seed and
NumPy/pandas environment. Different library versions may change serialized
floating-point values. Changing cohort size also changes the design-cell schedule
and therefore the RNG draws assigned to later tumors; a smaller cohort is not
necessarily a prefix of the 4,000-tumor cohort.
