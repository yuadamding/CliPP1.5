# CliPP1.5

CliPP1.5 clusters mutations from **one bulk tumor sample**, using a binomial
likelihood with latent multiplicity and one fixed mutation chain. It fits at
most ten occupied blocks, without a clonal constraint. The conditional estimator
is preserved from `15d397b`; this is not the historical complete-graph pipeline.

## Install and run

Use Python 3.12+ and a C++17 compiler for a source install. Required runtime
packages are NumPy ≥2.2.6, SciPy 1.18.x and pandas ≥2.0. **R and NVIDIA packages are not required.**
Linux x86_64 CPU is the validated target; CUDA and the container recipe remain
experimental until separately executed. See [validation scope](CONTRIBUTING.md).

From this checkout:

```bash
python -m venv .venv
. .venv/bin/activate
CLIPP_USE_CUDA=0 python -m pip install .
clipp doctor
clipp validate sample/sample.snv.txt sample/sample.cna.txt sample/sample.purity.txt
clipp fit sample/sample.snv.txt sample/sample.cna.txt sample/sample.purity.txt \
  --device cpu --sample-id example --output /tmp/clipp-example
clipp verify /tmp/clipp-example
```

Choose a **new** output path. The supplied example has 487 retained SNVs; its
reference selects K=3, q=3, with score approximately 4550.464737. Its biological
truth is unknown. `--clusters 3` fits one capacity; `--max-clusters 10` compares
capacities 1 through min(10, retained N). K is a capacity, not a required cluster
count.

```python
from clipp import fit, load_result
from clipp.config import FitConfig

fit("snv.tsv", "cna.tsv", "purity.txt", "new-result",
    config=FitConfig(sample_id="tumor-1", device="cpu"))
result = load_result("new-result")  # independent verification before reading
print(result.clusters)
labels_at_k3 = result.partition(3)["labels"]
```

Inputs are autosomal, use normal CN=2 and require inclusive, nonoverlapping CNA
intervals. A completed run has `COMPLETE.json` and passes verification. Schema 3
stores the selected mutation and cluster tables once, with compact, replayable
parameters for the other fits and candidates.

SNV columns are `chromosome_index`, `position`, `ref_count`, `alt_count`, with an
optional unique `mutation_id`. CNA columns are `chromosome_index`, `start_position`,
`end_position`, `major_cn`, `minor_cn`, `total_cn`. Purity is one scalar in (0, 1].
Positions are 1-based; only chromosomes 1–22 are accepted. Duplicate loci and
overlapping CNA intervals fail validation. Invalid counts, zero depth and missing
CNA coverage are excluded and recorded in `preprocess_result/input_ledger.tsv`.

CP divided by purity is CCF. Label 0 is the highest-CP block, **not evidence of
clonality**. Mixture weights differ from assigned mutation proportions. The
observed-mixture BIC-form score is evaluated at conditionally fitted centers;
it does not certify global optimization or biological accuracy.

Every multiplicity from 1 through major CN receives equal prior mass. A pooled
pilot freezes the chain; proposals undergo conditional center refitting, adjacent
coarsening and boundary refinement. The score marginalizes cluster and multiplicity
with `(2q-1) log(N)` complexity. Full multiplicity support is preserved, so a high
major CN can increase memory across the dataset; subsampling still initializes
and scores all retained rows. No fixed-memory or total linear-time claim is made.

The repository contains the fitting package and sample. Extended documentation,
tests and benchmark tooling are kept in the external validation archive. Current
CI checks packaging and sample fits; see [development and validation](CONTRIBUTING.md).

## Attribution

The [AGPLv3 license](LICENSE) and attribution to Kaixian Yu, Yujie Jiang,
Shuangxi Ji and Yuxin Tang are retained. Removed historical R code remains in
[the pinned history](https://github.com/yuadamding/CliPP1.5/tree/19f310b873ccb0c5a959478018408e6297de1e70/legacy).
Cite the exact source/native build and [CITATION.cff](CITATION.cff). Historical
CliPP accuracy claims do not automatically apply to this estimator.
