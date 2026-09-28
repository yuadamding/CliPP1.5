# Completed Experimental CliPP1.5 performance

All **3,097 planned development cases** were evaluated on September 28, 2026,
after execution completed at **2:44:30 p.m. CDT**. The evaluated implementation
is the CUDA soft-mixture estimator with the conservative structural selector.
It improves every assessed aggregate endpoint over Previous CliPP1.5 in each
cohort, including CN-first4K. It does **not** yet outperform PyClone-VI on every
target: SimClone and Phylogic sMF CCC remain lower.

The fixed inventory covers 2,661/4,000 eligible CN-first, 385/756 SimClone and
51/500 Phylogic tumors. It includes repeatedly inspected development cases and
repeated simulation structures. It is neither an independent confirmation set
nor the entire 5,256-tumor benchmark. No production-default adoption follows
from publishing this implementation or report.

## Matched comparison

Every method within a cohort uses identical cases and mutation IDs. CN-first
has 1,056,696 matched mutations, SimClone 252,370 and Phylogic 98,503. Separate
CN-first retained-population comparisons include 1,066,951 mutations and also
improve every assessed aggregate endpoint against Previous CliPP1.5.

ARI and sMF CCC are higher-is-better; CCF MAE and sMF MAE are lower-is-better.
Means give each tumor equal weight. Experimental CCF is the published mixture
component estimate, or the preserved original partition refit when the
structural selector retains it. These are not PyClone posterior means.

| Cohort | Method | Mean ARI | CCF MAE | sMF CCC | sMF MAE |
|---|---|---:|---:|---:|---:|
| CN-first4K | Experimental CliPP1.5 | 0.703385 | 0.046869 | 0.964825 | 0.041346 |
| | Previous CliPP1.5 | 0.690626 | 0.050775 | 0.944505 | 0.048574 |
| | PyClone-VI | 0.660342 | 0.051878 | 0.949641 | 0.051534 |
| | CliPP | 0.598842 | 0.065883 | 0.832938 | 0.093528 |
| | PhylogicNDT | 0.631276 | 0.057162 | 0.837227 | 0.095105 |
| SimClone1000 | Experimental CliPP1.5 | 0.633205 | 0.056125 | 0.764812 | 0.100673 |
| | Previous CliPP1.5 | 0.609395 | 0.067167 | 0.696255 | 0.123892 |
| | PyClone-VI | 0.606504 | 0.062021 | 0.793843 | 0.103193 |
| PhylogicNDT500 | Experimental CliPP1.5 | 0.344468 | 0.106188 | 0.527140 | 0.150707 |
| | Previous CliPP1.5 | 0.310791 | 0.119962 | 0.330348 | 0.225425 |
| | PyClone-VI | 0.340456 | 0.106661 | 0.615501 | 0.136454 |

PyClone-VI is the only external comparator in the frozen SimClone and Phylogic
inventories. The CN-first comparison includes all three external methods.

For tumors with more than one truth cluster, mean ARI changes from Previous to
Experimental as follows: CN-first **0.651316 to 0.665696**, SimClone **0.407941
to 0.444030**, and Phylogic **0.282660 to 0.317712**. Corresponding PyClone-VI
values are 0.626501, 0.403560 and 0.313536.

Clonal labeling uses the occupied group closest to CCF 1, with canonical
mutation-ID tie breaking; fitting remains unconstrained. Predicted sMF counts
mutations outside that group. Truth sMF counts mutations with CCF below
`1 - 1e-12`. CCC is computed across tumors, not per sample. Identical constant
vectors have CCC 1; any other comparison involving a constant vector has CCC 0.

## Individual gains and losses

Counts below use an absolute tie threshold of `1e-12` and show
**Experimental better / worse / tied**. Per-sample sMF compares absolute error,
since an individual tumor has no across-tumor CCC.

| Cohort | Comparator | ARI | CCF MAE | sMF absolute error |
|---|---|---:|---:|---:|
| CN-first4K | Previous CliPP1.5 | 329 / 18 / 2314 | 344 / 4 / 2313 | 246 / 91 / 2324 |
| CN-first4K | PyClone-VI | 1691 / 650 / 320 | 1832 / 828 / 1 | 1639 / 584 / 438 |
| SimClone1000 | Previous CliPP1.5 | 82 / 9 / 294 | 81 / 20 / 284 | 70 / 19 / 296 |
| SimClone1000 | PyClone-VI | 150 / 50 / 185 | 252 / 132 / 1 | 132 / 64 / 189 |
| PhylogicNDT500 | Previous CliPP1.5 | 21 / 2 / 28 | 20 / 3 / 28 | 19 / 4 / 28 |
| PhylogicNDT500 | PyClone-VI | 26 / 18 / 7 | 29 / 22 / 0 | 21 / 23 / 7 |

Compared with Previous, **103 CN-first, 35 SimClone and seven Phylogic tumors**
lose on at least one of these metrics. Most have tradeoffs: only three, nine
and zero, respectively, lose without improving on any of the other metrics.
The selector preserves all original estimates in 2,624 cases; their assessed
metrics are identical. All 473 selected replacements reached the declared EM
fixed point. Unconverged exploratory candidates remain recorded; none of this
establishes global search optimality.

Experimental has **zero false splits among 433 true-single-cluster tumors**.
False single-cluster collapses decrease from 28 to 14 in CN-first, 51 to 38
in SimClone and 11 to five in Phylogic. PyClone-VI has 12, 35 and five,
respectively. PhylogicNDT recovers the correct cluster count in more CN-first
tumors than Experimental (2,026 versus 1,946). Thus the continuous-metric gains
do not justify claiming superiority on every measure or every sample.

The SimClone maximum CCF MAE increases slightly against Previous, from
0.461425 to 0.464647, despite better mean and 95th/99th percentiles. Against
PyClone-VI, Experimental SimClone sMF error has worse 95th/99th percentiles
despite a slightly lower mean. Full error tails are retained in the JSON evidence.

## Multiplicity

| Cohort | Eligible supplied-CN loci | Previous macro-F1 | Experimental macro-F1 | Coverage |
|---|---:|---:|---:|---:|
| CN-first4K | 402,917 | 0.780530 | 0.785364 | 1.0 |
| SimClone1000 | 99,361 | 0.797657 | 0.937787 | 1.0 |
| PhylogicNDT500 | 5,187 | 0.271003 | 0.339959 | 1.0 |

These are pooled exact-class macro-F1 scores against the original simulator's
integer multiplicity, on supplied single-state CN other than 1/1, including
balanced amplification. Micro, weighted and per-class results are retained.
Phylogic converted single-state CN can differ from original mixed CN; these
supplied-CN diagnostics do not by themselves satisfy the final mixed-CN
multiplicity acceptance contract.

## Reproduction and boundaries

[PERFORMANCE.json](PERFORMANCE.json) preserves full-precision summaries,
per-sample counts, error tails, selection diagnostics and hashes of the source
evaluation artifacts. The estimator and selector source match the completed
run bindings. All five recovered cases are included through the resolved result
index; the separate shutdown replay is not an additional evaluated tumor.
The [recovery record](RECOVERY.md) records allocated-CUDA checks separately.

Full local evidence is under
`results/experimental-pool36-20260928-v1/comparison-complete-20260928T200251Z`:
`REPORT.md`, `metrics/per_case.tsv`, `paired_samples.tsv`,
`samples_with_any_loss.tsv`, `true_k_strata.tsv`, `paired_pyclone.pdf`,
`EVALUATION_RUN.json` and the original source/output authorities. These ignored
operational artifacts are not distributed as part of the source repository.
`results/CURRENT_MIXTURE_STUDY.json` points to that complete evaluation.

All assessed aggregate development non-regression checks against Previous pass,
on both matched and retained populations. This does not complete the
[three-cohort acceptance contract](../../docs/THREE_COHORT_ACCEPTANCE.md).
Independent confirmation, full-cohort comparisons and the remaining sMF deficits
are still outstanding. No fitting settings or acceptance thresholds changed
for this evaluation.
