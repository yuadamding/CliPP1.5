# CliPP1.5 precision and recall on CN-first4K

All **4,000 tumors**; primary population is the **1,983,489 matched mutations** from the four-method comparison. Rates below pool confusion counts across tumors. Each classification row treats its named class as positive.

| Evaluation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Clonal mutation classification | 0.918684 | 0.956572 | 0.937245 |
| Subclonal mutation classification | 0.929332 | 0.870886 | 0.899160 |
| Same-cluster mutation pairs | 0.854255 | 0.917824 | 0.884899 |

Truth clonal mutations have CCF=1. Predicted clonal mutations belong to the occupied cluster closest to CCF 1, with canonical mutation-ID tie breaking, selected within the evaluated population. Fitting remains unconstrained.

Confusion counts: **1,145,897 true clonal called clonal**, **52,023 true clonal called subclonal**, **101,428 true subclonal called clonal**, and **684,141 true subclonal called subclonal**.

Pairwise precision measures whether predicted same-cluster pairs share a truth cluster; recall measures recovery of true same-cluster pairs. Only unordered distinct pairs within the same tumor are counted. Equal-tumor mean pairwise precision/recall/F1 are **0.824146 / 0.884991 / 0.850961**. For true-K>1 tumors they are **0.801463 / 0.870157 / 0.831737**.

Full-retained-population rates on all **2,003,720 mutations** are also reported separately in the JSON. Per-tumor rates with zero denominators remain undefined, with defined/undefined counts reported explicitly. No fits were run.

All truth, assignment and center hashes were rechecked. ARI, sMF, clonal designations and mutation populations reproduce the preceding evaluation. Pair counts also passed an independent scikit-learn confusion-matrix check for every tumor and both populations.

[Full-precision evidence](CNFIRST_PRECISION_RECALL.json) includes confusion counts, pooled and equal-tumor rates, true-K strata, and parent evaluation hashes. Per-case metrics and the detailed report are retained under `results/experimental-full29-20260928-v3/performance-cnfirst4k-20260929-v1/precision-recall-v1`.
