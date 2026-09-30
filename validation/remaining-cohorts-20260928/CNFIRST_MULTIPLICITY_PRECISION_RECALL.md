# Exact multiplicity estimation on CN-first4K

All **4,000 tumors**, using the same saved CliPP1.5 guarded results as the
complete cohort comparison. The primary set contains **761,813 CNA mutations**,
with **100% call coverage** and no missing multiplicity calls. CNA means
`(major_cn != 1) | (minor_cn != 1)`, including balanced amplified states such
as 2/2. Ordinary 1/1 loci are excluded from the primary result.

| Exact multiplicity class | Truth count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 1 | 280,206 | 86.23% | 83.56% | 84.88% |
| 2 | 240,851 | 76.77% | 79.36% | 78.04% |
| 3 | 160,699 | 74.79% | 76.84% | 75.80% |
| 4 | 80,057 | 78.39% | 74.62% | 76.46% |
| Macro average | 761,813 | **79.05%** | **78.60%** | **78.79%** |
| Micro average | 761,813 | **79.87%** | **79.87%** | **79.87%** |

Counts are pooled across tumors before computing each class's precision and
recall. Macro averages give the four classes equal weight. Micro precision and
recall equal exact-call accuracy here because every eligible mutation has one
call: **608,498 / 761,813** are correct.

Across all **2,003,720 retained mutations**, including ordinary 1/1 loci,
micro precision and recall are **92.35%**. On the previous four-method matched
CNA population of 741,694 mutations, macro precision/recall/F1 are
79.54% / 79.00% / 79.24%; micro precision/recall are 80.30%. These supplementary
denominators must not be mixed with the primary full-retained CNA result.

The truth target is the simulator's integer multiplicity, sampled uniformly
from 1 through the previously generated major CN. The prediction is the saved
guarded integer multiplicity: either the preserved complete-graph partition
call or MAP multiplicity conditional on the selected mixture component.
This is separate from clonal/subclonal classification and pairwise clustering.

Every consumed assignment, truth, fitting input and experiment receipt was
rechecked against its frozen SHA-256 binding. Mutation joins and multiplicity
support were validated; every input locus has one CN state. Confusion matrices
and rates agree with independent scikit-learn calculations. The earlier
2,661-case CNA multiplicity F1 results reproduce exactly. No models were refit.

[Full-precision evidence](CNFIRST_MULTIPLICITY_PRECISION_RECALL.json) records
per-class counts, confusion matrices, coverage and evaluation provenance. The
detailed report and per-case tables are under
`results/experimental-full29-20260928-v3/performance-cnfirst4k-20260929-v1/multiplicity-precision-recall-v1/`.
