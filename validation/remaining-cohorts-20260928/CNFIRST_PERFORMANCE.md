# Complete CN-first4K performance

All **4,000/4,000 tumors** were evaluated on **September 29, 2026, 10:28 AM CDT**, using **1,983,489 identical matched mutations** across methods. Current Experimental CliPP1.5 leads the four-method comparison in mean ARI, multi-cluster ARI, sMF CCC, CCF MAE and sMF MAE.

| Method | Mean ARI | Multi-cluster ARI | sMF CCC | CCF MAE | sMF MAE | Correct K |
|---|---:|---:|---:|---:|---:|---:|
| CliPP1.5 | 0.706684 | 0.668850 | 0.962030 | 0.046256 | 0.041176 | 2951/4000 |
| PyClone-VI | 0.663430 | 0.628766 | 0.950728 | 0.051187 | 0.050993 | 2431/4000 |
| CliPP | 0.604554 | 0.563144 | 0.837163 | 0.065233 | 0.092249 | 2522/4000 |
| PhylogicNDT | 0.632880 | 0.585808 | 0.836337 | 0.056821 | 0.094723 | 3072/4000 |

Against PyClone-VI, CliPP1.5 has higher ARI in **2,596** tumors, lower in **932**, and ties in **472**. CCF MAE is lower in 2,748 tumors and sMF absolute error in 2,522. CliPP1.5 makes **zero false splits among 457 true-single-cluster tumors**. PhylogicNDT recovers the exact cluster count more often (3,072 versus 2,951); CliPP1.5 also has 14 false collapses to one cluster versus 13 for PyClone-VI. These gains are aggregate, not universal per-tumor or per-endpoint superiority.

The 2,661 previously evaluated outputs and 1,339 new outputs have identical 46-file scientific inventories and fit/selection policies. Every output/input hash was checked; the earlier 2,661-case metrics reproduce within 2.1e-16. There are 3,476 preserved complete-graph partitions and 524 selected new mixture structures, all selected replacements at the declared EM fixed point. Source commit: `fc9d349761de68d8c8771fe5d92137eabfc6e784`.

Clonal designation remains the occupied cluster closest to CCF 1; fitting is unconstrained. sMF CCC is across paired tumor fractions. Means give each tumor equal weight. CCF uses published mixture component or preserved partition-refit values; PyClone-VI uses unrounded posterior means. Matching excludes 20,231 of the original 2,003,720 mutations because of CliPP retention. No method was refitted.

PyClone-VI used binomial likelihood, 20 starting clusters and 1,000 restarts. PhylogicNDT used 1,000 iterations. All comparisons used saved fits, with local CPU evaluation and read-only remote output import.

This is the complete CN-first cohort, including repeatedly inspected development cases. It does not constitute independent confirmation or completion of the separate three-cohort accuracy requirement.

[Full-precision evidence](CNFIRST_PERFORMANCE.json) contains per-method summaries, paired counts, error tails, source inventory and report hashes. Full per-case tables, result/source bindings and a three-page comparison PDF are retained under `results/experimental-full29-20260928-v3/performance-cnfirst4k-20260929-v1`.

The separate [precision/recall supplement](CNFIRST_PRECISION_RECALL.md) reports
clonal/subclonal mutation classification and pairwise clustering, with pooled
confusion counts, per-tumor averages, both mutation populations and true-K strata.

The [multiplicity precision/recall supplement](CNFIRST_MULTIPLICITY_PRECISION_RECALL.md)
evaluates exact integer multiplicity on all 761,813 retained CNA mutations.
Macro precision/recall/F1 are 79.05% / 78.60% / 78.79%, with 100% call coverage.

The [ten diverse PyClone-VI comparison cases](CNFIRST_DIVERSE_PYCLONE_CASES.md)
cover large ARI gaps in both directions, all true-K/depth/purity/CNA levels,
over-splitting, collapse to one cluster and conflicting ARI/sMF rankings.
Their ten-page PDF includes CCF histograms and mutation assignment matrices.
