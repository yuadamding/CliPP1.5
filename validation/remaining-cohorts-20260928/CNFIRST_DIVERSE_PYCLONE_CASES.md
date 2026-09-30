# Ten diverse CN-first4K discrepancies against PyClone-VI

The user requested ten cases with large differences, then asked for broad scenario coverage. This panel selects one case per specified diagnostic scenario while maximizing summed absolute ARI gaps and requiring ten distinct depth/purity/CNA settings.

It covers true K=1–4; depths 100, 200 and 500; purities 0.4, 0.6 and 0.9; and CNA rates 0.1, 0.4 and 0.7. Six cases favor CliPP1.5 by ARI and four favor PyClone-VI. Ten of the 27 design cells are represented. These deliberately selected examples are not a cohort-average performance estimate.

| Case | Scenario | CliPP1.5 ARI | PyClone-VI ARI | Δ ARI |
|---|---|---:|---:|---:|
| 500_1_0.9_0.1_rep23 | Single cluster: PyClone-VI false split | 1.0000 | 0.0000 | +1.0000 |
| 500_2_0.9_0.7_rep57 | True K=2: CliPP1.5 advantage | 0.8804 | 0.2262 | +0.6542 |
| 500_2_0.4_0.4_rep58 | ARI favors CliPP1.5; sMF error favors PyClone-VI | 0.9343 | 0.3520 | +0.5823 |
| 200_2_0.4_0.7_rep122 | True K=2: PyClone-VI advantage | 0.0488 | 0.6185 | -0.5697 |
| 500_3_0.6_0.7_rep139 | True K=3: CliPP1.5 advantage | 0.6531 | 0.3341 | +0.3190 |
| 200_2_0.4_0.1_rep2 | CliPP1.5 collapses to one cluster | 0.0000 | 0.3095 | -0.3095 |
| 500_3_0.4_0.1_rep59 | True K=3: PyClone-VI advantage | 0.6226 | 0.8625 | -0.2399 |
| 200_4_0.9_0.4_rep79 | True K=4: CliPP1.5 advantage | 0.6975 | 0.4925 | +0.2050 |
| 100_2_0.4_0.7_rep88 | PyClone-VI collapses to one cluster | 0.1711 | 0.0000 | +0.1711 |
| 200_4_0.6_0.1_rep143 | True K=4: PyClone-VI advantage | 0.3963 | 0.5637 | -0.1673 |

The unconstrained ranking is retained separately: 31 true-K-one cases tie at ARI gap 1. The diverse panel includes one of them, selected by greatest PyClone-VI sMF error, plus both-method advantages at true K=2,3,4; collapse to one cluster by each method; and an ARI/sMF ranking disagreement.

The ten-page PDF shows truth and both methods’ CCF histograms, plus assignment matrices with counts and percentages within each truth cluster. All plotted ARIs, CCF errors and sMF errors were recomputed from hash-verified saved outputs and reproduced the parent report within 1e−12. No fits or active jobs changed.

[Selection evidence](CNFIRST_DIVERSE_PYCLONE_CASES.json) records cases, metrics and hashes. Full plots, matched mutation IDs, raw-output paths and per-case metrics are under `results/experimental-full29-20260928-v3/performance-cnfirst4k-20260929-v1/diverse-clipp15-pyclone-10cases-v1/`.

[Follow-up deep investigation](CNFIRST_TEN_CASE_INVESTIGATION.md) explains the partition-selection failures, PyClone's exact genotype model, multiplicity/CCF ambiguity and metric disagreements using the saved outputs.
