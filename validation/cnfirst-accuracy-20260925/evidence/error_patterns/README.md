# CN-first4K: frozen 193-case error-pattern audit

This is a read-only analysis of the validated export in
`CNfirst4K_CliPP15_performance_20260925T181236Z`, whose scheduler snapshot is
September 25, 2026, 1:12:36 PM CDT. It does not describe later completions. It
uses all 41,712 retained mutations in 193 tumors, with exact mutation-ID joins
to the locally staged truth. Fits, live jobs, source code, and published
results were not changed. No refits were performed.

## Main evidence

**The largest error concentration is copy-number/multiplicity ambiguity,
especially misclassifying genuinely clonal CNA mutations as lower-CCF
subclones.** This pattern is much stronger than a generic low-depth problem.

| Mutation population | Mutations | Mean absolute CCF error | Mean signed CCF error | Assigned label 0 |
|---|---:|---:|---:|---:|
| Diploid, true clonal | 15,001 | 0.02080 | -0.02080 | 97.61% |
| CNA, true clonal | 11,292 | 0.17364 | -0.17364 | 66.60% |
| Diploid, true subclonal | 8,670 | 0.07838 | +0.04250 | 16.08% |
| CNA, true subclonal | 6,749 | 0.11222 | +0.00762 | 13.16% |

CNA means `(major_cn != 1) | (minor_cn != 1)` and includes balanced
amplifications. CNA loci account for 43.25% of evaluated mutations and 73.27%
of total absolute CCF error. Estimates here are final-refit CCFs.

The relationship with simulated multiplicity is specific:

| Major CN | True multiplicity | True-clonal loci | Correctly assigned label 0 | Mean CCF bias |
|---|---:|---:|---:|---:|
| 2 | 1 | 1,179 | 43.94% | -0.29693 |
| 2 | 2 | 1,194 | 98.74% | -0.01294 |
| 3 | 1 | 1,172 | 40.19% | -0.35518 |
| 3 | 2 | 1,217 | 58.09% | -0.16396 |
| 3 | 3 | 1,186 | 98.82% | -0.01255 |
| 4 | 1 | 1,215 | 36.79% | -0.39030 |
| 4 | 2 | 1,149 | 44.30% | -0.28853 |
| 4 | 3 | 1,166 | 62.69% | -0.11418 |
| 4 | 4 | 1,209 | 99.26% | -0.01257 |

These outputs are consistent with fitting alternative integer multiplicities
at lower CCF, but this audit alone does not identify the exact likelihood or
candidate-selection mechanism. Source-level and likelihood counterfactuals
are needed to establish that mechanism.

The parent's subsequent exact-call audit in
`../bound_analysis/multiplicity_diagnostic.tsv` confirms that 3,596 of 3,772
misassigned truly clonal CNA mutations (95.33%) have an inferred final-refit
multiplicity greater than their simulated multiplicity. This was independently
reconciled against the table here, strengthening the aliasing interpretation.

## Number of clusters is only part of the problem

Mean selected K is 6.539 versus true K 2.549. Of 1,262 inferred clusters,
721 contain five or fewer mutations (378 are singletons). Yet those small
clusters contain just 1,381 mutations, 3.31% of all mutations.

Two descriptive, truth-independent postprocessing probes help separate
inflated K from incorrect large-cluster membership. They are diagnostics,
not accepted changes or cross-validated policies:

* Dropping clusters with at most five mutations reduces mean K from 6.539
  to 2.803 but mean ARI only rises from 0.4850 to 0.4968. This changes the
  evaluated population and therefore is not a fair production improvement.
* Connected merging of neighboring refitted centers within CCF distance
  0.05 reduces mean K to 3.756 but mean ARI stays 0.4850. Thresholds 0.01
  and 0.02 likewise give virtually no ARI improvement. There are no center
  duplicates within 1e-6 that would explain the cluster excess.

The high-K output includes erroneous large clusters. For example,
`500_2_0.9_0.7_rep74` has two true clusters, complete search, selected K=19,
ARI=0.1501. Its public label 0 contains 95 truly clonal mutations. Two
additional CNA-only clusters at CCF 0.514 and 0.479 contain respectively
48 and 22 mutations; 40 and 18 of those mutations are also truly clonal.
The two actual truth centers are 1.0 and 0.64568. These are substantial
misassignments, not just small splits around a correctly inferred center.

## Dataset and operational strata

| CNA setting | Cases | Mean ARI | Mean per-case CCF MAE | Mean sMF absolute error |
|---|---:|---:|---:|---:|
| 0.1 | 45 | 0.6905 | 0.05557 | 0.07586 |
| 0.4 | 69 | 0.5243 | 0.08337 | 0.10524 |
| 0.7 | 79 | 0.3336 | 0.11223 | 0.19657 |

The CNA-associated decline persists within every multi-cluster true-K
stratum and every depth stratum. On the identical retained mutation
intersection, CliPP1.5's ARI deficit versus original CliPP grows from
0.0268 at CNA=0.1 to 0.1427 at CNA=0.7. The corresponding deficit versus
PyClone-VI grows from 0.0608 to 0.2190. See `matched_cna_strata.tsv`.

At depth 100/200/500, mean selected K is 4.23/6.43/9.04, while mean
diploid CCF MAE improves to 0.0687/0.0359/0.0177. CNA CCF MAE is
0.1519/0.1482/0.1619. Higher read depth therefore does not remove the
dominant CNA error in this selected subset.

All 24 true-K-one tumors are correct. For true K=2/3/4, mean ARI is
0.4389/0.4059/0.3663 and mean inferred K is 5.36/8.63/9.69. Among K=2
tumors, truth separation still matters: mean ARI is 0.1016 for gaps <=0.25
(4 cases), 0.3177 for gaps 0.25–0.5 (40), and 0.6063 for gaps >0.5 (37).
This helps explain underclustering but cannot explain the large CNA-specific
downward errors and excess K throughout the cohort.

Complete searches (151) have ARI 0.5146, mean selected K 6.95, and CCF MAE
0.0901. Incomplete searches (42) have ARI 0.3787, K 5.05, and MAE 0.0838.
Incomplete search is a separate limitation; it does not account for the
poor complete-search results or the bulk of overclustering.

A100 and H100 cases differ in composition and are not paired experiments.
Their observed ARIs are 0.4667 (100 cases) and 0.5046 (93 cases). A small
additive descriptive regression adjusting categorical K/depth/purity/CNA
setting/search status estimates H100-minus-A100 ARI=0.0223 with HC3 95%
interval [-0.0321, 0.0767]. There is no positive evidence of a family-specific
accuracy failure here; this is not a GPU determinism test.

Changing the reported estimator does not rescue accuracy: mean per-case
CCF MAE is 0.1412 for pilot, 0.1276 for selected penalized raw estimates,
and 0.0887 for final refits. The refit already improves the headline number.

## Reproduction and limits

### One-pass fixed-center reassignment diagnostic

`fixed_center_reassignment.py` asks whether changing membership alone,
without moving any center, can improve the result. It evaluates every mutation
at every existing final-refit center using the frozen source's exact observed
likelihood and original feasibility bounds. The best center is chosen using
either likelihood alone or likelihood plus the log of that cluster's original
mutation fraction. Truth is used only afterward to evaluate the choices.
There is one assignment pass, no iteration, no center refitting, no invented
CCF center, and no truth-based tuning.

| Diagnostic | Mean ARI | Multi-cluster ARI | Mean K | CCF MAE | sMF CCC | sMF MAE |
|---|---:|---:|---:|---:|---:|---:|
| Published result | 0.4850 | 0.4119 | 6.539 | 0.08870 | 0.7831 | 0.13577 |
| Likelihood-only assignment | 0.5084 | 0.4386 | 6.534 | 0.08599 | 0.7923 | 0.12496 |
| Fixed original cluster-size prior | 0.6100 | 0.5546 | 3.000 | 0.06313 | 0.8637 | 0.08143 |
| Fixed cluster-size+1 prior | 0.6116 | 0.5565 | 3.067 | 0.06300 | 0.8642 | 0.08105 |

The unsmoothed size-prior assignment improves ARI in 146 cases, worsens it
in six, and ties in 41; 4,419 of 41,712 memberships change. All 24 true-K-one
tumors stay unsplit. Complete-search ARI improves 0.5146 to 0.6456;
incomplete-search ARI improves 0.3787 to 0.4818. The size+1 result is included
as a conventional smoothing sensitivity probe, not a tuned winner.

For `500_2_0.9_0.7_rep74`, the size-prior assignment changes 115/230
memberships, improves ARI 0.1501 to 0.8099, reduces occupied K from 19 to 4,
CCF MAE from 0.1960 to 0.0213, and sMF error from 0.4130 to 0.0087.

This provides direct evidence that there are useful memberships outside the
published fusion-path partition even with the existing CCF centers. The
mutation likelihood alone favors too many ambiguous assignments; the size
prior substantially helps. It does **not** establish that the revised
partition wins the production score or has a fit certificate. It is a
diagnostic candidate, not an accepted fit, and cannot inherit the original
KKT or refit qualification. The source/score audit must independently check
the production selection rule before recommending integration.

### Plots

The main fourth page uses the parent's stronger exact-production-score
diagnostic in `../reassignment/`, superseding the exploratory fixed-prior
heuristic for presentation. With the published centers fixed, score-decreasing
membership moves improve score and ARI in 152/193 cases and worsen ARI in none.
Mean ARI is 0.6397, CCF MAE 0.05930, sMF CCC 0.8744, and occupied K 2.472.
This evidence remains an offline diagnostic without refit/raw certification.

`CNfirst4K_CliPP15_error_diagnostics.pdf` contains four pages, each also
saved as a separate PDF and PNG:

* `cna_depth_accuracy`: ARI/CCF MAE by CNA setting and depth.
* `clonal_multiplicity_recall`: clonal-label recall by CN and multiplicity.
* `representative_membership_contingency`: truth counts, inferred CCFs,
  and observed CNA composition for the representative 19-cluster failure.
* `exact_score_reassignment`: paired ARI/CCF accuracy and truth-versus-estimated
  sMF under exact-score membership moves, without moving any CCF center.

The separately retained `fixed_center_reassignment.pdf/png` shows the earlier
one-pass empirical-prior heuristic and is not the combined PDF's fourth page.

Run `analyze.py` with the `ml1` interpreter to regenerate primary per-case,
per-cluster, per-mutation, and stratum diagnostics. `case_diagnostics.tsv`,
`cluster_diagnostics.tsv`, and `mutation_diagnostics.tsv` retain the
individual evidence. Additional descriptive tables are stored beside them.

These are the same 193 early finished, small tumors (200–232 mutations)
as the original performance report, not a representative 4,000-case final
panel. Association and diagnostic postprocessing do not establish causality.
Production recommendation needs the source/likelihood audit and independent
fit validation. Nothing here authorizes changing or restarting current fits.
