# Partial SimClone1000 and PhylogicNDT500 comparison

The September 29, 2026, **10:43 a.m. CDT** snapshot includes all **400/756
SimClone** and **71/500 PhylogicNDT** validated CliPP1.5 tumors. PyClone-VI has
completed all 1,256 eligible tumors and is compared on the identical completed
subset and mutation IDs: 297,694 SimClone and 143,916 PhylogicNDT mutations.
There are no missing truth IDs in these evaluated populations.

| Cohort | Method | Mean ARI ↑ | sMF CCC ↑ | CCF MAE ↓ | sMF MAE ↓ | Correct K |
|---|---|---:|---:|---:|---:|---:|
| SimClone1000 | CliPP1.5 | 0.628441 | 0.755885 | 0.057444 | 0.103908 | 215/400 |
| SimClone1000 | PyClone-VI | 0.602012 | 0.788191 | 0.062902 | 0.105732 | 214/400 |
| PhylogicNDT500 | CliPP1.5 | 0.383496 | 0.551183 | 0.098882 | 0.140731 | 12/71 |
| PhylogicNDT500 | PyClone-VI | 0.380088 | 0.623440 | 0.099436 | 0.128980 | 21/71 |

CliPP1.5 has higher mean ARI and lower mean CCF error in both subsets. The
Phylogic ARI advantage is small. PyClone-VI retains higher sMF CCC in both and
better Phylogic sMF error and exact cluster-count recovery. These results do not
establish superiority on all three cohorts or full-cohort performance here.

Paired ARI outcomes, expressed as CliPP1.5 better / worse / tied, are
**158 / 52 / 190** for SimClone and **38 / 22 / 11** for PhylogicNDT. Among
multi-cluster tumors only, mean ARI is **0.443357 vs 0.403763** on SimClone and
**0.361615 vs 0.358003** on PhylogicNDT. Both methods avoid false splits in all
133 SimClone true-K-one tumors; each splits one of the four Phylogic K-one cases.

The estimator is the guarded soft-mixture experiment from commit
`fc9d349761de68d8c8771fe5d92137eabfc6e784`. It preserves partition-refit estimates
in 327 cases and selects converged new mixture structures in 144. Fitting
remains unconstrained. The comparison includes the earlier 436 tumors plus 34
ordinary GPU completions and managed-memory SimClone case `simhr8dhc`. Its
allocator overlay retains a separate qualified source inventory. No numerical
or scheduling policy changed during this evaluation.

Both methods' previous 436-case metrics reproduce within 1e−12. All selected
saved outputs, identities and ancestry were revalidated. PyClone-VI validation
checks all 1,000 restarts and posterior/TSV agreement; scoring uses unrounded
posterior centers. Settings are binomial density, 20 starting clusters and
1,000 restarts. Matching uses the same retained mutations and corrected
SimClone normal-CN=2 inputs. Only true cluster labels and CCFs enter these
metrics; mixed-CN metadata does not affect them.

Clonal means the occupied cluster closest to CCF 1; no evaluated fit has an
exact clonal tie. ARI and CCF/sMF absolute errors are equal-tumor means. sMF
CCC is Lin concordance across tumors. Identical constant vectors score one;
other constant-vector cases score zero. The full report also includes
mutation-pooled clonal/subclonal precision and recall and per-case metrics.

The report, separate cohort PDFs and hash-bound artifacts are under
`results/experimental-full29-20260928-v3/performance-twocohort-20260929-v1/`.
See [the small evidence record](TWOCOHORT_PERFORMANCE_20260929.json),
`REPORT.md`, `SUMMARY.json`, `per_case.tsv` and `EVALUATION_RUN.json` there.
The study pointer's `latest_twocohort_performance` identifies this dated
comparison; it does not replace the completed CN-first4K evaluation.
