# Why CliPP1.5 loses sMF accuracy to PyClone-VI on the current CN-first panel

**The main demonstrated cause is inadequate partition search: some tumors are
collapsed to one cluster, and the current refinement cannot create a new one.**
The strongest evidence is that alternative memberships score better under
CliPP1.5's own unchanged model, and an offline, truth-free cluster-birth probe
recovers most of the collapsed tumors without changing the score or likelihood.

This investigation freezes the **September 25, 2026, 6:48:02 PM CDT** comparison:
382 finished tumors and 87,398 mutations matched across all four methods. It
evaluates only the requested separate partition-search estimate from commit
`d44c3e728fa6f535f725719cc77a1dab9a96efc5`. These are early small tumors, with
200–263 original mutations, representing 9.55% of the 4,000-tumor cohort.
They do not establish final cohort rankings.

The [investigation artifacts](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/)
contain reproducible scripts, frozen input/output bindings, component replays,
case-level measurements and plots. Production source, completed outputs and
the running GPU campaign were not changed.

## 1. The clearest deficit is sMF tail error

| Metric | CliPP1.5 | PyClone-VI | CliPP1.5 minus PyClone-VI, paired 95% interval |
|---|---:|---:|---:|
| Mean ARI | 0.66730 | 0.65848 | +0.00882 [−0.00854, +0.02646] |
| Mean ARI, true K > 1 | 0.62287 | 0.62475 | −0.00187 [−0.01840, +0.01377] |
| Mean CCF absolute error | 0.052867 | 0.052753 | +0.000114 [−0.002098, +0.002404] |
| sMF CCC | 0.90179 | 0.94806 | −0.04627 [−0.08154, −0.01732] |

These are exploratory paired-tumor bootstrap intervals, with 30,000 resamples
and fixed seed 20260925. CCC is recomputed from paired tumor-level fractions
within each resample. The intervals describe this selected panel; they do not
correct for its completion bias or repeated exploratory comparisons.

CliPP1.5 wins ARI in 202 tumors, loses in 127, and ties in 53. It wins CCF MAE
in 236 versus 146. Thus it is not generally worse. A minority of large losses
outweigh many small gains for sMF concordance. The tiny multi-cluster ARI and
CCF differences are unresolved. They also reverse when these two methods are
compared on their full shared native population of 88,278 mutations instead
of the four-method intersection; the sMF deficit persists.

## 2. Twenty-seven collapsed tumors explain the net sMF error gap

CliPP1.5 selects one cluster for **27/337 true multi-cluster tumors**, compared
with **3/337** for PyClone-VI. Under the agreed closest-to-one convention, a
single occupied cluster is designated clonal and predicted sMF becomes zero.

These 27 tumors contribute **69.0% of CliPP1.5's total sMF squared error** and
**102.1% of its net excess squared error over PyClone-VI**. This is a decomposition
of squared error, not an additive decomposition of CCC. The other 355 tumors
slightly favor CliPP1.5 in squared error.

| Population | Tumors | ARI, CliPP1.5 / PyClone-VI | CCF MAE, CliPP1.5 / PyClone-VI | sMF CCC, CliPP1.5 / PyClone-VI |
|---|---:|---:|---:|---:|
| All matched tumors | 382 | 0.66730 / 0.65848 | 0.05287 / 0.05275 | 0.90179 / 0.94806 |
| True multi-cluster collapsed to K=1 | 27 | 0 / 0.34046 | 0.10870 / 0.06677 | 0 / 0.64642 |
| Other tumors, diagnostic only | 355 | 0.71805 / 0.68267 | 0.04862 / 0.05169 | 0.96574 / 0.96367 |

Removing bad cases is not a valid replacement benchmark. The last row only
locates the failure mechanism. Even among the remaining 310 true multi-cluster
tumors, CliPP1.5 has higher mean ARI: 0.67713 versus 0.64951.

| Case | True sMF | CliPP1.5 sMF | PyClone-VI sMF |
|---|---:|---:|---:|
| `100_2_0.6_0.7_rep57` | 0.72692 | 0 | 0.71154 |
| `100_2_0.4_0.1_rep63` | 0.57088 | 0 | 0.49425 |
| `100_2_0.4_0.1_rep138` | 0.51980 | 0 | 0.46040 |

The problem is not an incorrect numeric cluster label. In all 382 cases, the
cluster chosen as clonal also contains the largest number of true clonal
mutations. Relabeling cannot restore missing partitions, and these results
do not support restoring a clonal fitting constraint.

## 3. Search structure makes K=1 an absorbing state

The frozen implementation evaluates qualified raw-start memberships, then
refines **one best-scoring seed**. A refinement move transfers a mutation into
an existing occupied cluster. It may delete a singleton cluster but cannot
create a cluster. A K=1 seed therefore has no possible move.

All 27 collapsed tumors enter refinement with K=1 and remain at that fixed
point. Nineteen never receive a qualified raw candidate at the true K. Eight
do receive one, but its memberships score worse than the single-cluster fit.
Across all 337 multi-cluster tumors, only 89 have a raw proposal at the true K.
A complete graph does not by itself provide complete partition exploration.

The implementation evidence is in
[partition seed selection](src/clipp1d/cuda/partition_search.py) and
[existing-center refinement](src/clipp1d/cuda/refinement.py).
The hash-bound consumed copies and per-case candidate records are preserved in
the investigation artifacts.

## 4. Same-model counterfactuals separate search failure from score preference

For every one of the 382 tumors, I independently refitted three fixed
memberships using the frozen CliPP1.5 scalar refitter: its selected partition,
PyClone-VI's partition, and the true partition. These CPU component diagnostics
preserve the likelihood, multiplicity prior, original bounds, scalar tolerances
and partition score. They do not run new fusion fits. Truth is used only as a
retrospective diagnostic, never to propose production candidates.

**In 20 of the 27 collapsed cases, PyClone-VI's memberships have a lower score
under CliPP1.5's own model.** CliPP1.5 would accept them if its search found them.
An excessive penalty cannot be the sole explanation for those failures.

| Case | Best raw candidate at true K: score change from selected | PyClone partition refitted by CliPP: score change |
|---|---:|---:|
| `500_2_0.6_0.4_rep136` | +18.54 | **−261.46** |
| `100_2_0.4_0.1_rep138` | No K=2 candidate | **−92.05** |
| `100_2_0.4_0.1_rep63` | +5.00 | **−50.55** |
| `100_2_0.6_0.7_rep57` | No K=2 candidate | **−34.68** |

Lower scores are better. The first case has 210 matched mutations at true CCF 1
and 29 at CCF 0.7152. CliPP1.5 collapses them to center 0.9684, while PyClone-VI
recovers K=2 with ARI 0.8973. Its alternative clears the unchanged score by a
large margin: merely generating an arbitrary K=2 partition is insufficient.

Across the full panel, 57 PyClone partitions have lower CliPP scores, 273 have
higher scores and 52 tie within numerical tolerance. Among the 127 cases where
PyClone has higher ARI, 47 have lower CliPP scores and 80 have higher scores.
The latter establish remaining score-versus-truth disagreement; broader search
alone is not guaranteed to make every case more accurate. In finite noisy
data, even true memberships need not minimize a fitted selection criterion.

## 5. A truth-free cluster-birth probe recovers most of the missing signal

I tested a minimal proposal mechanism on **all 72 selected K=1 tumors**, which
includes the 27 false collapses and all 45 true single-cluster controls.
Eligibility used the inferred K, not truth. For each case:

1. Propose two centers using a fixed lower-center grid 0.05–0.95 and upper
   initializer1.0, assigning mutations by the unchanged marginal likelihood.
2. Independently qualify the scalar refit of every distinct proposed partition.
3. Refine the three best split seeds with the existing score for at most 20 rounds.
4. Retain the original selected partition unless a candidate has a better score.

The initializer at1 is only a proposal. Both centers remain freely refitted
within their original bounds; no clonal constraint is imposed. Neither truth
nor PyClone assignments generate these proposals. This is a CPU discovery-panel
experiment, not production integration or allocated-CUDA qualification.

The probe split **22/27 false collapses**, improved their ARIs, and introduced
**0/45 false splits** among the single-cluster controls. Five collapses remain.
Replacing only these 72 cases with the diagnostic selection, while retaining
the other 310 current results, gives:

| Metric on the same 382 tumors | Current CliPP1.5 | Birth-proposal diagnostic | PyClone-VI |
|---|---:|---:|---:|
| Mean ARI | 0.66730 | **0.69092** | 0.65848 |
| Mean ARI, true K > 1 | 0.62287 | **0.64964** | 0.62475 |
| Mean CCF absolute error | 0.05287 | **0.05018** | 0.05275 |
| sMF CCC | 0.90179 | 0.94826 | 0.94806 |
| sMF MAE | 0.05855 | **0.04758** | 0.05238 |
| Correct K | 244/382 | 261/382 | 238/382 |
| False splits in true K=1 | 0/45 | 0/45 | 4/45 |

This supports missing cluster proposals as a practical repair target. The
near-equal diagnostic CCC values do not establish superiority over PyClone-VI.
The probe was designed after inspecting this panel, handles K=1 only and has
not been validated on held-out cases, larger tumors, or production CUDA.

## 6. Incomplete searches, signal strength and model differences

Of the 160 incomplete partition searches, 83 have raw-path incompleteness only,
60 exhaust the four-round refinement budget only, and 17 have both. There are
zero unresolved observed candidate refits. Importantly, **11/27 false collapses
have complete searches**. Completeness refers to configured search coverage,
not global partition optimality. Adding 20 refinement rounds to 25 already-fixed
difficult cases changes nothing; K=1 needs a different move type.

A separate continuation experiment addressed the **77 cases that actually hit
the four-round budget**. All reached a fixed point within 11 additional rounds
(median 2), with no scalar failures. Scores improved in 52; ARI improved in 37,
worsened in 14 and tied in 26. Only one case changed K. Within these 77 cases,
mean ARI rose from 0.55891 to 0.57081 and CCF MAE fell from 0.08331 to 0.08055,
but sMF CCC fell from 0.92226 to 0.91054.

If those continuations alone replace their corresponding results in the full
382-case panel, ARI becomes 0.66970, multi-cluster ARI 0.62560, and CCF MAE
0.05231, while sMF CCC becomes 0.90037. Thus the round cap contributes to the
small ARI/CCF differences; it does not explain the main sMF gap. These are
separate diagnostic replacements, not combined with the birth-proposal table
above. Lower score is not a guarantee of higher truth accuracy.

Collapses are enriched for lower depth, lower purity and smaller subclones:
17/124 depth 100 tumors versus 1/126 depth 500 tumors; 18/126 purity 0.4 versus 3/124
purity 0.9. Within true K=2, median minority fraction is 0.132 in collapsed cases
versus 0.286 otherwise; median CCF separation is 0.354 versus 0.576. These are
associations, not controlled causal effects, and the depth 500 failure above
shows that absence of signal is not a sufficient explanation.

CliPP1.5's uniform integer multiplicity model and fixed CN denominator match
this CN-first generator directly. PyClone-VI uses different genotype states,
a sequencing-error term, soft assignments and posterior CCF estimates. Its
1,000 randomized restarts over 20 components explore a different search space
from CliPP1.5's deterministic fusion starts. Their individual contributions
would require separate ablations.

The observed data do not support a blanket multiplicity-model failure:
mutation-weighted CCF MAE on amplified loci with major CN ≥ 2 is 0.08628 for
CliPP1.5 versus 0.09510 for PyClone-VI. On diploid 1/1 loci it is 0.03389 versus
0.02811. These strata are confounded by case difficulty. Changing the likelihood
or multiplicity prior before repairing the demonstrated search gaps is not
supported by this evidence.

## 7. Recommended implementation order

1. Add independently qualified cluster-birth/split candidates while retaining
   existing candidates, the complete graph, likelihood, score and unconstrained
   fit. Give direct partitions their own provenance; do not inherit raw KKT
   certificates or selected penalties.
2. Refine a bounded set of strong seeds across K, instead of only the single
   best seed before refinement. Evaluate coordinated split/merge moves separately.
3. Treat refinement-budget exhaustion separately from missing cluster proposals;
   extend useful refinement with an explicit budget and honest coverage status.
4. Confirm the change on held-out/larger cases and true-single controls, with
   CPU/CUDA parity and runtime/memory checks before promoting production behavior.
5. Then study residual score/prior calibration and posterior-mean versus point
   estimation differences. Avoid tuning a penalty to conceal missed partitions.

## Verification and scope

Independent checks matched all 39 frozen source files to their launch hashes
and all 382 canonical input files to the snapshot. Fixed-center selected scores
reproduced within 8.73e−11, and independently refitted selected scores within
1.31e−10. ARI, sMF and K reproduced exactly; maximum CCF-MAE difference from
CPU scalar replay was 3.48e−8. All fixed-partition counterfactuals completed
without qualification errors.

The birth-probe labels, centers, matched IDs, non-increasing score and whole-panel
metrics were independently checked. Truth is read after proposal selection;
PyClone assignments are not used in that experiment. All 72 selected K=1 tumors
were tested and the other 310 estimates were retained. The 77 budget continuations
are a separate experiment. None of these component replays establishes GPU
runtime, full-fit speed, CUDA parity or held-out performance.

Detailed evidence: [paired statistics](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/statistics/findings.md),
[mutation anatomy](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/anatomy/FINDINGS.md),
[model/search audit](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/model/FINDINGS.md),
[anatomy PDF](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/anatomy/anatomy.pdf),
[birth probe script](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/birth_probe.py).

[Birth-proposal comparison PDF](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/anatomy/birth_diagnostic_comparison.pdf)
compares the unchanged production results with the offline diagnostic and
PyClone-VI. [Numerical verification](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/statistics/counterfactual_verification.json)
and [budget-continuation results](results/cnfirst-pool14-20260925-v3/pyclone-gap-investigation-20260925T235034Z/model/budget_replay/SUMMARY.json)
retain the corresponding checks.
