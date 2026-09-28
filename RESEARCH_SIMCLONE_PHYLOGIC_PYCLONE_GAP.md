# Why CliPP1.5 trails PyClone-VI on PhylogicNDT500 and SimClone1000

Evidence snapshot: **September 27, 2026, 4:33 PM CDT**. Investigation completed September 27, 2026. This is a retrospective study of the fixed finished-case intersection, not a refreshed progress report or a full-cohort comparison.

**The main findings are score-driven merging on Phylogic, a small set of one-cluster collapses driving SimClone's sMF deficit, and a separate multiplicity-mixture likelihood problem driving several severe SimClone CCF errors. Restricted search is real, but expanding proposals under the unchanged score cannot by itself recover most of the better PyClone reference partitions.**

[Five-page diagnostic PDF](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/CliPP1.5_vs_PyCloneVI_diagnosis.pdf) · [Frozen comparison](results/twocohort-lsf20-20260926-v1/comparison-pyclone-20260927T213302Z/COHORT_REPORT.md) · [Investigation evidence](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/INVESTIGATION_BINDING.json)

## 1. What is actually worse

The analysis covers **51 Phylogic tumors and 385 SimClone tumors**, with **350,873 matched mutations**. CliPP1.5 means the independently refitted partition-search output. PyClone-VI used binomial density, 20 available components and 1,000 restarts; CCFs come from full-precision posterior means, not rounded exports.

| Cohort / metric | CliPP1.5 | PyClone-VI |
| --- | ---: | ---: |
| Phylogic: mean ARI | 0.31079 | **0.34046** |
| Phylogic: sMF CCC | 0.33035 | **0.61550** |
| Phylogic: mean sMF absolute error | 0.22542 | **0.13645** |
| Phylogic: mean CCF absolute error | 0.11996 | **0.10666** |
| SimClone: mean ARI | **0.60939** | 0.60650 |
| SimClone: sMF CCC | 0.69626 | **0.79384** |
| SimClone: mean sMF absolute error | 0.12389 | **0.10319** |
| SimClone: mean CCF absolute error | 0.06717 | **0.06202** |

SimClone is not uniformly worse: its overall ARI is effectively tied, slightly favoring CliPP. Among its 254 true multi-cluster tumors, ARI is 0.40794 versus 0.40356. Its 131 single-cluster tumors have ARI 1 for both methods. Phylogic has only two single-cluster tumors; the cohorts' overall ARIs therefore reflect different difficulty mixtures.

The paired ARI win/loss/tie counts for CliPP are **16/28/7 Phylogic** and **113/83/189 SimClone**. Paired 5,000-resample intervals for the ARI difference are [−0.04819, −0.01285] and [−0.00660, +0.01259]. CCF-error and sMF-CCC differences favor PyClone in both cohorts in these conditional bootstrap analyses. Grouping repeated SimClone truth structures before resampling gives the same qualitative conclusions.

All headline means weight tumors equally; mutation-pooled analyses below are explicitly identified. The finished subsets are selected toward smaller tumors: median retained N is 1,980 versus 4,004 in the planned Phylogic cohort, and 480 versus 2,054 in SimClone. These are not full-cohort estimates.

## 2. The diagnostic that separates search failure from score preference

For every tumor I took three fixed memberships: the saved CliPP partition, PyClone's hard posterior assignments, and the known simulation truth. I then refitted their centers with **the same pinned CliPP likelihood, feasible bounds, score, and scalar tolerances**. Each diagnostic uses the full CliPP-retained population. All those mutations are truth matched in this snapshot.

This produced **1,308 qualified scalar partition refits**, with zero errors. The saved CliPP replay reproduces all 436 original ARIs and sMFs exactly; its largest score difference from the published result is **2.47 × 10⁻¹⁰** and largest CCF-MAE difference is 1.53 × 10⁻⁸. Score order is checked with the scalar lower/upper uncertainty bounds, not rounded printed values.

These are CPU component diagnostics, not new fusion fits or allocated-CUDA qualification. Truth memberships are retrospective references, never inference proposals.

| PyClone membership after refitting under CliPP | Phylogic, 51 | SimClone, 385 |
| --- | ---: | ---: |
| Lower CliPP score than saved partition | 3 | 12 |
| Overlapping score bounds | 7 | 188 |
| Higher CliPP score | 41 | 185 |
| Among PyClone's ARI wins: lower score | 3 of 28 | 7 of 83 |
| Among PyClone's ARI wins: higher score | **25 of 28** | **76 of 83** |

Among those higher-score ARI wins, **22 Phylogic and 52 SimClone cases have better likelihood but lose because their allocation/complexity penalty is larger**. Other cases can also prefer a different membership under the likelihood itself.

No PyClone membership has lower CliPP score in any of the **62 true-multicluster tumors that CliPP collapses to K=1**. The refitted truth partition also loses to the collapse in all 62; its likelihood improvement is outweighed by the penalty. Across all 436 tumors, no truth partition has strictly lower score than the selected CliPP partition. This does not make truth "wrong": latent generating labels need not maximize an objective on noisy observations. It establishes a mismatch between this criterion's preference and the evaluated latent-cluster accuracy.

There are demonstrated missed lower-score candidates, so search still matters. However, **simply supplying the exact better PyClone membership would not make the unchanged selector choose it in most observed ARI losses**. This test compares one external reference per tumor; it does not bound what all other possible partitions could achieve.

Evidence: [score attribution](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/search_audit/counterfactual_summary.json), [per-case changes](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/search_audit/counterfactual_score_deltas.tsv), [all refits](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/counterfactual/per_case.tsv).

## 3. Phylogic: merging distinct populations is the main deficit

CliPP selects fewer clusters than PyClone in **24 of 51 tumors**, the same number in 27, and more in none. Those 24 cases contribute **90.9% of the net ARI deficit, 81.1% of the sMF-MAE deficit and 82.5% of the CCF-MAE deficit**. This is an additive descriptive decomposition, not a causal effect of selected K.

Six tumors have CliPP K=1 while PyClone selects K=2. All six have complete declared searches, purities 0.26–0.44, mean depths 31.7–51.2, and at least 99.44% supplied diploid mutations. They contribute **48.6% of excess sMF squared error**. Weak separation at these count levels plausibly makes a split harder to justify, but the exact score replay identifies the actual preference.

The minimized score is:

```text
S = 2 × negative_log_likelihood + K log(N) − 1.4 × A
A = lgamma(K) − lgamma(N+K) + Σ lgamma(n_k+1) + lgamma(K+1)
```

It includes a substantial **cluster-allocation term**, not just K log(N). At fixed K, moving one mutation from a group of size a into a group of size b adds `1.4 log(a/(b+1))` to twice the likelihood change. Weakly supported mutations can therefore be absorbed into large groups. No hidden hard minimum cluster size was found.

Concrete fixed-membership comparisons, each against the saved CliPP partition:

| Case | CliPP K → PyClone K | Change in twice negative log-likelihood | Change in penalty | Total score change; higher loses |
| --- | ---: | ---: | ---: | ---: |
| Sim_500_407 | 1 → 2 | −1,079.13 | +1,543.19 | **+464.07** |
| Sim_500_406 | 1 → 2 | −1,970.09 | +2,013.06 | **+42.97** |
| Sim_500_77 | 3 → 3 | −712.64 | +978.64 | **+266.00** |

For `Sim_500_407`, ARI rises from 0 to 0.211 and refitted CCF MAE falls from 0.2430 to 0.1675 with PyClone memberships, but the existing score rejects that partition. Its truth partition has CCF MAE 0.0800 but an even larger score disadvantage. For `Sim_500_406`, truth has five groups; CliPP's single center is 0.7944 and sMF is zero, while PyClone's two groups recover much of the clonal/subclonal separation.

`Sim_500_77` shows that **the problem is not only the number of clusters**. Both methods select three groups. CliPP leaves only nine mutations in its lowest group and assigns 220 of 229 low-CCF truth mutations to its middle group. PyClone captures 211 of those 229 in its low group. The more accurate same-K membership improves likelihood but loses through the size-dependent penalty.

The exact CCF MSE decomposition supports this distinction. Replacing each estimated center by the mean true CCF of its predicted members separates within-group mixing from center displacement:

| Phylogic CCF MSE component | CliPP | PyClone | CliPP − PyClone |
| --- | ---: | ---: | ---: |
| Within-predicted-group truth variance | 0.033975 | 0.030840 | +0.003135 |
| Center displacement from group truth mean | 0.005352 | 0.006163 | −0.000810 |
| Total | 0.039327 | 0.037003 | +0.002324 |

The main Phylogic disadvantage is who gets grouped together; its conditional center displacement is actually smaller under this decomposition.

## 4. SimClone sMF: a few collapses dominate the gap

**Nineteen of 385 tumors have CliPP K=1 and PyClone K>1. They explain 94.7% of the net sMF-MAE gap and 95.8% of excess sMF squared error.** Eighteen of those searches are marked complete. Among 305 equal-K tumors, CliPP has slightly better ARI and sMF MAE; its sMF CCC is 0.8090 versus 0.7958.

The agreed designation rule assigns the occupied group closest to CCF 1 as clonal. A one-group result therefore has sMF zero even when its estimated CCF is below one. This is a consequence of the partition and the chosen estimand, not a label-permutation bug or a reason to restore the removed fitting constraint.

Example `sim3p5fhr`: 529 true mutations have CCF 0.840397 and only 30 have CCF 1. CliPP merges them at 0.847610. Its **CCF MAE is only 0.0150**, yet its **sMF is 0 instead of 0.9463**. PyClone's sMF is 0.9088. Under the unchanged CliPP score, the refitted PyClone membership loses by 94.27 and the truth membership loses by 131.83. Both models support the true centers when given the correct memberships in this diploid case.

Good aggregate sMF alone should not be mistaken for correct mutation-level clonality. PyClone's designated clonal group in this example also contains 37 true subclonal mutations and only 14 true clonal mutations. The error rates below make the differing tradeoff explicit:

| True-subclonal mutations assigned to the designated clonal group | CliPP | PyClone |
| --- | ---: | ---: |
| Phylogic, pooled mutations | 50.80% | 35.40% |
| SimClone, pooled mutations | 35.38% | 25.58% |

CliPP makes fewer mistakes in the reverse direction, classifying fewer true clonal mutations as subclonal. Its conservative splitting behavior lowers estimated sMF overall.

## 5. SimClone CCF: the likelihood itself can favor a wrong multiplicity/CCF mode

The five largest CCF losses account for **49.1% of the net mean CCF-error gap**. Every one has better ARI than PyClone and 90.4–100% CNA mutations. Thus their poor centers cannot be reduced to inaccurate memberships.

The same diagnostic was repeated on all five cases, supplying exact truth membership while retaining the CliPP likelihood:

| Case | Original CliPP CCF MAE | Original PyClone CCF MAE | CliPP MAE with truth memberships | Largest subclonal group: true CCF → CliPP refit |
| --- | ---: | ---: | ---: | --- |
| simn0vt5y | 0.3653 | 0.0636 | **0.3511** | 0.8404 → 0.4694 |
| simvf66y3 | 0.3217 | 0.0790 | **0.3018** | 0.8504 → 0.4762 |
| sim2kigh0 | 0.2935 | 0.1059 | **0.2192** | 0.8380 → 0.5178 |
| simbxvuhy | 0.2380 | 0.1134 | **0.2019** | 0.6710 → 0.3957 |
| simkhuc9f | 0.2149 | 0.0983 | **0.1736** | 0.6738 → 0.3927 |

Correct memberships retain **75–96% of the original CCF error**. On a 100-point grid, the low CliPP likelihood peaks beat the nearest true-CCF point by 89.81, 63.12, 205.23, 113.83 and 183.28 log-likelihood units, respectively. This is evidence of the model's preference, not just a solver landing in an inferior local well.

These five cases were selected retrospectively and do not estimate the cohort-wide frequency of this mechanism. Some truth distinctions are also intrinsically difficult: two subclonal CCFs in `sim2kigh0` differ by only 0.00336, and `simvf66y3` includes groups of only 2–5 mutations. Those limits matter for recovering every truth group, but do not explain the large dominant-group center shifts above.

### Why this happens

CliPP uses a uniform integer-multiplicity mixture, with:

```text
p(VAF) = purity × CCF × multiplicity
         / ((1−purity) × normal_CN + purity × tumor_total_CN)
```

The denominator is fixed with respect to CCF. At amplified loci, a lower CCF and a higher multiplicity can explain similar read proportions. Marginalizing uniformly over those possibilities can give enough combined probability mass to a lower-CCF mode to defeat the few unambiguous dosage anchors.

In `simn0vt5y`, all 529 mutations in the main truth group have multiplicity one. Of those, 517 have major CN above one, including 310 balanced 2/2 loci; only 12 mutations have major CN one. Moving from true CCF 0.8404 to the selected low mode gains **142.06** log-likelihood units on amplified loci and loses **51.57** on the anchors: a net gain of 90.49. Counts at the 310 balanced-amplified loci imply CCF × multiplicity about 0.8391, consistent with the supplied truth. No count-scaling defect is required to produce the failure.

PyClone's "binomial" model is different. Its mutation-before-CN genotypes mix normal-CN tumor cells without the mutation with altered-CN cells carrying it, making the denominator CCF-dependent. It also includes an after-CN multiplicity-one genotype with a fixed denominator, equal genotype masses and sequencing error 0.001. For the same fixed truth group in `simn0vt5y`:

| Likelihood profile | Best CCF on 100-point grid |
| --- | ---: |
| CliPP uniform multiplicity mixture | **0.46465** |
| Original stored PyClone genotype-mixture likelihood | **0.78788** |
| Oracle fixing known multiplicity one | **0.83838** |
| True CCF | 0.84040 |

PyClone favors the nearest truth grid point over the low CliPP grid point by **209.63 log-likelihood units**. Its grid optima for the other four dominant groups are 0.798, 0.758, 0.566 and 0.586—closer to truth, though still biased downward. Absolute likelihood levels across the two models include different constants and were not compared.

The diploid companion `sim3p5fhr` shares the same truth structure and purity; both models peak at 0.83838 for its main group. These simulations differ in CN and observed reads, so their contrast is not a controlled CN-only intervention. The within-case fixed-membership likelihood profiles provide the stronger evidence.

Across the entire audited snapshot, **all 88,597 SimClone and 45,132 Phylogic true-subclonal mutations have multiplicity one**; higher multiplicities occur only in true-clonal groups. This empirical generation pattern differs from a uniform multiplicity prior independent of CCF. It motivates a prior/genotype study, not imposing test-set truth or multiplicity one on real tumors. The revised CN-first generator deliberately uses a different mechanism, so any repair needs cross-generator validation.

For SimClone, exact mean CCF MSE decomposes into **−0.000226 within-partition advantage** and **+0.001985 center-displacement disadvantage**, producing the net +0.001759 error. This independently supports prioritizing the center/likelihood problem.

Evidence: [input and model audit](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/input_audit/REPORT.md), [five-case measurements](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/input_audit/top5_profiles/case_summary.tsv), [bound profile curves](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/input_audit/top5_profiles/LIKELIHOOD_PROFILES.json).

## 6. Search limitations remain, but do not explain everything

The active policy is `birth_mode=single_cluster`, one retained seed, one birth generation, four ordinary refinement rounds and twenty birth-refinement rounds. Direct reassignment moves only to existing groups and can delete groups. Birth runs only at K=1 and can create K=2; it cannot repair an already underclustered K=2 or construct K≥3 through successive births under this policy.

Observed records show:

- **4,221 qualified birth proposals** were already tried: 426 Phylogic and 3,795 SimClone. The worst collapsed examples each received nineteen qualified splits. Birth was not simply absent.
- Birth actually increased final K from one to two in 10 Phylogic and 42 SimClone tumors. No true single-cluster tumor was falsely split in this snapshot.
- **26 Phylogic and 100 SimClone underclustered cases entered birth above K=1**, so it was skipped. Another 9 and 33 remained underclustered after a successful K=1→2 birth.
- All 436 selected partition refits qualified; no observed partition-candidate refit was unresolved. Most underclustered cases had complete declared searches: 28/46 Phylogic and 121/184 SimClone.
- Incompleteness includes raw-path coverage and the four-round direct refinement cap. It does not mean the selected center refit failed. Conversely, "complete" means complete within the configured search, not globally optimal.

The external-partition test exposes 15 lower-score missed candidates, but five of the 12 SimClone candidates have worse ARI. Better optimization and better truth accuracy are separate outcomes.

For concrete search failures, `Sim_500_333` improves from K=2 to K=5, lowers the score by 224.65 and increases ARI from 0.3876 to 0.5007 with PyClone memberships. `simrlamd8` improves from K=2 to K=3, lowers the score by 288.60 and increases ARI from 0.4783 to 0.6445. Both original searches were marked complete. In contrast, `simleqhxu` lowers the score by 15.71 but reduces ARI from 0.7183 to 0.6457. See the [independent counterfactual audit](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/search_audit/COUNTERFACTUAL.md).

PyClone uses soft assignments, CCF posterior distributions and 1,000 randomized restarts. Those may help, but this investigation does not isolate their individual contributions. Claims that the deficit is simply fewer restarts, the GPU, or insufficient numerical precision are unsupported by these tests.

Source references and exact branch audit: [search report](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/search_audit/REPORT.md). The bound source is under [the original payload](results/twocohort-lsf20-20260926-v1/payload/source/src/clipp1d/cuda/partition_search.py), not an assumed current checkout.

## 7. Input and evaluation explanations checked

**The old SimClone normal-CN bug is not the current cause.** Both methods consume corrected normal CN=2. All 350,873 matched mutations agree on counts, supplied tumor/normal CN and purity.

PyClone fitted additional high-CN mutations that CliPP excludes: 2,135 SimClone mutations across 114 tumors, and 53 Phylogic mutations across seven tumors. Matching evaluation IDs does not equalize fitted populations. Nevertheless, the Phylogic deficit persists among the **44 tumors with identical fitted populations**: ARI difference −0.03241 and CCF-MAE difference +0.01495. This rules out population differences as a sufficient explanation, while a rigorously controlled rerun should still equalize inputs.

Original Phylogic truth has mixed CN or primary-CN disagreement at **8,200/98,503 mutations**. At 798 loci, true multiplicity exceeds the supplied major CN. These are genuine input/model inconsistencies affecting absolute accuracy. They do not explain most of the differential loss: **94.75% of CliPP's excess pooled absolute CCF error is on clean-CN loci**. On 85,866 clean diploid mutations, pooled CCF MAE is 0.11675 versus 0.10372.

A metadata defect was found: normalized Phylogic `truth.tsv` rows all write `mixed_cn=0`, although the original bound MAF contains mixed-CN loci. Current ARI/sMF/CCF evaluation did not use that field. This investigation obtained the correct flags from the original MAF; future preparation should preserve them in a newly versioned truth artifact. Frozen active inputs were not rewritten.

The closest-to-one designation is consistently applied without imposing CCF=1 during fitting. Refit centers can change the designation: `simsiijpa` is the one PyClone-membership replay whose sMF changes after a center tie. ARI is invariant to this refit. Original PyClone performance and CliPP-refitted PyClone membership performance remain distinct throughout this report.

## 8. What to test next, preserving the framework

The evidence changes the priority from "more proposals should solve the gap" to **separate experiments for the criterion, the likelihood, and the remaining search coverage**.

| Experiment | What stays fixed | Discriminating outcome |
| --- | --- | --- |
| Score/allocation calibration on diploid tumors | Likelihood, inputs, proposal inventory and center solver | Can a separately calibrated allocation penalty or mixture criterion recover clonal/subclonal separation without false splits? |
| Genotype/multiplicity model sensitivity on CNA-heavy tumors | Fixed memberships first; matched counts and bounds | Does the true-CCF neighborhood become preferred under a defensible generative model? |
| General birth and diverse retained seeds | Existing likelihood and score | Can truth-free proposals recover the demonstrated lower-score missed partitions? |
| Additional refinement only on exhausted cases | Source/model/score and initial candidates | How much remaining score and accuracy change is due to the round cap? |
| Matched-input control | Exact common fitted mutation population | How much does PyClone's extra high-CN population influence shared centers? |

Do not pick an allocation coefficient from these test outcomes and call it validated. Retain single-cluster controls and disjoint simulation structures; test both these generators and CN-first simulations. A score change can trade underclustering for false splits. A multiplicity-one rule tuned to these cohorts can be wrong for CN-first or real tumors.

The existing prior-perturbation proposal study restores the original likelihood for final scoring. That can help find a missed partition, but it **cannot make the original likelihood prefer a high-CCF mode that loses even with correct membership**. Addressing the five demonstrated center failures requires a separately identified model/estimator experiment, beyond proposal-only prior perturbation.

Keep unconstrained fitting and the agreed closest-to-one label. Preserve current candidates, record direct-candidate provenance, and qualify any production revision on allocated CUDA with a separate held-out accuracy panel. This investigation implements no scientific production change and launches no new full fits.

## 9. Reproducibility and boundaries

Consumed CliPP source: commit `da651879c2a33f481957ffae1d1c865d11ad39c8`, numerical fingerprint `55d6e45949e4dee2e8798ac81fb49510baffca9959104355db87d45acca1a523`. All 96 frozen source inventory entries, input/truth identities, the 436 original run receipts, partition tables, and effective PyClone HDF5/validation receipts were checked. Two independent audits reproduced all 872 original case/method CCF MAEs; all 872 MSE decomposition identities were checked.

The two original external-center arrays for `sim4xnvmi` and `simtag6jj` include CCF 1 outside CliPP's bound 0.999999. Their **qualified refits** are valid and are used in score comparisons. Their original `fixed_center_score` fields are diagnostic arithmetic, not feasible candidate witnesses. Per-case replay `input_mutations` denotes the retained diagnostic N, not original native N.

No active run, consumed payload, production source, output or receipt was modified. The existing unrelated prior-study worktree changes were preserved. Local arithmetic used spare CPU affinity and single numerical-library threads. No result here is a CUDA pass or global partition-optimality certificate.

Detailed evidence lives under [the investigation directory](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap):

- [Statistical decomposition and uncertainty](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/statistics/REPORT.md), with per-case and cluster-composition TSVs.
- [Input/genotype audit](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/input_audit/REPORT.md), with original-CN strata and likelihood profiles.
- [Search/source audit](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/search_audit/REPORT.md), with every recorded birth/refinement history and score decomposition.
- [Fixed-membership diagnostic script](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/counterfactual.py) and [completion receipt](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/counterfactual/COMPLETE.json).
- [Plot script](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/plot_investigation.py), [figure bindings](results/twocohort-lsf20-20260926-v1/investigation-20260927-pyclone-gap/FIGURES.json), and the final `ARTIFACTS.json` inventory.

Replay scripts preserve the original frozen panel. The original counterfactual generator uses no-clobber case publication; do not rerun it over completed outputs. Use a separate bound directory for any new experiment. Aggregate/plot scripts consume the completed tables without fitting.
