# Three-cohort accuracy requirement

User requirement recorded September 27, 2026: **do not sacrifice CN-first4K performance to improve SimClone1000 or PhylogicNDT500; target the best performance across all three.** This requirement governs future adoption decisions. It is an objective to demonstrate, not a result already achieved or a guarantee possible from a finite benchmark.

## Required comparison

Evaluate one source version and one scientific configuration on:

1. `CliPPSim4K_CNfirst_20260924`: 4,000 eligible tumors.
2. `SimClone1000_TSV`: 756 eligible tumors.
3. `PhylogicNDT500_TSV`: 500 eligible tumors.

Preserve the current published CliPP1.5 partition estimator as the reference. Keep fitted CCF unconstrained and label the occupied group closest to one as clonal. Use the same input eligibility and evaluation rules, and distinguish full retained populations from the common cross-method mutation subset.

Scientific settings must be shared across cohorts. Do not select an algorithm, multiplicity rule or penalty from the cohort name, case ID, true labels, true CCF or whichever method scored best for that tumor. Do not choose per-case seeds after inspecting outcomes. A frozen, truth-free ID hash used solely for reproducible randomness is allowed. A common data-dependent rule can use observed counts, purity and CN only if its specification is frozen before confirmation. Runtime/device settings and input paths can differ without defining different scientific estimators.

## No compensation across cohorts or metrics

Each cohort independently must show no observed degradation relative to the current baseline in mean ARI, mean ARI among true multi-cluster tumors, tumor-averaged CCF MAE, sMF CCC, sMF MAE, true-single-cluster false splits and false K=1 collapses. Require complete paired case/mutation coverage and validated results. Preserve the full retained mutation population as an additional baseline comparison; a narrower intersection with an external method must not conceal a regression on excluded mutations.

The target against available other methods is best or tied on the headline accuracy metrics, reported separately by cohort and endpoint. A pooled average, composite score, lower fitting objective or a favorable subset cannot substitute for these checks. Winning ARI does not imply winning sMF CCC or CCF error. Report cluster-count accuracy, stratified results, error tails, per-case wins/losses and uncertainty as well; aggregate non-regression does not mean every individual tumor improves.

For a candidate that changes multiplicity modeling, also require a separately bound non-regression comparison of CNA-only exact-class multiplicity macro-F1 and call coverage, with micro/per-class metrics reported. CNA includes balanced amplification; only CN 1/1 is diploid. Define the truth target on mixed-CN loci explicitly. Existing headline tables do not contain those calls, so the general metric gate cannot supply this model-change evidence by itself. Error tails are mandatory diagnostics here, not an unmeasured claim of tail protection; any additional tail acceptance threshold must be fixed before candidate evaluation.

No positive scientific regression allowance is authorized. The evaluator uses only a stated floating-point comparison epsilon. Uncertainty must be reported separately: a nonsignificant difference is not proof of equivalence, and an observed pass does not establish population-wide or statistical superiority.

Do not discard failed, slow or missing candidate cases to create a favorable intersection. Keep the evaluation inventory fixed before candidate execution. An incomplete search can still have a qualified published result; retain that status explicitly. Missing validated outputs block the coverage check.

## Baseline and current evidence

The current bound numerical source is `da651879c2a33f481957ffae1d1c865d11ad39c8`, fingerprint `55d6e45949e4dee2e8798ac81fb49510baffca9959104355db87d45acca1a523`. Local baseline evidence is frozen under `results/joint-cohort-guard-20260927-v1/baseline/`; its preparation inventory binds the original artifacts rather than replacing them.

| Saved comparison | Cases | Matched mutations | Available comparators |
| --- | ---: | ---: | --- |
| CN-first, September 27, 2026, 2:10 PM CDT | 2,661 | 1,056,696 | CliPP, PyClone-VI, PhylogicNDT |
| SimClone, September 27, 2026, 4:33 PM CDT | 385 | 252,370 | PyClone-VI |
| Phylogic, September 27, 2026, 4:33 PM CDT | 51 | 98,503 | PyClone-VI |

These are previously inspected development snapshots. They are not full-cohort or held-out acceptance evidence. The CN-first native CliPP population has 1,066,951 mutations in the same 2,661 cases; the four-method intersection excludes 10,255 mutations removed by original CliPP.

The current CN-first estimator leads its matched panel on mean ARI (0.69063) and CCF MAE (0.05078), but PyClone has higher sMF CCC (0.94964 versus 0.94451). Therefore even protecting current CN-first performance does not yet establish the user's best-across-metrics target. The [two-cohort diagnosis](../RESEARCH_SIMCLONE_PHYLOGIC_PYCLONE_GAP.md) identifies the separate score and likelihood failures to address in the other cohorts.

The canonical gate is [evaluate_joint_cohort_gate.py](../benchmarks/evaluate_joint_cohort_gate.py). Its bound manifest declares populations, expected counts, method/config identities and metric-table hashes. Candidate evidence is initially absent, so the baseline package cannot pass as a new algorithm. A baseline-versus-itself check is an engineering control, never an accuracy improvement or release qualification.

From the repository in `ml1`, inspect the baseline-only package with:

```bash
python benchmarks/evaluate_joint_cohort_gate.py \
  results/joint-cohort-guard-20260927-v1/baseline/manifest.json
```

It intentionally exits nonzero with `gate_passed=false`: candidate evidence is missing and the saved panels are partial. To evaluate a future candidate, create a new manifest binding its separately validated metric tables and the preserved baseline authorities. Never edit the baseline manifest into a candidate manifest. `--output NEW.json` publishes a new report without overwriting an existing one. Even a full observed gate pass leaves `release_ready=false`; source-bound numerical qualification and independent confirmation are separate evidence.

## Research direction that respects CN-first

CN-first samples multiplicity uniformly from `1..major_CN`, including subclonal and balanced-amplified loci. The audited older cohorts instead have multiplicity one at every true-subclonal mutation. Forcing multiplicity one globally could improve those older simulations by violating the CN-first mechanism. Likewise, lowering the allocation penalty globally could recover weak splits while creating false splits. Neither is an acceptable untested production repair.

A shared experimental model can **nest the existing uniform multiplicity model**, while allowing the observations to support a multiplicity-one enrichment within a latent group. One candidate family is:

```text
q(m | major_CN=A, group=c) = (1 - w_c)/A + w_c * indicator(m=1)
0 <= w_c <= 1
```

The uniform branch is `w_c=0`. Hierarchical shrinkage toward that branch and a declared model-selection cost are needed; unconstrained maximization can overfit. All groups must use the same rule, including the group eventually labeled zero. That public label must not determine the prior. A separately identified [experimental implementation](SOFT_MIXTURE_EXPERIMENT.md) now exists; it remains unqualified for production adoption.

Single-region observations often identify CCF times multiplicity more strongly than either separately. Even an adaptive family may be unable to distinguish a low-CCF/high-multiplicity explanation from a high-CCF/multiplicity-one explanation. Retaining the old model as an option protects candidate availability, not truth accuracy: a selector can still choose the wrong branch. The mandatory CN-first benchmark is what tests the tradeoff.

For example, with purity 0.8, normal CN 2 and tumor total CN 4, both `(CCF=0.4, multiplicity=2)` and `(CCF=0.8, multiplicity=1)` imply VAF `0.8*0.8/3.6`. At a locus supporting both multiplicities, the same read-count distribution cannot identify which pair generated it. Other loci and shared structure may resolve the ambiguity; a method cannot guarantee the correct answer in both situations from this locus alone. The requirement is therefore empirical superiority on the declared cohorts with preserved CN-first accuracy, not an impossible per-case mathematical guarantee.

Keep the complete-graph proposal framework, independent qualification, feasible CCF bounds and candidate provenance. Test three changes separately before combining them:

- **Search coverage:** general births and multiple strong seeds under the current model and score. Measure recovery of demonstrably missed lower-score partitions.
- **Partition criterion:** evaluate whether a common soft-mixture or predictive criterion can retain weakly separated populations without the current hard-allocation penalty's merging preference. Control search inventory to isolate selection from exploration.
- **Multiplicity model:** evaluate a nested, regularized family first with fixed memberships on development data, then with truth-free membership search. Test uniform and enriched generators, false-split controls and ambiguous cases. Correct memberships used in diagnostics must never enter production selection.

For any changed prior, denominator or criterion, give the model and experiment a new identity. Do not compare unlike raw objectives as if they were the same score, reuse a raw certificate across models, or silently inherit old GPU qualification. The current proposal-only prior experiment restores the original final likelihood; it cannot by itself change the demonstrated wrong-CCF likelihood preferences.

## Evidence needed before adopting a revision

Freeze candidate source, scientific settings, input inventories and evaluation rules before measuring its confirmation results. Treat all already inspected outcomes as development information. Group repeated SimClone truth structures when defining independent confirmation sets or uncertainty calculations; closely related tumors must not leak across tuning and confirmation.

Require both the complete declared three-cohort benchmark and separate confirmation evidence spanning the relevant generation mechanisms. Repeatedly choosing settings from the same benchmark is not independent validation. Preserve previous failed candidates and report all three cohorts, including results that contradict the proposed mechanism.

Some prepared full-cohort cases have mutations without unambiguous truth. Bind and report those exclusions explicitly, use the same scorable IDs for every paired result, and retain their counts in coverage reports. Do not invent labels or drop whole difficult tumors to imply complete truth coverage. The current 436-case SimClone/Phylogic snapshot has no such retained-mutation exclusions.

Allocated-CUDA numerical qualification, paired full-fit checks, complete output validation and acceptable execution costs remain separate from accuracy. CPU arithmetic or an evaluator unit test cannot satisfy those requirements. The existing separately authorized prior-perturbation study keeps its frozen source, datasets and historical gates. Its own `adoption_supported` field describes that study's older scope and is **insufficient for adoption under this new three-cohort requirement**. Do not modify its active payload or represent its CN-first-style controlled panel as confirmation on SimClone/Phylogic.

This requirement authorizes no change to active campaign settings, GPU ownership or scheduler caps. Research preparation and the joint evaluator leave current results and workers intact. No revised production estimator has passed this requirement yet.
