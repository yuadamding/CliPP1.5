# Why CliPP1.5 performs poorly on the finished CN-first4K cases

Investigation date: September 25, 2026. The fixed evaluation snapshot is
**September 25, 2026, 1:12:36 PM CDT**: 193 validated tumors, 41,712 mutations.
These are early-finished tumors with 200–232 mutations, not the whole 4,000-case
cohort. No production source, fit, scheduler state, or scientific setting was
changed by this investigation.

## Conclusion

The clearest demonstrated problem is **insufficient partition/membership
search**. The current fusion path often leaves a partition that can be
improved substantially under **its own unchanged likelihood and score**.
Multiplicity ambiguity and the graph built from single-mutation CCF modes
are strongly supported contributing mechanisms; the graph's causal contribution
has not yet been isolated by an intervention.

There is also a secondary statistical limitation: at low depth the existing
allocation penalty can prefer merging real clusters. This does not justify
changing the score first: most observed excess clusters disappear when the
existing score is actually optimized over additional memberships.

## Scope, definitions, and baseline performance

This is an investigation of the current **CN-first4K** campaign, not a new
evaluation of Regional-CN, SimClone, PhylogicNDT simulations, or the original
CliPPSim4K cohort. The 193 cases represent 4.825% of the planned CN-first
cohort. The campaign runs smaller inputs first, so completion is a selected
population. Later completions were not mixed into this fixed comparison.

ARI measures agreement of mutation memberships with simulation truth. CCF
errors here use final unpenalized, fixed-membership refits, not raw penalized
CCFs. sMF is the fraction of retained mutations outside the occupied cluster
closest to CCF=1. This cluster is labeled 0; fitting remains unconstrained.
CCC is calculated across paired per-tumor sMF values. CNA-only multiplicity
macro-F1 includes all non-1/1 CN states, including balanced amplification,
with all four multiplicity classes and complete call coverage.

The native CliPP1.5 population contains 41,712 mutations. For the comparison
below, every method is evaluated on the same **41,194 mutations in the same
193 tumors**, because original CliPP filters 518 mutations. PyClone-VI uses
binomial likelihood, 20 components, and 1,000 restarts; PhylogicNDT uses
1,000 iterations. Their fitted outputs are already saved and validated.

| Method, matched mutation population | Mean ARI | sMF CCC | Mean CCF MAE | Correct K |
|---|---:|---:|---:|---:|
| CliPP1.5 | 0.4893 | 0.7827 | 0.08770 | 24.35% |
| CliPP | 0.5838 | 0.8265 | 0.06821 | 61.14% |
| PyClone-VI | 0.6493 | 0.9411 | 0.05452 | 62.69% |
| PhylogicNDT | 0.6115 | 0.8334 | 0.05920 | 79.27% |

CliPP1.5's paired ARI wins/ties/losses are 45/26/122 against CliPP and
22/25/146 against PyClone-VI. The native-population ARI of 0.4850 differs
slightly from the matched value 0.4893; these populations must not be mixed.
All native diagnostic comparisons below retain the same 41,712 mutations.

Across native cases, 47 have correct K, 124 are overclustered, and 22 are
underclustered. All 24 true-K-one cases recover a single cluster. Among the
169 multi-cluster tumors, only 23 recover the correct cluster count.

| True K | Cases | Mean ARI | Correct K |
|---|---:|---:|---:|
| 1 | 24 | 1.0000 | 24/24 |
| 2 | 81 | 0.4389 | 12/81 |
| 3 | 46 | 0.4059 | 8/46 |
| 4 | 42 | 0.3663 | 3/42 |

The median recorded CliPP1.5 case wall time is 319.85 seconds. This mixes
A100/H100 execution and compilation/reuse conditions and is not a controlled
cross-method or cross-hardware runtime comparison.

## 1. A direct, truth-free test demonstrates missing better partitions

For every finished case, keep all published refitted CCF centers fixed. Try
moving one mutation from its current cluster to another occupied cluster,
accept the move with the largest strict decrease in the existing score,
and repeat until no such move improves it. Empty clusters disappear. No
true labels, true CCFs, true multiplicities, new centers, or other method's
results participate in these decisions.

The score is exactly the production score:

```text
S = 2*NLL + K*log(N)
    - 1.4*[logGamma(K) - logGamma(N+K)
           + sum_k logGamma(n_k+1) + logGamma(K+1)].
```

For a move from cluster a to b that leaves a occupied, the score change is
`2*(loss_b-loss_a) + 1.4*log(n_a/(n_b+1))`. Removing a singleton also changes
the K-dependent term. The diagnostic explicitly handles that change and
checks every accepted move against a full score recomputation.

| Metric, identical 193 tumors and mutations | Published result | Fixed-center score descent |
|---|---:|---:|
| Mean ARI | 0.4850 | **0.6397** |
| Mean ARI, 169 multi-cluster tumors | 0.4119 | **0.5886** |
| Mean final-CCF MAE | 0.08870 | **0.05930** |
| sMF CCC | 0.7831 | **0.8744** |
| Mean sMF absolute error | 0.13577 | **0.07063** |
| CNA-only multiplicity macro-F1 | 0.6551 | **0.7908** |
| Mean inferred K | 6.539 | **2.472** |
| Correct K | 47/193 | **119/193** |

Scores decrease in **152/193 cases**, including **123/151 complete searches**.
ARI improves in all 152, with zero decreases and 41 ties. All 24 true-K-one
cases remain unchanged. This moves 5,704 mutations between clusters.
The smallest positive score improvement is 5.496, far above floating-point
reconstruction differences. Median improvement across all 193 cases is 77.786.

These are feasible lower-score witnesses, not new production fits or globally
optimal partitions. All destination centers satisfy every mutation's original
bounds. Every original score matches its hash-verified `run.json` to <1e-6;
an independent reconstruction has maximum discrepancy 1.17e-10. The move
formula also received an independent code audit. Truth is loaded only after
the move search, for evaluation.

The result does not imply every metric improves in every tumor: CCF MAE
improves in 146 cases and worsens in six. Underestimated K increases from
22 to 43 cases, while overestimated K falls from 124 to 31. This procedure
cannot introduce a missing center and is not a complete model-selection
algorithm. It also has no raw fusion-solver certificate.

Example: complete-search case `500_2_0.9_0.7_rep74` has truth K=2, centers
0.64568 and 1. Its published result has K=19, ARI=0.1501, CCF MAE=0.1960.
The diagnostic gives K=2, ARI=0.8726, CCF MAE=0.0131, with a score decrease
of 470.228. No truth was used to obtain that improvement.

## 2. Multiplicity aliases distort both pilots and final memberships

The expected variant fraction depends on the product `CCF * multiplicity`:

```text
p = purity * CCF * multiplicity
    / [2*(1-purity) + purity*(major_CN+minor_CN)].
```

For example, CCF=1 with multiplicity=1 and CCF=0.5 with multiplicity=2
have identical binomial read probabilities when both multiplicities are
supported. This ambiguity is real even with accurate CN inputs and deep reads.

This equality concerns the conditional binomial terms for the indicated
multiplicities. The complete likelihood marginalizes every supported
multiplicity with its prior; the total likelihoods at CCF=1 and CCF=0.5
need not be equal because the other multiplicity terms also contribute.

In these results:

- CNA loci constitute **43.25% of mutations but 73.27% of absolute CCF error**.
- True-clonal label-0 recall is **97.61% in diploid loci versus 66.60% in CNA**.
- At major CN=4, clonal recall for true multiplicity 1/2/3/4 is
  **36.79% / 44.30% / 62.69% / 99.26%**.
- Of 3,772 clonal CNA mutations assigned outside label 0, **3,596 (95.3%)**
  have an overestimated final multiplicity. Lower CCF compensates for that
  higher multiplicity in the read-probability model.
- Mean ARI at simulated CNA settings 0.1/0.4/0.7 is
  **0.6905 / 0.5243 / 0.3336**. The pattern persists within depth and true-K
  strata, although these remain observational comparisons.

The graph uses inverse distances between independently chosen pilot CCFs.
Among low-multiplicity CNA loci (`true_m < major_CN`), **10,876/11,321
(96.07%)** pilots lie closer to an alternate multiplicity alias than truth.
Reconstructing the frozen graph rule from the saved full-precision pilots gives:

| Pair type | Mean normalized weight |
|---|---:|
| Same truth cluster, diploid to diploid | 2.5513 |
| Same truth cluster, low-m CNA to diploid | 0.07782 |
| Different truth clusters, low-m CNA to diploid | 0.14811 |

Within 128/169 multi-cluster tumors, the last pair type has greater mean
weight than the second; median within-tumor ratio is 1.679. Thus a complete
graph still has misleading geometry: biologically related mutations can be
weakly connected while alternate-mode neighbors receive stronger coupling.

This reconstruction demonstrates misleading weights; it is not a
device-hash-certified graph replay or a causal graph-ablation experiment.
Changing graph construction requires a separately evaluated scientific variant.

The observed CNA effect is substantial in both membership and CCF accuracy:

| Simulated CNA setting | Cases | Mean ARI | Mean per-case CCF MAE | Mean sMF absolute error |
|---|---:|---:|---:|---:|
| 0.1 | 45 | 0.6905 | 0.05557 | 0.07586 |
| 0.4 | 69 | 0.5243 | 0.08337 | 0.10524 |
| 0.7 | 79 | 0.3336 | 0.11223 | 0.19657 |

At depths 100/200/500, diploid CCF MAE improves from 0.0687 to 0.0359 to
0.0177, while CNA CCF MAE is 0.1519/0.1482/0.1619. Mean inferred K rises
from 4.23 to 6.43 to 9.04. These are descriptive, unpaired strata, but they
show that more read depth alone does not remove the dominant CNA error.

For truly clonal mutations, mean CCF error is 0.02080 in diploid loci and
0.17364 in CNA loci, downward in both groups. After exact-score reassignment,
the clonal CNA mean bias improves to -0.05184. CNA multiplicity overcalls
fall from 4,962 to 1,656, while undercalls increase from 1,186 to 1,942.
Overall CNA multiplicity accuracy improves from 0.6592 to 0.8006; the
tradeoff is not a uniform improvement for every multiplicity class or tumor.

True cluster separation also matters. Among K=2 tumors, mean ARI is 0.1016
for CCF gaps <=0.25 (four cases), 0.3177 for gaps 0.25–0.5 (40 cases), and
0.6063 for gaps >0.5 (37 cases). The smallest-gap stratum is too small for
a broad quantitative conclusion; it supports separation as a further source
of difficulty rather than an explanation of the whole deficit.

## 3. The implementation explains why these memberships survive

The frozen production implementation scores only one raw-objective winner
per lambda after fixed-membership refitting. Other starts can have useful
partitions without winning the raw penalized objective. There are no
post-path mutation-reassignment or merge/split candidates.

The bounded path normally has zero plus 25 logarithmically spaced positive
penalties. A `complete` status means the planned numerical search resolved,
not that all partitions or penalties were searched. Raw stationarity is a
claim about a different objective from the final partition score.

All 193 saved paths include a qualified K=1 candidate; none selects the
upper boundary or reports a truncated path. Simply adding larger penalties
is therefore not supported as the main remedy.

Source references in the frozen
`results/cnfirst-pool14-20260925-v2/a100/payload/source/src/clipp1d/` tree:

- `cuda/solver.py:379–407`: raw-start competition before partition scoring.
- `cuda/selection.py:67–166`: bounded path and winner selection.
- `cuda/graph.py:117–147`: fixed pilot-distance weights.
- `cuda/scalar.py:300–355`: pilot mode/tie selection and limited alternative mode.
- `cuda/partition.py:197–207`: score definition.

Independent same-likelihood diagnostics further separate the causes:

| Diagnostic | Cases with lower score | Mean ARI |
|---|---:|---:|
| Refit existing memberships again | 0 | 0.4850 |
| Greedy merges, refitting each proposed union | 134 | 0.5164 |
| Fixed-center score-decreasing mutation moves | 152 | 0.6397 |
| PyClone-VI memberships, refit with CliPP1.5 likelihood | 147 | 0.6409 |

The last row imports another method's partition and is evidence about
candidate coverage, not a standalone CliPP1.5 method. Scalar diagnostic
refits use dense grids plus bounded local minimization, with feasible
published centers included; they are not production scalar certificates.
The fixed-center witness does not depend on this approximate optimization.
The smaller merge-only improvement is consistent with mixed memberships that
whole-cluster merges cannot repair; greediness and score preferences can also
limit the merge diagnostic.

Two further diagnostic controls are informative:

- At the known true cluster centers, assigning mutations by the same
  marginalized likelihood gives mean ARI 0.6387 with equal cluster
  probabilities, or 0.7092 using known true cluster proportions. These use
  oracle information and are neither deployable results nor ARI upper bounds.
  They nevertheless show that read ambiguity alone does not explain the
  observed gap. Under the true-proportion rule, classification error is
  5.16% in diploid loci versus 16.36% in CNA loci.
- Fixing the true memberships and diagnostically refitting centers gives
  CCF MAE 0.00574. This shows that accurate centers are attainable under the
  observation model when memberships are supplied. Its ARI=1 is tautological
  and must not be presented as recovered accuracy.

An exploratory one-pass reassignment gives ARI 0.5084 using likelihood alone,
0.6100 using fixed original cluster-size probabilities, or 0.6116 with
size+1 smoothing. These heuristics are distinct from the stronger exact-score
move experiment: they do not guarantee a lower production score. Their
results are retained as controls, not combined with the exact-score results
or selected using truth.

## 4. The score also has a secondary low-depth limitation

For oracle true memberships, approximate refitting lowers the score in
136/169 multi-cluster cases but raises it in 33. Of those 33, 27 have depth
100, six depth 200, and none depth 500. A denser 4,097-point sensitivity
check of 68 non-improving truth/PyClone cases changes no comparison outcome.
This is evidence of a score/statistical limitation, not proof of a
globally certified oracle optimum.

Example `100_2_0.4_0.7_rep52`, N=204, depth=100, truth centers 0.74705 and 1:

| Score component | Selected K=1 | Refitted true K=2 membership |
|---|---:|---:|
| Twice negative log likelihood | 19815.859 | 19727.051 |
| K log N | 5.318 | 10.636 |
| Allocation penalty | 0 | 175.587 |
| Total | **19821.177** | 19913.274 |

Here the allocation penalty outweighs the evidence for the true split.
Conversely, small outlier clusters incur much less allocation cost than
balanced new clusters. At N=220, splitting `[220]` into `[110,110]` requires
NLL improvement 110.69, while `[110,110]` to `[110,109,1]` requires 8.51.
This explains why the score can permit small fragments while discouraging
weak biological splits. It does not make penalty retuning the first repair.

## 5. Alternative explanations checked

- **Input/conversion bug:** zero count, coordinate, CN, normal-CN, purity,
  truth, multiplicity-support, or truth-feasibility mismatches across all
  193 tumors. Input/truth hashes match staging receipts. PyClone-VI and
  PhylogicNDT receive the same scientific inputs with their documented
  allele-column conventions.
- **Wrong simulation likelihood:** CN-first draws multiplicity uniformly
  from 1..major CN, including balanced amplification. CliPP1.5 marginalizes
  the same support/prior and uses the matching purity/CN binomial scaling.
  Binomial standardized residual mean is -0.00952, mean square 1.0070.
- **Filtering mismatch:** matching original CliPP's retained population
  changes CliPP1.5 ARI only from 0.4850 to 0.4893.
- **Duplicate labels/centers:** no exact duplicate centers and no adjacent
  refit-center gaps below the 2e-5 fusion tolerance among 1,262 clusters.
- **Tiny clusters alone:** 721 clusters of size <=5 contain only 3.31% of
  mutations. Dropping them changes the evaluated population and raises ARI
  only to 0.4968; merging centers within 0.05 barely changes ARI.
- **Refit arithmetic:** all scores reconcile and alternate scalar refits
  of the same memberships yield no meaningful improvement (>1e-4).
- **Incomplete search alone:** 151 complete cases still overcluster heavily;
  123 admit the fixed-center lower-score witness.
- **Reporting raw instead of refitted CCF:** mean raw-CCF MAE is worse,
  0.1276 versus final-refit 0.0887.
- **Hardware:** no positive evidence of an A100/H100 accuracy difference in
  descriptive adjusted comparisons. This is not a paired determinism test.
  The one exported infrastructure failure is excluded from accuracy.

Complete-search cases have mean ARI 0.5146 (151 tumors), versus 0.3787 for
incomplete searches (42 tumors). Complete cases still have mean K=6.95 and
101/151 are overclustered. Incompleteness is a real separate limitation, but
cannot explain the main result.

The hardware comparison contains 100 A100 and 93 H100 cases, with raw mean
ARIs 0.4667 and 0.5046. A descriptive regression adjusting for categorical
K/depth/purity/CNA/search status estimates an H100-minus-A100 difference of
0.0223, with a 95% interval [-0.0321, 0.0767]. Different case assignment and
the absence of paired execution prevent a hardware-equivalence conclusion.
The excluded failed case, `500_2_0.6_0.7_rep62`, failed with a stale-file-handle
error in the shared Triton compilation cache. That failure affects coverage
and operations, not the measured accuracy of the 193 validated outputs.

## Recommended implementation order

September 25 implementation addendum: the
[optional partition estimator](docs/PARTITION_SEARCH.md) now implements
explicit-label refits, exact-score reassignment, all-start candidate scoring,
and separate provenance/publication. The
[new validation record](validation/partition-search-20260925/README.md) reports
matched fixed and alternating CPU replays and the pending CUDA/held-out gates.
The findings below retain the original discovery scope; they do not qualify
the new implementation.

1. Add truth-free mutation-reassignment and merge/split partition proposals
   after the path, preserving all existing candidates. Use the unchanged
   likelihood, original bounds, and existing score; run the existing qualified
   CUDA scalar refit on proposed memberships before final selection.
2. Score distinct partitions from all qualified starts, not only the
   raw-objective winner at each lambda. Retain separate direct-partition
   provenance and raw-reference certification. A new winner must not inherit
   another candidate's raw KKT/stationarity certificate or silently redefine
   the primary raw estimator.
3. Evaluate multiplicity-aware pilot/graph alternatives as explicit variants.
   Keep the complete graph and unconstrained fitting; do not force CCF=1 or
   use simulation multiplicity truth.
4. Only then investigate score calibration for low-depth, nearby clusters.
   Use a held-out case panel and report under/overclustering, ARI, CCF, sMF,
   multiplicity, search status, and runtime together.

The 193 cases used to develop these diagnostics are a discovery panel.
Their improvements are not independent validation, full-cohort performance,
or evidence that the new proposals have production CUDA qualification.

For production integration, first verify proposal-score arithmetic against
the independent reference, deterministic mutation/cluster-ID handling, and
feasibility of every reassignment. Keep all original candidates, and require
the final selected score to be no worse than the original winner. Then run
qualified CUDA scalar refits and validate output semantics for any direct
partition winner. Reusing an independently certified raw reference must be
explicit, not an inherited certificate for the new memberships.

Evaluate changes on a predeclared held-out set spanning true K, depth,
purity, CNA burden, and larger mutation counts. Report paired per-case gains
and regressions, under/overclustering, true-K-one false splits, CCF error,
sMF CCC/MAE, CNA-only multiplicity F1, unresolved-search rates, and added
wall time. A lower score is an engineering invariant for these proposals;
better biological accuracy still needs independent evidence. The fixed-center
diagnostic's inability to create missing centers and its increased
underclustering make split proposals and this validation particularly relevant.

## Evidence and reproduction

The run's base commit is `a8a3ef7aaaeda8dffa8ac5fccef1cc4b77aa71f5`; its
dirty-source numerical fingerprint is
`62ec9d1480563cda8ce5c192aff94c6763a9978a2fd73ea53b85c1975ed21e09`.
The frozen payload, not the current dirty worktree, is the scientific authority.

Selected summaries, per-case tables, source copies, and the diagnostic PDF are
preserved in the [tracked evidence bundle](validation/cnfirst-accuracy-20260925/README.md).
The complete local artifact set is under
[`../CNfirst4K_CliPP15_investigation_20260925/`](../CNfirst4K_CliPP15_investigation_20260925/README.md):

- `BOUND_DIAGNOSTICS.json`: hash-verified full run receipts and multiplicity
  outputs for the fixed 193 cases; collection was read-only.
- `reassignment_diagnostic.py`, `reassignment/`: exact same-score move
  diagnostic, every assignment, case metrics and summary.
- `analyze_bound_diagnostics.py`, `bound_analysis/`: public multiplicity-call
  reconstruction, diagnostic multiplicity calls and path coverage.
- `input_audit/`: independent input/generator/graph audit with source hashes.
- `scoring_diagnostic/`: refit, merge, external/oracle partition diagnostics,
  score decompositions and grid sensitivity evidence.
- `error_patterns/`: case/mutation/cluster tables and diagnostic PDF figures.

Scripts use `/home/yding1995/miniforge3/envs/ml1/bin/python`, with BLAS thread
counts set to one. The scalar diagnostic uses at most four analysis processes.
These are local evidence calculations, not resumed local CPU production fits.
