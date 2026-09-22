Production update: the validated best-seed design is now integrated in 0.3.0.
See README.md for the v4 candidate-provenance contract. The results below
remain the original, unmodified offline research evidence.

# CliPP2 mechanisms that fit the CliPP1.5 framework

**Recommendation: add a bounded secondary pool of contiguous partitions after
the existing fusion path, using adjacent weighted Ward proposals and boundary
refinement of the best candidate.** Preserve the current path, fixed pilot
order, adaptive edges, likelihood, occupied exact-one constraint, qualified
scalar refit and partition score.

This is an implemented **offline research prototype**, not a production release.
Production remains CliPP1.5 `efdb9ca`, version 0.2.2. The experiment and source
audits are under [the study directory](results/clipp2-transfer-study-20260921-v1/).
The source fingerprint is
`521e01efbd080b3ef7660661779d507afc16f4f35206800332aeefaa3a2edc16`.
CliPP2 baselines remain separately bound to `77525a6` or `92492b6`.

## What transfers

CliPP2's useful mechanism is its direct-partition proposal/refit pool. The three
better baseline partitions on the original eight cases were selected by
`final_phi_hessian_ward_cem`. All three are legal contiguous partitions of the
existing CliPP1.5 chain, but its saved fusion path never proposed them.

| CliPP2 mechanism | Adaptation for CliPP1.5 |
| --- | --- |
| Ward proposals from pilot and qualified raw coordinates | Merge only adjacent intervals of the original chain; use a heap and interval moments. |
| Likelihood-based CEM reassignment and refitting | Move boundaries between neighboring occupied blocks, then refit and compare the full original score. |
| Union of raw and direct candidates | Retain the original path winner; accept a direct candidate only when its qualified score improves. |
| Separate raw-reference and direct-partition provenance | Record the actual proposal source and parent; never attach a raw KKT certificate or raw lambda to a direct winner. |

Unrestricted all-pairs Ward, nonadjacent CEM membership, and sorting again by
new CCF estimates would change the framework. Neither a denser lambda grid nor
a faster raw solver supplies the missing general candidate route. The earlier
sim1y9z5f analysis specifically ruled out the target split on the exact optimal
two-block fusion path under that model.

## Implemented prototype

The executable candidate logic is
[chain_prototype_best_seed.py](results/clipp2-transfer-study-20260921-v1/chain_prototype_best_seed.py),
with [adjacent Ward](results/clipp2-transfer-study-20260921-v1/adjacent_ward.py).
It uses only NumPy/SciPy and existing CliPP1.5 scalar/model/score functions.

1. Start with the unchanged, qualified production partition.
2. Build two adjacent Ward hierarchies in the fixed chain: one from the
   qualified marginal pilot and one from the selected qualified raw vector.
   Positive expected-curvature weights define proposal geometry; they are
   not a claim that the mixture likelihood is convex.
3. Propose K=1..min(12,N), deduplicate cuts, and perform the original constrained
   likelihood refit. Existing raw partitions with K>12 remain eligible.
4. Select the best candidate and refine only that seed for at most four
   deterministic boundary sweeps. At each boundary, score all feasible positions
   with centers fixed, and propose the best position plus immediate neighbors
   of the current cut. Each proposal receives a qualified refit that profiles
   the clonal block across all occupied blocks.
5. Accept only a score decrease exceeding twice the two refit gaps plus 1e-8.
   Keep the incumbent when numerical bounds overlap. Cap distinct partition
   attempts at 256 and cached scalar intervals at 512. Cache scalar records,
   not copied per-interval model matrices.

The graph and numerical state remain linear in N. Adjacent Ward uses O(N log N)
heap work and O(N) state. A fixed-cut boundary scoring pass uses linear row
work; sequentially moved boundaries can revisit rows, giving O(KN) worst-case
row work per sweep. Qualified mixture refits remain an additional cost and
must be measured separately. No all-pairs distances or all-interval score
matrix is constructed.

## Results and development boundary

The study evaluated **44 distinct SimClone tumors**: the original eight with
existing source-matched fits, followed by 36 fresh source-bound production fits.
All 36 new fits succeeded. Original eight, first 18, and final 18 are disjoint.
The final 18 contain 1,965 retained mutations. Both methods use identical
retained IDs, and every copied input, truth source and baseline output is bound
to its hash and original source.

The first prototype refined four seeds. It repaired the three original misses,
but introduced three false splits among ten true-one-cluster cases in the first
18-case panel. CliPP2 also makes those three false splits. All original results
are preserved in `holdout-results/`; they are not replaced by the revised run.

After seeing those results, the prototype changed **one policy value**:
`refinement_seeds=4` to `1`. Consequently the first 26 cases are development
data for this revised variant. A fresh 18-case confirmation set was selected
from the remaining 53 completed cases using only input-size/CN strata and a
fixed hash order. Five deterministic fill cases were needed because several
strata had been depleted. The revised source was frozen before evaluating this
confirmation set. It is a selected small-input panel, not full-cohort accuracy.

Both prototype variants recover the three original better partitions:

| Tumor | Production ARI | Prototype ARI | Result |
| --- | ---: | ---: | --- |
| sim1y9z5f | 0.0000 | 0.2737 | Adds the missing two-block split, then moves its boundary to 15. |
| sim9kisdo | 0.5688 | 0.7014 | Moves the boundary from 20 to 19. |
| sim5982qm | 0.5021 | 0.5141 | Moves the boundary from 33 to 31. |

The other five original cases retain their partitions, including simnwk1j1,
where the existing chain solution already beats CliPP2.

On the **fresh final 18-case confirmation panel**:

| Metric | Current CliPP1.5 | Best-seed prototype | CliPP2 baseline |
| --- | ---: | ---: | ---: |
| Mean ARI, eight true-K>1 cases | 0.4262 | **0.5405** | 0.5630 |
| Mean ARI, all 18 including degenerate K=1 cases | 0.7450 | 0.7958 | 0.7502 |
| False splits among ten true-K=1 cases | 0 | 0 | 1 |
| Mean tumor CCF MAE | 0.02636 | **0.02049** | 0.02172 |
| Mean absolute sMF error | 0.04692 | **0.02476** | 0.02715 |
| Pooled CNA-only exact-class macro-F1 | 0.98025 | 0.98476 | 0.99055 |
| Pooled CNA-only micro-F1 | 0.96524 | 0.97342 | 0.98364 |
| Pooled CNA-only weighted-F1 | 0.96546 | 0.97348 | 0.98361 |

The prototype lowered the score in six cases. ARI improved in four, worsened
in two, and stayed unchanged in twelve. Thus an improved score still does not
guarantee improved truth agreement for an individual tumor. The observed ARI
losses were simx0plrw (0.6193 to 0.5915) and simdh8zf5 (0.6959 to 0.6526).
CNA evaluation has full coverage on 489 rows, with truth multiplicities
1/2/3/4 supported by 319/159/10/1 rows. The two rare classes do not have broad
qualification. CCC conventions and exact-one versus designated-cluster sMF
are explicit in the machine-readable results.

Incremental proposal/refit time in the final panel had median **0.635 seconds**
and maximum **1.057 seconds**. These are separately measured additions using
saved production fits, not measured integrated full-fit runtimes.

## A benchmark-input defect must be separated from inference quality

The three initial-panel false-split cases revealed a normal-CN mismatch:
the original SimClone artifacts use normal copy number **2** on X/Y, whereas
the canonical converter writes **1** for male X/Y. Original serialized mutation
scaling values directly support the generating convention. This is specific
to this simulator/input conversion, not a general rule for biological sex
chromosome copy number.

The falsely split cases have only singleton multiplicity support; every
multiplicity call remains one. Independent binomial calculations reproduce
their likelihood and score changes, excluding a mixture-refit or score-code
explanation. Sex-chromosome rows provide a substantial part of the apparent
subclonal likelihood gain.

An offline counterfactual changed only X/Y `normal_cn` to 2 in memory, keeping
counts, purity, tumor CN, truth, original chain and the two tested partitions
fixed. It refitted all-one and the prototype's two-block partition:

| Tumor | K=2 score advantage with supplied input | Advantage after virtual normal-CN correction |
| --- | ---: | ---: |
| sim39w3sj | +2.530766 | **−15.187440** |
| simqm2w9u | +9.328434 | **−28.082222** |
| sim0zh5xa | +0.967536 | **−40.001014** |

Negative means the all-one partition wins. These six qualified fixed-partition
refits quantify the input confound for the tested comparisons. They are not
full corrected-input fits or an all-partition optimum. No canonical input or
historical output was changed. The confirmation panel also used the original
inputs, so its accuracy numbers remain conditional on that conversion.

Repair the simulator-specific conversion and requalify matching inputs before
using these results to calibrate score penalties or claim broad accuracy gains.
Do not compensate for this data defect by adding a minimum cluster size,
clonal attraction penalty, sex-specific estimator heuristic, or weaker numerical
gate to CliPP1.5.

## Numerical and scaling evidence

- Adjacent Ward matched an independent high-precision adjacent-merge oracle on
  388 small hierarchies. Nine invalid-input checks and a 20,000-point work/storage
  check passed.
- Both prototype policies passed independent checks: 46 cached refits were
  bit-identical to production centers/loss/gap/score, three infeasible partitions
  agreed, and 94 boundary scans matched brute-force likelihood/allocation
  ranking. Budget exhaustion and injected failures remained visible.
- Best-seed synthetic overhead was 0.008 s (easy N=100), 2.044 s (mixed N=100),
  0.023 s (easy N=1,000), and **4.379 s (mixed N=1,000)**. The latter contains
  333 mixed-CN mutations and 666 multiplicity-mixture mutations; it used 40
  partition attempts and 77 cached intervals. Independent final-partition
  production refits agreed. These are incremental measurements on one shared
  CPU core, not repeated full-fit speed distributions.
- The mixed synthetic cases each reject two clonal-infeasible Ward partitions.
  The research prototype conservatively records these in its failure list and
  sets proposal `search_complete=False`. They are proven infeasibility, not
  scalar nonconvergence. The original mixture-N=1,000 raw search remains
  incomplete; additional partition candidates do not erase that evidence.

## Concrete production integration

1. Add a private `partition_proposals.py` with adjacent Ward, immutable-model
   interval caching, deterministic boundary proposals and work counters. Reuse
   the production scalar refit implementation through a shared interval-provider
   interface; avoid keeping a second permanent copy of its arithmetic.
2. Extend `selection.py` after its current path. Preserve its winner and raw
   reference; evaluate the bounded direct pool and refine one winning seed.
   Treat the source cap and sweep/partition/cache limits as one versioned
   default policy, with no additional public mode switch.
3. Replace the implicit `(lambda, raw, refit)` association with explicit selected
   partition and raw-reference records. In `types.py`, `api.py` and `report.py`,
   make a direct winner's selected raw lambda/certificate absent. Keep any
   raw-seed parent lambda/Phi hash separate from its qualified refit evidence.
   Existing raw output columns need explicit reference semantics; silently
   assigning the old selected raw state to a new direct partition is incorrect.
4. Record chain/seed/cut hashes, candidate origin, parent identity, attained
   score, refit gap, rejection reason and work counts. Distinguish accepted,
   score-rejected, clonal-infeasible, numerically unresolved and budget-skipped
   proposals. Keep original raw-path completeness separate. Bump policy and
   receipt schema for the scientific selection change.
5. Harden the research helper's input validation to reject noninteger cuts
   before casting, and count scalar calls that raise before returning. Integrate
   adversarial tests for unequal boxes, mixed CN, ties, nonmonotone raw seeds,
   multiple exact-one blocks, budget exhaustion and failed proposals.
6. Qualify the integrated full fit on source-bound corrected SimClone inputs
   plus unchanged synthetic/control inputs, checking outputs, numerical status,
   score, truth metrics, total runtime and memory. The offline results establish
   feasibility and useful candidate coverage; they do not qualify an unbuilt
   production API/schema overlay.

Detailed evidence: [final-summary.json](results/clipp2-transfer-study-20260921-v1/final-summary.json),
[independent final audit](results/clipp2-transfer-study-20260921-v1/final-audit.json),
[source audit](results/clipp2-transfer-study-20260921-v1/clipp2-audit.md),
[design assessment](results/clipp2-transfer-study-20260921-v1/chain-design.md),
[normal-CN audit](results/clipp2-transfer-study-20260921-v1/normal-cn-source-audit.md),
[counterfactual](results/clipp2-transfer-study-20260921-v1/normal-cn-counterfactual.json).
