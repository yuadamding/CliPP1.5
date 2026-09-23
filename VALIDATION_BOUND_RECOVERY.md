# Exact failed-QP capture and bound-aware recovery

This study starts from `430db26cf07466e88e53c6e1a8fbe2be7b7b25e9`.
It preserves the likelihood, complete graph and weights, bounds, score, planned
starts, iteration budgets, original gap/KKT gates and unconstrained fitting.
The cluster closest to CCF 1 remains label 0. No cohort reruns are included.

The current candidate is
`a3aa1dc7719b743e0e9342291aa25dd21f7860632ade008a67768d8d5c40cf30`.
It is **not fully qualified**: 15/16 saved QPs, both 64-node full paths, the
mixed256 and mixed512 full paths, and the existing CUDA regression qualifier
pass. The below256 full path still has one unresolved QP, and below512 was not
run. Earlier successful checks below retain their own source identities and do
not qualify this candidate.

## Captured failures

The exact baseline source reproduced all 13 previously unresolved surrogate QPs
on allocated L40 GPUs. Each capture contains the literal problem, original
initialization, terminal ADMM state, returned state and last equality/refinement
states. The original compiled certificate was reproduced bit-for-bit from the
saved tensors inside the capture process. Successful diagnostic capture is not
successful inference: both original searches remain incomplete.

| Input | Failed starts captured | LSF job | Diagnostic result |
| --- | ---: | ---: | --- |
| `below_one`, 64 nodes | 2 | 77334933 | Original failures reproduced |
| `mixed_support`, 256 nodes | 11 | 77334934 | Original failures reproduced |

The source fingerprint is
`726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88`.
Input hashes match the original study; no pilots or graph were regenerated for
saved-QP replay.

## What the tensors establish

All 13 terminal gaps are dominated by **edge complementarity**, with zero
box-normal contribution. The largest curvature is about `9.27e8`. Equality
proposals can remove the edge gap while creating an infeasible fused group.

For below64, the terminal gaps are `0.1651274927` and `0.0798445089`, while
normalized KKT passes. Two edges account for 97.6% and 96.3% of those gaps.
The first saved refined proposal passes the gap gate but misses KKT by 2.10×.
In the second, the fused interior pair `{m000000, m000038}` requires internal
divergence `220.420699` across capacity `79.974233`. This proves that particular
fixed-primal candidate cannot be certified by any capacity-feasible dual.

For mixed256, edge complementarity accounts for at least
99.999999999708% of each terminal gap. Five of the eleven terminal states pass
KKT; six fail it. Three have a retained objective-admitted refinement, and all
three contain an interior group with a proven capacity-cut obstruction. The
other eight have no retained refined state. Every final tighter proposal is
correctly rejected for increasing the original objective.

These captures do **not** attribute the failures to the review's uniform
allocation among multiple bound normals. No captured admitted proposal has
that allocation freedom. The three-node counterexample demonstrates a separate
real limitation, addressed by projecting onto the permitted residual cone.
The original full-QP certificate remains the admission rule.

See [implementation and diagnostic contract](docs/BOUND_RECOVERY.md).

## Recovery on the exact saved QPs

The cone-aware repair alone recovered **0/13** original-start QPs. The additional
scale-consistent residual balance recovered **13/13**, with independent eager
and compiled original-QP certificates and unchanged initializations and budgets.

| Source | Below64 recovered | Mixed256 recovered | ADMM iterations per recovered QP |
| --- | ---: | ---: | --- |
| Cone repair alone | 0/2 | 0/11 | All exhausted 20,000 |
| Cone repair plus scale-consistent balance | 2/2 | 11/11 | Below: 1,024; mixed: 1,024–11,040 |

The cone-only source fingerprint is
`3d374f47e25067352250a2f29da35c83a99bbca68fa9aca84e64a8de828d762d`;
the combined repair is
`276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5`.
The source difference between these two experiments is the residual-balance
normalization. LSF jobs 77334993/77334994 ran the cone-only replays;
77334997/77334998 ran the combined repair. The three-node counterexample also
passed eager and compiled CUDA checks, including preservation of its valid dual.

The former balancing rule compared a primal residual in CCF units with a dual
residual in objective-gradient units. Multiplying an equivalent QP objective
by a constant could change the controller's decisions. That intermediate repair compared
`primal_norm` with `(rho / base_rho) * norm(adjoint(z - previous_z))`, where
`base_rho` is the unchanged initial median curvature. Factor-10 decisions,
doubling/halving, update frequency, rho limits and physical-dual rescaling remain
unchanged. This establishes controller scale covariance, not bitwise covariance
of every numerical trajectory or acceptance test.

Captured terminal `rho/base_rho` values range from `1/65536` to `1/16`.
The captures do not contain `previous_z`, so these measurements do not reconstruct
the exact historical balancing decisions. The matched original-start experiments
establish that the normalization resolves these thirteen QPs. They do not prove
that bound-aware repair is unnecessary on other inputs.

## Intermediate complete-path qualification

The combined source `276da27b...` recovers all thirteen original saved QPs,
but it has **not qualified all complete heterogeneous paths**. The imported
L40 results are:

| Input | Qualified starts | Complete penalties | Qualified winners/refits | LSF job | Full-path result |
| --- | ---: | ---: | ---: | ---: | --- |
| `mixed_support`, 64 nodes | 99/99 | 26/26 | 26/26 | 77335002 | Passed, including final export/publication and four matched fixed problems |
| `below_one`, 64 nodes | 98/99 | 25/26 | 26/26 | 77335008 | Incomplete: one later surrogate QP unresolved |
| `mixed_support`, 256 nodes | 97/99 | 24/26 | 26/26 | 77335012 | Incomplete: two outer starts unresolved; no failed QP |

For below64, path index 2/start index 2 reaches a later unresolved QP at lambda
`553.4678435191591`, outer iteration 1, after 20 accumulated backtracks and 22
QP calls. Recovery of the original saved QP allowed the outer algorithm to
advance; it did not guarantee that this later surrogate would qualify. Its
literal-state capture is a separate diagnostic step.

For mixed256, every attempted surrogate QP qualifies, but two outer starts do
not. Path index 1/start index 1 ends with `positive_descent` after 20 direction
restarts and 504 QP calls. Path index 7/start index 2 ends with
`positive_unresolved` after 24 QP calls. These are outer observed-likelihood
resolution failures, not remaining instances of the original failed-QP replay.
Indices in this section are zero-based, matching the saved path records.

Both incomplete paths still have qualified numerical winners and refits at all
26 penalties. The qualifier stops before final export, result validation and
publication because planned start coverage remains incomplete. Winner
qualification does not qualify the whole search. Neither below256 nor mixed512
is admitted by these results.

The selected numerical estimates in all three paths exactly match the saved
`430db26` comparisons: selected lambda, raw CCF, raw objective, refitted CCF,
cluster centers, labels and score. Pilot and graph-weight hashes also match.
Only mixed64 completed the final export/publication and matched-problem checks;
the other comparisons are read-only comparisons of unexported numerical
candidates from incomplete searches. `FULL_PATH_STATUS.json` in the study
directory binds the current and baseline receipts and full-path artifacts.

The existing CUDA regression qualifier also passed on this intermediate source
and the frozen baseline (job 77335003). This does not erase the heterogeneous
incomplete-search results above.

## Repairs identified by the complete paths

The later below64 capture (job 77335020) shows that all three equality tolerances
merge the same two interior nodes. They require internal divergence `220.420699`
across an edge capped at `79.974233`. Splitting only these provably obstructed
nodes preserves the 61-node fused group and the bound singleton. The replacement
must independently pass the original objective, gap and KKT checks; neither
reporting tolerance nor the number of equality proposals changes.

The exact CUDA pooled-state diagnostic for mixed256 (job 77335188) identifies
two separate outer failures. At mutation `m000243`, the selected inward gradient
is `1680540.48` while represented curvature is `1e-8`; the missing one-sided
curvature is `2.834615e12`. The repair adds that missing curvature only inside
the outer surrogate. Original likelihood terms, pilots and pooled starts remain
unchanged. The positive directional cut also stalls because a blanket roundoff
bound includes large strictly positive residuals. Conservative per-row intervals
before the negative-part reduction resolve that bound without changing the cut
objective, tolerance or budget. The negative cut then reveals genuine descent,
so this diagnostic is not a raw-stationarity or complete-fit pass.

The next intermediate source fingerprint was
`8627daf96b3b72be689a756445dfb60a7798a03d8bdd8d896f837eb1f11184c2`.
It recovered **all 14 QPs captured up to that stage** from their literal original initializations
on CUDA (jobs 77335248 and 77335252), with independent eager and compiled
original-QP certificates. The later below64 QP uses 1,792 ADMM iterations;
the original two use 1,024 each, and the eleven mixed256 QPs use 1,024–11,040.
Its subsequent complete-path results are historical evidence for that source:

| Input / check | Qualified starts | Complete penalties | LSF job | Result on `8627daf9...` |
| --- | ---: | ---: | ---: | --- |
| Existing CUDA regression qualifier | — | — | 77335291 | Passed |
| `mixed_support`, 64 nodes | 99/99 | 26/26 | 77335301 | Passed, including paired checks |
| `below_one`, 64 nodes | 99/99 | 26/26 | 77335363 | Passed |
| `mixed_support`, 256 nodes, attempt R | 98/99 | 25/26 | 77335381 | Incomplete: one new QP failure at the largest penalty |
| `below_one`, 256 nodes, attempt S | 98/99 | 25/26 | 77335394 | Incomplete: one QP failure at the first positive penalty |

The repaired mixed256 outer starts from attempt J now qualify. R's remaining
failure is a different surrogate: path 25/start 2, lambda `3631358527.4736257`,
outer iteration 0, first QP, no backtracks. Its gap is `248249951.44482526`
and KKT residual about `0.060606`; it exhausts 20,000 ADMM iterations without
an admitted equality repair. S fails at path 1/start 1, lambda
`137.6911252669794`, outer iteration 1, fourth QP, after two backtracks.
Its gap `1.0453191974868297e-10` passes the original gap gate, while KKT
`1.0822411227353396e-7` narrowly exceeds `1e-7`. The diagnostics' combined
inner-qualified flags must not be interpreted as separate gap-gate failures.

Both failures were captured exactly on frozen `8627daf9...`:

| Capture | LSF job | Receipt SHA-256 |
| --- | ---: | --- |
| [T: largest mixed256 QP](validation/cuda-bound-recovery-v3/attempts/capture-largest-mixed-t/imported/results/capture-largest-mixed256.json) | 77335395 | `c090549f5c9955d01f054255c965022dbdd31f021f22e08b480db31dbaa9163b` |
| [U: first below256 failure](validation/cuda-bound-recovery-v3/attempts/capture-first-below256-u/imported/results/capture-first-below256.json) | 77335398 | `910a09a7143e27cb4f27e693dec62a4e38db07adb078dc7015b16fa10df2bfc3` |

T confirms inflation 1 and backtrack index 0; U confirms inflation 4 and
backtrack index 2. These are literal captured values, not inferred from
cumulative counters. Both capture receipts denote diagnostic success while
their complete fits remain incomplete.

## Current per-node curvature normalization

T's curvature ranges from `34.8334` to `2.8346154720201875e12`, a ratio of
`8.13764e10`. The median is only `869.7184`. Dividing the entire dual residual
by that median makes the one very stiff node dominate a controller intended
to compare CCF displacement. The current candidate instead uses

```text
dual_norm = norm((rho / h) * adjoint(z - previous_z))
```

This applies the same diagonal curvature used by the surrogate to each node's
dual-gradient residual. The primal residual, rho initialization, factor-ten
comparison, factor-two updates, bounds and solver budgets remain unchanged.
It changes numerical conditioning, not `h`, target, capacities, initialization
or the original QP admission tests.

[LARGEST_MIXED_CONDITIONING.json](validation/cuda-bound-recovery-v3/study/LARGEST_MIXED_CONDITIONING.json)
records a read-only single-update reference: the boxed node solve has about
`1.21e-16` componentwise backward error, while the median and per-node
controllers would choose different next rho updates from the same terminal
state. This is not a reconstruction of historical balancing decisions or a
full CPU QP fit. The independent original-start CUDA replays test the repair:

| Current-source saved problem | Result | ADMM iterations | LSF job |
| --- | --- | ---: | ---: |
| T: largest mixed256 QP | 1/1 qualified | 10,240 | 77335399 (V) |
| Original mixed256 QPs | 11/11 qualified | 1,024–16,592 | 77335399 (V) |
| U: below256 QP | 0/1 qualified | 20,000 | 77335452 (W) |

V checks both eager and compiled original-QP certificates. Its largest-penalty
replay has eager gap `2.2351749777068638e-8` and KKT
`7.46795633598866e-8`. W remains failed under the unchanged gates: its eager
gap `1.5445872473525447e-10` passes, but KKT
`2.477685575358654e-7` does not. These saved-QP outcomes do not qualify a
complete likelihood search.

### From a local diagnosis to an exact mathematical obstruction

[BELOW256_REPRESENTABILITY.json](validation/cuda-bound-recovery-v3/study/BELOW256_REPRESENTABILITY.json)
examines the captured interior singleton `m000139`, keeping its neighboring
coordinates and edge ordering fixed. Its curvature is `1.313778508084449e13`,
and all incident edges are exactly saturated. The stationary root lies between
adjacent float64 values. Of the three neighboring floats examined, the closest
one already minimizes the saturated-flow residual but has normalized KKT
`1.0822411460429744e-7`, above `1e-7`. Moving enough incident flow to pass that
node test would incur at least `5.5731018777204295e-6` edge gap, over 241 times
the entire allowed gap `2.308534098106846e-8`, even after the stated row
roundoff allowance.

This first result is a **local fixed-order singleton diagnosis**, using literal-array CPU
arithmetic and a higher-precision root reference. It does not prove that every
float64 primal/dual configuration for the QP is infeasible, does not rule out
a different outer trajectory, and is not CUDA qualification. It explains why
continuing this fixed-candidate flow or refining an unrelated fused center
cannot repair this particular obstruction. W remains unresolved; no tolerance,
precision policy or success label has been changed.

The later, independently reviewed
[V3 rational bound](validation/cuda-bound-recovery-v3/study/BELOW256_RATIONAL_BOUND_V3.json)
establishes a stronger statement for the literal U surrogate: **no binary64
primal, with any feasible dual, can meet both original exact mathematical
gap and node-KKT gates**. Its SHA-256 is
`793afb93b5cf7fe9546dd8eeaba3def5c7dde5998f0cc55347e25446f97d5a9a`;
the [reproducer](validation/cuda-bound-recovery-v3/study/below256_rational_bound.py)
uses exact rational arithmetic on the captured binary64 inputs and policy
constants. A bound from the saved feasible gap and positive curvature confines
every potentially admitted primal to a region where node 139 stays interior
and its incident edge ordering cannot change. The gradient sign also forces
a positive adjoint for any KKT-admissible dual. The two floats bracketing the
stationary root require at least `5.572854941897914e-6` incident edge gap,
whereas the global admitted gap is at most `2.3085340981299313e-8`: a factor of
241.4023. Monotonicity excludes all farther binary64 coordinates.

This is an exact mathematical certificate obstruction, **not a formal proof of
every CUDA reduction and rounding context**. Actual eager/compiled gates and
the failed W replay remain separate execution evidence. It concerns this
literal surrogate, not every possible outer trajectory. It does not authorize
a tolerance change, alternative certificate, perturbed surrogate or successful
status for U. The earlier local diagnosis remains valid within its narrower
scope.

## Current-source qualification status

All entries here refer only to `a3aa1dc7...`, not the earlier `8627daf9...` passes.

| Check | Current-source status | LSF job |
| --- | --- | ---: |
| Original/later below64 QP replays | 3/3 passed, attempt X | 77335492 |
| Mixed64 paired full path | Passed: 99/99 starts, 26/26 penalties, final publication; X | 77335492 |
| Below64 full path | Passed: 99/99 starts, 26/26 penalties, final publication; X | 77335492 |
| Existing CUDA regression qualifier | Passed, including final public checks; Y | 77335501 |
| Mixed256 full path | Passed: 99/99 starts, 26/26 penalties, final publication; Z | 77336012 |
| Below256 full path | Incomplete: 98/99 starts, 25/26 penalties; AA | 77336027 |
| Mixed512 full path | Passed: 99/99 starts, 26/26 penalties, final publication; AB | 77336044 |
| Below512 full path | Not run: below256 predecessor remains incomplete | — |

AA retains one unresolved QP at path 1/start 1, outer iteration 1, after two
backtracks and four QP calls. Its numerical winners and refits qualify at all
26 penalties, but missing start coverage prevents final publication. Successful
mixed-family qualification does not admit the below512 stage or erase W's
failed original-start surrogate replay.

The [final source-bound comparison](validation/cuda-bound-recovery-v3/study/DIAGONAL_PATHS.final.md)
and [machine-readable evidence](validation/cuda-bound-recovery-v3/study/DIAGONAL_PATHS.final.json)
bind the completed attempts and baseline comparisons. Both 64-node paths and
mixed256 exactly match the saved baseline's selected raw and final estimates;
mixed512 has no corresponding baseline. Across the existing regression
comparisons, final-refit CCFs, centers, labels, scores and selected lambdas are
exact. Two raw comparisons are not bitwise exact: scaling N16 differs by at
most `2.7755575615628914e-17` in raw CCF; eager `all_bounds_below_one` differs by
`8.326672684688674e-17` in raw CCF and `2.842170943040401e-14` in raw objective.
The comparison records retain those differences. Single observed runtimes are
not repeated throughput measurements or a general speedup claim.

## Retained diagnostic failure and local checks

The first replay attempt, job 77334938, passed the three-node eager/compiled CUDA check but stopped
before original-start solves: a fresh compilation produced a one-ULP scale
difference and a small KKT reduction difference from the captured compiled
certificate. That diagnostic failure is retained; it is not a failed production
QP replay or permission to weaken numerical gates. Replay v2 requires the saved
eager certificate to reconstruct exactly, preserves the capture's original
bit-exact compiled proof, and reports fresh compiled differences while requiring
unchanged gap/KKT gate classifications. No acceptance tolerance was added.

The current `a3aa1dc7...` whole-suite local validation has **1,050 passing repository tests**,
with no failed or skipped tests;
lint, compilation and whitespace checks pass. The full redirected test log and
source/test/helper inventories are retained. Local CPU checks are not
allocated-CUDA qualification. The source-bound receipt is
`validation/cuda-bound-recovery-v3/local/LOCAL_VALIDATION.json`; its
`pytest-diagonal.log` hash is
`d9ffd82840fd06e128725cbfdf8451f56d5e2fbbdc3eba22b4c8349b8a690307`.
