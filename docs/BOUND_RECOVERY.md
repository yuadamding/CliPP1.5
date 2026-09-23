# Bound-aware dual recovery and failed-QP replay

This revision changes numerical recovery for the existing boxed, complete-graph
quadratic surrogate and its outer likelihood solver. It keeps the likelihood,
graph, capacities, feasible boxes, score, multistart coverage and qualification
thresholds. The three-node counterexample
motivates the repair; it does not identify the cause of any larger failed QP.
Allocated-CUDA capture and replay receipts are required for those conclusions.

Current candidate source
`a3aa1dc7719b743e0e9342291aa25dd21f7860632ade008a67768d8d5c40cf30`
uses per-node curvature in residual balancing. Both 64-node full paths, mixed256,
mixed512, the existing CUDA regression qualifier and 15/16 saved QPs qualify.
The below256 full path retains one unresolved QP; below512 was not run. Intermediate
`276da27b...` and `8627daf9...` evidence below keeps its original source scope;
no earlier pass is inherited by the current package. See
[the qualification report](../VALIDATION_BOUND_RECOVERY.md) for the source-specific
results and remaining qualification limit.

## Recovery problem and sign convention

For a fixed feasible equality candidate `x`, let

\[
g=H(x-t),\qquad r=g+D^\top q,\qquad (Dx)_{ij}=x_j-x_i,
\qquad D^\top q=-\operatorname{rowSum}(q).
\]

`q` is stored as a skew matrix. Each independent variable is one canonical
`i < j` edge. Nonfused edges have the fixed value
`q_ij = c_ij sign(x_j-x_i)`; fused edges satisfy `|q_ij| <= c_ij`.

The permitted stationarity residual cone is
\(\mathcal R_x=-N_{[\ell,u]}(x)\). Its component is nonnegative at a lower-only
bound, nonpositive at an upper-only bound, zero in the interior, and unrestricted
at a fixed coordinate. Membership uses exact original bounds. Recovery minimizes

\[
F(q)=\tfrac12\operatorname{dist}^2(g+D^\top q,\mathcal R_x).
\]

Write \(v=r-\Pi_{\mathcal R_x}(r)\). The implementation computes

| Coordinate | `v_i` |
| --- | --- |
| Lower bound only | `min(r_i, 0)` |
| Upper bound only | `max(r_i, 0)` |
| Interior | `r_i` |
| Fixed | `0` |

Fixed-coordinate gradients use the existing safe-target convention; their
residuals are unrestricted. This avoids unnecessary arithmetic with irrelevant
fixed-coordinate targets.

In canonical edge coordinates, \(\nabla F=Dv\). For an exact fused block of size
`k`, the complete incidence matrix has squared operator norm `k`. The residual map `r - projection(r)` for this closed convex cone is
1-Lipschitz, so the gradient has Lipschitz constant at most `k`. The update is

\[
q^+_{ij}=\operatorname{clip}_{[-c_{ij},c_{ij}]}
\left(q_{ij}-\frac{v_j-v_i}{k}\right)
\]

on fused edges, while nonfused edges remain fixed. In exact arithmetic,
\(F(q^+)\le F(q)-\frac{k}{2}\|q^+-q\|_2^2\) within each block. This norm
counts canonical edges once; the full skew matrix contains each edge twice.
Equal block sizes at both endpoints preserve skew symmetry.

A stationary recovery iterate can have positive `F` when the candidate cannot
be certified. Neither stagnation nor a small recovery residual is acceptance.
The original full-QP primal-dual gap, KKT residual, primal boxes and dual
feasibility remain decisive. Proposal repair is bounded to 1,024 steps; the
20,000-step per-QP ADMM budget and all numerical tolerances remain unchanged.

## Preserve an already valid certificate

Equality preparation first tests the incoming dual on the candidate under the
original QP gates. A valid dual is retained before applying a repair map. An
existing certificate can be reused only inside the same checkpoint, on the same
QP, primal values and dual. It never crosses an ADMM update or a surrogate change.

A valid near-fusion may still have tiny nonzero differences and unsaturated
edges. The implementation retains that valid result while trying the existing
tighter equality proposals, which can help the outer exact-kink audit. Exact
edge complementarity is a selection preference here, not another acceptance
gate; if tighter proposals fail, the valid candidate remains available.

## Make residual balancing consistent under objective scaling

The original thirteen captured terminal QPs have edge-complementarity gaps, not box-normal gaps.
Some equality proposals also contain capacity-cut obstructions, so repairing
their dual alone cannot certify those fixed primal values. Bound-aware recovery
does not by itself resolve those captured failures.

The previous ADMM controller compared the unique-edge primal residual

\[
r=\sqrt{\tfrac12\sum_{ij}((Dx)_{ij}-z_{ij})^2}
\]

with the node-gradient residual
\(s=\rho\|D^\top(z-z_{\rm previous})\|_2\). Multiplying the complete QP
objective by a positive constant `c` scales `h`, capacities, the physical dual
and `rho` by `c`, while leaving the target, boxes, `x`, `z` and scaled dual `v`
unchanged. The boxed ADMM update and edge threshold are unchanged in exact
arithmetic. However, `r` is unchanged and `s` scales by `c`, so the old comparison
could select a different rho update for equivalent iterates.

The intermediate `276da27b...` controller compared `r` with `s / base_rho`, using the fixed initial
`base_rho = median(h)` with the existing positive tiny-value floor. Both residuals
then have CCF units and remain unchanged under uniform objective scaling. The
implementation evaluated this as
`(rho / base_rho) * norm(D.T @ (z - previous_z))`. It retains the factor-ten
comparison, check frequency, factor-two rho updates, relative rho bounds, state
rescaling and initial rho. It adds no edge/node RMS normalization.

This invariance applies in exact arithmetic away from overflow, underflow and
activation of the tiny-value floor. It is not a claim of bitwise invariance under
arbitrary floating-point scaling, or a guarantee of convergence or speed. The
absolute-plus-relative gap and KKT gates remain unchanged; the absolute gap
term is intentionally not invariant under objective scaling.

The saved states contain terminal `rho` and `z`, but not `z_previous`. They
therefore cannot reconstruct historical balancing decisions or prove a causal
trajectory. The algebra establishes the controller's scale mismatch; exact
original-start CUDA replays and complete paths separately test the consequences
of correcting it. The study's `RESIDUAL_BALANCE_ANALYSIS.md` records the terminal
ratios and these limits.

### Per-node scaling for heterogeneous curvature

The median-based rule remains invariant to uniform objective scaling, but does
not account for a large ratio between individual node curvatures. After the
active-box curvature repair, the largest mixed256 penalty exposed this in
source `8627daf9...`: the literal surrogate has minimum curvature `34.8334`,
median `869.7184`, maximum `2.8346154720201875e12`, and a ratio of about
`8.14e10`. This is the distinct QP captured by
[T](../validation/cuda-bound-recovery-v3/attempts/capture-largest-mixed-t/imported/results/capture-largest-mixed256.json),
after complete-path attempt R left one unresolved start. Its exact initial
state, target, original capacities and corrected curvature are retained.

Current source `a3aa1dc7...` uses

\[
s_H=\left\|\operatorname{diag}(\rho/h_i)
D^\top(z-z_{\rm previous})\right\|_2.
\]

Each dual-gradient component is converted to a CCF displacement using that
node's own surrogate curvature. Uniform scaling of `h`, capacities and `rho`
still leaves the comparison invariant in exact arithmetic. Initial rho remains
the original median; rho limits, physical-dual rescaling, the factor-ten
comparison, doubling/halving and check frequency are unchanged. This controller
does not rescale the QP objective, replace the surrogate, adjust its input
initialization or change admission tolerances.

[LARGEST_MIXED_CONDITIONING.json](../validation/cuda-bound-recovery-v3/study/LARGEST_MIXED_CONDITIONING.json)
contains a literal-array CPU reference for one boxed update and the next
controller decision. Its small componentwise backward error supports that
specific node-update calculation, not every historical CUDA update. The
recorded next-step comparison is not a recovered past rho decision. Current
source CUDA replay V independently qualifies T from its original initialization
in 10,240 ADMM iterations and preserves qualification of the eleven original
mixed256 captures. Separate full-path attempt Z qualifies complete mixed256
fitting; saved-QP replay alone does not establish that result.

## Three further limitations exposed by complete paths

The source with normalized residual balancing,
`276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5`,
passed original-start replay of all 13 archived baseline QPs. Complete paths
then exposed three distinct limitations. These failures cannot be attributed
to the original uniform-normal counterexample or treated as additional failed
instances of the same saved QPs.

### An equality candidate can violate a necessary capacity cut

The new below-one64 failure is captured in
[capture K](../validation/cuda-bound-recovery-v3/attempts/capture-current-below-k/imported/results/capture-current-below64.json),
SHA-256 `0c63526d7e324f3d00125208165237df40800f49e2f919ef06f78711e752a4b6`.
Its original terminal gap is `1.1158742571527176e-7`, almost entirely edge
complementarity. The last equality candidate has a different problem: nodes
0 and 38 were fused into one interior block despite insufficient internal edge
capacity. Its gap is `4.512580548062884e-6`, entirely node quadratic, with KKT
residual `0.05183130144438088`. Changing only the dual cannot certify that block.

For a node in a proposed block, write `r_external` for its quadratic gradient
plus the fixed external-edge divergence, and `C_internal` for its total
incident internal capacity. An interior node requires
`abs(r_external) <= C_internal`. A lower-bound node requires
`r_external + C_internal >= 0`; an upper-bound node requires
`r_external - C_internal <= 0`. Fixed coordinates impose no such requirement.
These singleton cuts are necessary, not sufficient, for block feasibility.

The solver makes one bounded replacement proposal when a non-singleton block
violates this necessary condition beyond an arithmetic margin. It splits only
the offending nodes, retains the other equalities, uses the original iterate
to propose the newly separated ordering, and recomputes boxed centers. The
replacement must pass the original objective check, gap and KKT certificates.
A valid incoming certificate is preserved first. Reporting/fusion tolerance,
the likelihood partition score and the 1,024-step repair limit do not change.
The cut calculation proposes a candidate; it is never itself admission.

### A box endpoint can select a derivative whose curvature was omitted

The mixed256 path
[attempt J](../validation/cuda-bound-recovery-v3/attempts/qualification-mixed256-j/imported/results/mixed256-full.artifacts/full-path.json)
has **zero unresolved QP returns and two unresolved outer starts**. Both are
pooled starts. At path 1/start 1, all 504 QP trials were rejected across 21
outer iterations; 20 directional restarts provided the only accepted moves.
At path 7/start 2, all 24 trials were rejected at the original pooled state.
The empty objective tails record that no majorization trial was accepted.

[CUDA diagnostic L](../validation/cuda-bound-recovery-v3/attempts/diagnostic-outer-l/imported/results/pooled-outer-diagnostic.json),
SHA-256 `ba2c202a811daae970c52c640d097b6bdf1a48f3c83f5c3a64a93c2bc2a4ee85`,
uses the same frozen source, literal pilots and weights. It reproduces J's
path-7 pooled objective exactly: `4617151.7318359185`. The pooled-vector hash is
`fe070c58c04d8a848a2666ea9a9dc8c6c8317d02bfaa6b13dcb0484764d9af90`.
Both eager and compiled CUDA establish the same discrepancy at `m000243`:

| Quantity at the exact upper bound | Value |
| --- | ---: |
| CCF / original upper bound | 0.5939535596308417 |
| Represented curvature | 1e-8 |
| Selected inward one-sided gradient | 1680540.479615325 |
| Missing inward posterior curvature | 2834615472020.1875 |

The outer solver correctly selected the inward derivative at this clipping
kink, but the represented moving mask omitted its curvature. With the existing
base-curvature floor, 24 inflation trials start at 1 and reach only `2^23`.
This diagnoses a mismatched surrogate branch; L did not replay all 504 rejected
trials or prove every rejection has only this cause.

`bound_majorization_curvature` adds only the missing inward contribution at
an **exact active original-box clipping endpoint**, excluding fixed coordinates
and contributions already present in the represented moving mask. It uses the
already computed posterior. `likelihood()`, `TensorModel.terms()`, pilot
curvature, and the curvature-weighted pooled start calculation remain unchanged.
The original likelihood majorization, objective and QP checks still decide
acceptance; the correction increases no iteration or backtrack budget.

### A global rounding allowance can obscure a zero cut lower bound

At the exact path-7 pooled state, the original positive-direction cut has an
unadjusted lower bound of zero. Its global rounding allowance is
`2.941987824372397e-12`, exceeding the unchanged normalized tolerance
`1.8803196756525797e-12`. After 20,000 iterations, the reported lower bound is
still exactly minus that allowance. The raw audit returns `positive_unresolved`
before inspecting the negative direction.

For `r = a + D.T @ q`, the revised conservative bound first forms a lower
interval for each row with error
`Gamma * (abs(a_i) + sum_j abs(q_ij))`, where
`Gamma = 8 * (N + 2) * eps`. It takes the negative part of those row bounds,
then subtracts the final reduction error and the original-unit constant floor.
A row whose complete error interval is positive contributes exactly zero;
its large positive coefficients do not consume another row's allowance. The
production calculation additionally rounds the row subtraction and final lower
bound downward. Cancellation-sensitive rows retain their absolute-term error
bounds. Feasible skew/cap checks, both signed directions, the primal descent
certificate, threshold and cut-iteration budget are retained.

L's separate rowwise experiment qualifies the positive cut at iteration 0 with
lower bound `-4.308753143774573e-20`. Evaluating the negative cut diagnostically
then finds certified descent at iteration 16 under both formulas (normalized
attained value `-2.8833835598463406`). Thus the pooled state is **not stationary**:
the corrected positive certificate allows the negative descent to be discovered;
it does not qualify the raw state by skipping a direction.

K and L are source-bound diagnostic evidence retained in the immutable archive.
L establishes the exact
CUDA mechanism and its independent cut experiment, not a complete production
fit. Candidate references, individual QP replays, full-start coverage and full
path qualification remain separate claims. Neither K nor L is an admissible
predecessor for a larger-size fit.

## A remaining binary64 representability obstruction

The `8627daf9...` complete mixed256 and below256 paths both finish with one
unresolved QP: attempt R at path 25/start 2, and attempt S at path 1/start 1.
They are retained separately from J's two earlier unresolved outer starts.
Capture T preserves R; [capture U](../validation/cuda-bound-recovery-v3/attempts/capture-first-below256-u/imported/results/capture-first-below256.json)
preserves S exactly. Their hashes and outcomes are in the qualification report.

Current-source replay W of U still exhausts the original budget: its gap
passes, while KKT fails. The
[representability diagnostic](../validation/cuda-bound-recovery-v3/study/BELOW256_REPRESENTABILITY.json)
considers the captured interior singleton `m000139`, with curvature
`1.313778508084449e13`, fixed neighboring coordinates and fixed edge ordering.
All its incident edges are saturated. The nearest of three neighboring float64
CCFs has normalized KKT `1.0822411460429744e-7`, just above `1e-7`. At those
coordinates and that ordering, changing enough feasible incident flow to pass
the node test costs at least `5.5731018777204295e-6` edge gap, versus an entire
allowed gap of `2.308534098106846e-8`. The stated row roundoff allowance does
not close this difference.

This first diagnostic is a narrow local result from read-only CPU arithmetic on literal captured
tensors, including a higher-precision stationary-root reference. It is not a
proof of infeasibility across every possible float64 primal/dual state or every
outer trajectory, and it is not a CPU fit or CUDA qualification. Adjusting an
unrelated fused center or extending a fixed-candidate dual repair cannot remove
the demonstrated local obstruction. The current implementation retains the
failure; it does not broaden KKT/gap gates or claim that a small gap certifies
the whole QP.

The subsequent, independently reviewed
[V3 rational bound](../validation/cuda-bound-recovery-v3/study/BELOW256_RATIONAL_BOUND_V3.json)
extends the analysis to every binary64 primal with any feasible dual for the
literal U surrogate. Its SHA-256 is
`793afb93b5cf7fe9546dd8eeaba3def5c7dde5998f0cc55347e25446f97d5a9a`.
The [reproducer](../validation/cuda-bound-recovery-v3/study/below256_rational_bound.py)
interprets the saved binary64 problem and policy constants as exact rationals.
All curvatures are positive and all boxes have nonzero width, so the certificate
scale equals the primal objective plus a fixed constant. The saved feasible
gap bounds the optimum; any mathematical gate pass therefore has gap at most
`2.3085340981299313e-8`. Strong-convexity radii keep node 139 interior and
preserve every incident edge ordering throughout that admitted region. The
maximum gradient is below `-1269`, which forces the adjoint to be positive for
any node-KKT pass.

The two binary64 coordinates bracketing the stationary root then require
incident edge gaps of at least `5.572854941897914e-6` and
`0.00015524368526856393`; even the smaller exceeds the global gap allowance by
241.4023 times. The required flow adjustment increases monotonically away from
the root, excluding every farther binary64 coordinate. Thus **no binary64
primal with any feasible dual can satisfy both exact mathematical original
gates on this literal surrogate**. This strengthens the earlier local result;
it does not change that result's narrower scope.

The rational proof is not a formal interval verification of every CUDA
reduction or rounding context. Actual eager/compiled certificates and failed
replay W are separate execution evidence. The result does not exclude another
outer trajectory producing a different surrogate, and it authorizes no relaxed
gate, surrogate perturbation or successful status. This source retains U's
failure.

| Current-source stage | Status |
| --- | --- |
| Largest mixed256 and eleven original mixed256 saved QPs | Qualified individually by V |
| Original and later below64 saved QPs | 3/3 qualified by X |
| Newly captured below256 saved QP | Unresolved in W |
| Mixed64 and below64 full paths | Qualified by X: 99/99 starts and 26/26 penalties each, with final publication |
| Existing CUDA regression qualifier | Qualified by Y, job 77335501, including final public checks |
| Mixed256 full path | Qualified by Z, job 77336012: 99/99 starts, 26/26 penalties, final publication |
| Below256 full path | Incomplete in AA, job 77336027: 98/99 starts, 25/26 penalties |
| Mixed512 full path | Qualified by AB, job 77336044: 99/99 starts, 26/26 penalties, final publication |
| Below512 full path | Not run: below256 predecessor remains incomplete |

AA's unresolved start is path 1/start 1, outer iteration 1, after two backtracks
and four QP calls. Its 26 qualified winners and refits do not establish complete
start coverage or permit publication. The
[final comparison](../validation/cuda-bound-recovery-v3/study/DIAGONAL_PATHS.final.md)
preserves all current-source outcomes. Both 64-node paths and mixed256 exactly
match baseline selected estimates. Existing regression comparisons have exact
final-refit CCFs, centers, labels, scores and lambdas; raw differences remain
explicit for scaling N16 (maximum CCF difference `2.7755575615628914e-17`) and
eager `all_bounds_below_one` (CCF `8.326672684688674e-17`, objective
`2.842170943040401e-14`). Mixed512 has no matching baseline. Observed timings
are single runs, not repeated throughput measurements.

## Capture the actual failed surrogate

[The capture driver](../benchmarks/capture_failed_qp.py) admits only explicit
source profiles. The default `baseline430` profile retains frozen commit
`430db26cf07466e88e53c6e1a8fbe2be7b7b25e9`, production fingerprint
`726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88`, and
the original v1 receipt contract. Its return hooks preserve that solver's
arithmetic. Capture includes:

- Original `h`, target, boxes, literal capacities, primal start and physical dual.
- Terminal ADMM `x`, `z`, `v`, `q`, `rho`, separately from the returned state.
- Last attempted and last refined equality states, with their incoming duals.
- Source, policy, graph/pilot, input, lambda, path/start, outer iteration,
  backtrack and inflation identities.

The failed state is cloned at return and exported after the outer start returns.
Those timings are diagnostic and instrumented. JSON arrays carry file and exact
float64 byte hashes; exceptional nonfinite statistics use explicit lossless hex
encoding. Original compiled terminal certificates must reconstruct bit for bit
inside the original capture compilation context.

Run inside an authorized allocated-CUDA job with the frozen package on
`PYTHONPATH` and a valid compiler in `CC`:

```bash
python benchmarks/capture_failed_qp.py capture \
  --fixture below_one --nodes 64 --expected-failures 2 \
  --expected-source-sha256 726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88 \
  --device cuda:0 --out results/capture.json
```

The other admitted capture is `--fixture mixed_support --nodes 256
--expected-failures 11`. A capture receipt with `status=passed` confirms diagnostic
capture of the expected failures; its scientific fit remains incomplete.

The separate `normalized276da` profile binds production fingerprint
`276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5` and
v2 receipts. It admits only the unchanged `below_one` 64-node input with one
expected failure: path index 2, start index 2, outer iteration 1, 20 accumulated
backtracks, QP ordinal 22 and lambda `553.4678435191591`. Indices follow the
capture receipt. This profile uses `--source-profile normalized276da` and
`--expected-failures 1`, with its exact source fingerprint. It preserves the
new full-path failure separately from the original 13 captured QPs. A v2
diagnostic pass still cannot qualify a complete fit or a larger successor.

The `bound8627` v2 profile binds source
`8627daf96b3b72be689a756445dfb60a7798a03d8bdd8d896f837eb1f11184c2`
and admits one failure on each of two separate canonical 256-node inputs:
`mixed_support` at path 25/start 2, outer 0, QP 1, lambda
`3631358527.4736257`; and `below_one` at path 1/start 1, outer 1, QP 4,
lambda `137.6911252669794`, after two cumulative backtracks. Use
`--source-profile bound8627 --expected-failures 1` and the appropriate fixture.
The below256 per-outer backtrack index and inflation were not guessed from the
aggregate path record: U captures their literal values, 2 and 4 respectively.
Both receipt readers preserve the older source profiles unchanged.

## Replay, decompose, and qualify separately

[The replay driver](../benchmarks/replay_failed_qp.py) consumes a hash-bound capture
under the requested current package fingerprint. It never regenerates pilots,
weights or a surrogate. It first tests the literal three-node counterexample on
eager and compiled CUDA, then validates every saved terminal certificate before
running current-source QPs with their exact original starts and initial duals.
Replay v2 preserves the capture's exact compiled proof and requires fresh eager
reconstruction to match the separately saved eager certificate bit for bit.
Fresh compiled reductions can differ between compilation contexts: the receipt
reports their original and fresh bit patterns, signed differences, and values.
Both gap and KKT gate classifications must remain unchanged. This adds no
numerical tolerance and does not change production admission.

```bash
python benchmarks/replay_failed_qp.py \
  --capture results/capture.json --capture-sha256 CAPTURE_RECEIPT_SHA256 \
  --expected-source-sha256 CURRENT_PRODUCTION_SHA256 \
  --device cuda:0 --out results/replay.json --require-all-qualified
```

Every returned replay is independently checked by eager and compiled original-QP
certificates. The driver separately compares the archived uniform map with the
current cone map on captured equality candidates, without changing their primal
values or inputs. Those bounded experiments do not count as successful
original-start QP solves. Their source states and outcomes remain explicit.

Gap decomposition reports node-quadratic, box-normal and edge-complementarity
terms using the original stable safe-target/fixed-coordinate conventions. Raw
signed terms and the literal normal formula are separate diagnostics. Summing
components in a different order can introduce floating-point roundoff; it does
not replace the independently computed original certificate.

Default `status=passed` means the diagnostic finished and its identities and
checks passed. `all_replays_qualified`, `resolved_qps`, `unresolved_qps` and
`scientific_status` distinguish numerical outcomes. `--require-all-qualified`
fails admission if any original-start replay remains unresolved. Timed solve
intervals exclude exports and candidate diagnostics but may include lazy
compilation; they are not repeated throughput measurements.

Successful QP replay still does not qualify a full observed-likelihood search.
Complete heterogeneous paths, every planned start, final raw audits, refits,
selection and publication must be checked before larger stages are admitted.
