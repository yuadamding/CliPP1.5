# Changelog

## Unreleased — prepared QP work and heterogeneous CUDA qualification

Prepare fixed flow-polishing geometry once per candidate, share one equality
proposal and admission implementation, and reuse an already checked proposal
only inside the exact same ADMM checkpoint. Compile the combined node/edge ADMM
step and the prepared dual-flow repair step. Preserve the original objective,
boxes, dense graph, score, rho rule, numerical gates and iteration limits.

Report `qp_admm_iterations` across every attempted start and path candidate,
including unresolved work, while retaining per-start `inner_iterations`.
Add a separate benchmark for uninstrumented QP latency, kernel dispatch counts
and disjoint substage attribution. Extend synthetic complete-path qualification
to distinct heterogeneous mixed-support and below-one-bound inputs at 64, 256
and 512 mutations, with source-bound predecessor and baseline comparison gates.
See [the benchmark contract](docs/QP_BENCHMARKS.md); fixture availability alone
does not mean all sizes have passed allocated-CUDA qualification.

Matched allocated-L40 QPs reduce median latency by 24.5–30.8%, with identical
states and objectives in all 48 sample comparisons. The 877-test local suite
passes. Larger heterogeneous tests expose unresolved QP starts; their failures
and the corrected profiler's earlier attempts are retained in the
[QP validation report](VALIDATION_QP.md) and source-bound evidence archive.

## 0.5.1.dev0 — exact fusions, bounded reuse and complete measurement scopes

Preserve exact-value groups inside overwide tolerance runs instead of splitting
every node into a singleton. Apply the same rule to reporting, refit selection and
QP polishing. This fixes a partition defect that can change K, score and the
selected lambda. Advance the policy to `clipp1d_complete_cuda_unconstrained_v3`
and receipt schema to `clipp1d.cuda.run.v3`.

Move full model/graph/scalar snapshot reconciliation to owned stage boundaries
with metadata/version checks inside. Retain full exceptional-exit and publication
checks, and use the segment-reduction fast path only for validated immutable
lengths. Add shared-arithmetic loss-only and loss–gradient kernels, detect
analytical scalar groups before grid construction, and pack actual golden-search
wells with bounded padding. Add one qualified last-partition refit cache,
canonical singleton-pilot reuse and work counters. Initialize QPs with a qualified
dual within one start/lambda; cross-lambda continuation remains primal-only and
all new surrogates retain independent gap/KKT qualification.

The expanded 256-node QP check exposed a convergence limit in equality polishing.
Add bounded solver-only tighter fusion proposals and repeated affine-flow/cap
repair, subject to the original objective, gap and KKT gates. Keep the public
grouping rule and 20,000 ADMM iteration limit; report additional polishing work
separately and preserve the original failed GPU receipt.
The increasing-size test also exposed zero-step warm states that retained tiny
nonexact fusions, and consensus underconditioning from `median(h)/N`. Offer
certified equality polishing before warm admission, clear a stalled warm dual
after failed raw auditing, and initialize numerical rho from `median(h)` while
retaining residual balancing. Candidate flow repair need not start with an
already qualified dual; final objective, gap and KKT admission stays unchanged.
Preserve original-unit roundoff margins when normalizing directional cuts:
scale the constant error unit with the coefficients and stationarity threshold.
This removes an artificial high-penalty error floor while retaining cancellation
protection and the original absolute stationarity gate.

Measure synchronized numerical phases, final device qualification and export.
Capture CUDA peaks and compilation/work statistics after the last required device
operation. Receipts explicitly exclude their own serialization and durable
publication from elapsed time; returned `operation_metrics` separately records
post-publication completion. Check intended receipt bytes as well as table hashes
before atomic publication. Keep the dense graph, original likelihood/bounds,
unconstrained clonal fitting, raw-primary estimator and post-fit closest-to-one
label-zero rule.

The [archived `371003f` evidence](validation/371003f/README.md) is retrievable and
retains its original policy-v2 scope. It does not qualify this revision; current
acceptance and limitations belong in [VALIDATION_CUDA.md](VALIDATION_CUDA.md).

## 0.5.0.dev0 — complete graph on PyTorch CUDA

Replace the public chain fitting path with float64 PyTorch CUDA complete-graph
fusion, compiled tensor kernels, bounded ADMM surrogates and independent raw
audits. Preserve unconstrained fitting and post-fit nearest-to-one cluster label
0. Retain the marginalized likelihood, original bounds and partition-score
arithmetic. Raw penalized CCFs are primary; membership refits and their
multiplicity calls are explicitly secondary. Bind selected candidate provenance
and qualify publication under `clipp1d.cuda.run.v2`.

There is no CPU inference fallback. Historical chain modules/tests remain
reference-only, and historical CPU launchers reject this source. CPU numerical
references and actual CUDA qualification are reported separately in
[the archived validation evidence](validation/371003f/README.md); older GPU or
chain receipts do not qualify this revision.


## 0.4.1 — 2026-09-22

Designate the fitted cluster closest to CCF one as clonal public label zero.
Use its membership for primary sMF, retain exact-one membership as a separate
diagnostic, and record distance/ties/counts in receipt schema v6. The numerical
policy remains v5: CCFs, partitions, scores and solver constraints are unchanged.
Existing fits can be reevaluated without rerunning inference.

## 0.4.0 — 2026-09-22

Remove the occupied-clonal constraint from production raw fusion fits, separable
lambda-zero fits, partition extraction and independent final refits. Preserve the
original likelihood, boxes, frozen chain, score and numerical tolerances. Advance
policy and receipt schema to v5; sort all public labels by refitted CCF, leave
witness/designation fields null and the legacy clonal flag zero. Retain historical
constrained helpers only for offline regression comparisons.

Strengthen publication identity/certificate checks and JSON completion markers.
Repair exact integer input parsing, malformed-input handling and tolerance/budget
validation. Make benchmark validation and metrics explicit about old/new schemas,
nullable designation, raw-reference columns and direct-partition provenance.
See [validation](VALIDATION_UNCONSTRAINED.md).

## 0.3.0 — 2026-09-21

Add adjacent weighted Ward partitions and qualified boundary refinement beyond
the fusion path, with bounded storage/search budgets, separate candidate provenance
and an independent certified raw reference. Add four-cohort and OCCAMS runners.
This historical version retained the occupied-clonal constraint.

## 0.2.2 — 2026-09-21

Batch finite audit likelihood proposals with bounded storage, exact-order first
acceptance and duplicate reuse. Add a loss-only evaluator sharing the existing
log-kernel, and preserve all numerical policies and gates. Add differential
proposal comparisons and separate observed-dual/rejection diagnostics for the
retained outer-limit states. See [validation](VALIDATION_FINITE_PROPOSALS.md).

Document the two distinct causes of the matched SimClone discrepancies and the
limits of fusion-path search completion in [the investigation](VALIDATION_SIMCLONE.md).

## 0.2.1 — 2026-09-21

Reuse the common profile's forward/reverse reconstruction thresholds and share
polishing, dual recovery and certification with direct solves. Each accepted
vector has one immutable likelihood audit context across all witness anchors;
local proposals reuse old losses. Batch signed-interval prefixes with the
original arithmetic and deterministic tie rules.

Add audit-stage timing, bounded per-start progress diagnostics and actual
float64/longdouble precision metadata. Diagnose the six archived outer-limit
starts without changing the step cap, iteration budget, search policy or gates.
See [controlled comparisons and limitations](VALIDATION_REUSE.md).

## 0.2.0 — 2026-09-21

Move witness profiling into production common-surrogate MM with original boxes,
a freshly rebuilt profile at every backtrack, and at most four distinct primal
starts per penalty. Retain independent nonlinear enumeration only as an offline
reference. Reuse accepted curvature inflation, unchanged inner certificates and
trial likelihoods; remove unused direct-backend dual continuation. This explicitly
changes nonlinear search trajectories while preserving the likelihood, constrained
objective, chain weights, partition/refit/score rules and admission tolerances.
Numerical policy and run-receipt schema advance to v3; completeness describes
planned starts and path penalties, not independently optimized nonlinear witnesses.

Repair interior-normal gap cancellation, wide-range dual reconstruction, and
prefix-value cancellation. Add feasible interval audits across multiple occupied
witnesses. Retain eight hashed actual production failures as regression fixtures.
Benchmark workers capture bounded precise failures and progress inside interrupted
starts. Add CPU/thread controls, sample dispersion, penalty-normalization comparisons,
separate working-memory measurements and full-path/search-policy validation.
See [validation and remaining limitations](VALIDATION_SHARED.md).

## 0.1.1 — 2026-09-21

Repair clipping-endpoint and proper fused-subinterval qualification; use a stable
nonnegative quadratic gap, constant-free stopping scale and separate QP KKT gate.
Reuse chain-bound primal/dual continuation and qualified singleton pilots; add
exact same-slope scalar fits, lazy interval refinement and local clipping deltas.
Separate raw/refit/search statuses, retain selected raw objective/witness identity,
and expose stage timing and reuse counts. Numerical policy and receipt schema
advance to v2. The statistical objective, chain weights and score are unchanged.
Add targeted regressions and bounded hundreds/thousands scaling cases, and make
installation instructions portable.

Replace production primal–dual sweeps with a bounded weighted chain-TV message
solver, linear dual reconstruction and fixed-witness segment decomposition.
Provide shared prefix/suffix witness profiling for one common surrogate, with
independent branch-value validation. Keep the iterative solver as a benchmark
reference and separate topology, quadratic and full-fit timing evidence.

## 0.1.0 — 2026-09-21

Initial independent CPU float64 single-sample implementation in `CliPP1.5`:
validated long-table input; marginalized integer likelihood; qualified scalar
pilots; immutable adaptive chain; boxed primal–dual majorization solver;
streaming unknown-clonal-witness search; full deterministic penalty path;
contiguous constrained refits and compatible score; API, CLI and four-file
reporting; upstream fixtures, independent QP tests and evaluation tooling.
