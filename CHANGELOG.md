# Changelog

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
