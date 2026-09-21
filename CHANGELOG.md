# Changelog

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
