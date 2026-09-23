# Response to the af06b73 review

Retain `af06b73` as the numerical baseline. The review identifies no new confirmed
production-blocking defect and closes exceptional-start trace indexing.
All seven supplied control-flow/arithmetic checks passed locally after verifying
every ZIP member and matching the three source files to the exact Git blobs.
`review_af06b73.zip` retains the supplied checks and original result;
`independent_checks_rerun.json` records the independent local execution. These
controlled tests do not rerun CUDA inference.

The six-stage qualification and actual cohort-staging archives also passed their
existing integrity verifier. Five of six synthetic winners select lambda zero;
selected-output parity alone therefore gives limited evidence about penalized
solutions. `POSITIVE_PENALTIES.json` recomputes the more informative comparisons:

| Fixture | Qualified common positive penalties | Range of coordinate − scalar raw objective |
|---|---:|---:|
| below64 | 25 | 0 |
| mixed64 | 25 | −8.004e−11 to 1.455e−11 |
| below256 | 25 | −305.2993 to 0 |
| mixed256 | 25 | 0 |
| below512 | 25 | 0 |
| mixed512 | 25 | 0 |

Below256's improvement is at the exact shared lambda 137.6911252669794, where
scalar has one unresolved start. Both returned raw candidates qualify, while only
coordinate completes every planned start. This is an observed lower objective
and coverage improvement at that penalty; it does not prove global optimality or
better truth reconstruction. The literal saved QP remains unresolved under its
unchanged certificate thresholds.

Four timing stages are now available in the
[updated timing snapshot](../cuda-surrogate-timing-v2/README.md). The original
controller continues the remaining two. The
[paired empirical study](../../benchmarks/SURROGATE_EMPIRICAL.md) is prepared and
supervised behind all six timing terminals, with one research GPU job at a time.

**Policy decision: retain scalar as the production default.** Coordinate remains
an explicit research strategy pending the paired empirical results. A later
promotion requires a versioned decision that weighs completeness and accuracy
separately from latency. It must preserve the likelihood, graph recipe, bounds,
selection rule, qualification thresholds, literal-QP negative regression, and
historical source/policy identities. No running cohort source is changed here.
