# Implementation and qualification

`io → model → scalar → chain → solver → clonal → selection → api/report` is the
dependency direction. Low-level likelihood imports do not load model selection.
Arrays contain no singleton sample dimension. Compiled model, pilot and chain
arrays are backed by immutable bytes; solver workspaces are separate.

The scalar engine splits clipping breakpoints, seeds candidate-specific modes,
locates alternative wells, then refines intervals by a conservative lower bound.
One bound independently maximizes each candidate binomial kernel. A second uses
`f'' = E(curvature) - Var(score)` and a bound on the score range on a smooth
interval. An arithmetic margin lowers the computed bound. The budget is bounded;
an unresolved gap rejects the pilot/refit. This is float64 numerical qualification,
not a proof using directed-rounding interval arithmetic.

The outer solver uses positive expected-candidate curvature, floored at one,
and backtracks curvature until both true-objective descent and surrogate
majorization/descent hold. Its quadratic subproblem is

```text
min 0.5 sum h_i (x_i-U_i)^2 + sum caps_i |x_{i+1}-x_i|, lower <= x <= upper.
```

The difference and adjoint use slices; no incidence matrix is constructed.
Dual projection is `clip(q + sigma diff(xbar), -caps, caps)`. The primal proximal
step is exact coordinatewise clipping of
`(x - tau D.T q + tau h U)/(1 + tau h)`.
Initial steps are `tau=.49/median(h)`, `sigma=.49 median(h)`, with product .2401.
Steps remain fixed within each surrogate solve, with `xbar=2*x_new-x_old`.

For a dual-feasible q, let a=D.T q and z=clip(U-a/h,lower,upper). The dual lower
bound is `sum[.5 h (z-U)^2 + a*z]`. The primal minus dual gap qualifies the inner
solve at absolute 1e-10 plus relative 1e-11 of its primal objective. A materially
negative gap, nonfinite result, or exhausted iteration budget fails qualification.

When a surrogate step crosses a clipping kink, an exact breakpoint step may
replace it if it improves the true objective and satisfies both descent checks.
The qualified QP gap and the accepted breakpoint step's surrogate gap are reported
separately; the latter does not confer an inner-optimality claim.

The outer audit checks the componentwise box-normal residual normalized by
`1+abs(gradient)+abs(D.T q)`, the dual box, and nonzero-edge complementarity.
The threshold is 2e-5. Upward kinks use an admissible one-sided subgradient.
Exact clipping kinks receive one-sided coordinate and
fused-block checks and objective-decreasing restarts. An unresolved direction
returns a typed status instead of a smooth convergence claim. This is branch
stationarity, not global optimality. The lambda-zero case has separate scalar-gap
qualification because it is separable.

Memory remains bounded by a constant number of full raw states: current branch,
best branch, preceding lambda and selected candidate. Refitting caches only the
last partition. The fixed path stores scalar diagnostics and hashes, not a raw
state per candidate or witness. The clonal refit profile is computed in O(K).
Input parsing additionally requires storage proportional to the supplied CN rows.

## Validation boundaries

Tests exercise pinned upstream losses/derivatives/posteriors, interval bounds,
an independent OSQP chain oracle, witness release and screening, clipping,
original feasibility, ID permutation invariance and complete output/failure flows.
Fixture generation is separate from normal tests and requires the exact clean
upstream revision. A success receipt records incomplete witness/path coverage
explicitly; it does not claim that all candidates qualified.

`benchmarks/simulate.py` creates reproducible read-count data and truth.
`benchmark_scaling.py` runs each size in a fresh CPU process and records wall
time, peak process RSS, graph-array bytes, pilot/solver/refit time and numerical
coverage. `compare_clipp2.py` compares supplied final-refit tables with truth on
exactly matched retained IDs. It reports CCF error, ARI (including its true-K-one
flag), selected K and CNA-only multiplicity F1/coverage. Mixed-CN truth has an
explicit integer draw target in the simulator; real mixed-CN evaluation requires
an independently defined target.

Matched full-fit CliPP2 comparisons, large-cohort reconstruction accuracy,
memory scaling and end-to-end speedups are separate qualification work. No GPU
or remote scheduler execution is implied by the new package's CPU tests.
