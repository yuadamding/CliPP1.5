# Implementation and qualification

`io → model → scalar → chain → tv/solver → clonal → selection → api/report` is the
dependency direction. Low-level likelihood imports do not load model selection.
Arrays contain no singleton sample dimension. Compiled model, pilot and chain
arrays are backed by immutable bytes; solver workspaces are separate.

The scalar engine solves single-candidate, same-slope blocks exactly by pooling
counts, with the smallest feasible CCF on a clipping plateau. Heterogeneous slopes
and multiplicity mixtures use a bounded initial grid and candidate-mode seeds,
then lazily split promising intervals at clipping breakpoints or midpoints.
It retains the sorted breakpoints but does not evaluate every breakpoint eagerly.
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

The production inner solve uses convex functional dynamic programming:

```text
V_i(t) = .5 h_i (t-U_i)^2 + I_[lower_i,upper_i](t)
         + min_s [V_(i-1)(s) + caps_(i-1) |t-s|].
```

The derivative of the convolution clips the previous derivative to the two edge
slopes. Unequal unary curvatures, unequal edge weights, and mutation-specific boxes
are handled inside this recurrence. Box endpoints can create derivative jumps;
they are represented explicitly. The forward pass records two thresholds per
edge; backward reconstruction clamps the following coordinate between them.
This solves the same TV surrogate, with no segment-score table.

`tv.py` represents affine derivative segments with paired heaps and lazy deletion.
Each stage inserts at most two knots. Tail clipping permanently removes knots;
each heap entry is inserted and popped at most once. Thus this implementation,
**including box jumps**, takes O(M log M) heap work and O(M) storage. Backward
reconstruction and exact-block weighted-mean roundoff polishing take O(M).
These bounds concern one quadratic solve, not the nonconvex fitting pipeline.
The algorithmic background is convex TV message passing in
[Kolmogorov, Pock and Rolinek, sections 3–4](https://arxiv.org/abs/1502.07770).
The bounded extension here has its own implementation and tests.

A fixed witness at one separates the left and right free segments. Its incident
penalties are affine on CCF boxes, so each adjacent target receives `caps/h`.
The solve and certificate retain the original unaries and full objective; constants
are not dropped when comparing witnesses. Arbitrary frozen coordinates whose
incident penalties cross neighboring boxes use the full bounded recurrence.

Dual reconstruction propagates reachable intervals for `q_i` through the node
box normals, intersects them with edge complementarity intervals, then backtracks.
It is O(M), with only arithmetic slack in reconstruction; the independent gap and
KKT admission tolerances below are not relaxed. Numerical failure is unresolved;
production never falls back to the iterative reference.

`solve_quadratic_iterative()` retains the repaired first-order algorithm solely
for tests and attribution benchmarks. `inner_max_iterations` controls that reference.
Production `inner_iterations` counts direct solves (one forward/backward solution
per call); `inner_algorithm` and `last_inner_work` identify the method and knot work.

For a dual-feasible q, let a=D.T q and z=clip(U-a/h,lower,upper). The gap is
evaluated without subtracting complete primal and dual objective values:

```text
G = sum_i [.5 h_i (x_i-z_i)^2 + (h_i(z_i-U_i)+a_i)(x_i-z_i)]
    + sum_e [caps_e |(Dx)_e| - q_e (Dx)_e].
```

Frozen coordinates contribute zero and are omitted before target arithmetic.
Every displayed contribution is nonnegative in exact arithmetic. Materially
negative terms, nonfinite results, or a failed certificate reject qualification.
Only negative roundoff within an explicit arithmetic margin is clamped to zero.
The gap gate remains absolute 1e-10 plus relative 1e-11, now scaled by the free
quadratic energy above its separable boxed minimum plus TV energy on edges with
at least one free end. It excludes constants from frozen targets and fully fixed
edges. A separate normalized box-normal and dual-projection KKT residual must be
at most 1e-7. The dual residual uses `q-clip(q+Dx,-caps,caps)`; the raw audit below
also checks exact nonzero-edge complementarity.

When a surrogate step crosses a clipping kink, an exact breakpoint step may
replace it if it improves the true objective and satisfies both descent checks.
The qualified QP gap and the accepted breakpoint step's surrogate gap are reported
separately; the latter does not confer an inner-optimality claim.

The outer audit checks the componentwise box-normal residual normalized by
`1+abs(gradient)+abs(D.T q)`, the dual box, and nonzero-edge complementarity.
The threshold is 2e-5. Interior upward kinks use an admissible one-sided subgradient;
box endpoints use only the inward derivative (left at the upper bound, right at
the lower bound). At clipping kinks, an O(M) signed prefix/boundary scan covers
every feasible contiguous subinterval of every exact fused block. Frozen and
bound-blocked coordinates split the scan. This includes proper sub-blocks that
can move beside a frozen witness. A descending direction produces a decreasing
restart or an unresolved status. Local trial losses use only the affected
likelihood rows and internal/boundary edges; a full vector is copied only for an
accepted proposal. Additional finite plateau escapes inspect nearby transitions;
these nonlocal proposals do not establish global optimality. An unresolved direction
returns a typed status instead of a smooth convergence claim. This is branch
stationarity, not global optimality. The lambda-zero case has separate scalar-gap
qualification because it is separable.

Memory remains bounded by a constant number of full raw states: current branch,
best branch, preceding lambda and selected candidate. Refitting caches only the
last partition. The fixed path stores scalar diagnostics and hashes, not a raw
state per candidate or witness. The clonal refit profile is computed in O(K).
Input parsing additionally requires storage proportional to the supplied CN rows.
Continuation retains both primal and dual arrays in a chain-fingerprint-bound
`WarmState`. The dual is projected to the new penalty box, and each witness uses
fresh bounds from the original model. The direct quadratic minimizer is independent
of its start; retained primal continuation still initializes the nonlinear branch,
while the dual also supports the iterative attribution reference. Singleton refits reuse qualified pilot
results only after matching the mutation ID and exact likelihood/domain digest.
Unresolved witness lower bounds are rescreened against the final incumbent.

`profile_quadratic_witnesses()` computes prefix and suffix values at one for a
**single common quadratic**, then selects a witness and reconstructs only that
branch. It returns every eligible witness's relative objective and the common
boxed-unary constant separately, and binds the input arrays to a SHA-256 digest.
Its message work is O(M log M), versus M separate quadratic solves. Independent
per-witness solves test these values. The selected branch receives the full gap
and KKT audit, with an additional check against its predicted message value.
The nonlinear production search still enumerates witnesses: different branch
iterates have different curvatures and targets. This shared routine does not
remove that factor from full mixture fits or prove their global optimum.

Version 0.1.1 uses numerical policy `clipp1d_chain_v2` and receipt schema
`clipp1d.run.v2`; the likelihood, chain weights, partition tolerance and statistical
score are unchanged. A successful receipt has top-level `search_status` equal to
`complete` or `incomplete`. Each path record has separate raw/refit statuses and
retains raw diagnostics before attempting the refit. The selected raw objective,
chain-position witness index and witness mutation ID are explicit API/receipt fields.

## Validation boundaries

Tests exercise pinned upstream losses/derivatives/posteriors, interval bounds,
an independent OSQP chain oracle, witness release and screening, clipping,
original feasibility, ID permutation invariance and complete output/failure flows.
Fixture generation is separate from normal tests and requires the exact clean
upstream revision. A success receipt records incomplete witness/path coverage
explicitly; it does not claim that all candidates qualified.

`benchmarks/simulate.py` creates reproducible read-count data and truth.
`benchmark_scaling.py` runs each size in a fresh CPU process and records wall
time, peak process RSS, graph-array bytes, pilot/witness/start/inner/refit time,
scalar reuse counts and numerical coverage. Cases are run sequentially. A per-case
timeout kills only its worker and retains append-only progress; partial timing
counts include completed calls, and timeout RSS is an observed lower bound.
`status=timeout` and `search_status=not_completed` cannot be read as successful
publication. Default cases have 100 and 1,000 mutations, with easy single-candidate
and heterogeneous multiplicity-mixture scenarios.
`benchmark_chain.py` separately measures graph sweeps; complete-graph and chain
QPs with the same first-order algorithm; chain QPs with the direct solver; and
shared versus independently enumerated common-surrogate witness values. The
complete-graph arm is a NumPy reference, not the CliPP2 production solver.
Its arrays are confined to the benchmark. Edge counts, kernel ratios, certified
QP runtimes and full-fit runtimes are distinct evidence.
`compare_clipp2.py` compares supplied final-refit tables with truth on
exactly matched retained IDs. It reports CCF error, ARI (including its true-K-one
flag), selected K and CNA-only multiplicity F1/coverage. Mixed-CN truth has an
explicit integer draw target in the simulator; real mixed-CN evaluation requires
an independently defined target.

Matched full-fit CliPP2 comparisons, large-cohort reconstruction accuracy,
memory scaling and end-to-end speedups are separate qualification work. No GPU
or remote scheduler execution is implied by the new package's CPU tests.
