> Historical chain reference (through 0.4.1). The current production model and
> output contract are documented in [CUDA_FRAMEWORK.md](CUDA_FRAMEWORK.md).
> References to production below describe the historical revision.

# Implementation and qualification

`io → model → scalar → chain → tv/solver → fitting → selection → api/report` is the
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
min 0.5 sum h_i (x_i-U_i)^2 + sum caps_i |x_{i+1}-x_i|,
original_lower <= x <= original_upper.
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

Production imposes only the original coordinate boxes. Historical constrained
helpers retain fixed-witness decomposition solely for offline regression tests.

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
Interior boxed minimizers have exactly zero normal; computing it by subtracting
rounded large gradients can manufacture a negative gap term. Only active-bound
normals are evaluated. Message and dual-reconstruction accumulators use
`numpy.longdouble` guard digits where available. A rounded fused-block mean's normal-equation error
is distributed by curvature during dual reconstruction; independent float64 gap
and KKT checks still audit the represented solution at unchanged tolerances.
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
separately; the latter does not confer an inner-optimality claim. Unchanged
breakpoint/fusion proposals reuse the direct gap/KKT certificate. Changed proposals
receive fresh audits as required; the accepted trial's likelihood is cached across
all outer acceptance inequalities.

The outer audit checks the componentwise box-normal residual normalized by
`1+abs(gradient)+abs(D.T q)`, the dual box, and nonzero-edge complementarity.
The threshold is 2e-5. Interior upward kinks use an admissible one-sided subgradient;
box endpoints use only the inward derivative (left at the upper bound, right at
the lower bound). One immutable audit context binds the exact accepted vector and
model, sharing losses, posterior, gradient, one-sided derivatives, clipping masks
and fused boundaries across stationarity and interval audits. Stale contexts are
rejected. Local proposals reuse slices of its old losses. Loss-only calls share
the same candidate log-kernel and logsumexp arithmetic as full evaluation,
without constructing an unused posterior.

Finite proposals retain their original interval, breakpoint and offset order.
Likelihood evaluations batch at most 64 proposals and 4,096 mutation rows,
including singleton proposals from different mutations. One interval longer
than this row limit runs alone, so allocation remains O(max(M,4096) Cmax).
Per-interval reductions keep the original contiguous scalar summation order;
the first qualifying proposal is accepted, even if a later prefetched proposal
has a larger improvement. Directional backtracking starts with one proposal
before increasing its batch size.

The immutable scientific audit snapshot owns a separate bounded memo of at
most 4,096 likelihood deltas, keyed by exact interval and float64 value bytes.
Feasibility and TV penalties are recomputed for each proposed direction. This memo neither
changes the stored arrays nor survives a changed model or primal vector.
Speculatively evaluated proposals, considered candidates and cache hits are
different workload counts. Prefetch can add overhead to immediate acceptance;
component and full-fit measurements must report that tradeoff separately.

At clipping kinks, a signed prefix/boundary scan covers
every feasible contiguous subinterval of every exact fused block. Frozen and
bound-blocked coordinates split the scan. This includes proper sub-blocks that
can move while other coordinates remain at their original bounds. A descending direction produces a decreasing
restart or an unresolved status. Local trial losses use only the affected
likelihood rows and internal/boundary edges; a full vector is copied only for an
accepted proposal. Additional finite plateau escapes inspect nearby transitions;
these nonlocal proposals do not establish global optimality. An unresolved direction
returns a typed status instead of a smooth convergence claim. No clonal anchor
is frozen during these production audits.
For at least 64 float64 nodes, feasible exact-fused runs are batched by length;
each independent prefix sum follows the original sequential addition order.
Strict prefix minima keep the earliest start; equal final merits keep negative
sign and then the earliest stop. No numerical tie tolerance is introduced.
Prefix arithmetic and storage are O(M); bucket grouping has a conservative
O(M log M) work bound. Small or unsupported arithmetic uses the same original
scalar scan. Both paths retain identical coverage and acceptance thresholds.
These are numerical local qualifications, not global optimality. The lambda-zero case has separate scalar-gap
qualification because it is separable.

Memory remains bounded by a constant number of full raw states: current branch,
best branch, preceding lambda and selected candidate. Refitting caches only the
last partition. The fixed path stores scalar diagnostics and hashes, not a raw
state per candidate or witness. Each block is refitted independently; no exact-one override is applied.
Input parsing additionally requires storage proportional to the supplied CN rows.
Production continuation retains only the primal vector in a chain-bound
`PrimalWarmState`. Legacy `WarmState` input remains valid, but its dual neither
affects direct solves nor creates a distinct start. The offline fixed-witness
reference deduplicates primals after imposing each witness; production deduplicates
after projection onto the original coordinate boxes. Singleton refits reuse qualified pilot results
only after matching the mutation ID and exact likelihood/domain digest.

`fitting.fit_fixed_lambda()` calls `solver.solve_unconstrained()` for each
positive-penalty start. Each backtracking attempt calls the direct bounded-TV
solver with the original boxes, and every returned solution receives the gap/KKT
audit. After a successful step, the next iteration starts at half the accepted
curvature inflation (floored at one), then backtracks normally. A kink restart
resets inflation. Only this scalar scale is reused, never stale surrogate messages.
`largest_curvature_scale` records the largest accepted inflation.

The lambda-zero case uses the unmodified pilots and qualifies their scalar gaps
against the supplied policy. Its stationarity result is reported independently.
`RawFit.witness` is always `None` in production. Finite, correctly shaped initial
vectors are validated before clipping, so infinities cannot become valid bounds.

Version 0.4.1 retains policy `clipp1d_chain_v5`, search policy
`unconstrained_chain_multistart_v1` and uses receipt schema `clipp1d.run.v6`.
`search_complete` records qualification of every planned start; path and direct
proposal completion are separate. Compatibility profile/witness counters are zero.
The legacy constrained implementations in `clonal.py`, `solve_profiled()`,
`solve_branch()` and `profile_quadratic_witnesses()` remain covered by offline
regression tests; production does not import `clonal.py`. Historical validation
reports apply to their named revisions and do not qualify this changed model.

Successful publication first validates the independent raw reference, candidate
family, chain/model/partition identities, selected certificate, and agreement
among public labels, centers and per-mutation refits. Direct proposals retain
explicit seed and raw-parent lineage. A direct winner cannot inherit the raw
reference's certificate. JSON serialization is preflighted before writing fit
tables, and complete JSON markers are published atomically without overwriting
existing files. Only a successful marker with matching hashes establishes a fit.

Public label zero designates the block minimizing L2 distance to CCF one, with
chain-position tie breaks; remaining labels descend by fitted CCF. The designation
flag is one only for public label zero. `labeling.py` performs this post-fit operation
without changing centers or numerical certificates; raw witness fields remain null.
Evaluation distinguishes constrained v4, undesignated v5 and post-fit-designated v6
outputs. Current primary sMF counts retained mutations outside label zero.

## Validation boundaries

Tests exercise pinned upstream losses/derivatives/posteriors, interval bounds,
an independent OSQP chain oracle, witness release and screening, clipping,
original feasibility, ID permutation invariance and complete output/failure flows.
Fixture generation is separate from normal tests and requires the exact clean
upstream revision. A success receipt records incomplete planned-start/path coverage
explicitly; it does not claim that all candidates qualified.

`benchmarks/simulate.py` creates reproducible read-count data and truth.
`benchmark_scaling.py` runs each size in a fresh CPU process and records wall
time, peak process RSS, graph-array bytes, pilot/profile/start/inner/refit time,
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
Its arrays are confined to the benchmark. Both scripts bind allowed CPU affinity
and thread controls before loading NumPy, record them, and make no exclusive-host
claim. Kernel/QP timings retain samples and dispersion. Fixed QPs cover weak,
moderate and strong penalties, unequal caps, and both equal per-edge strength and
matched-total-cap normalization. Separate traced-memory passes include heap,
dictionary and reconstruction allocations; chain-array bytes and lifetime process
RSS are reported with their distinct scopes. Scaling workers retain up to eight
failed QP arrays with source identity and precise gate/arithmetic reasons for
replay, separate from compact fit receipts. Edge counts, kernel ratios, certified
QP runtimes and full-fit runtimes are distinct evidence.
`compare_clipp2.py` compares supplied final-refit tables with truth on
exactly matched retained IDs. It reports CCF error, ARI (including its true-K-one
flag), selected K and CNA-only multiplicity F1/coverage. Mixed-CN truth has an
explicit integer draw target in the simulator; real mixed-CN evaluation requires
an independently defined target.

Matched full-fit CliPP2 comparisons, large-cohort reconstruction accuracy,
memory scaling and end-to-end speedups are separate qualification work. No GPU
or remote scheduler execution is implied by the new package's CPU tests.
