# Finite proposal batching and stagnation diagnosis — September 21, 2026

Version 0.2.2 retains numerical policy `clipp1d_chain_v3`: objective, boxes,
graph, starts, witness profiling, iteration limits, partition/refit/score
settings and acceptance gates are unchanged. The executed package fingerprint is
`521e01efbd080b3ef7660661779d507afc16f4f35206800332aeefaa3a2edc16`.
The [committed evidence](benchmarks/evidence/finite-proposals-0.2.2.json) binds
the completed runs and their source, input, output, wrapper and environment
identities. The complete frozen source snapshot includes the new proposal module; its
receipt binds tracked and untracked files independently of the publication
commit. Baseline is unchanged `72ba3ed`, fingerprint
`30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055`.

## Implementation and decision parity

`model.loss` and `loss_at_rows` share the full evaluator's candidate log-kernel
and logsumexp arithmetic while omitting the unused posterior. Finite audit
proposals retain their original interval/breakpoint/offset order and first
acceptance rule. Batches are bounded by 64 proposals and 4,096 mutation rows;
one oversized interval runs alone. Per-interval sums preserve the scalar
reduction order. Singleton proposals can batch across mutations.

An audit's immutable model/primal/likelihood snapshot owns a separate memo
limited to 4,096 exact interval/value likelihood deltas. Anchor feasibility
and incident TV terms are not cached. A new context gets a new memo; identity
checks still reject stale contexts. Directional backtracking evaluates one
proposal first to avoid speculative work on its common immediate-acceptance
path. Later proposals may be evaluated speculatively, but consideration and
acceptance remain ordered.

The differential inventory comprises **240 distinct anchored audits**, including
the retained 200, smooth and clipping/plateau cases, multiple witnesses,
Armijo boundaries, early/late acceptance, complete rejection, fused intervals,
and clipped duplicate endpoints. Every consumed delta is bitwise identical;
generated prefixes, considered candidate order, first accepted restart vectors,
stationarity and qualification decisions match. The main 234-case receipt and
a supplementary run containing six new distinct duplicate cases retain their
own exact wrapper copies and hashes. Overlapping supplementary cases are not
counted again in the 240 total.

Five alternating paired component repetitions follow one unmeasured warmup per
arm, on CPU 2 with one numerical-library thread. Each timing starts with a
fresh audit context prepared outside the timer. Rejection-heavy, late-acceptance
and fused-block workloads in the main inventory improve approximately
**4.0–4.2 times**. Immediate
finite acceptance instead adds approximately **0.43–0.53 milliseconds** from
prefetch; this tradeoff is retained in the evidence.

For the 1,000-node rejection workload, 6,000 scalar likelihood calls become
95 batched calls. Across the main audit inventory, physical logsumexp calls
fall from 122,643 to 1,973. Physical kernel work, considered proposals and
memo hits are distinct counts. The measured maximum batch has 64 proposals
and 4,056 mutation rows; the cache reaches its 4,096-entry cap. Cache eviction
can limit cross-anchor reuse.

Bounded batching trades memory for fewer calls: tracemalloc peak for the
4,000-node all-rejected finite audit rises from 1.98 to 3.39 MB, excluding the
preexisting model/context. At 1,000 nodes it rises from 0.483 to 2.75 MB. These
are component traced allocation peaks, not full-process RSS reductions.

## Repeated full fits

Three paired mixture-1,000 experiments use the same seed-17 input, one CPU
(CPU 0), one numerical-library thread, fresh worker processes and a 300-second
cap. Arm order is baseline/revised, revised/baseline, baseline/revised.

| Arm | Repeat 1 | Repeat 2 | Repeat 3 | Median | Sample standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen 72ba3ed | 140.959 s | 154.783 s | 144.810 s | 144.810 s | 7.135 s |
| Revised 0.2.2 | 98.652 s | 97.249 s | 97.048 s | 97.249 s | 0.874 s |

Median elapsed time falls **32.8%**, corresponding to approximately **1.49x**
throughput. Observed ranges are 140.959–154.783 and 97.048–98.652 seconds.
These are three shared-host measurements per arm, not a population confidence
interval or a universal speed guarantee.

Every pair has identical input/model/policy/chain identities, all three output
table hashes, selected lambda, raw/refitted CCF vectors, labels, raw objective
and score. Recorded per-penalty statuses, starts, witnesses, profiles, iterations
and objective values match. All 26 penalties finish; the six outer-limit starts
and two unresolved penalties remain unchanged. Selected fits are qualified;
the full planned searches remain incomplete. Batching lowers evaluation cost
without dropping unresolved work or changing convergence behavior.

Separate paired easy-100, mixture-100 and easy-1,000 runs also preserve all
published tables and tracked search decisions. Those three searches complete.
An unexpectedly slow initial short-case run prompted two additional alternating
pairs; all observations are retained:

| Case | Baseline median (range) | Revised median (range) | Repeats per arm |
| --- | ---: | ---: | ---: |
| Easy 100 | 1.215 s (1.166–1.216) | 1.217 s (1.217–2.268) | 3 |
| Mixture 100 | 11.029 s (10.979–11.140) | 7.474 s (7.324–11.882) | 3 |
| Easy 1,000 | 6.073 s | 6.072 s | 1 |

Easy-case medians are essentially unchanged. The slow first revised small-case
observations are not discarded or attributed definitively to a measured system
cause. Across all four case types, **20 full fits / 10 paired comparisons**
preserve the recorded scientific outcomes. The easy-1,000 single pair provides
regression evidence, not a repeated-runtime estimate.

## Why the six retained starts stagnate

The diagnostic reads the six retained terminal states and their original
input/chain/source receipts. Unchanged `72ba3ed` replays reproduce their final
primal and dual arrays bitwise. All diagnostic work uses CPU 1 and one thread.
No saved production result is overwritten.

At a fixed primal, a forward chain reachability pass seeks duals satisfying
the original observed-gradient residual, box normals, exact nonzero-jump
complementarity and edge caps. No quadratic block-gradient correction is
applied to the likelihood gradient. Bisection estimates the smallest attainable
residual; independent sparse linear-programming feasibility checks corroborate
the result at the existing gate.

One of the six states, p01-s02, passes without moving the primal: the retained
dual residual is approximately **2.24716e-4**, while the reconstructed dual
gives **1.08940e-5**, below the unchanged **2e-5** gate. All selected/extreme
witness direction audits also pass. With exact nonzero-jump complementarity,
the other five retain minimum residuals approximately **2.84e-5 to 9.15e-5**
and fail. Their reachability obstruction occurs before every occupied clonal
anchor, so changing that anchor does not remove this obstruction. This
exact-complementarity diagnostic alone does not prove failure for every dual
allowed by the production gate's numerical edge tolerance. HiGHS' own
feasible-case dual exceeds the gate by about 4.2e-14;
it is corroborating numerical evidence, not the accepted certificate. The
separate reconstructed dual passes strictly.

The extended diagnostic distinguishes three constraint sets at the same six
unchanged primals:

| Dual constraint set | States passing the original residual and direction audits |
| --- | ---: |
| Exact dual boxes and exact nonzero-jump complementarity | 1/6 |
| Exact dual boxes, existing numerical jump-residual allowance | 4/6 |
| Existing numerical jump and dual-box residual allowances | 6/6 |

The last row changes no production tolerance, but its duals can lie slightly
outside exact caps. It shows that all six fixed primals can satisfy the
implemented first-order gate with different duals. It does **not** establish
exact dual feasibility, a Fenchel lower-bound certificate, or exact primal
stationarity. No such dual replacement is installed in production by this
efficiency revision.

Fixed-gate feasibility is cheaper than the 60-step diagnostic bisection:
median times over seven samples are approximately 0.50–0.56 ms for the five
exact-complementarity infeasible prefixes and 2.85 ms for the feasible
full-chain reconstruction, excluding the already prepared
likelihood context. These are diagnostic timings, not a production speedup.

Across **240** tail proposals, all **120 rejected lower-inflation trials**
fail only the **surrogate-descent** inequality after clipping-breakpoint
postprocessing. Likelihood majorization and the actual-objective check pass.
No substantive positive per-coordinate majorization excess remains after
guarded log-ratio evaluation. Maximum per-coordinate guarded positive excess
is at most about 6.7e-19;
the difference between ordinary loss subtraction and guarded kernel/clipping
arithmetic is at most 2.44e-11, far smaller than the roughly 3e-7
surrogate-descent failure.

The common QP itself is accurate and surrogate-descending. The failure arises when it
moves the zero-alternate-read mutation `m000831` away from its multiplicity-2
lower clipping breakpoint at approximately 1.75e-6. The ordinary gradient at
that kink is zero, while its right derivative is **40.00004000004**. It has
zero alternate and 140 reference reads. Presnap positive majorization excess
is concentrated at this coordinate, approximately 3.387 or 3.183 depending
on the penalty. For example, a lower-inflation QP moves it to approximately
0.08510 and reduces the surrogate by 1.25971, but increases the actual
objective by 1.92372. Breakpoint snapping restores **only this mutation**,
leaving the other QP coordinates moved. That snapped vector decreases the
actual objective slightly while increasing the surrogate by about 3.06e-7.
The original outer loop correctly rejects it and doubles curvature for the
entire chain. A direct solve at the previous witness reproduces the same
quadratic solution, excluding witness-profile selection error in this check.

Thus the observed terminal damping is driven by a clipping-kink/postprocessing
interaction. The post-snap coordinates do not justify coordinatewise curvature
inflation, and the evidence does not support increasing the interval step cap,
iteration budget or numerical tolerances.

## One separate penalized block-polishing experiment

The offline prototype optimizes the original likelihood plus the two incident
TV edges along each sweep-start exact fused block. It respects the original
boxes and an occupied clonal witness, accepts only actual penalized-objective
decrease, and applies the original full stationarity and interval audits.
It does not use the final unpenalized partition refit, permanently fix the
partition, or declare qualification from block optimization alone. Scalar
searches generate proposals; they do not confer a global nonlinear certificate.

One sweep gives passing exact-dual and full direction audits in **four of six**
states, with small primal changes around 2e-7–1e-6. One of these already passed
the exact-dual audit before polishing. A separately declared fresh 150-iteration
run of the unchanged solver qualifies only p02-s01 and p02-s02, each in one
iteration; p01-s01 and p01-s02 again exhaust the budget with the retained
surrogate dual. The two s03 states instead enter materially different
lower-objective regions, with decreases approximately 220.88 and 17.21, but
fail the full audits and the fresh solver qualification.

This is a mixed experimental result, not a universal stagnation repair or
qualification of a new production policy. It reinforces the need to address
dual handling and clipping-aware progress together. The released numerical
trajectory remains unchanged; the prototype and its failed attempts remain
offline, with separately bound evidence and budgets.

## Other qualification

All **358 tests pass** in `ml1`; Ruff, source/test/benchmark compilation and
whitespace checks pass. Focused regressions cover loss-only equality, proposal
order and first acceptance, exact duplicates, stale-context/cache isolation,
bounded allocation, and observed-dual interval/gate conventions.

The existing component regression also passes against frozen `72ba3ed`:
280 qualified quadratic profiles in both versions, 3,000 exact signed scans,
and 200 anchored audits with zero mismatches. This is separate from the new
finite-proposal inventory and from full-fit runtime measurements.

All four matched SimClone replays preserve all **12 public TSVs bytewise**,
along with input/model/policy/chain identities and selected decisions. This
efficiency change therefore preserves the discrepancies diagnosed in
[the separate SimClone investigation](VALIDATION_SIMCLONE.md).

The 0.2.2 wheel was built from the frozen source, installed into an isolated
target, and its example CLI invoked outside the checkout. Loaded package
fingerprint and public table hashes agree with the receipt. Wheel SHA-256:
`7808e41bd29cdbf5a8eae7bd9aca5d853d28ad836e08f7bd575d0d26cdbef11e`.

## Reproduction and evidence locations

Use `ml1` and fresh output directories. Numerical controls bind CPU affinity,
thread environment, interpreter and full source fingerprints. The host is
shared; affinity does not imply exclusive hardware.

```bash
mkdir -p results/72ba3ed-source
git archive 72ba3ed src/clipp1d | tar -x -C results/72ba3ed-source --strip-components=1
conda run --no-capture-output -n ml1 python benchmarks/compare_finite_proposals.py --help
conda run --no-capture-output -n ml1 python benchmarks/diagnose_observed_dual.py --help
conda run --no-capture-output -n ml1 python benchmarks/diagnose_rejections.py --help
conda run --no-capture-output -n ml1 python benchmarks/prototype_penalized_polish.py --help
```

Detailed local receipts are under `results/finite-proposals-20260921-v1/`,
`results/finite-proposal-components-v1/`, and
`results/finite-proposal-duplicates-v1/`. Archived unresolved-state fixtures
remain under `results/review-ff5b5c3-diagnosis/`. Each diagnostic wrapper checks
the identities it consumes and records its own hash. The full-fit repeat
controller preserves its initial wrapper-only failure before any fit was
started by that attempt; the successful controller has a separate log.
