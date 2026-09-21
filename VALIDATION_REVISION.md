# Numerical and chain-solver revision validation

This revision repairs the two qualification failures reported against initial
commit `4e9fcf62eb57908af3e15c45ddbd7bf035e10ad0`, then replaces production inner
iterations with a bounded weighted chain-TV solver. The likelihood, original
CCF boxes, adaptive weight rule, partition tolerance and selection score remain
unchanged. The package version is 0.1.1, numerical policy is `clipp1d_chain_v2`,
and run receipts use `clipp1d.run.v2`.

The [initial validation](VALIDATION.md) is retained as historical evidence. Its
passing tests did not cover the two numerical defects and cannot establish the
repaired solver's validity.

Snapshot: **2026-09-21 CDT**, local Intel Core i9-13900KS, Python 3.13.2,
NumPy 2.2.6, SciPy 1.18.0 in conda `ml1`.
Package-source SHA-256:
`e3478d0f33b59d02f5cf14f0ab103171073c0ce085c1966e41fa02645643da72`.
Validation precedes the revision commit; the complete per-file source identity
is recorded in the [committed evidence summary](benchmarks/evidence/revision-0.1.1.json).

## Correctness checks

The tests cover the supplied four-mutation endpoint reproduction; feasible
left/right derivatives at box endpoints; exhaustive signed-subinterval oracles;
stable gaps under frozen targets of 1, 1e6 and 1e9; independent primal/dual gap
arithmetic; and the OSQP epigraph oracle for unequal curvatures, unequal edge
weights, differing boxes and frozen interior coordinates.

Direct-solver checks additionally cover both witness endpoints and an interior
witness, arbitrary frozen boxes, zero edge penalties, singleton inputs, large
edge caps, failure on an invalid certificate, and knot/storage bounds at 100,
1,000 and 4,000 nodes. Shared prefix/suffix witness values are checked against
independent solves for every eligible witness of the same surrogate. A changed
surrogate receives a different identity.

Scalar/reuse tests compare exact fits with the general scalar engine, include
plateau ties, verify lazy breakpoint evaluation and source-bound singleton cache
reuse, compare local and full objective changes, and check warm dual projection
and witness release. Output tests preserve a successfully qualified raw branch
when a subsequent refit fails, with separate raw/refit/search statuses.

**122 tests pass**, including the independent OSQP checks. Ruff and compilation
pass. An additional 2,000 deterministic randomized bounded-QP stress cases all
qualified: maximum gap `1.04e-26`, maximum normalized KKT residual `2.00e-14`.
These are numerical checks, not directed-rounding proofs.

The supplied endpoint case now produces a decreasing restart from objective
`35.25563860203106` to `34.784964101451905`, and returns an **unresolved** branch,
not a qualified all-one vector. Its later QP dual reconstruction remains
unresolved; the reproduction is not claimed to be a completed fit. All three
frozen-target cases return exactly `(1,1,1)` with zero stable gap, zero KKT
residual and zero independently evaluated free-coordinate objective error.

The 0.1.1 wheel was built, installed into an isolated target, and its CLI fitted
the three-mutation example with complete search coverage. Table hashes and the
wheel's loaded source fingerprint match the receipt. Wheel SHA-256:
`68c764f0e9d69c8de174647870d764f192ae30256c6fad0458a43e920cf110c2`.

## Measurement boundaries

All measurements use local CPU float64 in conda `ml1`. The bounded runtime
scaling driver runs cases sequentially in fresh processes, using the same
seed-17 synthetic inputs for each solver revision. All intended penalty settings
and numerical gates are preserved. Each case has a 120-second process limit.
Timeout receipts preserve completed raw/refit stages and start-failure counts;
those receipts are not successful full-fit publications.

| Scenario | M | Process seconds | Finished raw penalties / initial 26 | Search | Failed completed starts |
| --- | ---: | ---: | ---: | --- | ---: |
| Easy | 8 | 1.27 | 26/26 | Complete | 1 |
| Easy | 16 | 3.07 | 26/26 | Complete | 0 |
| Easy | 100 | 32.52 | 26/26 | Complete | 0 |
| Mixture | 100 | 120.00 limit | 7/26 | Not completed | 2 |
| Easy | 1,000 | 120.00 limit | 2/26 | Not completed | 318 |
| Mixture | 1,000 | 120.00 limit | 1/26 | Not completed | 387 |

Failed starts are inner-qualification failures, not necessarily unresolved
witnesses: another start can qualify the same witness. The easy M=8 case illustrates
that distinction. Timeout counts cover completed calls through the last progress
snapshot; an interrupted call is not counted as a success or a numerical failure.
The corrected iterative baseline timed out at 120 seconds for all four larger
cases, after respectively 5, 2, 1 and 1 completed raw penalties (easy 100,
mixture 100, easy 1,000, mixture 1,000). Its source snapshot and receipts remain
under `results/review-iterative-source` and `results/review-scaling-large`.

The final easy M=100 fit selects K=3, final-refit CCF RMSE `0.03681` and ARI
`0.97015` against three generating centers. M=8 selects K=3 with ARI 1; M=16
still selects K=4 with ARI `0.91652`. These three successful easy cases are a
selected synthetic subset. The timed-out cases have no final-estimator accuracy
result. Easy cases contain no CNA-eligible mutations, so CNA-only multiplicity F1
is undefined, not reported as perfect.

Chain-array bytes (order, inverse order, weights) are 2,392 at M=100 and 23,992 at
M=1,000. Observed process RSS is 74,124–77,256 KiB in these larger cases; timeout
RSS is a lower bound through the last completed stage, not a complete memory
qualification. Numerical storage is linear by the implementation's knot/array
bounds; full witness enumeration still adds substantial work.

Graph kernels, fixed QPs, common-surrogate profiling and full fits are separate
experiments. The complete-graph benchmark is a NumPy attribution reference using
the same first-order update as its chain arm; it is not CliPP2's production
solver. Changing topology changes the TV objective. Shared message profiling
applies only to identical surrogate arrays and does not eliminate witness
enumeration from the nonlinear production search.

For M=1,000, all three fixed-QP arms qualified at the same gap/KKT thresholds:
complete graph with the common first-order solver took `1.376 s` (290 iterations),
chain with that solver took `0.02717 s` (720 iterations), and direct chain TV took
`0.01497 s` (one forward/backward solution). The two chain solutions differed by
at most `8.75e-10` in CCF. Their objectives match; the complete graph has a
different fusion objective. Kernel-only measurements are retained separately in
the evidence summary and must not be quoted as production fitting speedups.

At M=1,000, shared common-surrogate witness profiling took `0.03234 s`; enumerating
all 1,000 direct branch solves took `7.879 s`. All branches qualified and the
maximum witness-value disagreement was `2.27e-13`. This gain applies to the one
common quadratic, not the full nonlinear branch search.

Isolated scalar comparisons with the initial scalar module at M=1,000:

| Work | Initial seconds | Revised seconds | Agreement |
| --- | ---: | ---: | --- |
| Easy pilots | 7.254 | 0.0816 | Maximum loss difference `5.68e-14` |
| Mixture pilots | 12.688 | 10.335 | Maximum loss difference `5.68e-14` |
| Heterogeneous-slope zero-alt block | 0.2953 | 0.0130 | Equal attained loss; both qualified |
| 100 single-coordinate proposals | 0.00823 | 0.00592 | Maximum delta difference `1.16e-12` |

The lazy scalar case used 72 likelihood and 34 bound evaluations at both M=100
and M=1,000. Local proposal evaluation did **not** improve elapsed time at M=100
(`0.00587 s` versus `0.00531 s` full evaluation); its measured benefit is modest
at M=1,000, though it eliminates full-vector work for each rejected local move.
Pilot CCF differences from the initial approximate scalar engine were at most
`1.85e-8`. Timings are observations on this host, not universal speed factors.

The initial direct-solver experiment is retained under
`results/review-direct-small` and `results/review-direct-large`. It revealed that
unconditional roundoff widening of dual intervals could reject an optimal primal
solution under very large caps. The final implementation repairs only tiny empty
intersections; the independent gap and KKT gates remain unchanged. Final evidence
is recorded separately under `results/revision-final-*`.

Reproduction commands:

```bash
python -m pytest -q
ruff check src tests benchmarks
python -m compileall -q src tests benchmarks
python benchmarks/benchmark_scaling.py --sizes 8 16 --scenarios easy --outdir results/small-new
python benchmarks/benchmark_scaling.py --sizes 100 1000 --scenarios easy mixture --timeout-seconds 120 --outdir results/large-new
python benchmarks/benchmark_chain.py --outdir results/chain-new
python benchmarks/benchmark_reuse.py --outdir results/reuse-new
```

`benchmark_reuse.py` isolates the initial commit's scalar module against the
current unchanged likelihood engine. This is a source-bound scalar comparison,
not numerical qualification of the initial full fitting pipeline.

No matched CliPP2 full-fit, CUDA, real-data cohort or statistical accuracy claim
is made. Incomplete or timed-out searches remain a practical limitation even
when inner QPs and selected raw candidates are numerically qualified.
