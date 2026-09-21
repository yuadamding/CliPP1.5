# Shared-surrogate production validation — 2026-09-21

Version 0.2.0 integrates shared witness profiling into production. Every outer
step uses the original boxes, profiles all eligible witnesses of one quadratic,
and reconstructs/certifies the selected solution. Every backtrack rebuilds the
messages. At most four distinct clonal-feasible primal starts are attempted per
penalty. Independent nonlinear witness enumeration is an offline reference.

This changes nonlinear search trajectories. The likelihood, occupied-clonal
constraint, chain weights, partition tolerance, refit, score and numerical
admission thresholds remain unchanged. Policy `clipp1d_chain_v3` and receipt
schema `clipp1d.run.v3` distinguish the new search-completeness semantics.

## Source and verification

The executed package fingerprint is
`2db79fd19d8e05a1e386d133de37b60f19ce0be588717b46076de069c4750146`.
Experiments ran before the publication commit; their full package-file hashes
bind the implemented source independently of the then-current Git HEAD.
The [committed evidence](benchmarks/evidence/shared-0.2.0.json) retains source,
wrapper, input, output and fixture hashes, environment, controls and measurements.

All **229 tests pass** in `ml1` (Python 3.13.2, NumPy 2.2.6, SciPy 1.18.0).
Ruff, compileall and whitespace checks pass. Coverage includes original-box
witness release, fresh backtracking profiles, curvature-scale reuse, primal-only
start deduplication, unchanged-certificate reuse, independent QP oracles, smooth
and clipping descent beside multiple occupied witnesses, failure capture limits,
and full publication flows. Scalar/likelihood and prior qualification regressions
remain in the suite. No admission gate was widened.

The wheel was built, installed into an isolated target and invoked from outside
the repository. Its example fit completed with matching source and table hashes.
Wheel SHA-256:
`28e63bb6d80385056c0ee9aed678b7fb1e4c726510391a4445526ba8e63c1a91`.

## Numerical repairs and actual failure replay

Eight actual failed production QPs from archived `df44e6a` are retained as
[hashed fixtures](benchmarks/fixtures/df44e6a_failed_qps/README.md). All eight now
qualify. They failed the old gap calculation because a rounded cancellation
created a negative box-normal contribution at an interior minimizer. That normal
is mathematically zero and is now evaluated as zero.

Dual reconstruction uses guard digits and distributes a fused block's represented
mean rounding error by curvature. Independent float64 gap/KKT checks still audit
the returned primal and dual; an intentionally incorrect block remains rejected.
Prefix accumulation now integrates the previous clipped message and evaluates the
new unary directly at its threshold, avoiding repeated large-offset subtraction.
For an analytic 1,000-node case with true selected value 0.005, the profile error
is about 2.6e-19. `longdouble` supplies extra guard digits where the platform
supports them; output arrays remain float64.

A reproducible local stress replay uses 1,000 100-node QPs with curvature from
1e-2 to 1e9, caps from 1e-2 to 1e8 and alternating fixed witnesses. The archived
solver had 809 reconstruction failures; the revised solver has zero failed
qualifications. Maximum revised normalized KKT residual is 2.03e-8, below the
unchanged 1e-7 gate. Of 80 additional wide-range common profiles, the archived
solver had 26 qualification failures and the revision has zero. These are our
reproducible stress cases, separate from the reviewer's unavailable fixtures and
from the eight actual production failures; they are not a production failure-rate
estimate. The stress summary binds the final package fingerprint.

Future full-path benchmarks save at most eight actual failed surrogate fixtures
per case, with exact arrays, source/input identity, arithmetic or gate reason,
and original-versus-frozen box scope. Rate-limited progress retains completed
inner work even when a nonlinear start is interrupted. Profile time includes its
selected QP and must not be added to that nested QP time.

## Controlled full paths

Runs use one allowed CPU (CPU 0), six numerical thread environment limits set to
one before NumPy import, separate fresh workers, seed 17 and identical inputs.
The host is shared; affinity and thread controls do not establish exclusivity.
The reference uses the revised numerical solver, primal deduplication and the
old independent witness-search policy. Each case below has a 120-second limit.
Finished raw attempts include unresolved attempts; an interrupted next attempt
is recorded separately in the evidence.

| Case | Shared workflow | Finished raw attempts | Raw unresolved | Enumeration reference | Finished raw attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy, 100 | 1.37 s; complete | 26/26 | 0 | 36.47 s; complete | 26/26 |
| Mixture, 100 | 11.93 s; complete | 26/26 | 0 | timeout | 6/26 |
| Easy, 1,000 | 8.18 s; complete | 26/26 | 0 | timeout | 2/26 |
| Mixture, 1,000 | timeout | 15/26 | 2 | timeout | 1/26 |

Easy-100 has identical selected refit, labels and score in both policies. Its
observed wall-time ratio is 26.7; the changed search policy is part of this
comparison. A timed-out reference supplies no completed full-fit runtime ratio.
The three completed shared cases have no unresolved starts.

A **separate** mixture-1,000 run with a 240-second allowance finishes all 26
attempts in **170.88 s** and publishes a qualified selected fit. Its search is
explicitly **incomplete**: two penalties are unresolved and six of 97 starts hit
the unchanged outer-iteration limit. All 3,159 completed common QPs qualify;
there are no captured inner arithmetic/gap/KKT failures. Completed-start timing
attributes 89.70 s to shared inner solves and 63.05 s to feasible-direction/kink
audits. Completing the path does not erase its retained search failures.

Curvature reuse was introduced after a bounded profiling sample found repeated
backtracking dominating a slow start. The next common iteration tries half the
previously accepted inflation, floored at one; a kink restart resets it. Every
surrogate and acceptance inequality is rebuilt. The historical development run
without this reuse is retained locally and is not the source of the table above.

## Search-policy and selected-estimator comparisons

Sixty paired small-instance fits cover sizes 3 and 6, seeds 11/17/23, convex and
mixture likelihoods, and five penalty factors. Both policies produce qualified
candidates in all 60 pairs. Each policy has its own continuation state.

All 30 convex pairs agree in attained objective within 1.14e-13 and have identical
partitions. Mixture trajectories can differ: the largest shared-objective excess
is **0.1335078033** (380.5074670231 versus 380.3739592198, three mutations, seed 17,
penalty factor 0.01). One mixture pair has a different partition; the maximum
coordinate difference across the study is 0.44894. Neither finite-start policy is
a nonconvex global-optimization oracle. The reference is preserved to expose
these differences rather than asserting unchanged nonlinear results.

After checking successful publication and all table hashes, final-refit truth
metrics are:

| Shared fit | Selected K | ARI | CCF RMSE | CNA-only macro-F1 | CNA calls / eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy, 100 | 3 | 0.97015 | 0.03681 | n/a | 0/0 |
| Mixture, 100 | 5 | 0.70797 | 0.11382 | 0.71980 | 66/66 |
| Easy, 1,000 | 3 | 0.96743 | 0.03869 | n/a | 0/0 |
| Mixture, 1,000; extended incomplete search | 5 | 0.65868 | 0.12864 | 0.65188 | 666/666 |

CNA eligibility is `(major != 1) or (minor != 1)`. Mixed-CN truth explicitly means
the simulator's generating integer candidate. Micro/weighted/per-class F1, CCC,
coverage and clonal fractions are retained in the evidence. These selected
synthetic cases do not establish cohort accuracy or equivalence to CliPP2.

## Fixed QPs, kernels and memory

Three timing samples cover weak uniform, moderate unequal and strong unequal
caps, at 100 and 1,000 nodes. Both equal mean per-edge strength and matched total
cap strength are evaluated; these are different normalization questions and
neither makes the two graph objectives identical. All 36 chain-first-order and
36 direct-chain samples qualify. The complete-graph reference qualifies in 27
samples and reaches its three-second limit in nine. Its graph solver is a NumPy
attribution reference, not the CliPP2 production solver.

Selected 1,000-node per-edge medians illustrate the dependence on penalty regime:

| Regime | Complete first-order | Chain first-order | Direct chain TV |
| --- | ---: | ---: | ---: |
| Weak uniform | 0.259 s | 0.00449 s | 0.0221 s |
| Moderate unequal | 1.792 s | 0.00711 s | 0.0157 s |
| Strong unequal | three-second timeout | 0.1818 s | 0.0117 s |

Thus direct TV is not universally faster than a rapidly converging first-order
chain solve. The production improvement primarily removes repeated nonlinear
witness optimization. For a single common 1,000-node QP, shared profiling takes
0.03261 s median (0.03247–0.03276) versus 13.84988 s for all independent witnesses
(13.84888–13.85691). Every witness qualifies; maximum value disagreement is
2.85e-14. This measures one common surrogate, not a full-fit speedup.

Five kernel samples retain dispersion. Complete-graph medians/ranges are
0.00448 s (0.00439–0.00477) at 1,000; 0.11410 s (0.08810–0.20097) at 2,000; and
1.50436 s (1.22657–1.60108) at 4,000. The large size-dependent jump persists under
these process controls; its cause is not established. Extreme kernel ratios are
not used as a representative fitting-speed claim.

Memory is measured separately from timing. At 1,000 nodes and moderate unequal
per-edge caps, direct TV has 192,223 bytes of newly traced peak working allocation
(including heaps, knot dictionaries, reconstruction and audit work), plus 39,992
bytes of preexisting QP inputs. The corresponding complete first-order working
peak is 24,102,988 bytes, with graph-index and input arrays reported separately.
Tracemalloc can miss unregistered native allocations. RSS is a process-lifetime
bound, not an isolated solver peak; earlier dense-kernel allocations remain in
its high-water mark.

The frozen chain's 23,992 bytes are only order/inverse-order/weights. At 1,000
mutations, compiled model arrays occupy 49,000 bytes in the easy case and 83,000
in the mixture case. Fresh full-fit observed RSS is about 74–77 MiB, covering the
interpreter, libraries, model and working state. These distinct scopes must not
be presented as one total-memory estimate.

## Reproduction

Use fresh output directories and `ml1` in this workspace. Commands below also
work with `python` in an environment containing the declared test dependencies.

```bash
conda run -n ml1 python -m pytest -q
conda run -n ml1 ruff check src tests benchmarks
conda run -n ml1 python -m compileall -q src tests benchmarks
conda run -n ml1 python benchmarks/replay_robustness.py --outdir results/replay
conda run -n ml1 python benchmarks/benchmark_scaling.py --outdir results/shared --sizes 100 1000 --timeout-seconds 120
conda run -n ml1 python benchmarks/benchmark_scaling.py --outdir results/reference --sizes 100 1000 --timeout-seconds 120 --reference-enumeration
conda run -n ml1 python benchmarks/benchmark_scaling.py --outdir results/extended --sizes 1000 --scenarios mixture --timeout-seconds 240
conda run -n ml1 python benchmarks/compare_search.py --outdir results/search --sizes 3 6 --seeds 11 17 23 --penalty-factors 0 .01 .1 1 100
conda run -n ml1 python benchmarks/benchmark_chain.py --outdir results/chain --sizes 100 1000 --kernel-sizes 100 1000 2000 4000 --profile-sizes 100 1000 --repetitions 3 --kernel-repetitions 5 --first-order-seconds 3 --profile-seconds 30
```

The replay tool optionally accepts an archived baseline package directory.
Historical 0.1.1 evidence remains in [its original report](VALIDATION_REVISION.md).
No original CliPP2 fit, GPU benchmark or remote scheduler work was performed.
