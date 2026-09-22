# Reconstruction and audit reuse — 2026-09-21

Version 0.2.1 retains the common-surrogate multistart architecture and numerical
policy `clipp1d_chain_v3`. It reuses the two profile passes' reconstruction
thresholds, shares one immutable likelihood context across each accepted vector's
audits, and batches signed-interval prefix sums. Direct solves and profiles use
the same polishing, dual recovery, gap and KKT finalizer. No extra messages are
built after witness selection.

The likelihood, boxes, chain, starts, witness rule, partition/refit/score settings,
150-iteration limit, 1e-3 initial interval step cap and numerical gates remain
unchanged. This is not a new solver mode. Reordering mathematically equivalent
reconstruction arithmetic can still change float64 values and later nonlinear
trajectories; component equivalence does not establish identical full searches.

## Controlled full fits

The executed 0.2.1 package fingerprint is
`30a019eba2c99dbfae23b26200164893ab6ac54d82d2ef60a1dec2ce4f8e4055`.
Experiments ran before the publication commit; full package-file hashes bind the
tested source independently of the then-current Git HEAD. The
[committed evidence](benchmarks/evidence/reuse-0.2.1.json) retains source, input,
wrapper, output and environment identities.

Each case uses seed 17, the same input and policy, one allowed CPU (CPU 0),
numerical thread limits of one, a fresh worker and a 300-second cap. Baseline and
revised panels run sequentially on a shared host. These are one measured full
path per arm/case, including worker startup, rather than distributions of repeated
full-fit runtimes.

| Case | Frozen `ff5b5c3` | Revised | Search status in both |
| --- | ---: | ---: | --- |
| Easy, 100 mutations | 1.366 s | 1.216 s | Complete |
| Mixture, 100 | 11.930 s | 11.130 s | Complete |
| Easy, 1,000 | 8.376 s | 6.073 s | Complete |
| Mixture, 1,000 | 172.809 s | 141.268 s | Incomplete |

All eight fits finish all 26 raw penalty attempts and publish qualified selected
fits. Mixture-1,000 retains two unresolved penalties and six outer-limit starts
in both arms. There are no observed or captured QP qualification failures.
The revised mixture-1,000 wall time is 18.3% lower in this paired measurement;
this is not a universal speedup claim.

Readback verifies input/truth, compiled model, chain, policy, source and output
hashes and exact retained-ID populations. In all four pairs, selected lambda,
raw/refitted CCF vectors, labels, raw objective and score are exactly equal.
Tracked per-penalty status, witness, start, profile and iteration counts also
match. Truth ARI/RMSE and CNA-only macro/micro/weighted/per-class F1 and coverage
are retained for each arm. This matched panel shows no changed outcome; it does
not prove that all nonlinear fits are invariant to reconstruction rounding.

Revised mixture-1,000 prepares 1,853 audit contexts for 4,076 anchor checks.
Context preparation takes 0.677 seconds, signed scans 1.608 seconds, and finite
searches 54.344 seconds. These substage times are distinct; the inclusive kink
timer must not be added to scan/finite-search times. Finite proposal evaluation
remains the dominant audit cost.

## Component comparisons

An independently generated inventory of 200 moderate and 80 wide-range common
quadratics qualifies in both versions. Selected witnesses, relative witness
values and duals are identical; maximum primal difference is 4.18e-16. All 3,000
random/adversarial interval scans match the original scalar interval, sign,
derivative and scale bitwise. All 200 cached-audit comparisons match frozen-source
stationarity, interval decisions, restart vectors and qualification decisions.
The audit inventory covers smooth points, clipping boundaries, plateaus and
multiple occupied witnesses. It uses identical inputs in each arm.

Each component timing has one unmeasured warmup and three alternating paired
repeats on CPU 0 with one numerical thread. The evidence retains every sample
and its range; these medians include profiling, selected reconstruction,
polishing, dual recovery and certification:

| Nodes | Rebuild selected messages | Reuse thresholds | Reduction |
| --- | ---: | ---: | ---: |
| 100 | 3.126 ms | 2.531 ms | 19.0% |
| 1,000 | 31.119 ms | 25.215 ms | 19.0% |
| 4,000 | 122.264 ms | 98.235 ms | 19.7% |

At 1,000 nodes, signed-scan component speed ratios are 8.82 for a long fused
block, 14.28 for singleton runs and 7.18 for mixed runs. These ratios are neither
full-fit speedups nor factors to multiply into the full-fit table. The numerical
inventory and timings are our own reproducible measurements, not the reviewer's
unavailable prototype artifacts.

## Diagnosis before changing policy

A frozen `ff5b5c3` replay of the existing synthetic mixture-1,000 input reproduces
all six outer-limit starts, all 3,159 qualified common QPs, and identical selected
lambda, raw objective, score and three output-table hashes. Its package fingerprint
is `2db79fd19d8e05a1e386d133de37b60f19ce0be588717b46076de069c4750146`.
The instrumented replay took 181.44 seconds; this includes diagnostic wrapper
overhead and is not used as a paired throughput measurement.

Three starts fail at each of lambda 3,651.9057716670955 and 7,303.811543334191.
Their last interval restarts occur at outer iterations 20–56. None has a restart
or witness switch in its last 20 iterations. Those tails retain curvature
inflation of 256 or 512, make approximately 2e-10–2.1e-9 maximum-coordinate steps,
and decrease objective by roughly 1e-9 or less per step. Final node residuals are
2.25e-4–4.54e-4, above the unchanged 2e-5 threshold; nonzero-edge complementarity
residuals are exactly zero. Repeated half-inflation retries return to the same
accepted inflation. This evidence identifies terminal high-curvature stagnation
in these cases. It does not support increasing the interval step cap or blindly
raising the iteration budget. These starts remain unresolved.

Exclusive audit timings from that replay are:

| Audit work | Seconds | Calls |
| --- | ---: | ---: |
| Likelihood and posterior evaluation | 47.94 | 871,095 |
| Finite-proposal work excluding its likelihood calls | 12.20 | 777,112 |
| Remaining kink control/setup | 7.07 | 4,076 |
| Signed interval scan | 4.94 | 3,214 |
| Finite breakpoint construction | 1.20 | 86,172 |
| One-sided derivative assembly excluding its likelihood calls | 0.35 | 5,164 |
| Remaining stationarity work | 0.11 | 1,950 |

The exclusive rows sum to 73.80 seconds. Finite proposal evaluation is 53.85
seconds **including** its nested likelihood work; adding that number to the
table would double-count it. The scan accounts for 6.7% of audit time and 2.7%
of the instrumented fit. Finite plateau proposals dominate the remaining audit
cost, so a scan component speedup cannot be presented as a comparable full-fit
speedup. Cached old loss slices remove repeated starting-interval evaluations;
new finite proposals still undergo the same likelihood and descent checks.

## Exact audit decisions and numerical precision

Audit contexts are immutable, bound to model identity and the exact primal
vector, and rejected when stale. Selected-witness and extreme-witness interval
coverage is preserved. The vectorized scan keeps independent sequential prefix
sums within each feasible fused run, the earliest start on an equal prefix key,
and negative sign then earliest stop on an equal final merit. It introduces no
floating-point tie tolerance. Small and unsupported arithmetic retains the
original scalar implementation. Prefix arithmetic and storage remain O(M);
bucket grouping has a conservative O(M log M) total-work bound.

Receipts now distinguish storage bits, explicit mantissa bits, significand bits,
exponent range and epsilon for float64 and longdouble, and record OS, machine,
Python, NumPy and SciPy. Storage size alone does not establish precision, and
longdouble representation is platform-dependent. See the
[NumPy `finfo` documentation](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html).
This run uses Python 3.13.2, NumPy 2.2.6 and SciPy 1.18.0. Its longdouble occupies
128 storage bits but has a 64-bit significand (63 explicit mantissa bits), with
epsilon `1.084202172485504434e-19`. Float64 has a 53-bit significand.
Numerical qualification here is local Linux x86_64 evidence only. Other platforms
must run the same stress replay and retain their actual precision receipt before
claiming equivalent qualification.

## Tests, stress replay and packaging

All **291 tests pass** in `ml1`. Ruff, compileall over source/tests/benchmarks,
and whitespace checks pass. Regressions cover exactly two message passes and one
finalizer, singleton/edge/frozen witnesses, immutable and stale audit contexts,
old-loss reuse, strict floating-point ties, failure-capture scopes, timer
accounting, precision receipts and output/hash mismatch rejection.

On the final source, all eight archived actual production QPs qualify, as do
1,000 deterministic wide-range 100-node QPs and 80 wide-range common profiles.
The largest stress KKT residual is `2.0222716667991292e-8`, below the unchanged
`1e-7` gate. These are the repository's captured production fixtures and generated
stress inventory, separate from the reviewer's unavailable fixtures. No claim
about other platforms or real-cohort accuracy follows from this replay.

The 0.2.1 wheel was built, installed into an isolated target and invoked from
outside the repository. Its example fit succeeds with complete search, the
expected final package fingerprint, and verified output-table hashes. Wheel SHA-256:
`1d338e94364985b09515cd7e697c72014640ecca9d8a083c59b26a625126e5ea`.

## Reproduction

Run with the `ml1` interpreter from this repository. Archive the baseline without
overlaying revised source files:

```bash
mkdir -p results/ff5b5c3-source
git archive ff5b5c3 src/clipp1d | tar -x -C results/ff5b5c3-source --strip-components=1
conda run --no-capture-output -n ml1 python benchmarks/benchmark_scaling.py --package-source results/ff5b5c3-source --sizes 100 1000 --scenarios easy mixture --timeout-seconds 300 --outdir results/reuse-baseline
conda run --no-capture-output -n ml1 python benchmarks/benchmark_scaling.py --sizes 100 1000 --scenarios easy mixture --timeout-seconds 300 --outdir results/reuse-current
conda run --no-capture-output -n ml1 python benchmarks/compare_revision_outputs.py --baseline results/reuse-baseline --revised results/reuse-current --output results/reuse-output-comparison.json
conda run --no-capture-output -n ml1 python benchmarks/compare_reuse.py --baseline-package results/ff5b5c3-source/clipp1d --baseline-sha256 2db79fd19d8e05a1e386d133de37b60f19ce0be588717b46076de069c4750146 --outdir results/reuse-components
conda run --no-capture-output -n ml1 python benchmarks/diagnose_outer_progress.py --frozen-package results/ff5b5c3-source/clipp1d --input-file results/reuse-baseline/mixture-n1000/input/tumor.tsv --outdir results/outer-diagnosis
conda run --no-capture-output -n ml1 python benchmarks/replay_robustness.py --outdir results/reuse-stress
```

Output directories must be fresh. Scaling uses one allowed CPU and one numerical
thread, configured before NumPy import, and immutable source/input/wrapper
receipts. Cases run in fresh workers on a shared host; affinity does not establish
exclusive hardware. Timing categories distinguish profiles, standalone quadratic
solves and nested finalizers. Completed selected fits and search completeness
remain separate claims.
