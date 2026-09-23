# PyTorch CUDA policy-v3 validation

**Passed for the source and synthetic fixtures below.** Evidence published 2026-09-22 19:37:59 CDT.
Package `0.5.1.dev0`; policy `clipp1d_complete_cuda_unconstrained_v3`; output schema `clipp1d.cuda.run.v3`.
Production source SHA-256: `ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26`.
Accepted scalar LSF job **77334328**, run `clipp2_clipp1d_cuda_20260922i`, queue `egpu`.
GPU: NVIDIA L40, capability 8.9; PyTorch 2.9.1+cu128, CUDA 12.8.
Interpreter: `/rsrch8/home/bcb/yding4/miniforge3/envs/ml1/bin/python3.13`; compiler: `/rsrch8/home/bcb/yding4/miniforge3/envs/ml1/bin/x86_64-conda-linux-gnu-gcc`.
Qualifier: 2026-09-22 19:25:38 CDT through 2026-09-22 19:35:58 CDT, 619.277 seconds.
Initial compiler admission: **2.014 seconds**, recorded separately by the qualifier.
Later lazy Torch specializations remain charged to their executing phases; fit totals are not compilation-separated.

Fitting remains unconstrained: no coordinate is forced to CCF one.
The dense complete-graph raw estimate remains primary; qualified membership refits provide the unchanged score.
Public label zero designates the refitted center closest to one, with canonical mutation-ID tie handling.

## Changes covered

1. Exact equal values stay together inside overwide tolerance-connected runs; canonical labels and diameter limits remain enforced.
2. Timings and GPU peaks include final device qualification/export; durable publication uses separate return-only metrics.
3. Owned model/graph stages reconcile full immutable snapshots at entry, exit and publication; hot loops retain metadata guards.
4. Loss-only/gradient kernels, analytical-first scalar lanes and packed golden wells reduce work while preserving scalar gap gates.
5. A bounded last-partition cache and canonically matched singleton pilots reuse qualified refits without transferring raw certificates.
6. QP dual initialization is graph-bound and restricted to the same start/literal lambda; every changed surrogate requalifies.

The N256 probe additionally exposed complementarity-limited QP convergence. Bounded finer active-set polishing
and projected dual-flow correction now propose repairs, accepted only by the original objective, gap and KKT gates.
Polishing work is counted separately from ADMM iterations; the production ADMM budget remains 20,000.
H additionally exposed an incorrectly scaled audit-roundoff floor. The corrected audit scales its error unit
alongside the objective and tolerance, preserving original-unit cancellation protection. Fresh job **77334328**
qualifies this correction; H remains stopped and unqualified.
QP gap: `1e-10 + 1e-11 * scale`; KKT: `1e-07`.
Scalar gap: `1e-07 + 1e-10 * abs(loss)`; fusion/raw stationarity tolerances remain `2e-05` / `2e-05`.

## Local and allocated-CUDA checks

The source-bound local suite passed **811 tests in 61.570 seconds**, with 0 failures and 0 skips.
Local warning: Local CUDA unavailable: NVML initialization warning; no CUDA acceptance claim. These are CPU reference/tracing tests, not GPU test counts.
Static checks: `git diff --check`: passed; `python -m compileall -q src tests benchmarks`: passed; `ruff check src tests benchmarks`: passed.
Grouping regressions cover the reported duplicates, randomized equality/diameter/canonical checks and QP polishing.
Reuse/integrity regressions cover changed identities, tensor mutation, scalar qualification and publication timing/readback.

| Allocated-CUDA comparison | Qualified comparisons |
| --- | ---: |
| Likelihood and derivative/reduced-kernel parity | 7 |
| Complete-graph QP parity | 8 |
| Cold/warm QP parity with new certificates | 4 |
| Shared-pilot, identical-graph, literal-lambda parity | 12 |
| Small complete eager/compiled default-path parity | 3 |
| Public default-path fit and durable output publication | 1 |

| Small fixture | Eager status / seconds | Compiled status / seconds | Max raw CCF error | Max refit CCF error |
| --- | --- | --- | ---: | ---: |
| `all_bounds_below_one` | complete / 195.270 | complete / 179.638 | 4.83225e-14 | 8.27218e-10 |
| `mixed_multiplicity` | complete / 40.772 | complete / 31.926 | 1.11022e-16 | 4.11487e-10 |
| `single_support` | complete / 6.236 | complete / 2.518 | 0 | 0 |

Labels and raw/refitted multiplicity calls agree for these small paths; objective, score and CCF parity remain separately checked.
Independent pilots can change adaptive graph weights and lambda references; their differences remain in the evidence.
Fixed-problem comparisons separately require identical graph weights and literal lambda, including nonfused positive penalties.

## Complete increasing-size inference

Each synthetic compiled fit completed its planned path, final independent qualification, export and host validation.
The total below runs through that final/export boundary; compilation history is included where incurred.

| Nodes | Search | Total seconds | Peak allocated MiB | Peak reserved MiB |
| ---: | --- | ---: | ---: | ---: |
| 16 | complete | 5.381 | 0.134 | 10.000 |
| 64 | complete | 11.902 | 0.840 | 10.000 |
| 256 | complete | 84.344 | 11.894 | 14.000 |

| Nodes | Pilot s | Graph s | Path s | Refit s | QP s | Audit s | Final qualification s | Export s | Stage integrity s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.659 | 1.435 | 2.902 | 0.343 | 2.398 | 0.169 | 0.008 | 0.002 | 0.000 |
| 64 | 0.003 | 0.001 | 11.808 | 0.041 | 11.305 | 0.169 | 0.005 | 0.002 | 0.000 |
| 256 | 0.007 | 0.001 | 84.134 | 0.059 | 83.626 | 0.175 | 0.005 | 0.002 | 0.000 |

QP/audit subtotals are inside path time; do not add them again. Stage-integrity time is read from `numerical_stages`.
Path time excludes refits. Totals also include preparation/validation overhead; rounded phases need not sum exactly.

| Nodes | Refits computed / reused | Singleton pilots reused | QP calls / dual initializations | QP polish steps |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 4 / 22 | 22 | 285 / 211 | 381 |
| 64 | 7 / 19 | 177 | 287 / 213 | 4,835 |
| 256 | 8 / 18 | 0 | 273 / 199 | 4,260 |

Refit computations include attempted work; cache hits require matching qualified memberships. Dual counts are initializations.
Peak CUDA allocation/reservation is captured after all required final device work, including audits, posteriors and transfers.

## Separate N256 quadratic resource check

Both repetitions qualified under the 20,000-ADMM-iteration production budget and unchanged gates.
This QP-only check does not replace the complete nonlinear fits above. First-shape compilation and warm repetition differ.

| Repeat | ADMM iterations | Polish steps | Seconds | Gap / allowed gap | KKT | Peak allocated / reserved MiB |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 576 | 37 | 1.826 | 1.60531e-18 / 1.00821e-10 | 3.29015e-14 | 7.623 / 8.000 |
| 1 | 576 | 37 | 1.825 | 1.60531e-18 / 1.00821e-10 | 3.29015e-14 | 8.125 / 10.000 |

Original F/job **77333999** remains failed: after 20,000 ADMM iterations its gap was
`4.87546e-10` above `1.00821e-10`, although KKT `3.22756e-09` passed.
[Its original receipts and reconstructible source](validation/cuda-review-v3/failed-attempts/f/FAILURE_SUMMARY.json) are preserved; this pass does not relabel F.
G/job **77334114** remains failed: N64 selected raw qualified, search `incomplete`; 3 planned penalties retained 5 unresolved starts. [Exact failure evidence](validation/cuda-review-v3/failed-attempts/g/FAILURE_SUMMARY.json).
H/job **77334195** was explicitly stopped and remains unqualified: N16/N64 completed; no completed N256 fit. A source-matched CPU trace identified the directional-audit scale defect; the later small-path/publication checks were not reached. [Exact failure evidence](validation/cuda-review-v3/failed-attempts/h/FAILURE_SUMMARY.json).
[H's original CPU terminal, events and source receipt](validation/cuda-review-v3/failed-attempts/h/cpu-reference/terminal.json) are separately labeled CPU-only diagnostic evidence.
The source-matched [post-fix CPU reference](validation/cuda-review-v3/cpu-reference/scaled-audit/EVIDENCE_SHA256.json) completed 26 penalties in 49.587 seconds.
Its saved driver, full path and [arithmetic derivation](validation/cuda-review-v3/cpu-reference/scaled-audit/AUDIT_SCALING_DERIVATION.md) support diagnosis only; these are not CUDA results or a GPU speedup.

## Matched scalar profiling

| Fixture | Baseline scalar-read proxies | Current scalar-read proxies | Maximum qualified loss difference |
| --- | ---: | ---: | ---: |
| `analytical32` | 40 | 17 | 0 |
| `mixed6` | 2,098 | 243 | 0 |

Baseline/current profiles use the same allocated job, fixture and profiling script; all scalar lanes qualify independently.
Loss differences satisfy combined scalar-gap bounds. Counts are CPU `aten::_local_scalar_dense` host-dispatch proxies
while inference executes on CUDA; they are neither measured GPU synchronization duration nor evidence of end-to-end speedup.
Compact receipts/counts are published; [large trace locations and recorded hashes](validation/cuda-review-v3/REMOTE_TRACES.json) remain remote.

## Measurement, evidence and limits

Receipt elapsed time ends at output preparation, excluding its own serialization/readback/fsync/atomic publication.
Return-only, tamper-evident `operation_metrics` separately records completed durable publication and is retained by the public qualifier.
The [final reconciliation](validation/cuda-review-v3/FINAL_VALIDATION.json), [GPU receipt](validation/cuda-review-v3/qualification.json),
[local test log](validation/cuda-review-v3/whole-suite.log) and [source reconstruction map](validation/cuda-review-v3/operational/source-reconstruction.json) bind this result.
[All-file hash inventory](validation/cuda-review-v3/SHA256.json) SHA-256: `6b740eb7b47a34fb1d274cd7338ae8aaaa21c429b28538b2f964d81b03476cd4`.
The [evidence index](validation/cuda-review-v3/README.md) maps exact original bytes; [prior policy-v2 evidence](validation/371003f/README.md) remains separate.

These are synthetic fixtures, not cohort accuracy or full-cohort reproduction results. Dense graph memory remains quadratic.
Complete means all bounded planned candidates qualified; it does not cover every lambda or prove global nonconvex optimality.
The results do not establish arbitrary-size capacity, a GPU speedup, or qualification of changed source/settings/hardware.
