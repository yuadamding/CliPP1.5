# Uninstrumented surrogate timing: first completed fixture

Snapshot: **September 23, 2026, 13:30 CDT** (America/Chicago).
One of six timing stages is complete; the serial LSF controller has advanced to
mixed-support64. This is not a full-study speedup claim.

| Fixture | Warm pairs | Scalar median | Coordinate median | Median paired ratio |
| --- | ---: | ---: | ---: | ---: |
| below_one, 64 mutations | 3 | 15.915 s | 15.702 s | 1.0136× |

All eight fits (one cold pair plus three warm pairs) completed their 99 starts,
audits, refits and public validation. Selected labels, raw/refitted CCFs, scores,
selected lambdas, pilots and graph hashes matched. The cold scalar/coordinate
times, 28.276/15.792 seconds, are retained separately and excluded from the warm
summary. Their cache/order asymmetry illustrates why cold or instrumented ratios
must not be called a general throughput gain. The warm result is a small observed
difference without a statistical significance claim.

Completed job **77342695** used one L40 under the unchanged numerical package
`40c5304905d3e86832e22832e768bc9fe61cb291351358680d9053e93dd9505c`.
The timing driver SHA-256 is
`1e3d8e86c78a37719942f4b5668b9775b2531a1ff5cd3adab5893dc7c22d0ca5`.
See [below64-b.json](below64-b.json) for timings, source/helper identities,
start counts, completeness and comparisons. Its ZIP contains every imported
artifact, original frozen source, public outputs and scheduler/worker receipts.

The first canary, **77342485**, failed because the new timing wrapper omitted
required fields in the public phase-time receipt. The failure and full artifacts
are retained in [below64-a.json](below64-a.json) and its ZIP. The corrected helper
was frozen in a new attempt after exact terminal reconciliation. The numerical
package, policies and admission thresholds did not change.

```bash
python validation/cuda-surrogate-timing-v1/verify.py
```

The verifier checks hashes and receipt linkage without CUDA or remote access.
`controller-started.json`, `controller-progress.json` and `operations/` preserve
the exact serial owner and helper source at this snapshot. Five remaining stages
are mixed64, below256, mixed256, below512 and mixed512. Each requires its already
completed, source-bound diagnostic coverage receipt. The controller stops on an
unexpected failure and never duplicates pending work. Live operational state is
under `results/cuda-surrogate-timing-20260923-v1`; this tracked snapshot is not a
live-status endpoint.

The timer includes synchronized `fit_tensor_model` with normal audits/refits.
There is no surrogate observer or candidate-event tracer. Input preparation,
device upload, extra graph checks, summaries, export, public validation and
artifact writes are outside it. Warm pairs alternate order and reuse compiler
caches, with a fresh model/fit each time. Only complete, publicly validated warm
pairs contribute speed ratios; incomplete searches remain separately reported.
Production scalar backtracking and the existing cohort runs are unchanged.
