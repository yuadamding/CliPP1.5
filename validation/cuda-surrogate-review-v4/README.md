# Coordinate-backtracking qualification and cohort staging evidence

All six declared coordinate-backtracking CUDA stages passed by **September 23,
2026, 12:46 CDT** (America/Chicago). These are actual allocated L40 fits, with
complete planned starts, final raw audits, membership refits and public output
validation. Scalar backtracking remains the production default. The four-cohort
run retains its separately frozen `a0e1406` source and scalar policy.

| Fixture | Mutations | LSF job | Scalar search | Coordinate search | Coordinate starts |
| --- | ---: | ---: | --- | --- | ---: |
| below_one | 64 | 77341022 | complete | complete | 99/99 |
| below_one | 256 | 77341055 | incomplete | complete | 99/99 |
| below_one | 512 | 77341109 | complete | complete | 99/99 |
| mixed_support | 64 | 77341170 | complete | complete | 99/99 |
| mixed_support | 256 | 77341215 | complete | complete | 99/99 |
| mixed_support | 512 | 77341258 | complete | complete | 99/99 |

Each path evaluated 26 penalties. In all six pairs, selected raw vectors, labels,
refitted CCFs, scores and selected lambdas were identical. Pilot and graph hashes
also matched. The scalar below256 winner matches despite unresolved other starts;
it is not a complete-search qualification. These are synthetic qualification
fixtures, not evidence of improved cohort accuracy or global optimality.

The unchanged saved below256 QP still fails: compiled KKT residual
`2.477685574248767e-7` and eager residual `2.477685575358654e-7`, against the
unchanged `1e-7` gate. Both gap checks pass. Exact problem, initialization and
returned states are retained in `below64-a.zip`. Completing a changed outer
trajectory does not solve this literal obstruction.

## Retrievable receipts and verification

[QUALIFICATION.json](QUALIFICATION.json) contains stage identities, outcome and
exact-lambda comparisons. The original experiment and terminal receipts are
directly readable under [receipts](receipts). [ARCHIVES.json](ARCHIVES.json)
binds seven ZIPs totaling 22.9 MB. The six experiment ZIPs contain **all** imported
artifacts, including full matrices, traces, failed QPs, inputs, public outputs,
LSF admission/startup/terminal records, retained scheduler terminal observations,
and the original frozen source bundles. Nothing from their imported artifact
inventories is omitted. Archives preserve exact file bytes and UTC timestamps.

```bash
python validation/cuda-surrogate-review-v4/verify.py
```

The standard-library verifier checks archive hashes, every imported artifact,
frozen inventories, package fingerprints, job/plan/receipt linkage, qualification
coverage, the retained negative replay and canary input/output identity. It reads
ZIP members without executing their code. It verifies evidence integrity; it
does not independently rerun CUDA numerical inference or contact LSF.

The experiment package fingerprint is
`40c5304905d3e86832e22832e768bc9fe61cb291351358680d9053e93dd9505c`,
matching `75c102f`'s package. Launch plans accurately retain their original
`a0e1406` plus dirty-overlay identity; sealed source and patch are included.
The original diagnostic driver SHA-256 is
`a8fd578596ef723a0ad1289276cac260b3f5d8b016a7475a576b491ed779a2fc`.
These receipts predate the exception-attempt bookkeeping correction; they are
not relabeled as execution of that later driver.

## Actual cohort worker adoption

`cohort-staging-canary.zip` includes remote readbacks of the frozen worker,
staging helper, full case manifest, plan, source, and actual canary job
**77339914**, case `001783` (`100_2_0.6_0.1_rep26`). Its 200 retained mutations
were fitted and validated with the original basename. All published tumor/sample
IDs and the retained-ID hash agree with the original manifest. The canary
completed with `search_status=complete` on September 23, 2026, **10:58 CDT**.

The worker calls `stage_case_input`, then `fit(staged_input_path(...))`.
Refitting occurs inside that fit. Its final `_validate_result` rereads the same
`staged_input_path`. The helper bytes equal the committed
`benchmarks/cohort_staging.py`; startup checks bind the immutable worker inventory
and CUDA package before execution. The verifier checks the actual source and
receipt chain, rather than inferring adoption from helper availability.

This recovery binds **112 validated imports plus 5,344 fresh cases**. The full
cohort payload ZIP, five inventoried Python bytecode caches and outputs for other
tumors remain external. `ARCHIVES.json` explicitly lists omitted sealed members
and their original hashes; the Python source files are all included. The canary
archive does not claim all-cohort completion or independently republish every
imported result. Original failed, wrongly named outputs were not relabeled or
imported. The cohort package fingerprint is
`0a4fbfc4e4f1d93cafb4cd06d4f126029de5d1f36d08a33c57d5aef25b22ac43`,
distinct from the research observer package above.

## Review correction and separate timing

Trace attempt IDs are now reserved on entry, independently of returned-call
counts. Attempted, returned, qualified and raised counts are separate and survive
failed trials. Regressions cover thrown solver and strategy-check exceptions,
then a later attempt in the same context, without overwriting the first trace.
The supplied ZIP's original Git-blob-verified checks independently reproduce the
old conditional collision; [review-reproduction.json](review-reproduction.json)
records that reproduction. Those dependency doubles are not CUDA evidence.

The full local suite passes **1,126 tests**, including the rational negative
regression and timing-boundary tests; lint and this archive verifier pass.

Qualification times include diagnostic observation and artifact work. They are
not speedup measurements. The separate
[`time_surrogate_cuda.py`](../../benchmarks/time_surrogate_cuda.py) study requires
the same-fixture completed coverage receipt and numerical package. It disables
the observer and candidate tracer, retains one cold pair separately, then runs
three warm pairs with alternating order. Only synchronized `fit_tensor_model`
time is compared, including normal numerical audits/refits. Input/upload,
additional graph checks, summaries, export, publication and file writes are
outside the timer. Incomplete work has no complete-fit speed ratio.

The [separate timing evidence](../cuda-surrogate-timing-v1/README.md) retains the
first canary's phase-receipt failure (job 77342485) and the corrected, completed
below64 stage (job **77342695**), without changing the numerical package. The
three warm pairs measured medians of 15.702 s coordinate and 15.915 s scalar,
with identical selected results. This small observed difference is not a general
speedup claim. The source-frozen serial controller continues through the five
remaining previously qualified fixtures, stopping on unexpected failure.
