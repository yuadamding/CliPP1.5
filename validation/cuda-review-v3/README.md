# CUDA policy-v3 qualification evidence

Published 2026-09-22 19:37 CDT from accepted
Seadragon scalar LSF job **77334328**, run `clipp2_clipp1d_cuda_20260922i`.
Production source SHA-256: `ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26`.

- [Final reconciliation](FINAL_VALIDATION.json): 811 local CPU-reference tests,
  strict allocated-CUDA case coverage, complete synthetic N16/64/256 paths,
  qualified production-budget QPs, phase timings, memory and reuse counters.
- [Actual GPU receipt](qualification.json) and [events](qualification.events.jsonl).
- [Local validation](LOCAL_VALIDATION.json) and [test log](whole-suite.log).
- [Matched scalar comparison](profiles/scalar-profile-comparison.json).
- [Operational plan](operational/PREPARED.json), [sealed inventory](operational/inventory.json),
  [worker](operational/worker.py), [accepted](operational/accepted.json),
  [startup](operational/startup.json) and [terminal](operational/terminal.json).
- [Exact original-to-published map](EVIDENCE_MAP.json) and [all-file hashes](SHA256.json).
- [Original F failure](failed-attempts/f/results/qualification.json) and
  [F terminal](failed-attempts/f/receipts/terminal.json) remain unsuccessful evidence;
  later qualification does not relabel that failed resource QP.
- [Original G failure](failed-attempts/g/FAILURE_SUMMARY.json) retains its incomplete
  N64 path despite a qualified selected raw candidate and earlier passed checks.
- [Stopped H](failed-attempts/h/FAILURE_SUMMARY.json) retains cancellation intent/result,
  completed N16/N64 fits and its source-matched [CPU-only trace](failed-attempts/h/cpu-reference/terminal.json).
  H has no completed N256 fit and does not qualify the subsequent audit-scale correction.
- The [post-fix CPU reference](cpu-reference/scaled-audit/EVIDENCE_SHA256.json) preserves its saved
  driver, source/plan, terminal, events, complete bounded path and [arithmetic derivation](cpu-reference/scaled-audit/AUDIT_SCALING_DERIVATION.md).
  Its source matches this accepted revision; its results are CPU diagnosis, not allocated-CUDA acceptance.

Original remote receipt/artifact bytes are unchanged. Remote `results/qualification.artifacts/`
maps to `artifacts/`; `results/scalar-profile-*` maps to `profiles/`;
`receipts/` maps to `operational/`; selected stderr logs map to `logs/`.
Absolute paths inside original receipts retain their original meaning.
`operational/LOCAL_PREPARED.json` is the controller's extended archive receipt;
`operational/PREPARED.json` is the exact plan hashed by the worker.
The [source reconstruction map](operational/source-reconstruction.json) binds the
baseline commit, exact sealed tracked patch and additional source files omitted
from that patch. Each preserved failed attempt includes the same reconstruction
material under its `sealed/` directory. Expected source-file hashes and archive
inventories remain available; full archives and untracked tests are not duplicated.

Large Chrome traces remain under the original remote run. [Their exact paths and
recorded hashes](REMOTE_TRACES.json) are retained; the publisher neither downloaded
nor re-read them. Compact counts files and profile receipts are included.
Scalar-read proxies decreased from 40
to 17 (`analytical32`) and
2,098 to
243 (`mixed6`) on matched inputs.
These are CPU profiler host-dispatch operation counts
while numerical inference ran on CUDA. They are not measured GPU synchronization
durations or an end-to-end speedup.

Peak GPU measurements follow required final device qualification and export.
Receipt elapsed time excludes writing/durably publishing the receipt itself;
return-only publication metrics are retained separately in the public qualification
record. Compilation/cache histories and numerical scopes remain explicit.

The [prior policy-v2 evidence](../371003f/README.md) remains separate. Neither this
synthetic qualification nor its increasing-size cases establishes cohort accuracy,
arbitrary-size scalability, global nonconvex optimality or a GPU speedup.
