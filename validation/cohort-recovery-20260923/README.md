# Cohort recovery evidence — September 23, 2026

This is a dated record of saved evidence, **not live status, launch authority,
or a claim that the cohorts have finished**. Times use `America/Chicago` (CDT).
Original UTC receipts and failed attempts remain unchanged. Reusable procedures
are in [operations](../../benchmarks/OPERATIONS.md). The ignored, local-only
[current-run pointer](../../results/CURRENT_RUNS.json) locates newer status.

## Historical outcome

At **2026-09-23 20:25:49 CDT**, the combined observer recorded:

| Cohort | Planned | Validated, complete search | Validated, incomplete search | Unresolved failures |
| --- | ---: | ---: | ---: | ---: |
| Regional-CN single-region | 200 | 24 | 58 | 0 |
| SimClone1000 | 756 | 125 | 7 | 1 |
| PhylogicNDT500 | 500 | 1 | 0 | 2 |
| CliPPSim4K | 4,000 | 399 | 41 | 3 |
| Total | 5,456 | 549 | 106 | 6 |

There were no output-validation errors in this snapshot. Incomplete-search
outputs are valid selected results with a distinct search status; they do not
establish exhaustive search or global optimality. Finished quick cases are a
selected subset, not full-cohort accuracy evidence.

The replacement Regional-CN run imported **82 validated cases and scheduled
118 unfinished cases**. Ten A100 workers were Ready and hardware verified.
The 25 local CPU workers and LSF pool (10 RUN, 2 PEND) were preserved. Authorized
ceilings were 10 A100 workers for Regional-CN, 25 local CPU workers and 15 total
LSF jobs for the other cohorts; pending and uncertain admissions count toward
their relevant ceilings. The separate 000029 canary repeat adds no cohort case.

## Incidents and repairs

**CPU identity capture.** CPU v3 recorded the hash of a transiently empty
`/proc/PID/cmdline` for case 001663. A later wall-limit identity check failed;
an existing drain reason obscured the new error, and the exception blocked
reconciliation of five exited successful workers. The observer also omitted the
still-live worker. Admission now requires a nonempty exact argv and consistent
birth/group identity. Reconciliation errors are recorded independently while
other workers continue to be processed. Failed admission cleans up its child;
exit/timeout races are handled without losing ownership. Independently verified
pidfd signalling enforced the expired original limit. Its original failure
receipt was retained, with the timeout explanation recorded separately.

The CPU v5 handoff preserved all 25 healthy fits and shared the core cap with
the draining parent. A handoff does not grant a fresh retry to an existing retry.
The system Python was used only for pidfd signalling because ml1 lacked that
API; numerical execution retained the frozen ml1 interpreter.

**A100 service loss.** Six workers on `hpcgpu11` reported unspecified CUDA
launch failures, followed by twelve CUDA initialization failures (error 802).
The old hardware probe ran outside classified exception handling, causing the
latter failures to lose their specific classification. Pods became NotReady,
were selected for taint eviction, and the Indexed Job reached
`BackoffLimitExceeded`. Its supervisor assumed scheduler completion and exited;
cleanup had not yet proved absence of the six terminating Pods.

The observations establish a shared accelerator/node incident. They do **not**
prove the underlying driver, fabric, hardware, or node-management cause.
The repair classifies per-case hardware-startup/device-loss failures and retires
the affected dispatcher without a pool-wide halt. Initial phase probes remain
outside that per-case boundary, and scheduler eviction can still interrupt other
Pods. The lifecycle records failed/partial completion, retains evidence on log
timeouts, and distinguishes pending cleanup from verified absence. New static
and GPU manifests excluded `hpcgpu11`.

Two exact Job/owned-Pod absence observations and parent-output revalidation
passed at **2026-09-23 20:17 CDT** before any overlapping retry was admitted.
No force deletion was used to convert uncertainty into apparent absence.
Replacement Job `clipp1d-regional-a10010-20260923e` activated at 20:18:18 CDT,
UID `59164b8f-3467-45f1-90c0-0b96a52801ce`. Its parent UID was
`d94fe901-e6f7-49d5-8a6d-f6956cac5284`. Static qualification, cleanup, GPU gates
and the real canary preceded expansion from one worker to ten.

**LSF wall limits.** Cases 000512, 001187 and 001282 exhausted six-hour fit
budgets in Jobs 77346183, 77346185 and 77346184. Scheduler DONE did not mean
validated fit success. One **18-hour retry per named case** was prepared, with
unchanged numerical budgets. At this snapshot the retry controller had submitted
zero jobs: it waited for both existing LSF owners and all accepted jobs to finish
reconciliation, then required independent qualification. Uncertain submissions
or an active refilling predecessor blocked admission. The observed queue maximum
was 30,240 minutes; it is a dated resource-contract fact, not a future default.

## Qualification and limits

- CPU identity repair: 31 focused tests passed, with Ruff and diff checks.
- A100 recovery: 40 focused repository tests passed, including lifecycle,
  dispatch and failure classification; the deferred controller passed blocked
  and successful handoff simulations. Four LSF admission/retry tests also passed.
- Allocated A100 static checks and injected invalid-cut audit checks passed.
  The audit rejected invalid internal inputs without a certificate while a valid
  stationary fixture remained certified. The earlier invalid-cut arithmetic
  root cause was **not proven**; this qualifies containment, not a root-cause fix.
- The synthetic compiled float64 complete-graph QP at N=4,143 passed capacity
  qualification, with 2,195,718,144 peak reserved bytes. This is bounded resource
  evidence, not a full-fit speed or cohort-accuracy result.
- The real 000029 canary retained its graph, labels, final CCFs and score exactly:
  maximum CCF error 0 and score delta 0. Its baseline numerical source was
  `40c5304905d3e86832e22832e768bc9fe61cb291351358680d9053e93dd9505c`.
  One paired case does not qualify every input, platform or future overlay.

Fitting stayed unconstrained; public label 0 followed `nearest_to_one_l2_v1`.
The graph, likelihood, numerical tolerances, iteration budgets and score settings
were unchanged by the node recovery. Imported and still-running older results
retain their own source identities; aggregate results are not relabeled as one
homogeneous revision.

## Identity and evidence retention

The repaired packages were dirty-source overlays on full Git commit
`a8a3ef7aaaeda8dffa8ac5fccef1cc4b77aa71f5`, not that clean commit alone.
Numerical source SHA-256 was
`b16f1fc8068351aeeb75dfe0f940a205d2699347f3ce0799f0e349cbf51aba99`.
**That hash does not identify the complete operational package.** CPU and A100
archives shared it while containing different worker/lifecycle helpers.

| Frozen artifact | SHA-256 |
| --- | --- |
| CPU identity-repair source archive | `5ea32199f7a1fc64a605110b77fd613a5b5abf7b48f3f1459dc430292257b873` |
| A100 node-repair source archive | `4024888eb04db10bb329f6d3b7e10f2ff774c4720a37d51d2331264e5ff135df` |
| A100 replacement complete payload | `2a0338a27d22b2ce292e86506e0d40065b93acb44ec550f30c6765ee7b519fb7` |
| A100 replacement plan | `0ae427feab0e42a7b89cbfdda90fc59c14d4a89003be0081a731c3a3cedf8740` |
| Deferred LSF retry complete archive | `b47a5626289236ca444d1d207d38d1117482a40d1d21923d468987c53b1e5905` |

The [compact evidence index](evidence.json) is a **derived summary**, not copied
raw receipts. It records exact source paths, actual-byte file hashes, selected
receipt values and extraction locations. Original receipts, inventories, plans,
archives, manifests and output hashes remain in the local ignored `results/`
attempts. No large logs, inputs, output tables or full inventories are duplicated
here. Future reuse must reopen and reconcile the original source/plan/environment,
input/output and ownership evidence; this summary alone cannot authorize it.

Primary local records:

- [CPU incident and recovery](../../results/cpu-identity-recovery-20260923-v1/README.md)
- [A100 incident and relaunch](../../results/a100-node-recovery-20260923-v1/README.md)
- [Historical combined snapshot](../../results/a100-node-recovery-20260923-v1/snapshot-20260924T012548216891Z/SNAPSHOT.json)
- [Deferred timeout retries](../../results/lsf-timeout-retries-20260923-v1/README.md)

These links require the original local workspace; `results/` is not shipped with
the repository. Keep this dated record when current-run pointers advance.
