# LSF and research recovery — September 24, 2026

This is dated execution evidence, not current status or completed-cohort
qualification. Times use America/Chicago. The local operational pointer is
[CURRENT_RUNS.json](../../results/CURRENT_RUNS.json); reusable rules are in
[OPERATIONS.md](../../benchmarks/OPERATIONS.md).

The later [follow-up recovery](FOLLOWUP.md) repairs an empirical artifact import,
retries two CPU timeouts and launches an isolated Regional-CN numerical replay.
Use the current pointer rather than treating the snapshot below as live status.

## Repaired execution

- Corrected held canary **77355084** from an effective 4 GB limit to its intended
  8 GB limit, with matching reservation. Preserved its ID, failed admission and
  asynchronous modification evidence; verified admission before release.
- The canary passed. New controller
  `/rsrch8/scratch/bcb/yding4/clipp2_clipp1d_fourcohorts_20260924a` passed the
  exact-parent qualification/canary gate and entered the panel phase.
- Retired the old, empty deferred-retry owner. The replacement queue contains
  589 unsubmitted cases, 13 confirmed timeout retries, and one conditional
  retry of the healthy draining case. Every timeout retry has one 1,080-minute
  job budget with unchanged numerical limits. A successful draining-parent
  result is reused rather than fitted again.
- Reserved 14 cohort slots and one serial research slot within the total cap
  of 15 LSF jobs. All 13 confirmed timeout retries were admitted once.
- Recovered successful timing Job **77346551** using its retained LSF terminal
  log and hash-verified application outputs after its scheduler record aged
  out. The fifth stage completed three warm pairs, with median paired speedup
  1.002570259; this is not a cohort performance improvement claim.
- Submitted the final timing stage as **77355910**. Its live continuation will
  release the eight-case empirical study only after all six timing stages
  validate. Preserved the empirical study's original September 26, 2026,
  19:00 CDT deadline and its original scientific source/inputs.

The numerical cohort source fingerprint remains
`b16f1fc8068351aeeb75dfe0f940a205d2699347f3ce0799f0e349cbf51aba99`.
The replacement uses the same worker, validator, numerical source, environment,
compiler, likelihood, complete graph, score and unconstrained fitting policy.
The job-name binding, resource contract and orchestration changed. Prior
successful outputs keep their original provenance. Research studies retain
their separately frozen source; they are not the current cohort source.

## Verification and saved snapshot

The focused verification passed **61 tests**: 56 repository tests and five
attempt-local lifecycle tests. Coverage includes admission mismatch, foreign
identity, actual zero-exit/missing-job behavior, retained terminal-log ownership,
launch-acknowledgement schema, transport-only retry policy, conditional retry,
successful-parent reuse and existing CPU/A100 recovery boundaries. Ruff and
`git diff --check` passed. No numerical implementation changed in this recovery.

At **September 24, 2026, 03:45:24 CDT**:

| Cohort | Validated | Planned | Unresolved original timeouts |
| --- | ---: | ---: | ---: |
| Regional-CN single-region | 159 | 200 | 0 |
| SimClone1000 | 145 | 756 | 4 |
| PhylogicNDT500 | 1 | 500 | 9 |
| CliPPSim4K | 602 | 4,000 | 0 |
| Total | 907 | 5,456 | 13 |

There were 25 active CPU workers, ten hardware-verified Ready A100 workers,
seven running/seven pending cohort LSF jobs, and one running timing LSF job.
The cohort refiller was healthy. All 13 original timeout cases had active or
pending retries; they remain failures until a retry validates. There were no
output-validation errors. Of the 907 accepted results, 195 had incomplete
searches. The fifth timing result was imported; the sixth was running, with
the empirical continuation alive and waiting.

## Evidence retention and limits

The ignored local root
`results/lsf-recovery-20260924-v1` contains source-bound plans, initial ownership
and timeout proofs, held-job amendment evidence, transfer/launch receipts,
controller tests, the new registry, and the saved aggregate snapshot.
`results/cuda-surrogate-{timing,empirical}-20260924-v4` holds the live research
continuations. Earlier recovery helpers remain preserved: v2 exposed the
launch/accepted JSON schema difference before any new research job was submitted;
the v3 import exposed the zero-exit missing-job behavior. Both defects have
regression tests. No failed collector identity was overwritten or reused.

The [evidence index](evidence.json) binds selected immutable artifacts and their
actual hashes. Mutable progress files are deliberately excluded. A restarted
controller, allocated GPU, accepted retry or passing canary does not establish
successful completion of remaining cases, full-cohort accuracy or global
optimality. Incomplete-search results retain their status. No scientific
tolerance, iteration budget, clonal constraint or score penalty was changed.
