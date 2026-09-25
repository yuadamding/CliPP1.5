# Follow-up recovery — September 24, 2026

This record covers the subsequent “fix all” request. The mutable live pointer
is `results/CURRENT_RUNS.json`; the new immutable registry and recovery evidence
are under `results/run-failures-recovery-20260924-v1`. Existing successful fits
and healthy workers were preserved.

## Research artifact import

Empirical Job **77355927**, case `000007`, passed on the GPU. Its collector
rejected the original `.clipp2.txt` input because that extension was absent
from the artifact suffix allowlist. `validate_import_paths` now accepts an
original input only at the sealed task's exact relative path and SHA-256;
unbound files keep the restricted suffix policy. Tests cover changed bytes,
unbound filenames, duplicate paths, traversal and symlinks.

The passed job was re-imported into a fresh local destination: 16 artifacts,
result SHA-256
`2b92756ce051b3395aa8b7d89219be1ae63b8df845d09c06cc7e5b7f7b500e8b`.
No repeat GPU submission was made for it. The replacement empirical controller
revalidated the six timing stages and all four completed empirical cases,
then submitted the fifth case as **77360198**. Its source, inputs, deadline
and one-job research allowance are unchanged.

## CPU timeout recovery

Sim4K cases `001850` (284 retained mutations) and `003026` (297) reached their
360-minute process limits without worker terminals. Each has one explicitly
authorized 1,080-minute retry; no numerical iteration or qualification limit
changed. The v5 controller stopped refilling and retained its 25 live fits.
The v6 controller borrows their occupied cores and starts queued work only as
cores are released. The total cap remains 25. A handoff does not grant extra
retries to other previously admitted work.

The successor is `results/four-cohorts-local-cpu25-20260924-v6`. Its sealed plan
binds the parent process identity, complete accepted set, deferred cases,
failure history and unchanged source archive. The first retry began while
24 parent fits continued.

## Regional-CN numerical diagnosis

Case `000109` failed its scalar pilot at batch offset 2,176 with a nonfinite
maximum gap. A literal 64-row eager CPU diagnostic qualified every row; this
does not qualify CUDA or identify the cause. One source-identical A100 replay
adds failure-only capture of the scalar inputs/results and returns the
original solver result without changing admission.

The retry is `results/regionalcn-scalar-recovery-20260924-v3`, remote
`/rsrch8/home/bcb/yding4/clipp1d_regional_scalar_20260924c`, GPU Job UID
`b8c1df68-c50c-44b2-b1c6-7a0238df4b46`. Its static gate passed; each launch
checks the original Job UID, completed indexes, zero automatic Pod retries and failed
case terminal before admitting one additional GPU within the Regional-CN
cap of ten. Healthy original queues remain intact. Earlier preparatory roots
are retained: v1 was never launched; v2 stopped on a missing helper import
before creating a Kubernetes Job. The corrected v3 helper passed undefined-name
checks before publication.

The numerical fingerprint remains
`b16f1fc8068351aeeb75dfe0f940a205d2699347f3ce0799f0e349cbf51aba99`.
This operational replay is not a claim that the nonfinite arithmetic cause
has been repaired. Failed evidence and any captured tensors remain available
for further diagnosis. The already running LSF retries `77355905` and
`77355907` were not duplicated. The fresh four-H100 OCCAMS run was preserved.

## Verification

**51 focused repository tests passed**, covering artifact import, CPU process
identity/handoff, failure classification and A100 dispatch/lifecycle. Ruff
passed for the changed transport helper/tests; `git diff --check` passed.
No production numerical source changed in this follow-up. CPU diagnostics,
CUDA resource gates and validated full fits remain separate evidence.

## Saved execution snapshot

At **September 24, 2026, 11:07:31 CDT**, the four simulation cohorts had
**1,089/5,456 validated** fits: Regional-CN 184/200, SimClone 161/756,
PhylogicNDT 20/500 and Sim4K 724/4,000. Of these, 253 retain incomplete searches.
Four original failures remain counted pending validated replacement outputs:
Regional-CN `000109`, PhylogicNDT `001390` and the two CPU timeouts.
PhylogicNDT `001379` recovered successfully during this repair.

There were 25 active CPU workers; 10 running and four pending cohort LSF jobs;
one pending empirical LSF job; eight original A100 workers plus the one-case
A100 replay; and four Ready OCCAMS H100 workers. OCCAMS had **9/570 validated**
fits, including one incomplete search, with no failures. Both aggregate
snapshots reported no output-validation errors. This is a dated snapshot,
not a claim that queued/running retries have completed.
