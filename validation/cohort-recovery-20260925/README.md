# Timeout and export recovery — September 24, 2026

This record uses America/Chicago display times; its directory date is UTC.
Current status is separate from this historical snapshot. Follow
[CURRENT_RUNS.json](../../results/CURRENT_RUNS.json) and
[OPERATIONS.md](../../benchmarks/OPERATIONS.md) for future monitoring.

## PyClone-VI: 4,000 validated CN-first results

Four fits passed inference but failed export validation because their standard
deviations contained blank values. The exporter used `E[x²] - E[x]²`; highly
concentrated posteriors with normalization errors around 1e-13 produced tiny
negative variances and NaN square roots. A frozen exporter overlay calculates
the nonnegative centered sum `sum(q * (x - mean)**2)`, retaining the original
mean and rejecting invalid probabilities.

The four cases were `500_1_0.9_0.7_rep72`, `500_4_0.9_0.7_rep91`,
`500_3_0.9_0.7_rep76` and `500_2_0.9_0.4_rep122`. Only previously blank
standard-deviation fields changed. Every fitted HDF5 file is byte-identical;
labels, CCFs, assignment probabilities and all other TSV fields are unchanged.
No fit or restart was repeated. The original validator checked all 1,000
restarts, best ELBO, posterior normalization, mutation coverage and exports.
The other 3,996 outputs were hash-reverified. Six focused export tests passed.

The immutable recovery root is
`/storage/CliPP2/PyCloneVI_runs/CliPPSim4K_CNfirst_20260924_export_recovery_v1`.
Its `RESULTS_MANIFEST.json` resolves all 4,000 effective results; its own
`results/` contains only the four reexports. The manifest SHA-256 is
`9da0496447cf96d31d789f5b0c6fa791001c4dd864ad574e77252b6131230c05`.
Completion was recorded at **September 24, 2026, 11:44:36 PM CDT**.
The PyClone-VI current pointer now exposes this manifest and corrected status
command while preserving the original inference root and failure evidence.

## CliPP1.5: preserve admitted jobs and extend timeout allowance

All six outstanding LSF failures were terminal `resource_timeout` results,
with about 357 minutes of application runtime under six-hour job limits and
a three-minute cleanup reserve. They were not numerical failures or OOMs.
The affected keys are `001368`, `001276`, `000905`, `001316`, `001235`
and `000591` (four PhylogicNDT500 tumors and two SimClone tumors).

The replacement plan contains 513 unsubmitted cases, six explicit retries and
14 conditional parent cases. Each new fit receives an 18-hour wall allowance.
Each bound timeout receives at most one retry; there is no automatic retry for
other failures or another 18-hour timeout. Parent successes are reused. The
identity-verified parent controller stopped refilling and continues draining
its 14 admitted jobs. No running or pending parent fit was cancelled.

The successor counts borrowed, pending and uncertain jobs against the total cap
of 15. The research reservation was released only after all six timing and
eight empirical stages had terminal completion evidence. Fresh queue checks
confirmed GB memory units, reservation semantics and a 30,240-minute hard
queue maximum; the original case-specific host memory contracts remain bound.

The new root is
`/rsrch8/scratch/bcb/yding4/clipp2_clipp1d_timeout_recovery_20260925a`.
Its plan SHA-256 is
`961a837a294d9181eff9225a715ca532c25b64591c069b753ebce8142b018598`.
The new controller launched on ldragon5, PID `1218981`, start ticks `65032129`.
First retry **77391536** (case `001368`) was accepted and released from hold.
At **September 24, 2026, 11:53:59 PM CDT**, it was pending; aggregate LSF
occupancy was 11 running and four pending. The remaining retries were queued
inside the controller. None was counted as a successful recovery at launch.

The numerical source fingerprint remains
`b16f1fc8068351aeeb75dfe0f940a205d2699347f3ce0799f0e349cbf51aba99`.
The worker, input population, environment, compiler, likelihood, graph, score,
numerical budgets and unconstrained policy remain unchanged. Exact-parent
allocated-CUDA qualification and full-fit canary evidence were revalidated and
reused. This is an orchestration and resource repair, not a new numerical
qualification or proof that all remaining cases finish within 18 hours.

## Monitoring, verification and preserved work

[The retry-lineage validator](../../benchmarks/lsf_retry_lineage.py) now handles
successive immutable retry generations. It requires the exact terminal timeout
parent, accepted job/key, receipt hash and conditional allowance; it rejects
duplicate successes, ambiguous generations and unsafe receipt paths. The new
status helper includes this validator and the current registry was advanced
atomically, retaining its prior copy.

Focused verification passed **25 tests**: 13 repository LSF admission/lineage
tests, six attempt-local handoff tests and six export tests. Source compilation,
targeted Ruff checks and `git diff --check` passed. These operational tests do
not establish numerical or cohort-wide accuracy.

The saved CliPP1.5 simulation snapshot has 1,153 validated outputs of 5,456:
192 Regional-CN, 739 original Sim4K, 46 PhylogicNDT500 and 176 SimClone.
Incomplete-search outputs retain that status. Six LSF timeout records remain
failures until their retries validate. User-stopped Regional-CN and local
CliPP1.5 CPU work were not resumed.

OCCAMS remained healthy at **September 24, 2026, 11:54:30 PM CDT**:
39/570 validated outputs (23 complete searches, 16 incomplete), all ten A100
and four H100 workers Ready, and no validation errors. The independent
PhylogicNDT CN-first campaign continued on 28 CPU workers without modification.

The local attempt directory `results/timeout-recovery-20260925-v1` retains the
sealed payload, scripts, tests, launch, handoff, transfer and snapshot receipts.
[evidence.json](evidence.json) binds selected immutable artifacts. Do not rerun
its successful stage or launch commands, hot-patch a live frozen controller,
erase original failures or infer completion from a scheduler acceptance.
