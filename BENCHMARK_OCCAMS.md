# OCCAMS runs

## September 24 reassignment: ten A100s plus four H100s

The user stopped Regional-CN and reassigned its ten A100 workers to OCCAMS.
Regional-CN's completed outputs remain preserved; its exact Job and Pods were
removed after stopping its supervisor. `results/CURRENT_RUNS.json` records the
stop override. Do not relaunch Regional-CN from an older registry assignment.

The user subsequently requested one shared pool for all fourteen GPUs.
`results/CURRENT_OCCAMS.json` identifies the draining predecessors and the
shared queue's A100/H100 workers. Two device-specific Kubernetes Jobs consume
the same 548-case queue through atomic claims; there are no fixed case subsets
or worker shards. The original predecessor retains its 18 accepted results
and four active fits at handoff. All 570 cases have unique ownership; small
qualification repeats are separate from cohort completion. The numerical
source and fit settings are unchanged from the fresh campaign below.

A100 routing requires the existing memory estimate to fit within 70% of a
conservative 39-GiB baseline; the per-case runtime guard still applies.
Each device pool must pass static, compiled FP64 capacity and separate real-fit
qualification before expansion. Workers claim the next eligible case when
free; H100s can claim from the entire queue. Claims are never stolen after a
crash. Replacements wait for exact predecessor Job/Pod and owner absence.
Reserved H100 frontier cases are released only after the old H100 generation
is absent. The combined allocation cap is ten A100s and four H100s throughout
handoff. See the pointer's report and status command for actual Ready/Running
allocation and gate outcomes. Memory remains per GPU for each individual fit.

## September 24, 2026: fresh four-H100 run

The user requested all OCCAMS inputs on four H100 workers, then explicitly
requested a full restart and deletion of existing results. The new attempt
imports **zero results** and includes all **570 single-region inputs** from
the reference manifest below. Original compressed input bytes, basenames and
retained mutation IDs were checked with the current CliPP1.5 reader.

Use `results/CURRENT_OCCAMS.json` for the exact attempt, source and read-only
status command. This campaign is independent of the active four-cohort
CPU/LSF/A100 campaign. The current production fit is unconstrained PyTorch
CUDA, float64, complete graph, with the closest-to-one refitted cluster labeled
0. OCCAMS has no supplied simulation truth: cluster count, sMF and numerical
status are descriptive results, not accuracy measurements.

On September 24, 85 files in 17 previous local CliPP1.5 OCCAMS result
directories were deleted at the user's request. File hashes and terminal
metadata were recorded in
`results/occams-h1004-20260924-v1/DELETE_PRIOR_OUTPUTS_INTENT.json`; deletion
was verified in `DELETE_PRIOR_OUTPUTS_COMPLETE.json`. Inputs, CliPP2 reference
results and unrelated simulations were retained. No old OCCAMS result is
eligible for import into this full restart.

Each worker uses one H100; memory is not pooled across workers. The unchanged
production preflight estimates `256*N*N + 8,388,608` device bytes and compares
that with 70% of currently free memory. Using a conservative 78 GiB device
baseline, 244 cases fall within this estimate and 326 exceed it. These are
planning estimates, not completed resource failures. Every scheduled case
still receives its actual runtime guard; failures are reported separately.
The largest input contains 32,504 retained mutations. Qualification probes
N=15,499, the largest input potentially admitted within the allowed 78–82 GiB
H100 profile, followed by a complete real small-case fit before scaling to four.

The first qualification Job was cleaned up before any scientific fit. Its
single-worker static manifest accidentally inherited Indexed completion mode;
Kubernetes inserted `JOB_COMPLETION_INDEX`, triggering the strict Pod-spec
guard. The replacement uses NonIndexed mode for static qualification and
Indexed mode with an explicit index field for the four execution queues.
Do not relax the admission comparison to hide unexpected changes.

## Historical September 21 CPU plan

Historical scope: this document records the September 21, 2026 source-bound CPU
plan and its historical output contracts. It is not live run status or authority
to launch or restart the plan. Statements about production and active jobs below
refer to that recorded attempt. The deletion and missing-path notes above
supersede historical statements about the availability of its fit outputs.

The current 0.5.0.dev0 [production framework](docs/CUDA_FRAMEWORK.md) requires
CUDA. The historical OCCAMS runner rejects the current CUDA-only package before
CPU admission or fitting. Historical evaluation must retain each attempt's
frozen source and output contract; substituting the current package does not
reproduce the original CPU run.

The user requested CliPP1.5 on the same inputs as CliPP2's OCCAMS Kubernetes
run, **after the current local simulation jobs finish**. This run covers all
570 single-region tumors with up to 25 local CPU workers, one fresh process,
one pinned CPU and one numerical thread per tumor. No OCCAMS fitting is
permitted during preparation or while the predecessor benchmark is active.

Local evidence is under
`results/occams-matched-20260921-v1/`. The deferred run's immutable plan and
source are in `full-run/plan.json` and `full-run/frozen/`; owner, activation,
status, failure and completion receipts are outside the scientific fit files.
The runner is [benchmarks/run_occams.py](benchmarks/run_occams.py).

## Inputs and reference results

The exact reference is Kubernetes Job
`clipp2-latest-occams-92492b6-20260920e`, UID
`c171580e-f8cf-4c2b-a909-17771958ca06`, using CliPP2 commit
`92492b6a85319370f768de7223ceab596aa368ed` and plan SHA-256
`dbc799599824a1bf0e545b54c9e8de63d9b9aa0cdb872f7adce872ccc5912536`.

The September 21, 2026, 9:48 PM CDT read-only snapshot found three Ready
workers, zero restarts and eight validated completed fits. The former five
worker queues were reduced to three; the removed queues include unfinished
cases. Completion of the reduced Kubernetes Job will not establish coverage
of all 570 tumors. This local run includes all 570 inputs, regardless of
their current remote scheduling state. Remote jobs were not changed.

All 570 canonical compressed TSV files were copied without conversion from
the input paths bound in the reference plan. Every remote hash and every
local readback hash matched. The twelve model columns contain coded mutation
and sample identifiers and counts/CN data. Production input parsing with
`max_major_cn=4` reproduced all 570 retained counts: **9,589,739 mutations**,
44–32,504 per tumor. Original compressed files total 102,917,202 bytes.
No SimClone-specific normal-CN repair applies to these OCCAMS inputs.

The eight available reference fits were imported only after validating their
source, input and plan identities, terminal/publication agreement, successful
scientific status, and hashes of the historical four-table output contract.
The local plan freezes this comparison snapshot. There is no remote polling
or automatic import of later reference results.

## Deferred execution

The predecessor is
`results/single-region-four-cohorts-20260921-v1/full-run`, plan SHA-256
`30dbf9bfbccfeb446431dfe2ac8e93822577cd418a9e94908c4e988f4fae395b`.
The waiting owner checks local receipts every 30 seconds. Admission requires:

1. The original predecessor plan and all **5,456 terminal cases**.
2. A hash-matched final summary and passing final audit, including the audit's
   case-metrics hash and status counts.
3. Exit of both the identity-bound predecessor controller and finalizer.

`HALTED.json`, a failed audit, changed identities or inconsistent counts
prevent admission. An absent/stopped controller by itself is never completion.
Numerical fit failures may be audited terminal outcomes; they are reported
and do not become successful fits. The owner has a persistent exclusive claim:
an uncertain or stopped attempt cannot silently restart or duplicate work.

Once admitted, tumors run in increasing retained-mutation count, with at
most 25 active fresh processes. Each has a 72-hour operational timeout,
matching the reference cohort's per-case ceiling. This changes no scientific
fit setting. Setup, process or validation failure stops new admission and
drains already active workers. No automatic retries are performed.

The production package is unchanged from the four-cohort integration;
the plan pins the committed runner, complete source file hashes, actual
`ml1` interpreter/dependency versions, input hashes and retained-ID hashes.
This is a local CPU run on a shared host, not a GPU execution comparison.

## Output and interpretation

Each successful case retains the four public CliPP1.5 files plus external
startup, process, metrics and terminal receipts. Validation checks source and
input identity, all table hashes, retained mutation IDs and
candidate/refit/raw-reference provenance. Legacy v4 outputs require their occupied
clonal designation; v5 outputs require no designation and zero clonal flags;
v6 designates only the fitted cluster closest to CCF one as public label zero. A direct partition
cannot inherit a raw KKT certificate.

The final audit rechecks successful output/metric identities before publishing
`COMPLETE.json`, `summary-final.json`, `case-metrics.tsv` and `REPORT.md`.
Progress is available in `status.json` and `summary-current.json`.

OCCAMS has no supplied clustering truth. Report cluster counts, runtime,
memory, incomplete-search status and subclonal fraction. Where a validated
CliPP2 reference exists, report partition ARI **as agreement**, mean absolute
final-refit CCF difference and all-exact-one sMF. Designated-cluster sMF applies
to historical constrained results and current nearest-to-one v6 results; it is
unavailable for the intermediate undesignated v5 schema. Current primary sMF is
the fraction outside the designated closest-to-one cluster.
These are not truth accuracy measures, and eight smaller completed reference
cases cannot establish full-cohort agreement. Different CPU/GPU environments
also prevent interpreting raw wall-time ratios as algorithm-only speedups.

## Validation

`tests/test_occams_queue.py` covers dependency/audit gates, PID reuse, source
drift, duplicate-owner exclusion, bounded CPU admission, failure draining and
a fresh-process synthetic worker/output check. The synthetic fixture is not
an OCCAMS fit. No real OCCAMS fit starts before the dependency is satisfied.
