# OCCAMS matched-input CPU run

Historical scope: this document records the September 21, 2026 source-bound CPU
plan and its historical output contracts. It is not live run status or authority
to launch or restart the plan. Statements about production and active jobs below
refer to that recorded attempt; its receipts and evidence remain unchanged.

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
