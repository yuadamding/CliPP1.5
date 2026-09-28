# CliPP1.5 operations and reusable lessons

This guide records lessons verified against the September 23, 2026 source and
saved recovery evidence. Recheck the pinned implementation when contracts
change. It is not a live status report or standing authorization to submit jobs.
The [dated incident record](../validation/cohort-recovery-20260923/README.md)
contains observed counts, source identities and qualification scope.

## Find the current owner before acting

On the original workstation, `results/CURRENT_RUNS.json` is the aggregate entry
point. It names the registry and its SHA-256, case-owner map, exact read-only
status command, latest saved snapshot and human report. Read that pointer;
check the registry hash before using it. Do not execute an arbitrary command
from an unreviewed file. Older `CURRENT_A100_RUN.json`, `CURRENT_LSF_RUN.json`
and `CURRENT_HYBRID_RUN.json` are pool-specific historical handoffs and may lag
the aggregate registry. `results/RUNS.md` is only a navigation aid.

Independent campaigns can be linked through `additional_run_pointers` in that
entry point. The September 24 OCCAMS campaign uses
`results/CURRENT_OCCAMS.json`; an all-runs status request must include its
separate exact-ID snapshot. It imports no simulation or earlier OCCAMS fits.
See [OCCAMS scope and memory limits](../BENCHMARK_OCCAMS.md).

The later September 24 user instruction stopped Regional-CN and transferred
its ten A100 workers to OCCAMS, preserving the existing four-H100 capacity.
Honor `regional_cn_execution_override` in `CURRENT_RUNS.json`; an older frozen
registry is historical authority for results, not permission to revive its
stopped workers. OCCAMS's v3 pointer lists its draining predecessors and the
two hardware groups consuming one shared queue, as requested subsequently.
A cap or installed continuation is not allocated capacity.

The September 26 multiplicity-prior proposal study has its own
`results/CURRENT_PRIOR_STUDY.json` pointer, linked through
`additional_run_pointers`. Its single LSF study slot is separate from the
20-job SimClone/Phylogic pool. Read the [frozen study and handoff](../validation/prior-perturbation-v1/README.md)
before interpreting its stages: qualification, 108 development comparisons,
12 diagnostic factorial cases, 382 pilot-only discovery cases, 324 confirmation
comparisons and nine cost fixtures are different units. Auxiliary failures do
not erase the preserved B result; incomplete extra-search coverage is separate
from publication failure. This experiment does not promote a new default or
revive any stopped Regional-CN/OCCAMS campaign.

The first prior-study canary exposed a cold-process CUDA initialization bug:
visibility checks and a separate worker probe do not initialize the fitting
child's allocator. Synchronize before resetting CUDA memory statistics, and
qualify actual fresh children before the parent owns an exclusive GPU context.
The [startup recovery](../validation/prior-perturbation-v1/recovery-20260926c/README.md)
also fixes the case-sensitive qualification receipt consumed by panel admission.
Use the current pointer for the retry; preserve the original qualification and
failure receipts, and bind benchmark hashes even when the numerical fingerprint
is unchanged.

For fixed-queue handoff, publish the supported drain flag before reserving each
queue's started prefix and first unstarted frontier. An already-admitted case
may create its directory after the flag; only keys beyond that frontier are
safe to transfer immediately. Preserve current fits. A successor sharing the
old GPU allocation must wait for exact predecessor Job/Pod and owner absence,
and reject any case-directory overlap before launch. Keep qualification repeats
outside cohort totals and verify cross-pool result uniqueness during status.

Shared OCCAMS dispatch uses atomic case-directory creation on the common PVC,
with a receipt binding the winning pool and Pod UID. A100 eligibility uses the
unchanged memory estimate; H100s consume the same queue. Never reclaim an
existing directory automatically after a timeout or missing receipt: its
owner may still be writing. Hardware-specific Job completion is distinct from
whole-cohort coverage, and status must count each shared result only once.

The ignored `results/` tree is local evidence, not part of a fresh Git checkout.
If it is unavailable, use the tracked evidence record to identify what is
missing; do not reconstruct live owners from historical PIDs or Job names.

For a status request, read receipts and take one exact-ID snapshot through the
approved access chain. Report date and time in `America/Chicago` with CST/CDT.
Preserve UTC evidence. Include allocation, validated/failed/planned counts,
incomplete searches, controller health, validation errors and a concrete blocker.
Use an ETA only when comparable completed runtimes support it. A small number
of quick completions is not a representative full-cohort sample.

Documentation-only work needs no remote session. For remote work, use the
installed [Seadragon skill](/home/yding1995/.agents/skills/seadragon/SKILL.md) and
only its requested scheduler route. CliPP2-specific skills and the neighboring
repository do not override CliPP1.5's unconstrained scientific contract.

## Soft-mixture GPU scheduling recovery, September 28

The expanded study initially restricted its qualification and case jobs to
L40 GPUs. The qualification waited nearly four hours while compatible A40
capacity was available. An authorized resource-only transition removed the
qualification's model restriction with `bmod -gpu
'num=1:mode=exclusive_process' 77453726`; that same job then started on an A40.
This is dated evidence, not a reusable job ID or live capacity observation.

LSF applied the accepted `bmod` asynchronously: the immediate `bjobs -UF`
response still showed the old request. Preserve the mutation response and
reconcile the effective request after rescheduling; do not repeat a successful
mutation merely because its first readback is stale.

The old panel controller was cooperatively stopped and its terminal identity
verified before installing a new immutable controller generation. The successor
uses the hardware model from the qualification's hash-verified allocation log
for both scheduler requests and actual worker-device checks. Qualifying one
model does not qualify another model. Numerical source, input/seed bytes,
candidate budgets, scoring, publication checks and the one-job cap remain fixed.
Follow `results/CURRENT_MIXTURE_STUDY.json` for the current generation and its
separate resource-change receipt. Do not edit the older frozen payload or restart
its retired owner.

## Experimental resource transfer, September 28

The user subsequently requested: “stop all Previous CliPP1.5 runs and use all
saved resources to test Experimental.” This supersedes the previous allocations
and the Experimental one-slot cap. All older baseline CN-first, SimClone/Phylogic
and prior-proposal controllers were stopped before their exact GPU children.
The ten A100 and four H100 Jobs and UID-owned Pods were verified absent; the
20 baseline LSF jobs and one prior-study job were terminated. Previously stopped
OCCAMS, Regional-CN and local CliPP1.5 work remains stopped. Other methods are
not covered by this stop instruction.

`results/experimental-pool36-20260928-v1` binds the transition. Follow
`results/CURRENT_MIXTURE_STUDY.json` for the active generation. The shared pool
uses 10 A100 workers, 4 H100 workers and a 22-slot rolling scalar LSF controller.
Every case receives exactly one atomic directory claim across schedulers; a
missing or failed receipt never releases its claim automatically. Pending and
uncertain LSF submissions retain capacity. One LSF job still fits one tumor.
Kubernetes completion indexes identify workers, not case indexes.

The old Experimental owner was stopped at a receipt-verified boundary of 603
validated cases. All are explicitly imported with their original source,
manifest, job and output hashes, leaving 2,494 of the unchanged 3,097 development
cases. Their original artifacts remain in scratch. New shared worker data lives
under the approved home project subdirectory because Kubernetes cannot see
scratch. No truth files are supplied to inference, and qualification repeats
are outside the cohort totals.

The numerical source remains the previously qualified soft-mixture/structural
selector. A40 qualification is reused only after exact source/environment checks.
A new LSF worker canary gates expansion to 22. Each Kubernetes family starts
with one worker, executes all three allocated-CUDA tests, checks three component
outputs against the bound A40 references and guarded outputs against independent
CPU references, then checks the largest input for capacity. Its exact supervisor
scales to the authorized count only after that family qualifies. Requested slots,
Running/Ready Pods and verified actual GPUs are reported separately.

The institutional admission webhook overrides a requested 1-CPU/32-GiB shape
with 24 CPUs and 100 GiB per A100 (200 GiB per H100). The first A100 dry run
correctly failed before Job creation. A separate immutable Kubernetes control
generation binds the enforced shape; it does not modify the already running
LSF owner or frozen numerical worker. Keep the failed dry-run evidence. PyTorch
still uses one compute thread. LSF retains its verified 1-CPU/32-GB contract.

The first A100/H100 qualification workers then failed before numerical tests:
the mounted `ml1` interpreter has no pytest. The successful A40 qualification
had supplied a private, hash-bound `test_dependencies` tree; copying only the
case-worker payload omitted it. Do not install into the shared environment or
treat this as a numerical failure. The next Kubernetes-only control generation
copies that exact dependency tree into its visible home directory, verifies
`python -S -m pytest --version` locally and remotely, and exposes it only to the
test subprocess. Use a new qualification namespace and new Job identities after
proving both failed Jobs and their owned Pods absent. The healthy LSF owner and
all validated results remain intact. Each successor worker records its control
generation hash in its ownership and validation receipts.

On A100, the first full SimClone qualification exposed an intermittent compiled
posterior row with almost zero mass instead of one. The independent EM
monotonicity guard correctly rejected the update. Eager likelihood/M-step replay
agreed with the recorded objectives; loosening the guard would hide malformed
posterior output. Returned-tensor mutation was excluded by a separate allocated
diagnostic. Ten successful compiler repetitions were insufficient: disabling
persistent reductions and epilogue fusion still failed at repetition 16 of a
longer stress test.

The qualified A100 `k8s-v5` execution wrapper additionally sets
`max_fusion_size=1` and `allow_buffer_reuse=False`, with a generation-specific
compiler cache. All mathematical source files and tolerances remain unchanged.
It passed 2,000 eager stability repetitions, 6,000 compiled posterior checks,
three allocated-CUDA tests, three paired publication cases and the N=2,180
capacity case, then scaled to ten. At that launch the H100 `k8s-v3` owner and
scalar A40 LSF owner retained their existing configurations. Do not generalize this
result into a diagnosis of a specific upstream compiler bug or apply a new
configuration to a running owner. See the
[qualification record](../validation/experimental-pool-20260928/README.md).

Compiled caches are shared within, and separated between, hardware families;
temporary directories and GPU ownership remain worker-specific. The memory
profiler must catch a successful helper `SystemExit(0)` before publishing peak
memory; otherwise a successful fit silently skips its telemetry. Unit checks
cover this, duplicate claims, uncertain scheduler state and complete activation
identity tests. All completed output hashes are read back before counting a case.

The later terminal snapshot had 3,092/3,097 validated cases: one A40 guard
failure, one H100 guard failure and three H100 shutdown interruptions. The
remaining A100 queue completed cleanly. The explicit `terminal-five-v1`
recovery uses new claims/outputs and binds every failed predecessor receipt;
never remove a failed original claim to put it back in the old queue. Both
affected GPU families must qualify the new compiler generation separately.
Original failures remain history even after successful replacement. Original
status scripts do not see recovery subdirectories; use the current pointer's
recovery status command and count each case once.

One original LSF job also reached RUNLIMIT after publishing validated output.
Its logs do not prove which process survived. A reproducible worker cleanup
gap allowed compiler descendants to outlive their Python leader. The repaired
`case_helpers.run_child` uses a scoped Linux subreaper and bounded cleanup of
its own child group on success, failure, timeout and interruption. Group
absence is recorded before validation; a valid table alone does not establish
successful scheduler exit. Keep the original output, retain the scheduler
anomaly, and use a separate paired replay to verify the shutdown repair.

The [terminal recovery](../validation/experimental-pool-20260928/RECOVERY.md)
completed all 3,097 cases and cleaned every worker. The subsequent
[complete evaluation](../validation/experimental-pool-20260928/PERFORMANCE.md)
includes all five replacement outputs via `RECOVERY_RESULT_INDEX.json` and
supersedes the older 1,754-case performance snapshot. Read the current study
pointer for the local report and per-sample tables; do not restart a retired
controller or resolve replacement output paths from the original input root.

## Preserve execution and scientific identity

| Identity | What it establishes | What it does not establish |
| --- | --- | --- |
| Full base commit | Git ancestry | Inclusion of dirty or untracked repairs |
| `src/clipp1d` fingerprint | Numerical package bytes | Worker, controller, staging or environment identity |
| Complete source archive, patch and inventory | Reviewed executable overlay | Correct input, environment or GPU admission |
| Input/truth hashes, original basename and retained-ID hash | Exact observations and evaluated population | Fit completion or numerical qualification |
| Plan, wrapper, interpreter, compiler, environment and image bindings | Intended execution contract | Actual allocated device or successful output |
| Worker startup and Pod/Job identity | Actual owned execution | Validation of the scientific result |
| Validated receipt plus matching output hashes | Accepted case output under its validator | Complete search or global optimality |

`git archive HEAD` alone drops uncommitted fixes and new helper files. Freeze
the complete authorized overlay, and bind controller/worker bytes separately
from the numerical fingerprint. Never hot-patch a healthy frozen owner or its
helpers. A change limited to orchestration can preserve numerical identity
while invalidating lifecycle/static qualification.

Retained successful results keep their original source, input, environment and
job provenance. Mixed-source recovery is explicit; do not relabel the aggregate
as a homogeneous latest-source benchmark. A full redo and an import-based
recovery are different experiments.

## Distinguish completion states

| State or receipt | Interpretation and next action |
| --- | --- |
| LSF `PEND`, suspended or unknown | Accepted work still occupies a slot; never duplicate it |
| LSF `DONE` | The wrapper exited successfully; inspect its case outcome, which can be a timeout |
| Kubernetes phase `Running` | Check Ready, deletion/termination, restart and hardware receipts before reporting a working GPU |
| Kubernetes Job `Complete` | Dispatchers exited; inspect coverage and per-case outcomes |
| Lifecycle `COMPLETE.json` | Recognized terminal outcomes cover the planned cases; some outcomes may still be failures |
| `validated_complete` | Accepted selected result and complete bounded planned search, not all possible penalties or a global optimum |
| `validated_incomplete` | Accepted selected result with unresolved planned search work; report separately |
| `scientific_failure` | No accepted result; preserve numerical diagnostics |
| `resource_failure` / `resource_timeout` | Memory or wall budget was exhausted; not a validated fit |
| `execution_failure` | An unexpected fit failure; retain the traceback and distinguish isolated from repeated failures |
| `infrastructure_failure` in the A100 worker | Retire that dispatcher and preserve its unstarted queue; independent GPUs can continue |
| Setup/input or output-validation failure | Stop new admissions and drain already admitted work |
| `FAILED.json` / `PARTIAL.json` | Preserve scheduler failure or uncovered cases; neither is successful cohort completion |
| `cleanup-pending.json` | Absence was not verified; no automatic retry permission or implicit cleanup watcher |

See [failure classification](cohort_failures.py), the
[A100 dispatcher](run_a100_cohort.py), [lifecycle](supervise_a100_cohort.py) and
[failure-isolation contract](COHORT_RECOVERY.md). The CPU controller has its own
allowed terminal states; do not assume every shared status is accepted by every
historical controller.

## CPU ownership and recovery

1. Admit a worker only after its command line is nonempty and exactly matches
   the intended argv. Check stable `/proc/PID/stat` birth ticks, process group
   and session around the command-line read. Reject PID reuse and exited tasks.
2. Persist admission and independent worker startup identities. PID existence
   alone is not ownership, and a hash of an empty command line is not admission.
3. Reconcile each active worker independently. Save an explicit error for a
   failing entry and continue reconciling other workers, including completed
   successes. A previous drain reason must not conceal a later exception.
4. Clean up and reap a child whose admission failed. Retain its failed-admission
   receipt; never lose an unregistered live process from capacity accounting.
5. Preserve healthy workers during handoff. Count borrowed parent cores and new
   workers together, require disjoint CPU affinities, and recheck host headroom.
   A live draining parent is expected until its final child is reconciled.
6. Deferred parent success is reused after output validation. Parent failure
   permits only the explicitly authorized retry; a new owner does not reset an
   existing retry budget. Keep old terminal receipts unchanged.

Encode remaining authorization explicitly in `deferred_parent_retry_keys`.
Omitting that field allows every eligible isolated parent failure to retry;
the controller does not independently reconstruct historical retry counts.

The September 23 incident combined an empty command-line admission hash with
controller-wide reconciliation failure, hiding five successful exited workers
and an overdue live worker. The recovery independently verified argv,
birth/group/session and worker startup before signaling through a Linux pidfd.
The `ml1` Python lacked `os.pidfd_open`; a system interpreter was used only for
that operational signal, never for numerical work. Do not generalize this
exception into mixing numerical environments. Preserve the original
`execution_failure` caused by an external stop and attach its timeout explanation
instead of rewriting history.

See [CPU controller](run_complete_graph_cpu.py),
[identity tests](../tests/test_cpu_process_identity.py) and
[handoff tests](../tests/test_cpu_cohort_handoff.py). Worker counts, wall/RSS
limits and reserved headroom belong to the frozen plan, not to a reusable
machine-wide constant.

## A100 node and lifecycle recovery

The September 23 attempt on `hpcgpu11` had six CUDA unspecified-launch failures,
followed by twelve CUDA initialization failures. This establishes a correlated
device/node execution incident; it does not establish a driver, fabric-manager
or hardware root cause. Exclusion of that node belongs to the replacement
attempt, not to a permanent cluster-wide blacklist.

The repaired per-case child runs its hardware probe inside the classified
failure boundary. A device failure retires the affected queue immediately,
instead of spending several cases on an unusable GPU. The initial `dispatch`,
`static` and `capacity` probes still run outside that per-case boundary; diagnose
those failures from phase and scheduler evidence. Repeated generic fitting
errors retain the separate three-consecutive-error pool-drain guard. A node
eviction can still cause the Indexed Job to fail and interrupt healthy Pods;
application failure isolation does not guarantee scheduler-level isolation.

Use this order for an authorized replacement:

1. Bind the exact old run name, UID, source/plan and supervisor identity. Capture
   terminal conditions, case outcomes and bounded owned Pod logs. Record missing
   or timed-out logs as unavailable; do not hide the Pod state.
2. Reconcile an existing delete intent before any further deletion. Use UID and
   fresh resourceVersion preconditions for an authorized initial delete. A
   foreground delete or terminating Pod is not proof the computation stopped.
3. Require exact Job and UID-owned Pod absence before overlapping retries.
   Do not force-delete unknown execution merely to unblock replacement. If
   absence is delayed, use a single bounded, identity-bound continuation that
   observes only the old identities and creates nothing before the gate passes.
4. Revalidate successful output hashes and the unfinished case set after
   absence. If outputs changed while waiting, rebuild the retry plan rather
   than duplicating newly completed cases.
5. Use a new absent root, immutable package and manifest. Apply diagnosed node
   exclusion to static and GPU manifests and audit actual admission. Prove
   worker mounts, ownership and visibility through the institutional template.
6. Run the invalidated static, CUDA-audit, capacity and paired-fit gates. Create
   suspended, bind the actual admitted UID/template twice, activate once, and
   let the bound supervisor expand to the authorized cap after qualification.

Kubernetes A100 worker inputs and outputs use the approved visible home/PVC
mapping. LSF defaults to approved scratch roots. Login-node path readability
does not establish worker visibility. Preserve old trees; do not migrate active
attempts or invent mounts. A remote-command yield is not failure: resume the
same invocation, or reconcile its receipts before retrying any transition.

The supervisor's bounded cleanup can end with `cleanup-pending.json`; it does
not itself wait indefinitely or launch replacement work. The September 23
deferred replacement used a separate source-bound local controller. Likewise,
its handoff receipt means launch succeeded, not that 118 cohort fits finished.
Do not reinvoke the normal lifecycle as an attach shortcut: its evidence writes
are no-clobber and supervisor startup expects the original suspended, zero-Pod
activation state. Recovery needs its own reviewed identity and transition.

## LSF throughput and timeout recovery

Use one tumor per scalar submission, one exclusive GPU and a fresh process.
Count Running, Pending, suspended, uncertain submissions and preserved old jobs
against the **combined** cap. Independent controllers cannot safely each infer
that unused slots are theirs.

Use login-shell LSF tools. For each new resource contract, verify memory units,
reservation/per-task semantics and the queue's maximum RUNLIMIT. Distinguish
host reservation, host hard limit, CPU affinity/thread count and GPU VRAM.
Smaller host reservations can improve placement without changing numerical
work, but need measured justification and verified effective admission.
Do not alter healthy work merely to make the queue ordering prettier.

Start quick-first by retained workload `(N*N*R, N, R, manifest_index)`, unless
user priority or reliable matched runtimes provide a better ordering. Memory
admission does not predict full-fit duration: many starts, penalties, refits
and scalar-search intervals can dominate even a small case. Cold compilation,
warm kernel timings, profiled timings and full-fit wall time are different
measurements. Shared-cache reuse does not establish numerical equivalence.

The three September 23 legacy timeouts exhausted a configured **360-minute
fit budget**, not the queue's observed 30,240-minute maximum. Each received one
new 1,080-minute retry with unchanged numerical budgets. Those limits are dated
contracts, not future defaults or completion guarantees. Their separate retry
owner waits for both existing owners to finish and all accepted jobs to reconcile;
it submits no retries while the refilling owner is active. An incomplete parent
stop or uncertain submission blocks that gate. Extending wall time must never
silently change solver iteration limits, score penalties or qualification gates.

### September 24 admission and research-collector follow-up

The September 23 reduced-reservation contract did **not** preserve its requested
hard limit in actual admission. A held canary submitted with `-M 8` and
`rusage[mem=4]` showed **MEMLIMIT 4 G**. Qualification with both values set to 8
had admitted 8 G. Use equal reservation and intended hard-limit values for this
contract, then verify the effective limit; do not bypass the admission guard or
claim that a requested hard limit was preserved merely from the `bsub` argv.
The [admission helper](lsf_admission.py) rejects mismatched requests before
submission and reports expected versus admitted values on failure. Record raw
admission observations **before** checking them, including rejected admissions.

An acknowledged `bmod` is asynchronous. Its first immediate `bjobs` response can
still show the previous values. Preserve the modification receipt, reconcile
the same held ID, and release only after the complete effective contract matches.
Do not repeat the modification or resubmit merely because readback is delayed.

The September 24 replacement admits the 589 unsubmitted cases and one bounded
1,080-minute retry for each of 13 observed terminal timeouts. The last healthy
parent case remains running; its result is reused if valid, and only a terminal
resource timeout permits its one conditional retry. The old separate retry
owner was retired without submitting jobs. Fourteen cohort slots plus one
serial research slot keep the combined LSF ceiling at 15, including Pending
and unknown jobs. This is a dated recovery contract, not new launch authority.

The timing collector lost its SSH connection before importing a subsequently
successful fifth stage. A collector failure is not a failed GPU task. Recover
the exact accepted job's terminal evidence before deciding to rerun anything.
Use bounded read-only calls and limited transport retries; do not retry launch,
submission or cancellation through the same mechanism. Preserve full transport
exit status and stderr so a banner cannot obscure the diagnostic.

`bjobs -UF` can return **exit 0 with empty stdout and “Job <ID> is not found” on
stderr** after scheduler history expires. This is not completion proof. The
recovery verifies the retained LSF terminal log's job ID, job name, owner,
timestamps and terminal outcome together with the application terminal and
artifact hashes. Keep that evidence explicitly labeled as a retained log, not
fabricated scheduler output. Launch receipts add `released_acknowledged` to the
accepted identity; compare the bound accepted fields rather than requiring the
two JSON objects to have identical schemas. See
[research evidence checks](research_transport.py) and the
[September 24 recovery record](../validation/cohort-recovery-20260924/README.md).

Research input artifacts keep their original basenames, including `.clipp2.txt`.
The collector must bind these exceptions to the sealed task's exact relative
input path and SHA-256. Do not broaden the artifact allowlist to arbitrary text
files. Reject traversal, symlinks, duplicate paths and changed bytes. A successful
remote task with an import failure needs a fresh local evidence destination and
re-import, not another GPU submission. Preserve its accepted job and terminal
identities, then resume the study after revalidating all predecessor imports.

CPU timeout recovery can hand off the refilling controller without stopping its
fits: the parent drains, the successor reserves each borrowed core until its
parent worker exits, and both retain the total 25-worker cap. Bind each retry's
prior terminal and budget explicitly. In the September 24 follow-up, cases
`001850` and `003026` received one 1,080-minute retry each after 360-minute
timeouts. This is a case-specific allowance, not a new unlimited retry policy.

A scalar-pilot diagnostic with a nonfinite maximum gap is not evidence that the
interval budget is too small. Preserve the original failure and capture the
literal failed tensors during a source-identical CUDA replay. An eager CPU pass
does not qualify the GPU path or establish the arithmetic cause. Failure-only
instrumentation must return the original result, with no fallback, changed
tolerance, mutation removal or inherited certificate.

## Inputs, evaluation and scientific interpretation

- Apply the user's [three-cohort accuracy requirement](../docs/THREE_COHORT_ACCEPTANCE.md)
  before adopting a revision: preserve CN-first4K and evaluate SimClone/Phylogic
  separately with one source/configuration. The joint evaluator requires bound
  paired populations and rejects missing candidate evidence. A passed older
  prior-study acceptance gate alone does not satisfy this expanded requirement;
  keep the frozen study's historical gates and active payload unchanged.
- Stage under `TASK/input/<original-basename>` using
  [cohort_staging.py](cohort_staging.py). Without `##tumor_id`, the reader uses
  the basename; renaming everything `input.tsv` changed every attempted Sim4K
  identity. Preserve bytes and verify parsed tumor/sample IDs and retained IDs
  before fitting. Never repair this by relabeling outputs or relaxing validation.
- Match the actual prepared cohort population: 200 original one-region
  Regional-CN tumors, 756 SimClone, 500 PhylogicNDT and 4,000 Sim4K, totaling
  5,456 for this campaign. A folder's nominal name is not its eligible count;
  do not project multiregion tumors into a different experiment silently.
- The SimClone generative contract used normal CN=2 on X/Y. Use independently
  hash-bound corrected copies; retain originals. Do not apply this correction
  to PhylogicNDT or real male sex chromosomes by analogy. See the historical
  [preparation and truth caveats](../BENCHMARK_FOUR_COHORTS.md).
- The supplied PhylogicNDT single-state CN can differ from original mixed-CN
  truth. Report that input/model mismatch separately from optimization failure.
  Match retained IDs and report ambiguous or missing truth coverage.
- Define raw versus final-refit CCF, selected candidate, filtering population,
  designated-cluster versus exact-one sMF, and the constant-vector CCC convention
  before comparing results. Nearest-to-one label 0 can have CCF below one.
  Do not combine old constrained/exact-one metrics with current designated-label
  metrics. Report cohorts separately; stratify true-K=1 false splits and true-K>1
  ARI. Identify incomplete-search results and failures in denominators.
- The current cohort validator defines predicted sMF as the matched-truth
  mutation fraction outside label 0, and truth sMF as the fraction with CCF
  `< 1 - 1e-12`. Report the full-retained predicted fraction separately when
  truth coverage is incomplete. **sMF CCC is across paired per-tumor sMF
  values**; one tumor has an sMF and error, not its own sMF CCC. The CCF metric
  helper returns CCC 1 for identical constants and 0 for every other case where
  either vector is constant. State the actual aggregation convention rather
  than relying on a plotting library's defaults. See
  [cohort validator](validate_cpu_cohort.py) and [metric helper](compare_clipp2.py).
- CNA-only means `(major_cn != 1) | (minor_cn != 1)`, including equal amplified
  CN. Specify the multiplicity truth target, exact-class F1 aggregation, missing
  calls and coverage. A metric on early easy cases is not full-cohort accuracy.
- Historical chain comparisons showed that missing partition proposals,
  contiguity and model/score preferences can each explain different errors;
  denser lambda sampling need not fix missing partitions. Those experiments
  do not establish the behavior of current complete-graph production. A better
  model score need not mean better truth ARI or sMF.
- The [September 27 SimClone/Phylogic investigation](../RESEARCH_SIMCLONE_PHYLOGIC_PYCLONE_GAP.md)
  separates those mechanisms on a frozen 436-case complete-graph panel.
  Refit an external partition with the original likelihood and score before
  attributing its absence to search: most better-ARI PyClone memberships lost
  under the existing allocation penalty. Five severe SimClone CCF failures
  retained low-CCF likelihood modes even with correct memberships; proposal-only
  prior perturbation cannot change that final likelihood preference. These are
  source-bound CPU component diagnostics, not CUDA or full-cohort qualification.
  Preserve original Phylogic mixed-CN flags when preparing future truth versions;
  the audited normalized truth wrote zero despite mixed CN in original MAFs.
  Never repair frozen truth in place or tune a production prior using test truth.

## Certification, verification and remaining limits

Invalid internal cut inputs and nonfinite observed gradients must produce an
unqualified diagnostic, not a certificate or invented restart direction. The
affected start stops while independent starts remain available. Caller-supplied
invalid graph data still fails explicitly. The original rare cut failure was
not reproduced by frozen replay/normalization probes: the repair establishes
containment, **not** a proven underlying arithmetic cause.

A paired qualification repeat has its own output identity and is excluded from
cohort completion totals. Compare source/graph identity, retained IDs, labels,
raw/refitted CCFs, multiplicity, score, objective, selected penalty, search status
and certificate semantics as appropriate. Exact parity for one preserved case
does not establish cohort-wide parity. An eager CPU comparison can perturb pilots
and graph weights, so distinguish independent full-fit agreement from a literal
fixed-graph/QP comparison.
The CPU adapter retains the schema name `clipp1d.cuda.run.v3`; that name is not
device evidence. Check `backend=cpu`, `numerical_device=cpu`,
`execution_scope=cpu_numerical_reference`, `compiled_inference=false` and the
adapter fingerprint before labeling an output or timing as CPU or CUDA.

Focused operational regression command, from the repository in `ml1`:

```bash
python -m pytest -q tests/test_cpu_process_identity.py tests/test_cpu_cohort_handoff.py \
  tests/test_cohort_failures.py tests/test_a100_cohort_dispatch.py tests/test_a100_lifecycle.py
python -m ruff check src tests benchmarks
git diff --check
```

The dated report distinguishes 40 repository tests from four attempt-local LSF
gate tests and the deferred-controller simulation. Do not advertise 44 as one
committed repository suite. Local CPU tests, allocated CUDA tests, synthetic
capacity, paired full-fit checks and cohort accuracy are separate gates. The
source-bound capacity pass at N=4,143 and one exact paired canary do not establish
full-cohort throughput, universal memory safety or global optimality.

For future maintenance, keep this guide reusable, the dated record historical,
and live registry/snapshots local. Update human navigation when the canonical
registry changes, preserve failed receipts, and never revive a retired owner
because a stale document still names it.

## Successive timeout generations and export-only recovery

Bind each retry generation to its own root and exact terminal parent proof.
A single global retry root stops being sufficient after a second recovery.
Use [lsf_retry_lineage.py](lsf_retry_lineage.py) to validate accepted job/key,
terminal resource-timeout status and immutable receipt hash; conditional retries
also require a bound one-retry amendment. A successful parent is reused and
must never be replaced by a retry. During a controller handoff, count preserved
running, pending and uncertain parent jobs against the successor's cap until
terminal reconciliation. Release a reserved research slot only after terminal
evidence for every study owning it. The September 24 evening recovery's
18-hour allowance is a dated resource contract, not a universal default.

For a failed tabular export, inspect the preserved fitted posterior before
rerunning inference. `E[x²] - E[x]²` can cancel to a small negative variance for
concentrated distributions. A centered sum can repair that export while keeping
the original mean; validate probability support and normalization explicitly.
Bind the original fit hashes, assert which fields may change and rerun the full
original validator, including restart evidence. Keep old failures and outputs.
Publish an effective result manifest that resolves original successes and new
exports, rather than treating the repair directory as a complete cohort.
Follow the current pointer's status command: the old immutable run can still
correctly report its original export failures after recovery succeeds.

See the [September 24 evening recovery record](../validation/cohort-recovery-20260925/README.md)
for the six LSF timeout cases and four repaired PyClone-VI CN-first exports.

### PyClone-VI posterior-variance validation, September 27

The SimClone/Phylogic campaign exposed a separate validation-only issue: its
exporter already used centered variance, while the frozen validator still used
`E[x²] - E[x]²`. Concentrated posteriors within the existing normalization
tolerance produced negative values around `-1.4e-12`. Three successful fits
were rejected even though their TSVs were finite and correct.

Use centered variance in both exporter and validator, preserving the original
mean and normalization tolerance. Revalidate the original best-of-1,000 fit,
restart logs, mutation coverage, posterior, assignments and export agreement;
compare output hashes before and after. Do not refit, renormalize, raise the
acceptance tolerance or modify a live frozen controller to repair this issue.

The local recovery record is
`/storage/CliPP2/PyCloneVI_runs/SimClone1000_PhylogicNDT500_20260927_validation_recovery_v1/README.md`.
It records 13 passing regression tests and full unchanged-output validation of
the three failures plus a passing control. Follow
`PyCloneVI_runs/CURRENT_SIMCLONE_PHYLOGIC.json`: its separate monitor publishes
an atomic effective manifest while the original 28-worker pool continues.
Original raw failures remain historical evidence. Matched comparisons must
consume effective validation receipts and retain both original and recovery
receipt hashes; an old raw progress count does not supersede those receipts.
