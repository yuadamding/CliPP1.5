# Cluster-birth integration — September 25, 2026

The first production-code port reproduces the tested K=1 repair under the
unchanged complete graph, marginalized likelihood, uniform multiplicity prior,
original bounds and partition score. It affects only the separately qualified
partition estimate. The primary raw estimator and raw continuation are retained.

Current numerical fingerprint:
`55d6e45949e4dee2e8798ac81fb49510baffca9959104355db87d45acca1a523`.
Base commit is `d44c3e728fa6f535f725719cc77a1dab9a96efc5`; this revision includes
uncommitted new files and changes. The complete qualification archive inventories
them; a Git archive of the base commit would not reproduce this implementation.

## Behavior and scope

- Default with `--partition-search`: the previous selected K=1 partition gets
  whole-group grid birth proposals and independently qualified refits. The
  three best split seeds receive at most 20 refinement rounds. The prior
  selected partition after its own refinement remains an explicit candidate.
- `--partition-birth off` isolates the prior estimator. `--partition-seeds 3`
  activates a bounded bank diverse in inferred K and memberships. The baseline
  and distinct bank seeds are refined before final selection.
- `--partition-birth any_cluster` separately enables conditional fixed-pair
  splitting of any occupied group. Global N/K, forced feasibility, stable
  likelihood-difference ordering and complete-tumor scores are used. A finite
  center bank is not a global partition search certificate.
- All provisional qualified candidates remain eligible for bounded exploration,
  including candidates that do not initially beat the incumbent. Only the
  lowest-score qualified result is published. Later recoverable failure keeps
  prior accepted improvement and the complete attempt history.
- Schema `clipp1d.cuda.run.v5` / `clipp1d.partition_estimate.v2` binds original
  raw reference, immediate parent, birth and refinement ancestry independently.
  A direct child inherits no raw certificate or fitted penalty. Score-interval
  ordering is distinct from an improvement in the published feasible score.

No active worker, frozen source bundle, original receipt or cohort result was
modified. This is implemented source, not a relaunch of the 14-GPU campaign.

## Verification completed locally

The final full repository suite passed **1,259 tests**. Ruff and `git diff
--check` passed. The suite includes 5,000 independent global split-complexity
checks, 1,000 exhaustive conditional assignment comparisons, forced/infeasible
and tied assignments, whole-group rescue, unconstrained child centers, bounded
seed diversity, a real multi-seed score reversal, score intervals, ancestry,
and preservation after later birth/refinement failures.

An additional CPU end-to-end numerical/publication check used a certified fused
raw reference at lambda 1000: both birth modes selected score 423.85548 versus
455.59576 for the original one-cluster partition and passed device/host ancestry
validation. This restricted-penalty engineering check is not default-path CUDA
qualification or a speed comparison.

`benchmarks/replay_birth.py` replayed the exact 382-case discovery snapshot with
the current source fingerprint. All 72 eligible single-cluster cases reproduced
the offline selected memberships, centers and scores. All 310 ineligible cases
kept their saved memberships and centers. The 1,233 qualified initial split
proposals matched the original count, order, center pairs and refitted scores;
the maximum score difference was zero. Proposal membership vectors and hashes
are now saved for independent CPU/CUDA comparisons.

| Measure, 382 matched tumors | Earlier partition estimate | Birth port on CPU | PyClone-VI |
|---|---:|---:|---:|
| Mean ARI | 0.66730 | 0.69092 | 0.65848 |
| Mean CCF absolute error | 0.05287 | 0.05018 | 0.05275 |
| sMF CCC | 0.90179 | 0.94826 | 0.94806 |
| Exact K | 244/382 | 261/382 | 238/382 |
| False K=1 among true multi-cluster | 27/337 | 5/337 | 3/337 |
| False splits among true single-cluster | 0/45 | 0/45 | 4/45 |

Twenty-two tumors escaped K=1 with improved ARI. This is distinct from exact-K
recovery: some true K=3/4 cases now have K=2. Twenty-one of the 22 improved CCF
MAE; one worsened slightly. Zero observed false splits does not establish zero
risk. The nearly equal CCC point estimates establish neither superiority nor
equivalence. These results remain discovery-panel CPU component evidence.

The birth-port pooled CNA-only exact-class macro-F1 is 0.797956 over classes
1–4, including amplified balanced CN. Truth is used for evaluation only.
See [CPU_REPLAY.json](CPU_REPLAY.json) for source/environment, metric conventions,
bundle/result hashes and the test-log binding. Large inputs, complete candidate
memberships and logs remain at
`results/birth-integration-20260925/cpu-final/REPLAY.json` and the sibling bundle.

## Independent validation protection

The old 36-case holdout contained two tumors subsequently used in the 382-case
discovery investigation:

- `100_4_0.6_0.1_rep130`
- `200_3_0.9_0.4_rep89`

They are explicitly excluded. [HELDOUT_PLAN.json](HELDOUT_PLAN.json) freezes the
34 surviving cases with the same input/truth hashes. Two original design strata
are therefore missing; no performance-selected replacements were added. The
original plan is preserved as historical evidence. Do not inspect these cases'
ongoing campaign results while tuning the birth policy and later call them
untouched validation.

The planned comparisons separate the previous estimator, extra iterations,
birth alone, multi-seed alone, their combination and generalized splitting.
They report collapse, false splits, exact K, ARI, CCF errors, sMF squared/absolute
error and CCC, multiplicity performance, paired regressions, numerical failures,
coverage and end-to-end cost. They have **not been run** on allocated CUDA.

## Allocated CUDA qualification

`benchmarks/qualify_birth_cuda.py` runs three paired default-path engineering
fixtures, a birth-derived publication on a certified fused raw reference, and
the frozen 382-case replay. Its controller creates no competing CUDA context;
only sequential allocated-GPU child processes initialize CUDA. This protects
exclusive-process LSF execution.

The prepared 2.87-MB archive is
`results/birth-integration-20260925/cuda-qualification.tar.gz`, with inventory
and resource plan in `CUDA_PLAN.json`. It binds complete source, new helpers,
tests, drivers, exact inputs and the dirty tracked patch. Requested resource
shape is one temporary LSF GPU, one CPU, 32 GB host memory and 60 minutes.
Queue units, maximum RUNLIMIT, environment and actual admission must be checked
before launching. The existing 10 A100 + 4 H100 campaign is preserved.

The user approved the temporary allocation. LSF job **77425633**
(`clipp1d_birth_0925a`, queue `egpu`) was released after admission checks and
was **RUN** on `gdragon005` at **September 25, 2026, 8:13:13 PM CDT**.
The scheduler confirmed one allocated CPU slot, one exclusive GPU, 32 GB host
memory and a 60-minute run limit. Hardware model and CUDA correctness remain
worker-result claims; scheduler RUN alone does not establish either.

[CUDA_SUBMISSION.json](CUDA_SUBMISSION.json) binds the authorization, source,
environment, compiler, wrapper, accepted job and initial snapshot. The original
archive was wrapped in a single `payload/` directory for safe publication;
the resulting transfer SHA-256 is
`a6df13f3549e2263af7f39a5f9a9e0c38a66689409c529185284f6b011863316`.
All 135 packaged files passed remote readback. The filesystem rejected the
no-replace directory rename with EACCES. After proving the final directory
absent and the staging inventory intact, exclusive directory/file creation
published the same bytes. Both the failed transition and its reconciliation
are retained. No existing run directory was overwritten.

Remote root:
`/rsrch8/scratch/bcb/yding4/clipp1d_birth_qualification_20260925a`.
Local operational receipts:
`results/birth-integration-20260925/lsf-qualification-20260925a`.
The live pointer is `results/CURRENT_BIRTH_QUALIFICATION.json`, also linked from
`results/CURRENT_RUNS.json`. Its status command makes one read-only exact-job
snapshot; Pending work must never be duplicated. Held-job `slots` fields may
be absent, so distinguish the explicit CPU request from actual dispatched
slots. The final running snapshot independently confirms one of each.

At this snapshot, allocated-CUDA qualification is **running, not passed**.
The separate 34-case held-out study has not run. The existing 10 A100 + 4 H100
campaign retains its previous source and settings.
