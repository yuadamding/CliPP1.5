# Experimental resource transfer — September 28, 2026

The user requested stopping all previous CliPP1.5 runs and transferring their
resources to Experimental. The fixed guarded soft-mixture development inventory
remains 3,097 cases: 2,661 CN-first4K, 385 SimClone1000 and 51 PhylogicNDT500.
This is not a new production default or independent confirmation.

The previous baseline CN-first Kubernetes supervisors were stopped before their
Jobs were deleted with exact UID/resourceVersion preconditions. Job and owned
Pod absence was verified for both families. The SimClone/Phylogic LSF controller
and prior-study controller were stopped before terminating their 20 + 1 jobs.
The original Experimental controller was retired; its last accepted job finished
successfully. All 603 completed cases passed source/input/output readback and
were explicitly imported without modifying or relocating their originals.

The new run root is
`/rsrch8/home/bcb/yding4/clipp1d_experimental_pool36_20260928a`.
The home-root exception is necessary for the approved Kubernetes PVC visibility.
Scratch originals remain intact. The local evidence root is
`results/experimental-pool36-20260928-v1`; the active pointer is
`results/CURRENT_MIXTURE_STUDY.json`.

The authorized allocation is 10 A100s, 4 H100s and 22 scalar LSF GPU jobs.
A single atomic queue makes case claims disjoint across both schedulers.
Interrupted/unknown claims remain reserved; there is no automatic retry.
Kubernetes indexes identify persistent single-GPU workers. LSF still submits
one tumor per scalar job. Both Kubernetes pools use the approved pinned image,
PVC, runAs identity and institutional resource shape. Scientific fitting remains
single-threaded and CUDA-only.

The model module SHA-256 is
`8cdf12ed24410f2a4a3e2185cbb5fb874dc4b6d71fc71b0b46e358c5b8e897b7`.
All scientific source files match the predecessor byte-for-byte. The run-plan
SHA-256 is `f4ff0feff66a215a15925c231ada66b1fc1c4539c860a5af6d853f9a7583ad35`;
payload inventory SHA-256 is
`4d22568e9b98f3ba1bfee5db61ba156db1da486f3227c59b90002833855c4394`.
Operational overlays have their own complete inventories; they do not replace
or modify the live LSF owner or numerical payload.

The initial A100 server dry run rejected admission's change from 1 CPU/32 GiB to
24 CPU/100 GiB; no Job was created. Control generation `k8s-v2` bound the enforced
resource template (H100 uses 200 GiB). Its allocated workers then found a missing
pytest package before any numerical tests. Both failed Jobs and owned Pods were
verified absent before recovery. Control generation `k8s-v3` supplies the exact
pure-Python test dependencies from the successful A40 qualification, only in the
test subprocess. `python -S -m pytest --version` passed locally and remotely.
Its generation SHA-256 is
`b415a007c0b7678ee7c1d90636cca38ffb8728dae24564d7a8dfc9064356adb5`.
Original failures and qualification directories are preserved.

The LSF canary passed and its controller expanded to the 22-slot cap. Each
Kubernetes family must separately pass three allocated-CUDA tests, three
component/guard publication comparisons and a largest-input capacity case
before its supervisor scales to 10 or 4. Qualification repeats are excluded
from study coverage. The actual activation, scale and hardware receipts—not
requested counts—determine allocated capacity.

Sixteen focused operational tests passed. They cover concurrent cross-scheduler
claims, retained uncertain submissions, strict admission resources, output
validation, activation identity and successful-exit memory telemetry. Ruff,
compilation and whitespace checks passed. Numerical qualification results belong
in the current attempt receipts; CPU operational tests are not GPU evidence.

Read the `status_command` in `results/CURRENT_MIXTURE_STUDY.json` for the current
status entry point. The original `check_status.py` does not include later
recovery directories. Original UTC receipts remain unchanged; user-facing
display uses America/Chicago. See `benchmarks/OPERATIONS.md` for reusable
recovery lessons.

The H100 generation passed all three CUDA tests and all three publication
comparisons, then expanded to four verified Running/Ready H100 workers. The
A100 passed its three CUDA tests and the CN-first/Phylogic publication cases,
but the SimClone case `simn0vt5y` failed the unchanged EM monotonicity gate.
A diagnostic captured the original failing state without modifying inference:
`dosage_quantiles_k1`, uniform multiplicity, iteration 3. One posterior row
(index 446 of 559) summed to approximately `2.55e-22` instead of one. The next
mixture mass became `558/559`, and the objective increased by about `0.993`,
far above the original `2.84e-9` monotonicity margin. Independent eager likelihood
and M-step evaluation agreed with the observed objectives and center update;
the malformed posterior is the issue, not the tolerance. Do not loosen the gate.

The first diagnostic completed and was exactly cleaned. A second one-GPU A100
diagnostic compared compiler configurations and checked for mutation of returned
posterior tensors, within the existing ten-A100 resource ceiling. Its bound
generation was `control/a100-diag-v2`; it also completed and was exactly cleaned.
LSF and the four qualified H100 workers continued the unchanged experiment.

The second A100 diagnostic reproduced intermittent posterior corruption from
compiled expectation alone. Returned tensors were not subsequently mutated by
the M-step. Default compilation and `inplace_buffers=False` both failed;
`triton.persistent_reductions=False`, `max_fusion_size=1`, and
`epilogue_fusion=False` each agreed with the eager expression to about `7e-15`
in ten repetitions. This isolates an execution problem in the compiled
reduction/fusion path; it does not establish the precise upstream compiler bug.

A100 recovery generation `k8s-v4` disabled persistent reductions and epilogue
fusion and used a separate compiler cache. Its longer stress test rejected the
configuration at K=1, repetition 16: maximum posterior error was approximately
0.999978 and normalization error was 1.0. No panel result was admitted from that
generation. Its exact Job and owned Pods were verified absent before replacement.
The generation SHA-256 is
`b70060917d3b4fd7b975e11f820e7f95c6f91bffee1cd39e12b2ed5ea1c0baa6`.
Ten successful repetitions were therefore insufficient qualification.

Generation `k8s-v5` also sets `max_fusion_size=1` and
`allow_buffer_reuse=False`. It retains strict compiled CUDA execution, every
mathematical source file, and all scientific tolerances. Its separate cache
is bound to generation SHA-256
`5e32c1686e8c290e23f4e8a3685e2fa8d6b7cc17be143038ee5bc1aabcbdc143`.
The captured input passed 2,000 eager stability repetitions and 2,000 compiled
repetitions each at K=1, K=2 and K=8. Maximum posterior discrepancies were
`6.44e-15`, `7.11e-15` and `8.55e-15`, respectively. This is kernel evidence;
the original three CUDA tests, three paired publication cases and largest-input
capacity case also passed. The three cases had identical published CCFs to the
bound A40 references; maximum absolute score discrepancy was `3.64e-12`.
The largest input retained 2,180 mutations and peak reserved device memory was
44,040,192 bytes. This small mixture-only footprint must not be extrapolated to
raw complete-graph fusion fits.

Qualification completed September 28, 2026, at 12:56 p.m. CDT. The exact
supervisor expanded Job `clipp1d-experimental-a100-20260928d`, UID
`99da97bf-a332-4665-8986-6b1130903fbb`, to ten workers. The 12:57 p.m. CDT snapshot
found ten Running/Ready A100 Pods, four Running/Ready H100 Pods, the healthy
22-slot LSF controller, and 1,236 validated cases (603 imported, 633 new), with
no new case failures. At 12:58:58 p.m. CDT, all fourteen Kubernetes workers also
had verified hardware receipts, and coverage had reached 1,283 cases (603
imported, 680 new). Seven scalar LSF jobs were RUN at that snapshot; the 22-slot
cap is distinct from simultaneous allocation. The LSF and H100 owners retain
their original qualified execution configurations. Local `HANDOFF.json` binds
this snapshot; later live counts belong in the current pointer.

The September 28, 2:15 p.m. CDT terminal snapshot reached 3,092 of 3,097
validated bundles. A40 case `02593` (`100_1_0.9_0.7_rep140`, Job 77470813)
and H100 case `02920` (`simu6nhvc`) failed the unchanged EM monotonicity check.
The H100 Job's fail-fast shutdown interrupted cases `02915`, `02921` and
`02922`. Both device families still used their original compiler settings;
the A100 `k8s-v5` pool completed cleanly. All original owners were terminal,
and exact Job/Pod and supervisor absence was reverified before recovery.

A separate A40 Job, 77469627 (`01059`, `100_4_0.4_0.7_rep36`), published a
fully validated bundle after 20.5 seconds but remained charged until RUNLIMIT
at 4,223 seconds. Its stderr was empty, so retained evidence does not identify
the surviving process. Inspection found a reproducible worker defect: child
cleanup only ran when the Python leader was alive, leaving orphaned compiler
helpers after a successful or failed leader exit. The repair makes the worker
a scoped Linux child subreaper, terminates and reaps its own child process
group on every return path, bounds TERM/KILL waits, and records group absence
before output validation. It does not signal unrelated groups. Real-process
tests cover success, nonzero exit, an orphan that ignores TERM, and timeout.

The user's “fix all” request authorized the fresh recovery attempt
`recoveries/terminal-five-v1`, with immutable control generation
`control/recovery-v1`. No original claim, numerical source, input, seed,
tolerance, failed receipt or successful result was overwritten. The generation
SHA-256 is `e4b712809974e74ae8bcac953eaa621d5837b7a79514a0d473b7218a71f80940`;
the recovery-plan SHA-256 is
`62d818a1744306f96240da1bb4fa5c96bb7870a7da53e9ddb634ed2c9a8c04ca`.
It applies the A100-qualified compiler configuration to separate A40 and H100
qualifications, with device/generation-specific caches. Bounded diagnostic
replays use the original configuration in isolated processes and can never
publish an accepted fit. New captured failures also receive 2,000-repeat
posterior checks per tested K under the repair. Mathematical likelihood,
starts, objective, EM tolerances, selection and clonal labeling are unchanged.

Twenty-two focused operational tests and Ruff passed before deployment.
Recovery-only tests reject changed predecessor evidence, reclassification of
a successful fit as a failure, and an unbound output directory. Allocated
qualification repeats the three CUDA tests, three paired full-fit/publication
cases and largest-input capacity check separately on A40 and H100. Kernel
stress and local operational tests do not replace those full-fit gates.

The first scalar recovery Job is 77471879. H100 recovery Job
`clipp1d-experimental-h100-recovery-20260928a`, UID
`a2bab056-0452-4336-96bd-277a05dfec4d`, activated September 28 at 2:33 p.m. CDT.
It begins with one qualifier and can expand to four single-GPU workers.
The original LSF controller remains retired. Case `01059` has an explicit
separate shutdown replay, excluded from new coverage. Final qualification,
recovery, scheduler-exit and cleanup outcomes belong in the attempt receipts
and current pointer; launch is not completion.

Recovery subsequently completed at **2:44:30 p.m. CDT on September 28**:
**3,097/3,097 validated**, five recovered, no unresolved cases and no active
workers. Both GPU families passed their allocated qualification. Both LSF
jobs exited DONE, and the H100 Job and owned Pods were verified absent.
See the [completed recovery report](RECOVERY.md) for diagnoses, limits, exact
identities and the resolved result index. The two numerical failures were not
reproduced in their three bounded default-configuration replays; do not claim
that their precise upstream causes were established.
