# Fresh-process CUDA startup recovery

The original qualification job **77438696** passed its eight fixtures on an
L40, September 26, 2026, 1:16–1:32 PM CDT. Its first development job,
**77438898**, failed before model upload at
`torch.cuda.reset_peak_memory_stats(device)` with `Invalid device argument`.
The study controller halted at 2:00 PM CDT, preserving the failure receipts.

`require_cuda('cuda:0')` verifies visibility and index bounds but does not
initialize CUDA. In the installed PyTorch 2.9.1, allocator-stat reset also
does not perform lazy initialization. The qualification process previously
performed other CUDA work before its full-path tests, hiding the failure in
the fresh child used for real cases. A worker startup probe runs in another
process and therefore cannot initialize the fitting child.

Both `run_case` and the independent cost-measurement entry point now
synchronize the selected CUDA device before resetting peak statistics. The
reset still precedes model upload, so model allocations count toward the peak.
No fitting equations, priors, proposal settings, inputs or numerical package
files changed. Complete process timing continues to include CUDA startup.

Qualification now executes comparison and timing entry points in separate
children that assert CUDA is initially uninitialized. These finish before the
qualifier parent initializes CUDA, preserving exclusive-device ownership.
The existing eight-fixture CPU/eager/compiled and full-path checks then run.
Fresh-process evidence and output hashes are recorded separately.

A second controller defect would have prevented panel admission after a
successful canary: it read lowercase `qualification.json`, while the worker
publishes `QUALIFICATION.json`. The handoff now verifies the uppercase receipt
against the qualification terminal hash before publishing panel admission.

Validation: **32 focused CPU tests passed** (25.56 seconds), including cold-start
ordering in both real entry points, model-upload peak coverage, exclusive-GPU
parent/child ordering, and successful/corrupted qualification handoff. Ruff
passed for all changed Python files. These checks are separate from the fresh
allocated-CUDA qualification required for this attempt.

Recovery is a new immutable attempt under
`/rsrch8/scratch/bcb/yding4/clipp1d_prior_perturbation_20260926c`.
Original receipts and outputs remain at the `20260926a` root. The intermediate
`20260926b` local preparation was not staged or launched. The new attempt
imports zero fits and reruns qualification before retrying the same development
canary. It retains the one-L40/two-CPU job cap, 823 fixed plus 12 dynamic tasks,
input hashes, scientific settings and December 25 study deadline. The separate
production cohort controllers are unchanged.

The numerical fingerprint remains
`4d320d0468f121e561a487b6176370e906845495944e998c6483e3741cc5ab0d`;
this does not identify the changed benchmark drivers. [PLAN.json](PLAN.json)
records their before/after hashes, the new complete source patch and historical
parent bindings. Use `results/CURRENT_PRIOR_STUDY.json` for the actual owner,
job IDs and current qualification state.
