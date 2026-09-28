# Expanded study: GPU scheduling recovery

The user requested action to get the queued study running. Qualification job
`77453726` had waited nearly four hours with an L40-only resource request.
The scoped LSF resource inspection found compatible A40 capacity. No numerical
code required an L40, but the panel's original scheduler and worker checks did.

The original panel controller was cooperatively stopped before changing the
resource contract. Its exact process identity was verified terminal, with zero
case jobs submitted. The qualification retained its job ID, source archive,
wrapper, inputs, one CPU, 32 GB host memory and 60-minute limit. Removing only
the model restriction via `bmod` allowed it to start on `gdragon038`. At
**September 28, 2026, 12:34 AM CDT**, its allocated worker reported **NVIDIA A40**
and PyTorch 2.9.1+cu128. Allocation is not qualification success.

The first readback after LSF accepted `bmod` still showed the old restriction.
A subsequent reconciliation proved the new effective request and running job.
The accepted modification was not repeated, and no duplicate job was submitted.
The immediate stale readback, accepted response and final confirmation are all
retained.

The successor is prepared in
`results/soft-mixture-expanded-20260928-v1`, with remote root
`/rsrch8/scratch/bcb/yding4/clipp1d_soft_mixture_expanded_20260928a`.
It selects its scheduler GPU model from the qualification's hash-verified
allocation log and requires the worker's actual device to match. The supported
models are A40 and L40; an unqualified or unsupported model fails admission.
Its startup checks require the predecessor retirement and resource-change
receipts. All 3,097 input/seed files, publication references, numerical source,
scientific policy and selection rules are byte-identical to the predecessor.
Only operation helpers and relocated manifest paths changed.

Validation: **11 operational tests passed**, including both GPU models, exact
resource admission, hardware-evidence tampering, unsupported devices, uncertain
submissions and duplicate prevention. Scoped Ruff, compilation and diff checks
also passed. CUDA qualification and the three publication canaries remain
required. The qualification and panel retain a combined cap of one submitted
job. Existing cohort campaigns remain unchanged.

The current pointer and its latest status receipt establish activation and
current progress; this record alone does not establish qualification completion
or expanded accuracy.

At **September 28, 2026, 12:40 AM CDT**, qualification had **passed**: all
three allocated-CUDA tests ran successfully, and all three observed-input
CPU/CUDA pairs had identical labels, multiplicities, CCFs and scores. The
replacement controller was alive and its first publication canary,
**77460530**, was **RUN** with an A40 request. No expanded case had yet reached
validated completion. The qualification repeats are excluded from the planned
3,097-case denominator. See the [bound recovery snapshot](RECOVERY_SNAPSHOT_20260928.json).
