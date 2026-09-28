# Initial study handoff

Snapshot: **September 26, 2026, 12:59:44 PM CDT**.

- Controller alive on ldragon4, PID 793856 (exact invocation identity in the local pointer).
- LSF qualification job **77438696**: **PEND**, zero execution-error bytes.
- One exclusive L40 slot, two CPU slots, 16 GB host reservation for qualification; 120-minute hard limit.
- Development **0/108**; confirmation **0/324**. No CUDA pass or scientific improvement is claimed.
- The frozen controller starts the development canary only after qualification, then the remaining declared stages.
- Existing cohort controllers and their frozen payloads were not modified.

Baseline: `da651879c2a33f481957ffae1d1c865d11ad39c8`. Experimental numerical SHA-256:
`4d320d0468f121e561a487b6176370e906845495944e998c6483e3741cc5ab0d`.

Remote root: `/rsrch8/scratch/bcb/yding4/clipp1d_prior_perturbation_20260926a`.

Local pointer: `results/CURRENT_PRIOR_STUDY.json`; immutable phase receipts are in
`results/prior-perturbation-v1/lsf-20260926a`. The aggregate run index links this
study separately from the 20-worker two-cohort pool. The 34 older protected cases
are excluded, and no evaluator truth files were staged to the fit workers.

Validation: **1,286 full-suite tests passed**, plus **seven targeted tests passed**
after the last evaluator/graph-provenance changes. Ruff, compilation and diff
checks passed. These are CPU component and integration checks; allocated-CUDA
qualification remains pending.

Do not reissue a submission because it is Pending. A new status request uses the
exact accepted IDs in the controller progress and immutable admission receipts.
Do not rerun the consumed preparation, reserve, upload, publication or launch phases.
