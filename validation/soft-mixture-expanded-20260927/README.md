# Expanded soft-mixture development test

The L40-only owner described below is retired. The
[September 28 resource recovery](RESOURCE_RECOVERY_20260928.md) records a passing
A40 CUDA qualification and activation of its successor; the current pointer
remains the authority for live progress.

The September 27 follow-up requests substantially more cases than the original
36-tumor screen. The fixed expanded inventory contains all **3,097** tumors in
the previously saved matched comparisons: **2,661 CN-first4K, 385 SimClone1000,
and 51 PhylogicNDT500**. This includes the original development cases and is
neither independent confirmation nor the complete three-cohort benchmark.

The estimator and structural selector are unchanged. The local operational root
is `results/soft-mixture-expanded-20260927-v1`; source/input/selection hashes and
the separate truth inventory are recorded there. Current execution belongs to
`results/CURRENT_MIXTURE_STUDY.json`. One additional LSF GPU slot covers sequential
CUDA qualification, three final-output publication canaries and the rolling
one-case jobs. Existing campaign allocations and outputs are preserved.

Engineering checks cover exact resource admission, durable ambiguous-submission
handling, duplicate prevention, unknown job states, scheduler-history expiry,
bounded runtime admission, publication label/CCF/multiplicity identity, immutable
receipts and safe remote-artifact path resolution. The relocated and cached
evaluation replay reproduces all original 36-case headline and multiplicity
metrics exactly. These checks do not establish CUDA execution or accuracy on
the expanded inventory. Final allocated-CUDA status and new case results must
come from their own receipts.

The two initial qualification attempts failed in test setup: missing pytest
(`77453319`), then a missing `py.py` compatibility shim in the test bundle
(`77453503`). Both failed attempts remain intact. The complete dependency bundle
passes a site-packages-disabled import check locally and remotely; shared conda
packages and numerical source are unchanged.

Attempt `77453550` executed the tests: two mixture comparisons passed, but the
mixture center update following a successful full fusion fit hit a Triton
dominance error for identical slopes. A separately bound execution repair uses
six compiled blocks of six bisection updates for that shape, preserving all 36
updates, endpoint checks and scientific policy. Its 15 reference/selector tests
pass. Fresh CUDA qualification is required; the earlier partial pass is retained
as failure evidence. All 3,097 input/seed inventories are prepared, including
3,550 corrected Phylogic mixed-CN annotations with other truth fields unchanged.

The strict per-cohort scientific gates remain in force. Missing or failed cases
cannot be dropped from the planned denominator. Keep retained and cross-method
matched populations separate, report CNA-only multiplicity and cluster-count
errors, and retain the original screen's SimClone sMF CCC regression. This
experiment does not authorize production adoption.

At **September 27, 2026, 8:41 PM CDT**, the staged controller was alive on
`ldragon4`, waiting for fresh CUDA qualification job **77453726** (`PEND`).
No expanded case fits had completed. After successful qualification, three
publication canaries precede the remaining scalar jobs automatically. The
qualification and case jobs share the same one-job admission cap. See
[the launch snapshot](LAUNCH_SNAPSHOT.json) for the exact process identity and
immutable status receipt; this dated observation is not a live status claim.
