# Preserve cohort input identity during LSF staging

The September 23 four-cohort wrapper renamed every payload to `input.tsv`.
CliPP1.5 uses `##tumor_id` metadata when present and otherwise derives tumor ID
from the input basename. All 4,000 CliPPSim4K inputs use the latter contract.
The 14 attempted Sim4K fits published tumor ID `input`, then failed the wrapper's
strict output-ID check. SimClone, Regional-CN and PhylogicNDT carry explicit
metadata. All 112 validated SimClone results retained their correct identities.

Use `cohort_staging.stage_case_input` and `staged_input_path` for cohort workers.
The manifest records the original `input_filename`; the input is unpacked under
`TASK/input/<original-basename>`. Exact input and truth bytes remain unchanged.
The helper verifies the input hash, tumor/sample IDs, retained mutation count
and retained-ID hash using the actual production reader before fitting. Refitting
and output validation must read that same path. Do not change output labels or
relax identity checks to compensate for renamed inputs.

Recovery must use a new immutable run root and validate the predecessor's stopped
controller and terminal accepted jobs. Import only hash-checked, independently
validated successful results; keep their original source and job provenance and
never submit them again. Failed outputs stay historical and are not relabelled or
imported. Revalidate every cohort's actual staging contract before resubmission.
Classify output validation failures explicitly, preserve their traceback, and stop
new submissions while already accepted work drains.

The fixed recovery has 112 imported fits and 5,344 fresh cases (14 retries plus
5,330 previously unsubmitted cases), under the original total 15-job limit.
It preserves the complete-graph CUDA inference fingerprint, thresholds, input/CN
filters, unconstrained fitting and closest-to-one public label 0. It uses a fresh
allocated CUDA qualification and then the smallest remaining real case, a failed
200-mutation Sim4K input, before opening the rolling submission window.

The local regression reproduces the generic-filename failure and verifies the
preserved filename, explicit metadata, byte identity, retained population and
fail-closed staging checks. Full staged-input preflight covered all 5,456 inputs;
these CPU I/O checks are separate from the allocated CUDA qualification.

The [retrievable staging evidence](../validation/cuda-surrogate-review-v4/README.md#actual-cohort-worker-adoption)
contains the actual frozen worker/helper, full case manifest and remote readback
of canary job 77339914. Its original basename, 200 retained mutation IDs and
published tumor/sample IDs match; fitting, internal refitting and final validation
use the same staged input. Its plan retains the original `a0e1406` source and
112 imported fits. This canary does not establish completion of the full cohort.

Use the local `results/CURRENT_RUNS.json` aggregate registry for current owners,
assignments and the exact read-only observer. The older pool-specific
`results/CURRENT_LSF_RUN.json` pointer describes a historical handoff and may
lag that registry; both are intentionally excluded from Git. See
[operations and lessons](OPERATIONS.md) for recovery rules. Use the tracked
evidence link above for review; historical receipts must not be overwritten.
