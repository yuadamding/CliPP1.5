# Multiplicity-prior proposal study v1

The [September 26 startup recovery](recovery-20260926c/README.md) records the
passed original qualification, subsequent cold-process canary failure, focused
repair and fresh immutable retry. Original receipts remain historical evidence.

This is an opt-in experiment on baseline
`da651879c2a33f481957ffae1d1c865d11ad39c8`. It does not change production defaults,
restore a clonal fitting constraint, or modify the active cohort payloads.
The attached September 26 protocol is implemented as a separate proposal study.
There is no claim of improved accuracy before the confirmation results exist.

## Frozen comparison

- **R**: original raw fit and its original-prior membership refit.
- **B**: current partition estimator, `birth_mode=single_cluster`, seed bank 1,
  and the existing exact-one grouping convention.
- **P**: B plus eight qualified pilots from four ID-keyed directions and their
  opposites at eta 0.03. Deduplicate at maximum coordinate difference 1e-8,
  including the original pilot; retain at most four starts by farthest RMS
  distance, breaking ties by direction index. Solve from those coordinates at
  three positive penalties from R's path, freshly under the original prior and
  original graph. Refine each distinct extra membership for four rounds without
  another birth generation. Preserve B's complete candidate records and winner.
- **D**: B plus the four prescribed VAF/multiplicity starts, deduplicated without
  padding, with the identical penalties and extra-refinement budget.

Every auxiliary model and certificate has its own identity. The experimental
entry point accepts coordinates only, starts a fresh solve, and independently
audits its raw result under the target model, graph and absolute penalty.
Auxiliary qualification/OOM failures leave B eligible and mark coverage incomplete.
Fatal source, device or output-integrity errors are failures, not fallback fits.
Raw tables remain byte-identical across arms; extra partition tables are separate.

A lower original raw objective at the same penalty, a lower original partition
score, and better scientific accuracy are separate outcomes. Candidate traces
record score intervals and the existing stronger refit-ordering test as well as
attained-score improvement. No perturbed-prior certificate is reused under the
original model. Fitting and selection never read the local truth files.

## Data and order

`PLAN.json` binds the code, proposal/evaluation seeds, source patch, manifests,
settings, budgets and acceptance gates. Frozen datasets are under
`results/prior-perturbation-v1/datasets`; earlier draft manifests are retained,
and their input hashes were verified to match the frozen regeneration.

1. Local Stage 0 tests and allocated-CUDA qualification on eight fixtures covering
   the six mechanism families. The isolated alias explicitly brackets both wells
   for xi=(-1,+1)/(+1,-1), eta 0.01, on CPU/eager CUDA/compiled CUDA.
2. The **108 development tumors**, N=400, with the prescribed 108 design cells,
   generation seed 2026092601. Run R/B/P/D. IDs do not encode true K.
3. The separate **12-tumor sensitivity factorial**: pilot-ambiguity tertiles,
   then hash-pick four per tertile; eight directions, S00/S10/S01/S11, common
   original starting bank and a single common positive absolute penalty.
4. **382 archived discovery tumors**, with input SHA verification, pilot-only
   diagnostics at eta 0.01/0.03/0.05. The protected 34 cases are excluded.
5. **324 fresh confirmation tumors**, same 108 cells times three replicates,
   generation seed 2026092602. The code and primary policy are frozen before
   their outcomes are inspected. R is also run to independently verify the
   reference path; B/P/D remain the confirmation accuracy comparison.
6. **Nine cost fixtures**, N=256/1000/4000, three per size, seed 2026092603.
   One cold and three cache-warm executions per arm, rotated sequentially on the
   same GPU. Each execution is a fresh process. Separate per-arm cache paths
   prevent another arm's compilation from being free. P and D pay for input
   preparation, all B work, auxiliary attempts, refitting, qualification,
   publication and process startup. Report cold and warm costs separately;
   adoption gates use all 36 paired fixture/repetition ratios. Memory means
   measured peak allocated CUDA bytes, with reserved bytes reported separately.

The LSF controller uses **one additional study slot**, one exclusive L40 and two
CPU slots per scalar job. Pending and uncertain submissions occupy that slot.
Qualification precedes a real development canary and the remaining study.
A development publication loss halts before confirmation. No retries, lambda-path
extensions for the factorial, per-tumor tuning, or default-policy promotion are
automatic. This study does not consume or modify the existing cohort controllers.

## Evaluation and diagnostics

`benchmarks/evaluate_prior_perturbation.py` joins exact mutation IDs to local
truth, verifies public-file hashes, and reports ARI, refitted CCF MAE/RMSE,
cluster-count errors and multiplicity class metrics. R's raw CCF is reported
separately. Bootstrap 10,000 paired tumor resamples within the 108 design cells,
seed 2026092699; perturbation directions are never statistical replicates.
With one tumor per cell, development within-cell bootstrap intervals are
uninformative, explicitly flagged, and cannot support adoption.

Confirmation gates are the supplied ARI/MAE/P-versus-D, cluster-count,
publishability, incumbent, correctness and cost thresholds. Missing evidence
prevents adoption. The study does not choose eta or seeds from confirmation.

`benchmarks/oracle_prior_perturbation.py` performs evaluator-only true-membership
refits for unpublished cases or P/D cases with worse ARI or CCF MAE than B. It
compares score intervals with the selected partition to distinguish a missed
lower-score truth partition from the criterion preferring an inaccurate result.
These diagnostic refits never enter the fitting candidate bank.

## Commands after result import

```bash
# Each results root contains case-ID subdirectories and their validated arms.
python benchmarks/evaluate_prior_perturbation.py \
  --dataset results/prior-perturbation-v1/datasets/development \
  --results IMPORTED_RESULTS --outdir NEW_DEVELOPMENT_REPORT \
  --qualification IMPORTED_RESULTS/qualification/QUALIFICATION.json

python benchmarks/prior_perturbation_diagnostics.py select \
  --baseline IMPORTED_RESULTS \
  --manifest results/prior-perturbation-v1/datasets/development/FIT_MANIFEST.json \
  --outdir NEW_FACTORIAL_SELECTION.json

python benchmarks/time_prior_perturbation.py \
  --manifest results/prior-perturbation-v1/datasets/performance/FIT_MANIFEST.json \
  --results IMPORTED_RESULTS --outdir NEW_COST_REPORT

python benchmarks/evaluate_prior_perturbation.py \
  --dataset results/prior-perturbation-v1/datasets/confirmation \
  --results IMPORTED_RESULTS --outdir NEW_CONFIRMATION_REPORT \
  --qualification IMPORTED_RESULTS/qualification/QUALIFICATION.json \
  --performance NEW_COST_REPORT/PERFORMANCE.json

# Requires an allocated GPU; separate evaluator task, never a production fit.
python benchmarks/oracle_prior_perturbation.py \
  --dataset results/prior-perturbation-v1/datasets/confirmation \
  --results IMPORTED_RESULTS --outdir NEW_ORACLE_REPORT
```

The new local tests are CPU component and serialization checks, including an
explicit CPU harness of the experimental driver. They do not qualify CUDA.
Allocated-CUDA evidence belongs to the immutable study attempt and must match
its actual source fingerprint, input manifests and environment. Use the study
pointer `results/CURRENT_PRIOR_STUDY.json` for its exact receipt-bound owner.
