# Three-cohort guard verification — September 27, 2026

The user's requirement is to preserve CN-first4K while targeting the best accuracy on CN-first4K, SimClone1000 and PhylogicNDT500. The [acceptance contract](../../docs/THREE_COHORT_ACCEPTANCE.md) and [joint evaluator](../../benchmarks/evaluate_joint_cohort_gate.py) implement the evaluation boundary. This change does not implement a new fitting model or establish improved accuracy.

## What was implemented

- Independent cohort and endpoint checks; gains elsewhere cannot compensate for a regression.
- A versioned accepted-baseline source/configuration contract and required comparator coverage.
- One candidate scientific source/configuration across all three cohorts.
- Exact case, mutation-set, input/truth inventory and metric-artifact bindings.
- Separate protection of all scorable retained mutations when cross-method comparison uses a narrower population; explicit truth exclusions reconstruct the original retained identity.
- Complete eligible-cohort counts of 4,000 / 756 / 500 for the full observed gate.
- Fail-closed behavior for missing, failed, malformed or inconsistent evidence.
- Separate observed non-regression, best-among-required-comparators, engineering self-check and full-coverage results. The report never treats these as statistical, CUDA, held-out or release qualification.

The required metrics are mean ARI, multi-cluster mean ARI, CCF MAE, sMF CCC and sMF MAE. False-split and false-K=1 safeguards additionally protect the accepted baseline. The fixed `1e-12` comparison epsilon is arithmetic allowance, not permission for a scientific performance loss.

## Verification

The focused suite passed **25 tests**. It includes a full-size synthetic inventory; a tiny CN-first loss despite large gains on the other two cohorts; missing candidate/comparator cases; mixed scientific settings; artifact tampering; invalid metrics; full-population errors hidden behind a matched subset; explicit truth exclusions; source-baseline substitution; comparator omission; and the established constant-vector CCC conventions.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 31 /home/yding1995/miniforge3/envs/ml1/bin/python -m pytest -q \
  tests/test_joint_cohort_gate.py
python -m ruff check benchmarks/evaluate_joint_cohort_gate.py tests/test_joint_cohort_gate.py
git diff --check
```

Ruff and the whitespace check passed. These tests use local metric arithmetic and synthetic tables; they run no inference and provide no GPU qualification.

The real baseline package contains **14,177 per-case method/population rows**, with nine aggregate summaries reproducing their original saved comparisons within `1e-12`. It includes 2,661 CN-first cases, 385 SimClone cases and 51 Phylogic cases. CN-first has both the 1,056,696-mutation matched panel and the 1,066,951-mutation retained panel.

The integration check verifies:

- With candidate bindings absent, the checker reports exactly three missing-candidate errors and displays the correct baseline summaries.
- With the baseline supplied against itself, all paired non-regression and retained-population checks pass exactly.
- That self-check remains explicitly engineering-only, with `gate_passed=false`, `full_cohort_coverage=false`, `best_among_declared_comparators=false` and `release_ready=false`.

Final evaluator SHA-256:
`0921bdac13c871170e47f4a19387336d1a4dd628f7819d1d78667383475d4535`.

Local hash-bound integration evidence is under
`results/joint-cohort-guard-20260927-v1/BOUND_BASELINE_VALIDATION.json`.
The final 25-test run is bound in `JOINT_GATE_TESTS.json` beside it, with
`JOINT_GATE_TESTS.log`, the exact command, source/test hashes and CPU affinity.
The baseline authorities, summaries and input provenance are under its `baseline/` subdirectory. The no-clobber integration script is `verify_bound_baseline.py`; reruns require a new output directory. The ignored results tree is local evidence, not guaranteed to exist in a fresh checkout.

The current snapshots are inspected development data, not full-cohort or fresh confirmation evidence. No candidate revision has passed the three-cohort requirement. All existing production sources, consumed study payloads, fit results and active workers were preserved.
