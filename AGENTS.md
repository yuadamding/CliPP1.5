# CliPP1.5 agent guide

This repository provides single-region, unconstrained, complete-graph inference
using PyTorch CUDA in float64. The Python package and CLI are named `clipp1d`.
It is a different scientific contract from the adjacent **CliPP2** repository.

## Read the smallest relevant context

- Model, outputs and certification: [README](README.md), then
  [CUDA framework](docs/CUDA_FRAMEWORK.md) and the exact pinned source.
- Simulation: [CliPPSim4K generator](docs/SIMULATION.md). Generate major/minor CN
  first, then multiplicity uniformly on 1..major CN, including balanced amplified
  loci. Preserve the original joint CN distribution and record the revised model
  identity; simulation truth does not constrain fits.
- Partition-search development: [separate estimator contract](docs/PARTITION_SEARCH.md).
  `--partition-search` retains the primary raw output and adds independently
  qualified partition tables. It is opt-in pending allocated-CUDA and held-out
  evidence; never relabel CPU replay as CUDA qualification.
- Status, recovery or cohort evaluation: [operations and lessons](benchmarks/OPERATIONS.md).
  On this workstation, start from `results/CURRENT_RUNS.json`; do not infer the
  active attempt from directory names or an old progress message.
- September 23 recovery evidence and its limits:
  [dated recovery record](validation/cohort-recovery-20260923/README.md).
- CPU runs: [explicit benchmark adapter](benchmarks/CPU_COMPLETE_GRAPH.md).
  Input staging: [basename and ID contract](benchmarks/LSF_COHORT_STAGING.md).
- Historical chain papers, plans and `VALIDATION*.md` files are source-scoped
  evidence. They are not current production specifications or launch commands.

## Scientific boundaries

- **Do not restore a clonal fitting constraint.** Every CCF retains its original
  feasible bounds. The refitted cluster closest to CCF 1 receives label 0, with
  canonical mutation-ID tie breaking. Labeling does not alter the fit.
- The public API is CUDA-only and fails explicitly on unavailable CUDA or
  compilation failure. The authorized CPU benchmark adapter uses eager PyTorch
  complete-graph equations; it is neither automatic fallback nor the old chain.
- The selected raw penalized CCF is primary; unpenalized membership refits are
  secondary summaries and supply the partition score. State which estimate an
  evaluation uses. Existing cohort wrappers score final-refit CCFs.
- Qualified selected results can have `search_status=incomplete`. Do not erase
  unresolved searches, infer global optimality, or give a candidate another
  candidate's certificate. Preserve graph, likelihood, bounds, search, score and
  numerical tolerances during operational repairs.
- A package version or commit alone does not identify a dirty run. Bind the
  numerical fingerprint **and** complete source/worker/plan/input/environment
  inventory, including new untracked helpers. Preserve old source lineage.

## Workspace and safe operation

- Work directly here: `/storage/CliPP2/CliPP1.5`, also reachable through
  `/data/CliPP2/CliPP1.5`. Use conda `ml1`; receipts bind the actual interpreter.
- `src/clipp1d/cuda/` and `cuda_api.py` implement production fitting;
  `benchmarks/` contains explicit adapters and frozen-worker templates;
  `tests/` includes independent references and recovery regressions.
- `results/` is ignored local operational evidence. Keep reusable documentation
  and small reviewed evidence records outside it. Never edit frozen payloads,
  consumed launch scripts, original receipts or completed outputs in place.
- Status/documentation tasks do not fetch source, launch, scale, cancel, migrate
  or continuously poll jobs. Read the installed Seadragon skill's matching route
  before remote work. Use exact receipt-bound identities and the approved chain.
- Follow execution overrides in `results/CURRENT_RUNS.json`. On September 24
  the user stopped Regional-CN and reassigned its ten A100 workers to OCCAMS,
  alongside four H100s, using one shared memory-aware case queue. Older
  Regional-CN and local-CPU assignments are retired;
  do not revive them from historical registries. Other cohorts retain their
  separate LSF ownership and total cap of 15 jobs. Draining workers and
  Pending/unknown jobs still occupy their caps.
- On September 25 the user stopped **all OCCAMS** work and reassigned the
  ten A100 plus four H100 shared pool to **CN-first 4K**. Follow
  `results/CURRENT_CNFIRST.json` and [the campaign note](BENCHMARK_CNFIRST.md).
  OCCAMS output preservation is not authorization to resume its old workers.
- The user then explicitly requested **skip qualification** for this CN-first
  campaign. Its v2 launch starts the full 10 A100 + 4 H100 pool directly and
  records qualification as skipped. Do not restore the prelaunch gate for this
  attempt. Hardware, source/input integrity and ordinary fit/output validation
  remain required; this override does not change scientific tolerances.
- Preserve healthy work, retire a refilling owner before replacing it, and
  verify exact Job/Pod absence before overlapping Kubernetes retries. A
  `halted_draining` process is not refilling; a pending cleanup is not absence.
- Display all status times in `America/Chicago`, with date and CST/CDT. Preserve
  UTC evidence and use the IANA zone for conversion.

## Verification

From this repository, using the bound `ml1` environment:

```bash
conda run -n ml1 python -m pytest -q
conda run -n ml1 python -m ruff check src tests benchmarks
git diff --check
```

Run only checks appropriate to the change. An affinity-sensitive test can
require two available CPUs; do not interpret a one-core test harness as a
production regression or disturb occupied worker cores. Local tests do not
qualify CUDA. Numerical changes require the relevant allocated-CUDA and paired
full-fit gates; capacity, kernel timing and full-fit speed are separate evidence.
See the operations guide for focused recovery commands and exact test scope.
