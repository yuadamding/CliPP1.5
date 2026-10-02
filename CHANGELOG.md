# Changelog

## 1.6.0 — unreleased compact CPU revision

Preserves the fixed-chain estimator and frozen golden outputs. Replaces mandatory
R preprocessing with differential-tested Python, removes mandatory psutil, retires
legacy entry points/options and removes bundled simulation tooling. Schema 3
stores selected tables once and every candidate as replayable chain parameters.
The single verifier checks source coordinates, exact configuration/attempts, all
recorded candidate scores/ranks and native diagnostics. Adds `validate`, a small
verified result reader and numerical-stack receipts. Validation used isolated
benchmarks and separate strict-reference/portability contracts, preserved in an
external archive; docs, tests and benchmark directories are intentionally removed
from this checkout. CI covers builds and example fits. Historical schema 2 uses its
pinned verifier. CUDA/container and broad-cohort superiority remain unqualified.
Python 3.12+ and SciPy 1.18.x are required after the Python 3.10/SciPy 1.15.3
stack failed candidate-partition equivalence; the failure and golden are retained.

## 1.5.1 — unreleased engineering revision

Preserves the 15d397b fixed-chain model, initializer, conditional refit, candidate
search and BIC-form score. Adds a src package, CLI/API, CPU-default source builds,
ABI 3 native identity, strict autosomal input validation, row ledger, transactional
runs, independent verification, complete API timings, current-estimator tests,
CI and deterministic simulation/benchmark tooling. Output schema 2 adds identity,
CCF/purity and assignment-proportion fields. Ambiguous inputs and reused output
paths now fail explicitly. Archives unreachable legacy R postprocessing.

CUDA availability now distinguishes missing device/driver from execution errors;
GPU metadata and opt-in differential tests are provided. Revised CUDA and Docker
execution are unqualified. No broad-cohort accuracy or speed improvement is claimed.

## 15d397b — 2026-10-01

Replaced the earlier repository pipeline with the conditional fixed-chain
estimator. This commit is the frozen numerical baseline for this revision.
