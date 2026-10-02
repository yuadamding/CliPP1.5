# Changelog

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
