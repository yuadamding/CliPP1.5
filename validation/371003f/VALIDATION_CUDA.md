# PyTorch CUDA revision validation

Snapshot: 2026-09-22 17:00 CDT. Package `0.5.0.dev0`; policy
`clipp1d_complete_cuda_unconstrained_v2`; output schema `clipp1d.cuda.run.v2`.

## Result

**707 local tests passed, and actual CUDA qualification passed** on an allocated
NVIDIA L40 in Seadragon scalar LSF job **77333496**. The local production
source, qualification script and package configuration match the frozen GPU
bundle exactly. No numerical admission tolerance was loosened.

Fitting remains unconstrained with respect to clonality. Public cluster 0 is the
secondary refitted center closest to CCF one. The selected raw penalized vector
is primary; refitted CCFs and their multiplicity calls are separate summaries.

## Local checks

The whole repository suite passed **707 tests in 64.81 seconds**, with no
failed or skipped tests. Ruff, compileall and `git diff --check` passed. The sole
warning was unavailable NVML on the local CPU reference host. This host has
PyTorch 2.9.1+cu128 but no usable CUDA device. CPU numerical references and tracing
are not presented as GPU evidence. Public admission rejected the no-CUDA host
without falling back to CPU.

## Allocated GPU checks

The qualified environment used PyTorch 2.9.1+cu128, CUDA 12.8 and the receipt-bound
ML1 interpreter/compiler. Qualification took 622.25 seconds and passed:

- Seven independent likelihood/derivative probes, including original bounds,
  exact clipping breakpoints and adjacent represented values.
- Eight bounded complete-graph QP configurations in both eager and compiled CUDA,
  requiring primal-dual gap and KKT admission.
- Three full default-path fixtures in both modes, with every planned start and
  refit qualified; labels and raw/refitted multiplicity calls agreed.
- 12 additional comparisons with identical literal pilots, graph weights and
  penalties, including nonzero penalties with nonfused states.
- The actual public TSV input/fit/publication path, complete default search,
  receipt/source/graph identities, exact table hashes and readback. All original
  upper bounds in that fixture were below one.

| Fixture | Eager seconds | Compiled seconds | Max raw CCF difference | Max refitted CCF difference |
| --- | ---: | ---: | ---: | ---: |
| `single_support` | 8.40 | 7.18 | 0 | 0 |
| `mixed_multiplicity` | 65.82 | 47.07 | 1.11e-16 | 4.11e-10 |
| `all_bounds_below_one` | 209.04 | 185.31 | 4.83e-14 | 8.27e-10 |

These small-fixture times include differing compilation/cache histories and are
not a GPU speedup benchmark. Independently derived pilot/graph/path differences
are retained explicitly. Their selected discrete grid positions agreed; the
additional fixed-graph checks compare the same statistical problem. Exact device
path recipes are checked separately from host `math.ldexp` rounding differences.

## Bounded resource measurement

A separate 256-node QP measured 32,640 unordered edges with a
256-iteration budget. Dense storage remains quadratic. Recorded
allocated memory is process-wide peak CUDA allocation during each repeat, not a
large-cohort capacity estimate.

| Repeat | Seconds | QP qualified | Peak allocated MiB |
| --- | ---: | --- | ---: |
| 0 | 0.806 | False | 7.62 |
| 1 | 0.801 | False | 8.12 |

Both resource repeats exhausted the reduced 256-iteration measurement budget
without meeting the QP certificate. They are runtime/memory measurements only,
not numerical qualification at 256 nodes. Production retains its unchanged
20,000-iteration maximum.

## Preserved failures and repairs

| Attempt / LSF job | Observation and subsequent repair |
| --- | --- |
| A / 77331996 | Exact clipping derivative mismatch; canonical tensor division and conservative scalar-bound rounding repaired. |
| B / 77332064 | Dynamo recompile limit on changing scalar shapes; bounded structural compiler banks added, with global cache limits unchanged. |
| C / 77332387 | Fits completed; qualifier incorrectly required identical independently scaled lambda values. Qualification now separates adaptive paths and fixed-problem parity. |
| D / 77332893 | Qualifier treated host `math.ldexp` as the exact CUDA recipe; device recipe is now checked exactly and host ULP differences are recorded. |
| E / 77333496 | All required CUDA qualification checks passed. |

All failed attempts and their original receipts remain preserved. Production
source is unchanged between C, D and E; the last two repairs affect qualification.

## Reproducible evidence and limits

Production source SHA-256:
`50c4d2a16b361bf34943a91350f664ac0a354c84d89e36c67ff03f59c37e0ee7`.

Frozen source archive SHA-256:
`88eefd13c934d477ebb4311074e813db5ebcdcfc87efd53e266eed86fb2efaac`.

[Final validation receipt](results/complete-graph-migration-20260922-v1/FINAL_VALIDATION.json),
[whole-suite log](results/complete-graph-migration-20260922-v1/whole-suite-final-e.log), and
[actual CUDA receipt](results/complete-graph-migration-20260922-v1/gpu-qualification-e/imported-diagnostics/results/qualification.json)
retain exact source, environment, compiler, inputs, tests and output identities.
The original dirty repository and supplied overlay were preserved before integration.

This qualifies the tested synthetic CUDA paths and public output contract.
It does **not** establish full-cohort accuracy, large-input scalability, a GPU
speedup, or global optimality of the nonconvex likelihood. No cohort was rerun and
no existing frozen cohort worker was modified for this qualification.
