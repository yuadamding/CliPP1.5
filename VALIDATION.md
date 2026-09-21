# Local implementation validation

Historical initial-release record. The subsequent review found two qualification
defects in this source. These original observations are preserved below; they
must not be treated as validation of the repaired solver. See
[revision validation](VALIDATION_REVISION.md) for the regressions and new evidence.

Snapshot: **2026-09-21 CDT** (`America/Chicago`).
Environment: conda `ml1`, Python 3.13.2, NumPy 2.2.6, SciPy 1.18.0.
Execution: local CPU, float64.

Package-source SHA-256:
`da62da1bc86cfd2c8b8ec4a21c9f89a0de9b42cc1f14989a337ce97a47ce2c5b`.
Validation preceded the initial commit; per-file source hashes in each run receipt
bind the implementation. The upstream reference is the clean CliPP2 commit
`77525a6875e698f6834cb73e6d2a1c27af2af8bf`, which was not modified.

Verified locally:

- 49 tests: upstream likelihood fixtures, scalar bounds, independent OSQP chain
  quadratic oracle, witness search, clipping plateaus and exact kinks, refits,
  input permutations, public reporting and evaluation conventions.
- Ruff, Python compilation, workflow YAML parsing and source import isolation.
- Editable installation in `ml1` and a built wheel installed into a separate
  target; the wheel's CLI fitted the example and published all four outputs.
- Output-table hashes and source identities reconciled with `run.json`.

The wheel's three-mutation example selected two clusters. Its evidence is in
`results/wheel-example-final/`. The wheel is
`dist/clipp1d-0.1.0-py3-none-any.whl`, SHA-256
`f0e514bd1ebd17e32f5c0b4ed63b0f2d86c382a12aee4b6684b2682cb70b3502`.
Generated results and build products are ignored by Git.

Two small fresh-process synthetic checks used the exact package source above:

| Retained M | Edges | Chain array bytes | Fit seconds | Peak process RSS (KiB) | Selected K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 7 | 184 | 10.96 | 72,592 | 3 |
| 16 | 15 | 376 | 54.70 | 73,008 | 4 |

Evidence: `results/scaling-qualified/scaling.json` and its per-size receipts. The
chain byte count includes order, inverse order and weights; RSS includes the
Python process and libraries. These tiny checks verify execution and array
accounting, not asymptotic end-to-end scaling or comparative speed.

Both selected raw candidates passed their numerical gates and completed their
witness search. The full path was not completely qualified: the eight-mutation
case had incomplete witness coverage at two penalties; the sixteen-mutation
case had seven such penalties, including three with no qualified raw branch.
Those attempts remain explicit in the receipts and do not inherit the selected
candidate's qualification. The latter simulation has three generating centers
but selected four blocks, so this is not evidence of perfect reconstruction.

No paired CliPP2 full-fit, GPU, real-data cohort or large-scale statistical
qualification was performed. The committed comparison tools support that
separate work without submitting remote jobs.

Reproduce the software checks from this repository:

```bash
conda run -n ml1 python -m pytest -q
conda run -n ml1 ruff check src tests benchmarks
conda run -n ml1 python -m compileall -q src tests benchmarks
```
