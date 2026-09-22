# Unconstrained production and repository audit

Validated 2026-09-22 11:17 CDT, using the local `ml1` Python environment on CPU.
This report applies to the working-tree package **0.4.0**, based on
`f90ac33e34f37a87ee498046cefa98c791df05dd`, with package source fingerprint
`71db1c7e61ecab1bbf18a963cd9195921ec42e87ceb01ead8e32389a4e4b1307`. The base commit alone does not include these edits.

## Scientific change

Production requires no occupied CCF-one cluster. Lambda zero retains the marginal
pilots; positive penalties solve the original boxed chain problem without a frozen
witness; every final block is independently refitted. Extraction treats exact-one
and near-one values with the same fusion tolerance. All public labels descend by
refitted CCF, with chain-position tie breaks. A center can naturally equal one.

The original likelihood, probability-safe boxes, immutable chain, Ward/boundary
proposal rules, score arithmetic and numerical tolerances are retained. Policy and
receipt schema advance to v5. Witness/designation fields are null and the legacy
`designated_clonal` table column is zero. Historical constrained helpers remain
available only for offline regression work and are not called by production.

## Audit repairs

- Integer counts and copy numbers are validated before float conversion. Inputs
  such as `1.000000001` and `9007199254740993` are rejected rather than rounded.
  Malformed gzip, UTF-8 and TSV quoting produce input errors; tumor metadata IDs
  are validated; blank inert annotation columns are allowed.
- Nonfinite/incorrectly shaped solver starts are rejected before clipping. Invalid
  tolerances, iteration budgets, fusion tolerances and partition boundaries fail
  explicitly. Reused zero-penalty pilot gaps must satisfy the requested policy.
- Publication checks label/center/refit consistency, occupied partitions, score
  components, pilot identity, raw-reference identity, candidate family and raw
  certificate lineage. Boundary refinements retain their actual seed provenance.
  A direct winner has no inherited raw certificate.
- Entire receipts are serialized before fit-table writes. Complete JSON markers
  are published atomically without overwriting existing evidence. This uses hard
  links; output filesystems without hardlink support, including exFAT, are not
  supported. I/O failures may leave partial TSVs; a successful receipt and matching
  hashes remain necessary to establish publication.
- Benchmark tools distinguish constrained v4 and unconstrained v5 results,
  including nullable designated-cluster metrics, zero designation flags,
  `raw_reference_ccf`, direct-partition objectives and free-solver instrumentation.
  Historical constrained diagnosis/replay tools require their pinned source;
  importing the historical integration replay no longer launches fits.

## Verification

The full regression suite passed **490 tests in 26.91 seconds**. It includes the
pinned upstream likelihood fixture, independent OSQP quadratic comparisons, an
independent SLSQP exact-binomial/TV comparison, original-bound and clipping tests,
all-subclonal fits, direct candidate provenance, publication failure tests and
benchmark metric/queue regressions. Ruff, compilation and `git diff --check` pass.
Tests used CPUs 30–31 with numerical-library threads fixed at one.

A wheel was built without dependency changes and installed into a fresh isolated
target. Its package source hashes match the audited source. CLI checks ran from
outside the source tree and verified all output table hashes:

| Input | Refitted centers | Search |
| --- | --- | --- |
| Committed example | 1, 0.2625 | Complete |
| Sixteen mutations, two subclonal populations | 0.75, 0.25 | Complete |
| Three mutations, every original upper bound below one | 0.10 | Complete |

A truncated gzip CLI input returned exit status 2 with a typed `InputError`
receipt and no fit tables. The wheel reports version 0.4.0.

Local detailed evidence is under
`results/repo-audit-unconstrained-20260922-v1/`: `pytest.log`,
`source-provenance.json`, `package-smoke.json`, the built wheel and per-case outputs.
The wheel SHA-256 is `6c9d5317eedfdde30f5c52debd050decd335520fd59a5fa097be34d288af27b9`.

These checks establish local implementation correctness within the exercised
cases. They do not establish cohort accuracy, speedups, global optimality or
cross-platform numerical qualification. Existing frozen runs and their historical
results were not rewritten; the previous cohort scores are not unconstrained
results. No cohort or remote jobs were launched by this audit.
