# Engineering validation — October 2, 2026 (America/Chicago)

This revision protects the estimator from `15d397b`. It is not a GPU-qualified,
container-qualified, broad-cohort accuracy claim or published release.

## Claims and evidence

| Claim | Executed evidence | Limit |
| --- | --- | --- |
| Legacy numerical behavior preserved | Frozen original source/binary run before refactor; all 10 capacities preserve labels, CPs, weights and scores exactly in ml1 | One 487-SNV legacy smoke sample; biological truth unknown |
| Broader numerical equivalence | 24 independent truth-labelled cases, all 10 capacities compared against original executable | Small panel, includes correctly specified and robustness cases; no biological superiority claim |
| Input/output/numerical contracts tested | Pytest suite covering primitives, R validation, failure injection, build identity and verified fits | CPU only; GPU skips listed separately |
| Installable without NVIDIA | Clean wheel and sdist builds/installs in fresh Python environments; example executed and verified outside checkout | Linux x86_64 Python 3.13 exercised locally; remote CI matrix not yet run |
| Read-only installed code works | Full installed-wheel test suite with package files/directories non-writable; external result paths | Mutation test uses a separately writable disposable copy |
| Simulation protocol is reproducible | 24-case development and 24-case protected-seed panel; generator/truth/receipts retained, failures included | Source frozen before evaluation; engineering confirmation replays are explicitly identified |
| Timing is complete for the declared scope | Per-stage and API entry-through-publication timing, sampled process-tree RSS | Not process-startup time or exact peak allocation; GPU peak VRAM unmeasured |

The final local source suite and the fresh installed-wheel suite each passed
**83 tests, with 6 CUDA tests skipped**. The installed suite ran with read-only
package files outside the checkout. The final source distribution installed in
a separate clean environment, ran all ten example capacities, and independently
verified K=3, q=3 and BIC-form score 4550.46473695031.

The [local validation receipt](/storage/CliPP2/clipp15_repo_revision_20261002/VALIDATION.json)
records source/build and artifact hashes; it is a local evidence location, not a
public archive. Distributable source archives include this guide, fixtures and
tests, while full generated runs remain outside version control.

## Retained execution evidence

The local evidence root is
`/storage/CliPP2/clipp15_repo_revision_20261002/`, outside source control. Its
`baseline/manifest.json` binds all original tracked files and the CPU binary.
`baseline/full_example/` retains the original sample run. The checked-in
`tests/fixtures/legacy_cpu_baseline.json` contains the small regression reference.
`tests/fixtures/synthetic/` contains separate generated truth, never fitting input.

`development-panel-v1/` and `protected-panel-v1/` each contain 24 completed,
independently verified cases with no fit failures. `baseline-protected-v1/`
compares all 240 capacity results: labels agree exactly and maximum CP, weight
and BIC-form differences are zero. This comparison used identical ml1 dependencies
and one numerical-library thread. Panel time medians were about 2.83 and 2.98
seconds, including verification; these are observations, not performance targets.

Final audit added only rejection of inconsistent text row widths and quoted
purity tokens, preventing implicit row-name coercion. Confirmation on the same
protected seeds is an **engineering replay**, not a new unseen holdout. It does
not tune the estimator or change the generator, prior, chain/search or score.
The final evidence receipt identifies that replay and its exact source/build.
All 24 confirmation cases completed and verified. Comparison against the original
executable again matched all 240 capacity results exactly: assignments, CPs,
mixture weights and scores. The final code in the tested wheel and source checkout
was hash-compared to the code bound by these confirmation manifests.

Clean installation used a fresh Python 3.13 environment without NVIDIA packages;
NumPy 2.5.3, SciPy 1.18.1 and pandas 3.0.6 also passed the installed numerical
regression, separately from ml1's NumPy 2.2.6, SciPy 1.18.0 and pandas 3.0.3.
The first read-only suite found a test-copy permission defect; that failed log is
retained, and the repaired test changes permissions only on its disposable copy.

## Unqualified combinations and deferred scientific claims

Six CUDA-only differential/full-fit tests are deliberately skipped without an
explicit allocated GPU. No new CUDA execution is claimed. Docker and Podman were
not available, so no built image digest or container execution is claimed. The
GitHub CPU matrix is configured but has not been remotely executed as part of
this local revision. No changes have been pushed by these validation steps.

Broad accuracy/scalability on CN-first4K, SimClone1000 and PhylogicNDT500 requires
separate, source-pinned comparison with population alignment and failure counts.
The small panel exposes some difficult balanced-amplification and multiplicity
aliasing behavior; successful execution is not successful biological recovery.
Those scientific limitations are preserved rather than hidden by changed defaults.
