# Review checklist disposition

Review/checklist source snapshot: `15d397b680ed3392a2ec3bbead1e5d5b460448ff`.
This is the local engineering revision of that estimator, not a published release.
Concrete execution evidence and limits are in [validation.md](validation.md).

| Review area | Implemented change | Evidence / remaining boundary |
| --- | --- | --- |
| Freeze estimator | Separate model/initializer/search/score identities; normative methods; frozen original sample outputs | All K=1..10 sample memberships/centers/weights/scores; protected panel comparison against archived original executable |
| Input/CNA ambiguity | Explicit autosomes, inclusive integer coordinates, exactly one purity scalar, strict widths/CN/identity validation; overlap rejection | R preprocessing tests, original sample matching and row ledger; genome assembly is declared, never inferred/lifted over |
| Traceable population | Original input copies, original row/string IDs, matched segment and exclusion reasons, canonical retained data | Independent Python reconstruction of counts/interval matching; zero-depth/no-CNA/empty cases tested |
| Native identity | Private package resource; build/source/binary hashes, ABI 3, embedded build/model/config checks | Wrong/stale source rejected; newer unrelated library ignored; absent-NVIDIA clean install passes |
| Environment flags | Shared Boolean spelling contract, explicit CPU build default and runtime conflicts | Python/C++ tests cover missing, empty, false/true and invalid flags |
| Package/API/CLI | pyproject, wheel/sdist resources and dependencies, entry points, import-safe legacy wrapper, external output | Fresh installs outside checkout; read-only installed-package execution; mandatory new run path |
| Failure/publication | Isolated attempts, exclusive output lock, copied inputs, atomic directory publication, completion marker last | Failure injection, reused paths, missing diagnostics, corrupt files, changed chain/raw diagnostics and stale Best_K tests |
| Independent verifier | Independent binomial enumeration, scores/weights/posteriors, memberships/CP/CCF/chain, raw diagnostics and exact parent identity | Every successful API run verified before publication; no global-optimality or authenticity claim |
| Numerical regression | CPU C ABI for likelihood, projection and stable chain QP; recorded constants | Finite differences, surrogate curvature, exhaustive DP/projection, independent dense QP/KKT and high-rho tests |
| Search semantics | Likelihood/refit/scoring/selection/output modules separated without changing arithmetic | Nonmonotone DP, incumbent preservation, adjacent exact-center merging, zero-weight repair, q/K and score counting tests |
| Subsampling | Explicit MT19937 seed+replicate policy, indices/IDs, full initialization and full-N refit/score | No replacement, deterministic quotas/replay, midpoint expansion and full-data score tests |
| Initializer | Serialized finite-grid gap/budget/compression controls, including active refit interval | Permutation/duplicate/grid compression/budget tests; grid certification stays separate from global fit claims |
| CUDA | Explicit source build, availability versus execution-failure handling, metadata and allocation-only differential/full-fit tests | **Unqualified here**; six CPU-environment skips are not GPU evidence; actual allocated execution remains a release gate |
| Container | Complete-source CPU image recipe, non-root execution, external work directory | **Unqualified here**; Docker/Podman unavailable; image digest/execution cannot be fabricated |
| Profiling | Stages and API wall time through publication; process-tree RSS sampling | Recorded on real local runs; excludes import and final marker-write time; GPU peak VRAM unmeasured; no claimed speedup |
| Scientific validation | Versioned independent truth generator; development/protected seed splits; per-case failure-aware evaluator | Small correctness/robustness panels executed; broad current-cohort accuracy and formal noninferiority remain unestablished |
| Docs/citation/license | Current quick start, model/input/output/migration/reproducibility/limitations guides; changelog, citation and contribution/release policy | AGPLv3 text preserved; legacy author attributions retained; no historical accuracy claims transferred |
| CI/artifacts | CPU matrix and manual review-artifact workflow; source-only archives and checksums | Local checks run; remote CI, publishing and tags are not implied or performed |
| Future optimization | Accurate serial CPU/hybrid CUDA and memory limits documented | Algorithmic/threading/cache redesign deferred until separately measured and regression-qualified |

The original checklist's CUDA/container gates explicitly permit an unqualified
label instead of unsupported execution claims. Large scientific benchmarks and
future optimization are separate experiments, not implied by successful packaging.
No unrestricted reassignment, changed multiplicity prior, changed score penalty,
clonal constraint or multi-region model was added.
