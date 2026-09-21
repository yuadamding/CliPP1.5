# Upstream provenance

Reference repository: `/data/CliPP2/CliPP2` (`CliPP2` contributors).
Pinned full commit: `77525a6875e698f6834cb73e6d2a1c27af2af8bf`.
The clean local checkout was verified before fixture generation. No source fetch
or modification of that repository was needed. Port date: 2026-09-21.

CliPP1.5 is an independent implementation named `clipp1d`, following the supplied
single-region design. The original GNU Affero General Public License v3 text is
preserved verbatim in `LICENSE`; adapted likelihood code carries an attribution
notice. No upstream runtime imports, submodule, or full repository copy is used.

| Reference | Inherited contract | Local changes |
| --- | --- | --- |
| `io/tumor_txt.py` | Twelve-column schema, repeated-observation validation, CN state normalization, major-CN filter, bulk scaling and upper bounds | CSV reader, exactly one sample, explicit missing/zero-depth exclusions |
| `core/objective.py` (`_compile_integer_candidates`, `_emission_kernel_numpy`, `candidate_terms_numpy`, `_observed_reduction_numpy`) | Uniform integer mixture, clipped probabilities, zero derivatives on plateaus, posterior expected candidate curvature | One-dimensional observation arrays, NumPy/SciPy only |
| `core/scalar.py` | Multiple wells, clipping boundaries, attained value/lower bound/gap qualification | New bounded interval search, with candidate-maxima and score-variance Taylor bounds |
| `core/clonal.py` | Occupied exact-one constraint and original-domain eligibility | Streaming scalar witness branches; no retained state per witness |
| `core/bic.py` (`_dirichlet_exact_partition_log_mass_and_uncertainty`, `fixed_partition_dirichlet_score`) | Nominal K log M and alpha-one exact-partition term at weight 0.7 | Explicit `clipp2_compatible_partition_score_v1` arithmetic |

The adaptive chain, implicit operators, primal–dual solver, safeguarded outer
iteration, deterministic penalty path, contiguous-block extraction and reporting
are new. Complete-graph flows, ALM/box-QP code, hybrid candidates, Torch/CUDA and
historical modes were not copied.

Version 0.1.1 replaces the production inner solver with a new bounded weighted-TV
functional message implementation and linear dual reconstruction. The initial
primal–dual method remains an attribution reference only. Algorithmic background:
[Kolmogorov, Pock and Rolinek, Total Variation on a Tree (2016)](https://arxiv.org/abs/1502.07770).
No external solver implementation was copied; the box handling, certificate and
common-surrogate profiling are qualified by the independent tests described in
[revision validation](VALIDATION_REVISION.md).

`tests/fixtures/upstream_reference.npz` contains loss, gradient, expected candidate
curvature, posterior, slopes and original upper bounds evaluated by the pinned
upstream on `mixed_cn.tsv`. The adjacent JSON binds its hash, input hash, exact
source-file hashes and interpreter. It includes single-state, equal-CN,
mixed-CN, clipping-plateau and near-one upper-bound cases. Regeneration is explicit:

```bash
conda run -n ml1 python tests/fixtures/generate_upstream_reference.py /data/CliPP2/CliPP2
```

The generator rejects a different commit or dirty upstream. Normal package tests
load only the fixture. Future upstream changes require a deliberate tested port.
