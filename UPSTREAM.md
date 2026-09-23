# Upstream provenance

Reference repository: `/data/CliPP2/CliPP2` (`CliPP2` contributors).
Pinned full commit: `77525a6875e698f6834cb73e6d2a1c27af2af8bf`.
The clean local checkout was verified before fixture generation. No source fetch
or modification of that repository was needed. Port date: 2026-09-21.

CliPP1.5 is an independent implementation named `clipp1d`, following the supplied
single-region design. The original GNU Affero General Public License v3 text is
preserved verbatim in `LICENSE`; adapted likelihood code carries an attribution
notice. No upstream runtime imports, submodule, or full repository copy is used.

The pinned source and fixtures below document the original port. The current
0.5.1.dev0 implementation uses a complete graph and PyTorch CUDA float64;
the earlier chain implementation is historical reference code. See the
[current CUDA framework](docs/CUDA_FRAMEWORK.md) for the production contract.

| Reference | Inherited contract | Local changes |
| --- | --- | --- |
| `io/tumor_txt.py` | Twelve-column schema, repeated-observation validation, CN state normalization, major-CN filter, bulk scaling and upper bounds | CSV reader, exactly one sample, explicit missing/zero-depth exclusions |
| `core/objective.py` (`_compile_integer_candidates`, `_emission_kernel_numpy`, `candidate_terms_numpy`, `_observed_reduction_numpy`) | Uniform integer mixture, clipped probabilities, zero derivatives on plateaus, posterior expected candidate curvature | Single-region observation arrays; original NumPy/SciPy port, now also implemented in PyTorch CUDA tensors |
| `core/scalar.py` | Multiple wells, clipping boundaries, attained value/lower bound/gap qualification | New bounded interval search, with candidate-maxima and score-variance Taylor bounds |
| `core/clonal.py` | Historical occupied exact-one constraint | Removed from production in 0.4.0; original probability-safe bounds retained; constrained helpers are offline references only |
| `core/bic.py` (`_dirichlet_exact_partition_log_mass_and_uncertainty`, `fixed_partition_dirichlet_score`) | Nominal K log M and alpha-one exact-partition term at weight 0.7 | Explicit `clipp2_compatible_partition_score_v1` arithmetic |

At the original port, the adaptive chain, implicit operators, primal–dual solver,
safeguarded outer iteration, deterministic penalty path, contiguous-block
extraction and reporting were new local implementations. Upstream complete-graph
flows, ALM/box-QP code, hybrid candidates, Torch/CUDA and historical modes were
not copied as part of that port. This describes the provenance of the historical
chain versions, not the architecture of the current release.

Version 0.1.1 replaced the chain inner solver with a new bounded weighted-TV
functional message implementation and linear dual reconstruction. The initial
primal–dual method remains an attribution reference only. Algorithmic background:
[Kolmogorov, Pock and Rolinek, Total Variation on a Tree (2016)](https://arxiv.org/abs/1502.07770).
No external solver implementation was copied; the box handling, certificate and
common-surrogate profiling are qualified by the independent tests described in
[revision validation](VALIDATION_REVISION.md).

Version 0.5.0.dev0 replaced the production chain with a frozen adaptive complete
graph, a CUDA bounded ADMM surrogate solver and membership-based likelihood
refits. It fits within the original boxes without an occupied CCF-one constraint;
the refitted cluster closest to one receives public label zero after fitting.
Historical chain solvers and their validation records do not establish CUDA
qualification.

Version 0.5.1.dev0 corrects the overlay's exact-fusion partition fragmentation and
adds local stage-integrity controls, reduced-output likelihood kernels,
analytical-first scalar processing, packed scalar wells, bounded refit reuse and within-lambda
dual initialization. These changes do not import another upstream solver or
restore a clonal fitting constraint. Policy and receipt schema advance to v3;
measurement now separates completed CUDA work, output preparation and returned
durable-publication timing. The
[retrievable prior CUDA evidence](validation/371003f/README.md) remains bound to
`371003f` and policy v2, rather than qualifying these changes.

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
