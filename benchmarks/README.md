# Current-estimator validation protocol

`clipp.simulation` is a new versioned CN-first generator for small, truth-labelled
single-sample tests. It is not the historical 4K generator and does not claim to
reconstruct that cohort. First draw major/minor CN, then uniform multiplicity in
1..major, then binomial reads at the assigned CCF. The receipt records all settings,
PCG64 seed, NumPy version, ascertainment and file hashes. Truth stays separate from
all inputs passed to `fit`. The legacy `sample/` does not have invented truth.

Protocol `chain_engineering_panel_v1` is a staged panel of diploid, LOH, balanced
amplification, high major CN, rare/close components, no-clonal cases, physical
endpoints and true cluster counts above the capacity cap. Correctly specified
cases are separate from purity error, nonuniform multiplicity, overdispersion and
ALT ascertainment. Large CNA-error/cohort panels remain a separate scientific
validation task rather than being silently mixed into this small engineering test.

Development and protected splits use different fixed seed offsets. Freeze the
estimator, generator and script hash before looking at protected results. If an
engineering defect changes tested source, retain the failed run, freeze the repair
and explicitly declare a new validation attempt; do not call reused seeds unseen.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/run_panel.py --split development --replicates 2 \
  --device cpu --output /tmp/clipp-development-v1
# After freezing the source/protocol:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/run_panel.py --split protected --replicates 2 \
  --device cpu --output /tmp/clipp-protected-v1
```

The runner admits one case at a time, retains failures and writes attempt/status
rows, ARI (true-K-one/constant-partition convention: 1 when both are constant),
true/fitted cluster counts, signed over/underclustering, CCF MAE, rare-cluster
best-match F1, CNA-only multiplicity macro precision/recall/F1, matched mutation
counts, full API time, sampled RSS and median/p95 runtime. CNA means anything
except 1/1, including balanced amplification. Macro classes use the union of
observed truth/prediction classes, with zero undefined precision/recall; raw
truth and posterior tables permit different predeclared summaries. Missing output
is a failed attempt, never a perfect or omitted case. Absent rare/CNA groups are
NA, not perfect scores. This script does not train or tune the estimator.

For a broad scientific study, predeclare distributions/sample sizes, external
method settings, aligned retained populations and uncertainty scopes. Archive
large data/results outside source control with hashes. Separate failure rates
from accuracy among completed cases. Stratify by depth, SNVs, purity, major CN,
true cluster count and rare/close components. Compare source-pinned full pipelines,
not isolated kernel times. The included panel is deliberately small and cannot
support broad accuracy superiority or a three-cohort noninferiority claim.
