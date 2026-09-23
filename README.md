# CliPP1.5: PyTorch CUDA complete-graph inference

Version **0.5.1.dev0** implements single-region observed-count fusion on a
complete graph using **PyTorch CUDA and float64 tensors**. There is one production
backend. CPU devices, unavailable CUDA and compiler failures produce explicit
errors; fitting never falls back to CPU.

Fitting is **unconstrained with respect to clonality**: every CCF stays inside its
original likelihood bounds, and no mutation or cluster is forced to CCF 1. After
selection, the refitted cluster nearest CCF 1 receives public label **0**. Ties are
resolved by the smallest canonical mutation ID. This designation changes labels
only; it does not merge clusters or change their CCFs.

The selected raw penalized CCF vector is the primary estimator. Membership-based
unpenalized refits are secondary summaries and supply the existing partition
score. Raw and refitted multiplicity calls are reported separately. Only qualified
complete-graph path candidates enter selection; unresolved starts, penalties or
refits remain visible through `search_status=incomplete`.

Policy v3 preserves exactly equal raw CCFs in the same group, including inside
overwide tolerance runs. It also reuses qualified membership refits and singleton
pilots, limits full integrity checks to owned numerical stage boundaries, and
reuses QP dual initializations within a lambda. QP polishing prepares fixed group
geometry once per proposal; combined ADMM and flow-repair steps use strict
compilation with independent admission checks. Bound-aware repair preserves a
valid incoming certificate and lets permitted box-normal residuals adapt to
edge capacities. The dense graph, likelihood,
original bounds and qualification tolerances remain unchanged.

## Installation and use

Use a CUDA-enabled PyTorch installation compatible with the allocated GPU. Set
`CC` to an available C compiler before starting Python. Installation does not
replace an existing shared CUDA stack.

```bash
python -m pip install --no-deps --no-build-isolation -e .
clipp1d fit --input-file examples/example_tumor.tsv --outdir results/cuda-example --device cuda:0
```

```python
from clipp1d import fit
result = fit("tumor.tsv", outdir="results/tumor", device="cuda:0")
print(result.raw_phi)       # primary estimate
print(result.refitted_phi) # secondary summary
print(result.search_status)
```

The twelve-column TSV input contract, copy-number filtering, marginalized
multiplicity likelihood, original probability-safe bounds and partition-score
arithmetic are retained. Complete-graph weights are frozen after the independent
pilots. Dense edge storage is quadratic in mutation count; memory admission fails
explicitly when the estimated workspace exceeds the configured share of free GPU
memory. This preflight does not guarantee that later compiler allocations fit.

The schema-v3 `run.json` receipt records separate numerical, final qualification,
device export and output-preparation phases. GPU peaks are read after the last
required CUDA work. Receipt `elapsed_seconds` ends before receipt serialization
and durable publication; the returned `result.operation_metrics` separately
reports completion after publication. See the
[measurement scopes](docs/CUDA_FRAMEWORK.md#timing-and-memory-scopes) before using
these fields for benchmarks.

## Validation and evidence

```bash
python -m pytest -q
python -m ruff check src tests benchmarks
PYTHONPATH=src python benchmarks/qualify_cuda.py --device cuda:0 --out results/cuda-qualification.json
```

CPU tensor tests compare the numerical equations with independent references.
They do **not** qualify CUDA compilation or execution. The qualification command
requires an actual allocated CUDA device and records compiled/eager results,
source identity, numerical certificates, runtime and memory. A small qualification
run does not establish cohort accuracy or scalability at all input sizes.

The [QP benchmark contract](docs/QP_BENCHMARKS.md) separates uninstrumented
latency from profiling overhead and describes the larger heterogeneous
mixed-support qualification. `qp_admm_iterations` totals work across all
attempted starts and candidates, including unresolved attempts.

The historical chain numerical modules remain available for reference tests.
They are not called by the public fitting API. Historical CPU cohort launchers
reject this CUDA source; existing frozen runs and their result evaluators keep
their original contracts. Do not replace the source beneath running workers.

See [the CUDA framework](docs/CUDA_FRAMEWORK.md) for the model, certification and
output semantics. The [bound-recovery study](VALIDATION_BOUND_RECOVERY.md) records
exact failed-surrogate diagnoses and the current qualification status; its
[diagnostic contract](docs/BOUND_RECOVERY.md) explains capture and replay.
The [coordinate-backtracking evidence](validation/cuda-surrogate-review-v4/README.md)
qualifies all six declared synthetic CUDA stages and verifies actual cohort
staging adoption. It retains the negative literal-QP result and keeps scalar
backtracking as the production default. Diagnostic qualification timings do not
establish throughput; the separate timing driver disables tracing.
Its [first completed timing stage](validation/cuda-surrogate-timing-v1/README.md)
retains cold and warm pairs separately and excludes incomplete work from ratios.
The historical [QP optimization report](VALIDATION_QP.md) records `430db26`'s
matched timing, numerical parity and failed heterogeneous stress cases;
its [evidence archive](validation/cuda-qp-v3/README.md) retains every attempt.
The earlier [validation report](VALIDATION_CUDA.md) distinguishes local
reference tests from allocated GPU qualification for the `daaf50a` baseline; its
[source-bound receipts](validation/cuda-review-v3/README.md) include the final
acceptance and preserved failed attempts. The retrievable
[prior CUDA evidence for `371003f`](validation/371003f/README.md) qualifies its
policy-v2 source only. It does not qualify this revision. Historical chain
documentation and receipts likewise retain their original scope.

The original scientific provenance and AGPL-3.0 license are retained.
