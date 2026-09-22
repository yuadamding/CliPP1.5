# CliPP1.5 / `clipp1d`

CliPP1D estimates mutation CCFs and clusters mutations from one tumor sample using
an observed-count likelihood and adaptive fusion along a fixed chain ordered by
unpenalized marginal CCF estimates. At least one occupied cluster is constrained
to CCF exactly one.

Its Python package and command are `clipp1d`. It is an independent NumPy/SciPy
implementation, with CPU float64
arrays and guarded internal message accumulation, with no runtime dependency on CliPP2, PyTorch, or a graph library.

```bash
git clone https://github.com/yuadamding/CliPP1.5.git
cd CliPP1.5
python -m venv .venv
source .venv/bin/activate
python -m pip install .
clipp1d fit --input-file examples/example_tumor.tsv --outdir results/example
```

Requires Python 3.10 or newer. On Windows, activate the environment with
`.venv\Scripts\activate` instead.

```python
from clipp1d import fit

result = fit("tumor.tsv", outdir="results/tumor")
print(result.cluster_centers)
print(result.raw_phi)       # selected penalized candidate, retained-mutation order
print(result.refitted_phi)  # public estimator, same retained-mutation order
print(result.search_status) # complete or incomplete; separate from fit validity
```

The CLI exposes only `--input-file`, `--outdir`, `--max-major-cn` (default 4),
and `--verbose`. Use a fresh or empty output directory. Running directly from
the source tree without installing also works with `PYTHONPATH=src python -m clipp1d`.

## Input

Plain or gzip TSV with the twelve CliPP2 long-table columns:

```text
mutation_id sample_id alt_count ref_count count_observed purity normal_cn
segment_id cn_state_id cn_state_fraction allele_a_cn allele_b_cn
```

Exactly one sample is required. IDs remain strings. CN states repeat one read
observation; repeated fields and segment-wide complete state sets are validated.
Fractions must be positive and sum to one. Copy numbers are nonnegative integers,
with `allele_a_cn >= allele_b_cn`. Extra columns are inert. Optional CliPP2
`##key=value` metadata precedes the header.

Any original state above the major-CN cutoff excludes the mutation, even if its
fraction is tiny. Missing counts and zero depth are excluded from this
single-sample chain and reported with reasons. Zero alternate reads remain
informative. Retained all-zero CN is unsupported. The multiplicity support is
uniform over `1..min(4, maximum major CN)`, independently of the input cutoff.
The original probability-safe upper bounds determine clonal eligibility.

## Method and outputs

Counts → qualified marginal pilots → frozen adaptive chain → clonal-witness
shared-surrogate fusion fits → contiguous blocks → constrained likelihood refits → partition score.
There is no hard ordering constraint, reordering during fitting, nonadjacent
merge, Ward/CEM proposal, or permanent highest-pilot clonal witness.

The public estimator is the constrained likelihood refit of the selected
chain-generated partition. Labels are public labels: `0` is the designated
clonal block; other blocks are ordered by decreasing refitted CCF. Disconnected
blocks remain distinct even if their centers coincide.

| File | Contents |
| --- | --- |
| `mutation_clusters.tsv` | All input IDs, retained/exclusion status, zero-based chain rank, pilot/raw/refitted CCFs, label |
| `cluster_centers.tsv` | Public label, size, refitted CCF, designated-clonal flag |
| `mutation_multiplicity.tsv` | Retained IDs and posterior MAP multiplicity conditional on refitted CCF |
| `run.json` | Policy, source/input/table hashes, environment, score terms, scalar gaps, numerical and start/path coverage |

Excluded numeric fields are `.`. Failed validation or numerical admission writes
`run.json` with `status: failure` and no fallback fit tables. A publication I/O
failure can leave partial files; only a successful `run.json` and matching table
hashes establish a published result. Existing evidence is never overwritten.
Receipts retain UTC timestamps. Convert displays with `America/Chicago`.

## Numerical claims and limits

The inner convex quadratic is qualified by a nonnegative-term primal–dual gap
and a separate KKT residual. Frozen-coordinate constants cannot inflate the gap
tolerance. The nonconvex
raw branch is separately checked for box/dual feasibility, componentwise
stationarity, edge complementarity, and feasible signed subinterval clipping
directions, including proper sub-blocks beside a frozen witness. Scalar searches
return attained losses, conservative float64 lower bounds and gaps. These are
numerical qualifications, not formal interval-arithmetic proofs.

Production uses one common surrogate per outer step: prefix/suffix messages
profile every eligible witness under the original boxes, then reconstruct and
certify the selected minimizer using their retained reconstruction thresholds.
Each accepted vector shares one likelihood context across all witness audits.
Backtracking rebuilds both messages. The previous
witness is released; observed-likelihood majorization and descent remain mandatory.
At most four distinct, clonal-feasible primal starts are tried per penalty.
Continuation carries only the primal vector; differences in unused duals never
create a new start. This changes the nonlinear search policy relative to 0.1.1,
while preserving the likelihood, constrained objective, weights, score and
numerical tolerances. Independent nonlinear witness enumeration is retained only
as an [offline reference](benchmarks/reference_enumeration.py).

A qualified candidate can be selected when another start or penalty is unresolved.
The CLI, `FitResult.search_status`, and top-level `run.json.search_status` report
`complete` or `incomplete`. Each penalty records separate `raw_status` and
`refit_status`, preserving raw diagnostics if the refit fails. `search_complete`
means all planned primal starts qualified; `path_search_complete` also requires
every path penalty and refit to qualify. Neither means independent nonlinear
witness enumeration or a global optimum. `search_profile_calls` counts common
QP profiles; `search_surrogate_witnesses_profiled` counts witness values across
those profiles. The zero-penalty case reports its qualified separable scalar-gap
search separately. Version 0.2.1 uses numerical policy `clipp1d_chain_v3` and receipt
schema `clipp1d.run.v3` to make these changed search semantics explicit.
The selected raw objective and witness mutation ID are retained in both the API
result and receipt; `raw_witness_index` is a zero-based **chain** position.
`global_optimality_proven` remains false. The zero-penalty separable solution also
reports its global scalar gap. The selected partition refit reports its own gap.

Graph storage and graph operations are linear in retained mutation count M.
Numerical state is O(M Cmax + M), with Cmax ≤ 4; each primal–dual iteration is
O(M) in the retained reference solver. Production uses a direct bounded,
weighted TV message pass with O(M log M) heap work, O(M) storage and a linear
dual certificate pass. A fixed clonal witness separates its two sides.
Shared prefix/suffix profiling and one selected reconstruction cost O(M log M)
per common surrogate, with O(M) storage. Sorting costs O(M log M). Full runtime
also depends on scalar search, penalties, starts, outer iterations and backtracks;
production has no independent witness-enumeration factor. No unconditional linear
fitting-time or speedup claim is made. Ambiguous multiplicities can produce an
unreliable pilot order. Heap/message arithmetic uses `numpy.longdouble` guard
digits where available; output arrays and all admission tolerances remain float64.
The unchanged gap and KKT audits reject unqualified arithmetic.

The adaptive floor/weight policy and CliPP2-compatible score are development
choices. Statistical adequacy and end-to-end speed relative to CliPP2 require
matched data qualification. The repository supplies that comparison tooling;
unit tests alone do not establish cohort accuracy or GPU qualification.

## Development and evaluation

In an activated Python environment:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
ruff check src tests benchmarks
python -m compileall -q src tests benchmarks
```

For this project's local development environment, the equivalent example uses
the preconfigured `ml1` conda environment:

```bash
conda run -n ml1 python -m pip install --no-deps --no-build-isolation -e .
conda run -n ml1 python -m pytest -q
conda run -n ml1 ruff check src tests benchmarks
conda run -n ml1 python -m compileall -q src tests benchmarks
```

The test extra supplies pytest, Ruff and OSQP. OSQP is used only as an independent
small chain-QP oracle. Tests use committed upstream likelihood fixtures and do
not import CliPP2. CI mirrors these checks on Python 3.10 and 3.12.

```bash
python benchmarks/simulate.py --mutations 30 --outdir results/simulation
python benchmarks/benchmark_scaling.py --sizes 100 1000 --scenarios easy mixture --timeout-seconds 120 --outdir results/scaling
python benchmarks/benchmark_chain.py --sizes 100 1000 --outdir results/chain-attribution
python benchmarks/compare_clipp2.py --help
```

The comparison tool consumes already validated CliPP2 outputs; it does not submit
remote work. See [formulation](docs/formulation.md),
[implementation](docs/implementation.md), [current validation](VALIDATION_REUSE.md),
[historical 0.1.1 validation](VALIDATION_REVISION.md),
and [upstream provenance](UPSTREAM.md). Scaling receipts retain partial stages
and explicit timeouts; a timeout does not produce a qualified full-fit result.

Released under the upstream GNU Affero General Public License v3; see [LICENSE](LICENSE).
