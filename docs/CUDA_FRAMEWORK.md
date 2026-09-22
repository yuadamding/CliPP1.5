# PyTorch CUDA complete-graph framework

Package: `0.5.0.dev0`. Policy: `clipp1d_complete_cuda_unconstrained_v2`.
Output schema: `clipp1d.cuda.run.v2`.

## Statistical contract

For the canonical observed-count negative log likelihood, fit

    F_lambda(phi) = sum_i f_i(phi_i)
                   + lambda * sum_{i<j} w_ij * abs(phi_j - phi_i)
    original_lower_i <= phi_i <= original_upper_i.

There is no occupied-clonal constraint, witness profiling, attraction to one or
post-fit clonal projection. Inputs whose upper bounds are all below one remain
valid. The host input compiler preserves the existing CN mixture, uniform integer
multiplicity support capped at four, clipping and original feasible bounds.

Complete-graph fusion is a changed statistical model relative to the historical
chain. The marginalized observed likelihood can be nonconvex. A finite-start
qualified candidate is not a proof of global optimality.

## Execution boundary

CPU work parses and validates TSV input, compiles canonical arrays, constructs
source hashes, makes orchestration decisions from scalar status checks, and
serializes the final result. Mutations are canonically ordered by string ID
before uploading the numerical model.

PyTorch CUDA tensors hold the float64 likelihood, posteriors, one-sided
derivatives, curvature, pilots, grouped scalar refits, graph weights, ADMM states,
raw certificates, path scores and final multiplicity computations. Pure tensor
hot kernels use `torch.compile(fullgraph=True, dynamic=True)`. Each of the four
compiled kernels has its own per-fit LRU bank of up to 32 structural families.
These separate singleton dimensions, dimension-equality patterns, layouts and
aliases using independent Python code objects; numeric extents remain dynamic
within each family. Global Dynamo cache limits stay unchanged. Receipts
record family counts and evictions. Compiler errors propagate. No
CPU optimization, min-cut or fitting fallback is available through the public
API. Python control and device synchronization remain at qualification checks;
this is not a claim that the entire dynamic fit is one captured CUDA Graph.

Low-level tensor functions accept CPU tensors for independent numerical tests.
Such evidence is explicitly recorded as CPU reference execution.

## Fixed graph and storage

The immutable graph uses every unordered pair:

    delta_ij = abs(pilot_i - pilot_j)
    floor = max(1e-8, 0.1 * median(positive adjacent sorted-pilot gaps))
    r_ij = 1 / max(delta_ij, floor)
    w_ij = r_ij / mean_{a<b}(r_ab).

If no positive gaps exist, the floor is `1e-8`. All equal pilots give unit
non-diagonal weights. The median averages the two central values for even counts.
Sorting supplies a spacing statistic only; it does not restrict graph edges.

Weights are symmetric with zero diagonal. Edge primal/dual matrices are
skew-symmetric, with

    D(x)[i,j] = x[j] - x[i]
    D'(q) = -row_sum(q).

Energies use half the full matrix sum, representing exactly `M*(M-1)/2` edges.
This dense layout stores both orientations and uses quadratic memory. Identity
checks and private snapshots reject modified graphs, numerical models and stale
continuation states, including modifications through tensor `.data`.

## Convex surrogate and raw qualification

Each safeguarded outer step solves

    Q(x) = 0.5 * sum_i h_i * (x_i - target_i)^2
           + sum_{i<j} caps_ij * abs(x_j - x_i),
    caps = lambda * weights,

on the original box. ADMM keeps weights in `caps`, so `D'D = M I - 11'`.
The bounded node update solves a diagonal-minus-rank-one system through the
scalar equation

    x_i(s) = clip((b_i + rho*s)/(h_i + rho*M), lower_i, upper_i)
    s = sum_i x_i(s).

The implementation uses sorted entry/exit breakpoints and safeguarded roundoff
polishing. It is not an unconstrained update followed by clipping. Numerical
`rho` starts at `median(h)/M`; it does not change statistical lambda.

Surrogate admission requires both the stable primal-dual gap and KKT residual.
Frozen-coordinate constants do not inflate gap tolerances. Equality polishing
must pass an objective-nonincreasing check with a float64 roundoff margin and
fresh certificates for the original QP; it cannot create a reporting-only adjustment. Original tolerances remain unchanged.

An independent raw audit uses observed one-sided derivatives, exact signed
contributions from nonfused edges and TV terms on fused edges. Both positive and
negative feasible-box directions are checked without witness restrictions. For

    min a' t + sum_{i<j} c_ij*abs(t_j-t_i), 0 <= t <= allowed,

a feasible dual yields the lower bound

    sum_i min(a_i + (D'q)_i, 0) * allowed_i.

Arithmetic margins account for absolute dual-row contributions before
cancellation. This check covers arbitrary mutation subsets, including groups
noncontiguous in any pilot order. An attained descent direction is backtracked
against the original objective. An unresolved audit is not success. Raw
stationarity, inner QP qualification, scalar refit gaps and global optimality are
separate claims.

## Memberships, selection and labels

Partitions derive from raw fitted CCFs. Tolerance-connected runs whose total
diameter exceeds `fusion_tol` split conservatively into singletons. Exact-one
and near-one values remain separate. Canonical groups are ordered by their
smallest mutation ID, not by adjacency in a fixed chain.

Each arbitrary membership group is independently refitted inside the intersection
of its original bounds. No group is forced to one. Analytic or interval-search
scalar certificates qualify pilots and refits. Clipping-aware interval bounds
account for represented endpoints, midpoint rounding and likelihood sensitivity.
These are float64 numerical qualifications, not interval-arithmetic proofs. The
existing score is

    2*L_refit + K*log(M) - 1.4*log_partition_mass.

The score selects a lambda candidate while preserving its raw penalized vector.
Within a lambda, qualified starts are compared using the raw objective. Equal
partition scores prefer fewer occupied groups, then the smaller lambda. There
are no direct Ward, boundary-search or CEM winners in this production revision.
Lambda-zero qualification validates the separable scalar gaps and their actual
likelihood. Nonzero penalties require inner and independent raw certificates.
Unresolved starts, penalties or refits make the path status incomplete even if
another candidate can be published.

For more than one mutation, the default path is zero plus
`lambda_reference * 2**k` for `k=-12,...,12`. The reference uses the balanced
gradient of a pooled boxed pilot quadratic:

    lambda_reference = max_{i<j} abs(g_i-g_j) / (M*w_ij).

This constructs a path scale; it is not an exact fusion threshold for the
nonconvex likelihood. A degenerate reference uses `1e-3`; a singleton uses only
lambda zero. If the uppermost candidate wins, up to three additional doublings
are attempted. `search_status=complete` means every start and refit on that
bounded planned path qualified. It does not mean all penalties were explored:
`provenance.numerical_stages` separately reports `selected_at_upper_boundary`,
`extension_limit_reached` and `path_truncated`.

Public cluster 0 is the secondary refitted center with smallest L2 distance to
one (absolute distance in a single region); ties use the canonical minimum
mutation ID. Other labels descend by refitted center. Labeling does not change
memberships, raw estimates or refits. A designated cluster may have CCF below one.

## Outputs and provenance

A successful output directory contains:

- `mutation_clusters.tsv`: original mutation IDs, exclusions, canonical
  `node_index`, pilot CCF, primary `raw_ccf`, secondary `refitted_ccf` and labels.
- `cluster_centers.tsv`: occupied cluster sizes and secondary refitted centers.
- `mutation_multiplicity.tsv`: separate `raw_multiplicity_call` and
  `refitted_multiplicity_call` columns for retained mutations.
- `run.json`: the versioned receipt and searched-path records.

`node_index` is bookkeeping in mutation-ID order, not a fusion adjacency. Use a
fresh or empty output directory; existing results are not overwritten.

The run receipt distinguishes raw primary CCF, secondary refitted CCF, their
conditional multiplicity calls, original bounds, graph/model/source identity,
partition identity, selected penalty, raw certificates, refit gap and path
coverage. The `designated_clonal` output is a membership designation, not a claim
that raw or refitted CCF is exactly one. `raw_witness_mutation_id` is null.

Output validation precedes successful publication. A failure or an unresolved
candidate cannot inherit another candidate's certificate. Old chain receipts
and old constrained CUDA schema-v1 receipts retain their original meaning.

## Validation boundaries

See [the current evidence](../VALIDATION_CUDA.md). Numerical CPU references and
CPU fullgraph tracing do not establish generated CUDA kernel correctness. Actual
GPU qualification must bind the exact source, environment, input, compiler and
allocated device, with eager/compiled numerical comparison and public output
validation. Small synthetic qualification does not establish full-cohort
accuracy, large-input memory behavior or a GPU speedup.

The retained historical chain files and their test-only API adapter support
regression comparisons. They are not a production backend. Legacy CPU launchers
must reject the CUDA revision before starting new work; old frozen workers and
historical evaluators remain source-bound.
