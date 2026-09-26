# PyTorch CUDA complete-graph framework

Package: `0.5.1.dev0`. Policy: `clipp1d_complete_cuda_unconstrained_v3`.
Output schema: `clipp1d.cuda.run.v3`.

An opt-in [separate partition estimator](PARTITION_SEARCH.md) now adds explicit
membership refitting, all-qualified-start candidate scoring, and sequential
exact-score reassignment, and separately qualified whole-group birth proposals.
Its v5 output extension preserves the primary raw
estimator and the baseline v3 table semantics. The baseline contract below
continues to apply; the new estimator has separate provenance and qualification.

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
hot kernels use `torch.compile(fullgraph=True, dynamic=True)`. Each of the eight
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

Full snapshot reconciliation occurs at owned numerical stage entry and exit,
including exceptional exits. Inside a stage, identity, metadata and tensor
version checks avoid repeated full-array comparisons. Explicit
`validate(full=True)` still forces full reconciliation. Scalar group reductions
use `segment_reduce(..., unsafe=True)` only inside an owned validated stage whose
immutable group lengths have already been checked; outside it, the reduction's
independent length checks remain enabled. Gap, KKT, finite-value and publication
gates are unchanged. Integrity-check counters distinguish these operations.

The likelihood kernels share log-kernel arithmetic but expose loss-only,
loss–gradient and full-term interfaces. Scalar search and objective comparisons
request only the outputs they need. Full terms remain available for curvature,
one-sided audits and final posteriors; the other compiled kernels implement the
bounded node update, edge update, combined ADMM step, prepared flow-repair step
and QP certificates. Compilation covers these functions, while data-dependent
admission remains in Python. `fullgraph=True` applies to each compiled function,
not the entire fit; see the [PyTorch compilation contract](https://docs.pytorch.org/docs/stable/generated/torch.compile).

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
`rho` starts at `median(h)` and is then adjusted by residual balancing. Dividing
by all nodes can undercondition consensus on much smaller active fused groups;
this numerical parameter does not change statistical lambda or the objective.

Surrogate admission requires both the stable primal-dual gap and KKT residual.
Frozen-coordinate constants do not inflate gap tolerances. Equality polishing
must pass an objective-nonincreasing check with a float64 roundoff margin and
fresh certificates for the original QP. Original tolerances remain unchanged.

The larger qualification exposed a case where conservative tolerance grouping
missed approximate fused blocks and one cap-clipped dual correction left a KKT
residual. A bounded QP-only fallback tries tighter equality proposals and repeated
affine-flow/cap projections. These proposals change neither public partition
extraction nor the original optimization problem: only a candidate passing the
same objective, full gap and KKT checks can return. The 20,000 ADMM iteration
limit remains; polishing rounds are bounded and reported separately from ADMM
iterations, including the original one-step equality corrections.
At sparse ADMM checkpoints, and before admitting an otherwise gap-qualified raw
or zero-step warm state, the fallback tries `fusion_tol`, `fusion_tol/16` and
`fusion_tol/256`. Each candidate must pass objective descent before at most 1,024
flow/cap projections. Its initial dual need not already satisfy the gap or KKT
gate: repairing that dual is the purpose of these projections. Every accepted
candidate still passes the original full gap and KKT gates. Singleton proposals
without internal flow freedom and exact nonqualifying projection fixed points
stop early. These smaller tolerances are
numerical proposal settings only; exported memberships still use `fusion_tol`.
An unsuccessful repair remains unqualified, including when bound-normal
allocation prevents its projections from converging.

Each equality proposal now owns its fixed group geometry, bound-active masks,
quadratic gradient, nonfused-edge values and a copy of the caps. Repeated flow
steps update only the dual against those prepared values. The context exists
only inside that proposal; another candidate or surrogate prepares new geometry.
The original QP arrays independently supply every certificate. Within one exact
checkpoint, refinement can reuse the first default-tolerance proposal and its
initial certificate rather than preparing and qualifying them twice. The reused
first projection consumes one of the same 1,024 allowed steps.

One pure-tensor ADMM step combines the node right-hand side, bounded rank-one
solve and edge update. It reuses the fixed-coordinate-safe weighted target for
that QP. A second compiled step applies prepared flow correction and cap
projection. Qualification checkpoints, residual balancing, numerical rho and
the production iteration limits are unchanged; no fixed iteration blocks or
edge-storage migration are introduced.

A qualified QP dual can initialize the next surrogate or backtrack within the
same start and lambda. This state is bound to the frozen graph and literal
lambda, projected onto the current edge caps, and rescaled using the new ADMM
`rho`. Changed curvature and targets receive fresh solves and certificates;
even a rejected likelihood trial supplies no inherited certificate. Storage is
bounded to the current dual state. Across lambdas, continuation remains
**primal-only**. Dual initialization can change the finite-iteration trajectory.
When a raw audit fails and a directional restart cannot be accepted, clear the
warm dual before the next surrogate. This prevents repeatedly admitting an
unchanged QP state whose tiny nonexact fusions fail the raw nonsmooth audit.

An independent raw audit uses observed one-sided derivatives, exact signed
contributions from nonfused edges and TV terms on fused edges. Both positive and
negative feasible-box directions are checked without witness restrictions. For

    min a' t + sum_{i<j} c_ij*abs(t_j-t_i), 0 <= t <= allowed,

a feasible dual yields the lower bound

    sum_i min(a_i + (D'q)_i, 0) * allowed_i.

Arithmetic margins account for absolute dual-row contributions before
cancellation. If the cut coefficients are divided by a positive scale `S`,
divide both the stationarity threshold and the constant roundoff unit by `S`.
Thus the original-unit bound `Gamma*(1 + absolute summands)` becomes
`Gamma*(1/S + absolute scaled summands)`, where `Gamma = 8*(N+2)*eps`.
Keeping a unit constant after normalization would incorrectly inflate the
original-unit error floor by `S` and prevent exact high-penalty stationary states
from qualifying. The absolute coefficient and dual terms also cover rounded
division and feasible-cap projection errors; cancellation protection remains.
Both lower-bound qualification and attained descent retain the original
absolute stationarity tolerance. This check covers arbitrary mutation subsets, including groups
noncontiguous in any pilot order. An attained descent direction is backtracked
against the original objective. An unresolved audit is not success. Raw
stationarity, inner QP qualification, scalar refit gaps and global optimality are
separate claims.

## Memberships, selection and labels

Partitions derive from raw fitted CCFs. Tolerance-connected runs whose total
diameter exceeds `fusion_tol` split conservatively into **exact-value groups**.
Exactly equal values always share a label, including duplicates inside such an
overwide run. For example, `[.3, .3, .300015, .300030, .300030]` at tolerance
`2e-5` gives `[0, 0, 1, 2, 2]`. The QP's first equality proposal uses this rule;
its additional tighter proposals are solver-only and require fresh certificates.
Exact-one and near-one values remain separate. Canonical groups are ordered by
their smallest mutation ID, not by adjacency in a fixed chain. The v2 singleton
fallback could split exact fusions; this correction can change K, scores and the
selected lambda.

Each arbitrary membership group is independently refitted inside the intersection
of its original bounds. No group is forced to one. Analytic or interval-search
scalar certificates qualify pilots and refits. Clipping-aware interval bounds
account for represented endpoints, midpoint rounding and likelihood sensitivity.
These are float64 numerical qualifications, not interval-arithmetic proofs. The
scalar engine identifies same-slope, single-candidate analytical groups before
building search grids, separates them from general groups, and restores
canonical group order. General groups refine only detected seed wells in a
bounded padded batch, preserving seed order and tie behavior. Interval lower
bounds, rather than golden-search initialization, qualify general scalar fits.

A cache retains only the last qualified membership refit, alongside the selected
best candidate. Reuse requires the same immutable model, policy and canonical
membership vector; tensor snapshots guard cached values. Returned refits are
independent copies. Qualified singleton pilots can supply the matching canonical
mutation's scalar result, including when other groups are noncontiguous.
Matching an array position in another model or policy is insufficient. Every
raw lambda candidate still receives its own qualification, and final publication
reconciles the selected raw memberships, refit loss, gaps and score.

`refits_computed` counts attempted new membership refits, including unresolved
ones; `refits_reused` counts cache hits. `singleton_pilots_reused` counts qualified
singleton lanes reused in new refits. Scalar work counters separately record
evaluated rows/proposals, kernel calls, analytical/general groups and packed-well
work. These counters describe work, not a measured speedup. The existing score is

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
and CUDA schema-v1/v2 receipts retain their original meaning. Before atomic
directory publication, the prepared receipt and TSVs are read back and checked
against their expected hashes, then files and directories are synchronized.

## Timing and memory scopes

`provenance.phase_seconds` records the following scopes. Device phase boundaries
synchronize CUDA so they measure completed work, not just queued operations.

| Field | Scope |
| --- | --- |
| `input_preparation_seconds` | API entry, source/input validation, canonical compilation and device admission |
| `device_upload_and_compile_seconds` | Model upload and initial compiler probe; later specializations are charged to the phase executing them |
| `pilot_seconds` | Scalar pilots and qualification for singleton reuse |
| `graph_build_seconds` | Frozen graph, pilot curvature and lambda reference |
| `path_seconds` | Planned-path work excluding membership refit time |
| `refit_seconds` | New refits, singleton reuse and cache validation/hits |
| `stage_integrity_seconds` | Outer numerical-stage entry/exit reconciliation and boundary bookkeeping |
| `final_device_qualification_seconds` | Independent selected-state reconciliation, including the nonzero-lambda raw audit |
| `device_export_seconds` | Posterior calls, graph/model hashing, final transfers and final snapshot checks |
| `output_preparation_seconds` | Host validation and prepared TSV tables, or in-memory result validation when no output directory is requested |

`provenance.numerical_stages` also carries all-start `qp_seconds` and
`audit_seconds`, their call counts, `qp_dual_warm_starts`, `qp_dual_warm_resets` and
`qp_polish_iterations`. `qp_admm_iterations` explicitly totals all attempted QPs
across all starts and penalties, including unresolved attempts; each start also
retains its original `inner_iterations`. The selected raw diagnostic's legacy
`inner_iterations` remains selected-start work, not the all-start total.
The two time fields are subsets of
`path_seconds`, not additional nonoverlapping phases. `numerical_wall_seconds`
covers the pilot/graph/path/refit stages and their boundary reconciliation; it
excludes final publication work. Detailed directional-audit diagnostics are
retained with the raw start records.

Peak CUDA allocated/reserved bytes, compilation statistics and numerical work
counters are captured **after** final device qualification, posterior evaluation,
hashing and transfers. `numerical_completed_utc` and
`through_device_export_seconds` mark that device-complete boundary. They do not
claim durable output publication.

The receipt's `elapsed_seconds` and `output_prepared_utc` end after host validation
and TSV preparation but before serializing the receipt itself, readback, fsync and
atomic publication. Their `elapsed_scope` states this exclusion. After successful
publication, the returned `FitResult.operation_metrics` contains
`publication_seconds`, `publication_completed_utc` and elapsed time through that
boundary, excluding return bookkeeping. These metrics are **return-only**: the
receipt cannot time its own durable completion. An external caller may preserve
them in a separate receipt. With no output directory, the return-only metrics
instead describe completed in-memory fitting.

## Validation boundaries

See [the current evidence](../VALIDATION_CUDA.md). Numerical CPU references and
CPU fullgraph tracing do not establish generated CUDA kernel correctness. Actual
GPU qualification must bind the exact source, environment, input, compiler and
allocated device, with eager/compiled numerical comparison and public output
validation. Small synthetic qualification does not establish full-cohort
accuracy, large-input memory behavior or a GPU speedup.

The [retrievable `371003f` evidence](../validation/371003f/README.md) contains the
original source-bound policy-v2 GPU receipts and test log. It does not qualify
this policy-v3 grouping and efficiency revision. Current CUDA acceptance must
use the current source; no speedup or larger-cohort claim follows from reducing
work in the implementation.

The retained historical chain files and their test-only API adapter support
regression comparisons. They are not a production backend. Legacy CPU launchers
must reject the CUDA revision before starting new work; old frozen workers and
historical evaluators remain source-bound.
