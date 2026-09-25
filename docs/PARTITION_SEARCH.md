# Separate CUDA partition estimator

This revision implements the search-repair foundation described in the
[CN-first investigation](../RESEARCH_CNFIRST_FAILURES.md). It is explicitly
opt-in while CUDA execution and held-out scientific validation are pending.
The observation model, uniform multiplicity prior, bounds, allocation score,
graph, solver tolerances and raw continuation rule are unchanged.

```bash
clipp1d fit --input-file tumor.tsv --outdir new-output --device cuda:0 --partition-search
```

The original raw estimator remains primary. Its three TSVs retain their original
meaning and baseline path winner. Enabling the additional estimator produces
schema `clipp1d.cuda.run.v4`, adding:

- `partition_mutation_clusters.tsv`: explicit memberships, partition CCF,
  closest-to-one label 0, and multiplicity conditional on that partition CCF.
- `partition_cluster_centers.tsv`: occupied sizes and partition centers.
- `run.json.partition_estimate`: separate candidate search, provenance,
  qualification, raw reference, and array identities.

`FitResult.partition_estimate` provides the same separate estimate in memory.
Old evaluators reading `mutation_clusters.tsv` continue to measure the baseline;
they must explicitly choose the new tables to evaluate the repair. The raw
reference penalty is not a fitted penalty parameter of a direct partition.

## Implementation and invariants

1. `cuda/partition.py:refit_labels` canonicalizes labels by first mutation,
   preserving exact memberships even if distinct groups have equal or nearby
   centers. Every group intersects its members' original bounds. Empty
   intersections fail. The existing scalar solver supplies fresh qualification.
   Feasible incumbent centers can improve the scalar incumbent but supply no
   inherited certificate. The vector-based raw refitter delegates to this API.
2. `cuda/solver.py:fit_lambda` optionally streams each qualified start to a
   callback. The raw-objective winner still supplies the next penalty's primal
   warm state. Partition score never changes continuation or path extensions.
3. `cuda/partition_search.py` scores distinct memberships from qualified starts,
   retaining one winning raw reference, one refit cache, and at most four
   within-penalty membership vectors. It does not retain a history of dense duals.
   The original baseline winner remains an explicit candidate and wins exact
   score/K/penalty ties. Failure of an optional proposal preserves that baseline
   and marks partition-search coverage incomplete.
4. `cuda/refinement.py` computes blocked mutation-by-center likelihood costs with
   the existing CUDA loss-only kernel. It evaluates move deltas in parallel,
   accepts one deterministic improving move, updates counts, and recomputes the
   full score. Empty destinations and infeasible centers are masked. Sequential
   acceptance prevents interacting moves from being accepted against stale sizes.
5. After each sweep, explicit memberships receive qualified scalar refits.
   Moves and refits alternate for at most four rounds, with at most 10,000 moves
   per round. Budget exhaustion is separate from raw-path completeness; it is
   not a convergence certificate. Costs refresh after centers change. No split
   or graph/score variant is silently included.

For a non-singleton source, the score change is
`2*(loss_destination-loss_source) + 1.4*log(source_size/(destination_size+1))`.
Deleting a source singleton additionally changes occupied K. Both branches are
tested against an independent complete-score calculation. Every accepted move
must decrease the full production score by more than the stated numerical margin.

## Qualification and provenance

Raw-derived candidates must retain the relationship between raw coordinates and
their extracted memberships, plus an independently audited raw solution and
qualified scalar refit. Direct partitions require their own membership identity,
original-bound feasibility, fresh scalar qualification, exact score, and proposal
history. A direct partition has `raw_qualified=false`, `raw_certificate=null`,
and `selected_lambda=null`; its separately named `raw_reference` retains the
actual parent state, penalty, and certificate.

Device snapshots cover candidate arrays, scalar certificate fields, and search
metadata. Export independently audits the parent raw state and both refits.
Host publication validates array identities, canonical labels, clonal designation,
score/tie selection, baseline preservation, source identity, and separate coverage.
Tests reject altered scalar bounds, changed candidate/search identities, and
attempts to attach a raw certificate or penalty to a direct winner.

## Exact-one grouping is a separate controlled variant

The baseline raw extractor retains its existing exact-one exception explicitly,
so the reassignment comparison does not silently include a grouping change.
Use `--generic-partition-grouping` together with `--partition-search` to remove
that exception for additional candidates. The primary baseline still uses its
original rule. This is cluster extraction, not a clonal fitting constraint.

Tests distinguish `[0.49999,0.5]` and `[0.99999,1]` under the legacy rule and
confirm symmetric treatment under generic grouping. On the saved selected raw
vectors in the 193-case discovery panel, the two extraction rules yield the
same memberships in all 193 cases. This is not an ablation of every path start.

## Evidence and reproduction

`benchmarks/replay_partition_search.py` accepts a portable, hash-bound bundle of
canonical input TSVs, published memberships/centers, independent expected
assignments, truth for evaluation, and matched-mutation masks. It invokes no new
fusion fits. `--device cpu` is explicit tensor-reference evidence;
`--device cuda:0` uses compiled production kernels. `--alternating-refit` adds
the qualified scalar and alternating-membership stages, reporting failures and
budget exhaustion. It never uses truth to generate proposals.

The CPU fixed-center replay reproduces all 193 archived memberships, with
maximum score discrepancy `5.82e-11`. Native ARI is 0.639734. On the identical
41,194 mutations used for the method comparison, ARI is **0.644738**, CCF MAE
**0.058322**, and sMF CCC **0.874458**. The corresponding PyClone-VI values are
0.649334, 0.054521, and 0.941077. Comparable mean ARI does not imply equivalent
sMF or overall accuracy. This remains discovery-panel evidence.

With the production four-round alternating budget, matched ARI increases to
**0.657245**, CCF MAE decreases to **0.053656**, and sMF CCC increases to
**0.883593**. Correct K rises to 122/193. Against the baseline, matched ARI
improves in 151 cases and worsens in one; CCF MAE improves in 150 and worsens
in eight. Underclustering remains 43/193, compared with 22/193 originally.
All 24 true-single-cluster cases retain one cluster. Forty cases exhaust the
four-round budget; 153 reach the membership fixed point. None has a scalar
refit failure. These are CPU component replays from frozen baseline memberships,
not new fusion fits or an evaluation of all-start candidate coverage. The
[validation record](../validation/partition-search-20260925/README.md) preserves
paired per-case metrics, source identity, and the predeclared held-out plan.

`benchmarks/qualify_partition_search_cuda.py` prepares three small engineering
fixtures and compares complete default-path fits with the feature off, on, and
with generic grouping. It checks the preserved raw outputs and graph, separate
publication schema, source/array identities, non-worse partition score, runtime
and memory. It requires an allocated CUDA device and compiled inference.
It does not establish full-cohort accuracy or A100/H100 equivalence.

The next scientific comparison must use the predeclared held-out panel, including
larger tumors, paired identical mutation populations, and per-case gains and
regressions. Merge/split, graph, and score changes remain later separately
attributable experiments. No discovery improvement is promoted to CUDA or
held-out qualification.
