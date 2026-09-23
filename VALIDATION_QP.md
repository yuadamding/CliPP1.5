# CUDA QP preparation and compilation validation

The optimized QP implementation preserves the qualified results on the existing
complete-path fixtures and the new heterogeneous 64-node mixture fixture. Paired
CUDA measurements show lower QP latency and fewer kernel dispatches. The harder
below-one 64-node fixture remains incomplete on both sources, and the new
256-node mixture fixture also failed complete-path admission. These failures
remain part of the evidence; this is not qualification of simulation cohorts.

The publication index is [validation/cuda-qp-v3/README.md](validation/cuda-qp-v3/README.md).
The retrievable operational evidence is under
[validation/cuda-qp-v3](validation/cuda-qp-v3), with the
scope and staged admission rules in [STUDY_PLAN.json](validation/cuda-qp-v3/study/STUDY_PLAN.json).

## Source and unchanged scientific contract

| Identity | Value |
| --- | --- |
| Baseline commit | `daaf50ad5a2ae7301e9b54b9c31e87d87b24cfa6` |
| Baseline production SHA-256 | `ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26` |
| Qualified current production SHA-256 | `726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88` |
| Policy | `clipp1d_complete_cuda_unconstrained_v3` |

The complete dense graph, observed likelihood and multiplicity support, original
boxes, adaptive weights, score, grouping policy and numerical gates are unchanged.
Fitting remains unconstrained; postfit labeling assigns cluster 0 to the refitted
cluster nearest CCF 1. The 20,000-iteration per-QP ADMM budget, 150 outer-iteration
budget, rho initialization and residual balancing are unchanged. There is no CPU
fallback or weakened gate when CUDA compilation or numerical qualification fails.

The implementation prepares proposal-local flow geometry once and reuses it in
pure-tensor repair steps. It compiles the full ADMM update and flow-repair update
through the existing bounded compilation banks. A duplicate equality proposal and
its certificate are reused only within the same checkpoint, before any change to
the surrogate, iterate or rho; original objective, full-gap and KKT admission are
retained. There are no fixed iteration blocks or cross-QP geometry caches.

New `qp_admm_iterations` counters include every attempted start and unresolved
penalty. Legacy `inner_iterations` retains its selected-start meaning; polishing
work is reported separately. See [QP benchmark methodology](docs/QP_BENCHMARKS.md)
and [CUDA framework](docs/CUDA_FRAMEWORK.md).

## Local checks

The final local suite passed **877 tests**, with no failures or skips, in 28.84 s.
Ruff, compileall and `git diff --check` passed. The single NVML warning does not
establish local CUDA availability. These are CPU reference, tracing, contract and
static checks; allocated CUDA evidence is separate.
[Local source-bound receipt](validation/cuda-qp-v3/local/LOCAL_VALIDATION.json).

## Existing allocated-CUDA qualification: A

LSF job **77334451** passed the complete existing qualifier on NVIDIA L40,
float64, PyTorch 2.9.1/CUDA 12.8. It covers eager/compiled likelihood and clipping,
reduced-output kernels, QP cold/warm certificates, the 256-node resource QP,
complete default paths at 16/64/256 nodes, controlled fixed-graph penalties, and
public fitting, final export and output publication.
[Qualification receipt](validation/cuda-qp-v3/attempts/qualification-a/imported/results/qualification-current.json).

Against the archived baseline, the three scaling fixtures have exactly equal
pilots, graphs, penalty paths, selected raw/refitted CCFs, labels, scores and
objectives. Every planned path is complete. Their total ADMM work is unchanged;
duplicate polishing work decreases:

| Nodes | ADMM iterations, each source | Baseline polish steps | Current polish steps |
| --- | ---: | ---: | ---: |
| 16 | 6,048 | 381 | 353 |
| 64 | 20,128 | 4,835 | 4,354 |
| 256 | 25,904 | 4,260 | 3,860 |

These historical full paths ran in separate allocations. Their elapsed times are
not a controlled full-fit speedup comparison.
[Scientific and work reconciliation](validation/cuda-qp-v3/study/PAIRED_RECONCILIATION.json).

## Paired QP measurements: B3

LSF job **77334593** ran the two frozen sources serially in one exclusive L40
allocation on identical literal QP inputs. Each source/fixture has one warmup,
five uninstrumented latency samples, and two separate profiler calls. All **48**
QP returns passed independent original-problem CUDA certificates. Primal solutions
and objective values are exactly equal across all recorded samples and sources.

| QP fixture | Baseline median, ms | Current median, ms | Baseline kernel dispatches | Current kernel dispatches |
| --- | ---: | ---: | ---: | ---: |
| resource64 | 25.521 | 17.651 | 2,075 | 1,385 |
| resource256 | 309.467 | 215.913 | 18,825 | 12,494 |
| mixed64 | 115.999 | 87.565 | 8,212 | 5,960 |

Separate profiler attribution below reports actual kernel/copy duration in milliseconds,
baseline → current. It excludes CUDA user-annotation mirrors and recursively
attached timing. Stage ownership is disjoint; unattributed work remains explicit
in the receipts. These instrumented durations are not latency measurements.

| Stage | resource64 | resource256 | mixed64 |
| --- | ---: | ---: | ---: |
| ADMM update/residual balance | 3.103 → 2.400 | 55.701 → 48.781 | 12.660 → 9.827 |
| Equality proposal/preparation | 0.854 → 0.894 | 10.322 → 10.624 | 3.403 → 4.113 |
| Dual-flow repair | 0.385 → 0.006 | 3.879 → 0.107 | 1.542 → 0.096 |
| Certificate | 0.646 → 0.463 | 60.517 → 40.883 | 2.768 → 2.773 |

The baseline flow range includes rebuilding geometry; current one-time geometry
preparation is charged to the equality range. Interpret those two categories
together. The table does not imply that every stage or every full fit speeds up.
[Paired reconciliation and complete samples](validation/cuda-qp-v3/study/PAIRED_RECONCILIATION.json).

The earlier B profiler measurements were invalidated because the helper counted
CUDA annotation mirrors and recursively attached durations as work. Original
receipts and independently recorded numerical/latency evidence are retained, but
the invalid profiler results are not used above. B2 failed closed on ambiguous
CPU profiler correlation ownership before a valid paired measurement completed.
[B invalidation](validation/cuda-qp-v3/attempts/attribution-b/operational/MEASUREMENT_INVALIDATION.json)
and [B2 failure](validation/cuda-qp-v3/attempts/attribution-b2/imported/receipts/terminal.json).

## Heterogeneous mixture qualification: C

LSF job **77334538** passed both complete 64-node `mixed_support` paths and four
controlled shared-pilot/shared-graph literal-penalty replays. The fixture includes
distinct counts, depths and slopes, support sizes 2–4, clipping and competing
likelihood wells. Independent pilots, graph weights and selected caps were
exactly equal. Selected lambda was `10.003106095025691`; raw/refitted CCFs,
centers, labels, multiplicity calls, score and raw objective were exactly equal.
Both sources passed final device qualification and output publication.
[Mixed64 reconciliation](validation/cuda-qp-v3/study/MIXED64_RECONCILIATION.json).

Single full-workflow observations include journaling and lazy compilation. They
provide coverage and scientific parity, not repeated full-fit throughput evidence.
The `mixed64` QP timing fixture in B3 is a separate, bounded quadratic problem.

## Retained limitations: D, E, F and G

The `below_one` 64-node fixture failed complete-path admission on both baseline D
(job **77334573**) and current F (job **77334588**). Both produced qualified
winner raw/refit candidates at all 26 penalties, but only 24 penalties resolved
every planned start. The same two pooled starts failed the original QP gap gate:

| Lambda | Baseline final QP gap | Current final QP gap | Allowed gap, approximately |
| --- | ---: | ---: | ---: |
| 276.7339217595795 | 0.16512749270769345 | 0.16512749270921298 | 3.79e-9 |
| 553.4678435191591 | 0.07984450893451846 | 0.07984450893457537 | 3.95e-9 |

Both residuals individually pass the KKT threshold, which does not override the
failed gap test. Across all 99 starts, status, objective histories, ADMM counts,
outer counts, QP calls and backtracking counts match exactly. Every per-penalty
winner objective, score, cluster count and completion flag also matches exactly;
selected pilots/graph/raw/refitted CCFs/labels are unchanged. Total ADMM work is
127,440 on each source, spread across multiple QPs; total polish work decreases
from 73,680 to 71,490. The incomplete result was not finalized or published.

This reproduces an existing convergence limitation. Failed surrogate tensors were
not exported, so the receipts do not establish a precise final-gap decomposition.
The `below_one` 256/512-node stages remain blocked by their 64-node prerequisite.
[D diagnosis](validation/cuda-qp-v3/attempts/below64-d/operational/DIAGNOSIS.json)
and [D/F comparison](validation/cuda-qp-v3/attempts/below64-f/operational/FDIAGNOSIS.json).

The 256-node `mixed_support` runs E (current, job **77334585**) and G (baseline,
job **77334594**) also returned incomplete paths and failed final
export/publication admission. Both have the same 11 unresolved starts among 99,
across the same ten penalties. All 26 winners and refits qualify, but this does
not resolve the failed starts. Selected pilots, graph, raw/refitted CCFs, labels,
centers, objective, score and lambda match exactly. This reproduces a baseline
convergence limitation; the 512-node stage remains blocked.

On E, failed starts consumed 135.50 of 157.38 seconds of QP time and 163,619 of
165,849 polishing steps. This is a concrete next diagnostic target, not evidence
for weakening the gap or KKT gates. Failed surrogate tensors were not retained,
so the exact terminal gap decomposition needs a separately instrumented replay.
[E failure receipt](validation/cuda-qp-v3/attempts/mixed256-e/imported/results/mixed256-current.json).
[E/G comparison](validation/cuda-qp-v3/attempts/baseline256-g/operational/DIAGNOSIS.json).

No cohort rerun, tumor accuracy claim, general 512-node qualification or changed
objective/scientific policy follows from this study. The passed QP and complete
path cases, incomplete stress cases and invalidated measurements remain separate.
