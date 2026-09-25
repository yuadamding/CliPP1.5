# Partition-search revision: implementation and reference evidence

The optional partition estimator is implemented on top of the existing PyTorch
CUDA likelihood and qualified scalar refitter. The primary raw estimator,
continuation rule, graph, prior, bounds, score, and numerical tolerances remain
unchanged. [The contract](../../docs/PARTITION_SEARCH.md) specifies the API,
candidate families, separate output tables, and grouping ablation.

This record is **CPU reference validation**, not allocated-CUDA execution,
independent scientific validation, or promotion of the partition estimator to
the primary output. No active cohort payload was changed. The base commit is
`75f4eae89d2b3875dca702fcf47dfc07f86dc738`; the tested dirty-source fingerprint is
`4e5c1c9fdb20529a3c48b9b7c6536442b9b28b8cb2e4021a52fb4d6019d1f95b`.

## Numerical and publication checks

- Full repository suite: **1,244 passed**, 35.22 seconds. The single NVML warning
  reflects the unavailable local CUDA device. See [the test log](pytest.log).
- Ruff passed for `src`, `tests`, and `benchmarks`; `git diff --check` passed.
- Independent complete-score comparisons cover singleton deletion, occupied K,
  infeasible destinations, deterministic ties, and the stale-count batch trap.
- Explicit labels preserve noncontiguous memberships and equal-center distinct
  groups. Empty feasible intersections fail; scalar incumbents retain fresh
  qualification. Tests cover all-start callbacks and unchanged continuation.
- Device and host validators reject changed membership/scalar identities,
  inherited raw certificates, false search completeness, and direct candidates
  misrepresented as raw-derived. A roundoff stop with no accepted moves remains
  incomplete. Optional proposal failure retains the qualified baseline.

## Discovery replay on identical mutation populations

The portable bundle freezes the original 193 tumors, inputs, published
memberships/centers, independent diagnostic assignments, truth for evaluation,
and matched-mutation masks. Reassignment does not consult truth. All **193 fixed
partitions match** the independent diagnostic, with maximum score discrepancy
`5.82e-11`. All fixed-center sweeps reach the declared one-move stopping rule.

The matched population contains **41,194 mutations** in the same 193 tumors:

| Metric | Published baseline | Fixed-center moves | Alternating qualified refits | PyClone-VI reference |
|---|---:|---:|---:|---:|
| Mean ARI | 0.489326 | 0.644738 | 0.657245 | 0.649334 |
| Mean CCF MAE | 0.087695 | 0.058322 | 0.053656 | 0.054521 |
| sMF CCC | 0.782728 | 0.874458 | 0.883593 | 0.941077 |
| sMF MAE | 0.134759 | 0.070059 | 0.062786 | — |
| CNA-only pooled macro-F1 | 0.661492 | 0.798566 | 0.801183 | — |
| Correct K | 47/193 | 119/193 | 122/193 | — |
| Underclustered | 22/193 | 43/193 | 43/193 | — |
| Overclustered | 124/193 | 31/193 | 28/193 | — |

PyClone-VI values are the already-bound comparator from the original
[investigation](../../RESEARCH_CNFIRST_FAILURES.md), not a fresh run. Similar ARI
does not establish equivalent sMF accuracy. Native-population metrics on 41,712
mutations remain separate in [REFERENCE_REPLAY.json](REFERENCE_REPLAY.json).

Against baseline, alternating refits improve matched ARI in **151 cases**, worsen
it in **one**, and leave 41 unchanged. CCF MAE improves in **150**, worsens in
**eight**, and remains unchanged in 35. Absolute sMF error improves in 137 and
worsens in 14. All 24 true-single-cluster tumors remain unsplit in every stage.
[Paired case metrics](paired_case_metrics.tsv) retain every gain and regression;
delta columns are new minus published. The JSON also includes CNA-only pooled
micro/weighted/per-class F1, supports, and call coverage. CNA includes balanced
amplification; only CN 1/1 is excluded. sMF uses the occupied cluster nearest
CCF one as the designated clonal cluster, with canonical-ID ties; all other
memberships are subclonal. True-single-cluster ARI follows sklearn's convention.

The alternating replay starts from a freshly qualified refit of the original
memberships and uses the production **four-round** budget. **153 cases** reach
the membership stopping rule; **40 exhaust the round budget**. There are no
scalar refit failures. Budget exhaustion is retained as incomplete search.
This component replay does not rerun the fusion path or measure gains from
scoring all qualified starts. Existing-center moves cannot create a missing
center; the increased underclustering remains a substantive limitation.

Median CPU wall time for the replay loop is 0.157 seconds per tumor. This includes
input/model preparation, the independent fixed sweep, the alternating stage, and
metric calculation. It is **not added full-fit runtime or CUDA acceleration**.
Earlier exploratory receipts under `cpu_alternating_all` included an extra
initial sweep; they are preserved but are not the four-round evidence here.

Disabling exact-one separation changes **0/193 saved selected raw partitions**.
The generic grouping option remains a separate variant; this observation does
not cover every raw path start.

## Reproduction and remaining qualification

Use the `ml1` interpreter and the portable local input bundle:

```bash
PYTHONPATH=src python benchmarks/replay_partition_search.py \
  --bundle results/partition-search-discovery-20260925-v1/DISCOVERY_BUNDLE.json \
  --outdir results/partition-replay-new --device cpu --alternating-refit
```

The bundle SHA-256 is
`771317d9b80175e5f95a1025d19ed711921225c4d5d827674a79f7e0bbdb03cb`.
[MANIFEST.json](MANIFEST.json) binds the exact raw replay, test, input, capacity,
and environment evidence. Large input and per-move records remain external;
their absence from Git must not be treated as an empty or completed run.

`benchmarks/qualify_partition_search_cuda.py` prepares three paired full-path
engineering fixtures with the feature off, on, and with generic grouping. It
requires allocated CUDA and checks raw parity, graph identity, compilation,
separate publication, score preservation, runtime, and memory. The same discovery
replay can run with `--device cuda:0`. Both remain **not run** on this source.
One LSF device does not establish A100/H100 equivalence.

[HELDOUT_PLAN.json](HELDOUT_PLAN.json) predeclares 36 tumors excluded from the
193-case discovery panel: one per depth × true-K × mutation-count bin, with
purity/CNA balancing and a deterministic hashed tie break. It includes 236–800
mutations per tumor and binds input/truth hashes. Selection used no method
performance. Baseline, partition-search, and generic-grouping fits must use
identical inputs and mutation populations. This held-out panel is **not run**.

The read-only LSF capacity snapshot on **September 25, 2026, 2:36 PM CDT** found
11 running and four pending jobs, occupying the authorized 15-job cap. The user
then approved one additional temporary GPU. [CUDA_SUBMISSION.json](CUDA_SUBMISSION.json)
records job **77420786**, submitted at **3:05 PM CDT** with one CPU, 32 GB host
memory, and a 60-minute RUNLIMIT. At **3:06 PM CDT** it was Pending because its
GPU reservation was unavailable. Submission is not CUDA qualification. The
existing cohort owners and frozen payloads remain in place. Merge/split,
graph, and score variants remain later, separately attributable experiments.

The frozen qualification package includes the complete numerical source,
tests, both CUDA validation drivers, input bundle, source inventory, and dirty
tracked patch. Package revision 2 changes only the unlaunched runtime wrapper:
its GPU probe exits before fitting begins, avoiding a competing supervisor
context under LSF's exclusive-process allocation. The first staged archive is
preserved and was never submitted. Future status starts from
`results/CURRENT_PARTITION_SEARCH_QUALIFICATION.json`, also linked by the current
runs index. The additional slot authorization applies to this job only.
