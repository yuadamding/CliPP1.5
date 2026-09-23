# Explicit complete-graph CPU evaluation

`fit_complete_graph_cpu.py` evaluates the current complete-graph implementation
using eager PyTorch CPU float64 tensors. It shares the production likelihood,
pilots, graph construction, scalar backtracking, planned search, qualification
gates, partition refit, score, output generation and output validator. It does
not use the historical chain solver. Fitting remains unconstrained; the refitted
cluster nearest CCF 1 receives label 0.

This is an explicitly selected benchmark adapter. The public CUDA API still
rejects CPU devices and never falls back to CPU. The shared receipt format is
named `clipp1d.cuda.run.v3`; that name alone does not establish CUDA execution.
CPU receipts explicitly record `backend=cpu`,
`execution_scope=cpu_numerical_reference`, `numerical_device=cpu`,
`compiled_inference=false`, and the adapter's SHA-256. CPU results do not qualify
CUDA compilation or GPU throughput.

## Frozen cohort workers

`run_complete_graph_cpu.py` operates a source-bound local pool. Its plan binds
the package, adapter, interpreter, numerical environment, manifest, input ZIP,
CPU assignment and disjoint CPU/GPU partition. The controller verifies the full
ZIP; workers verify each staged input, its original filename, parsed identities,
retained mutation population and truth hash. `validate_cpu_cohort.py` validates
the output tables, provenance, selected candidate and final-refit cohort metrics.

Each case gets a fresh process, one CPU affinity and one numerical thread.
The pool uses 25 workers with a 3 GiB estimated numerical workspace budget,
4 GiB RSS limit per worker and 12 GiB host headroom. The controller monitors
memory and original per-case wall limits. Resource and numerical failures retain
their receipts; unexpected setup, execution or validation failures stop refilling
and drain admitted work. Outputs and submission receipts are never overwritten.

The LSF handoff preserves every previously accepted job and completed import.
Only unsubmitted cases within the CPU memory estimate go to the local pool.
Larger cases remain assigned to LSF. The replacement GPU submitter holds the
original controller lock and retains the original GPU source, workers and
15-job cap. The immutable partition prevents duplicate case ownership.

The frozen controller entry point is:

```bash
/path/to/bound/python -B RUN/source/benchmarks/run_complete_graph_cpu.py control RUN
```

`RUN` must already contain a qualified, hash-bound plan, inventory, partition,
source snapshot and absent worker output/receipt paths. This command is a
single launch, not a resumable or repeatable submission shortcut. Read existing
process and terminal receipts before recovering any interrupted attempt.

## September 23, 2026 qualification

The local attempt `results/four-cohorts-local-cpu25-20260923-v1` binds package
source from commit `af06b7305b1928aeeabeac5ab5c5b454d2369c3f`, with source SHA-256
`40c5304905d3e86832e22832e768bc9fe61cb291351358680d9053e93dd9505c`.
No production numerical source was changed for CPU execution.

A 200-mutation cohort canary completed all 26 path candidates in 230.267 seconds
on one CPU. Against its existing GPU result, labels, final-refit CCFs and score
matched exactly; the maximum raw CCF difference was 5.313e-9. Selected penalties
and graph hashes differed slightly after floating-point differences in the
adaptive pilots, so this is an independent full-fit comparison, not a literal
fixed-graph/QP equivalence claim. It establishes one matched canary, not
cohort-wide CPU/GPU parity. The GPU comparison retains its original source
identity in the comparison receipt.

Evidence is under that attempt's `qualification/QUALIFIED.json`,
`qualification/COMPARISON.json` and `SOURCE_BINDING.json`. The full local test
suite passed 1,130 tests; four additional handoff tests cover partition ownership
and unsettled submissions. These local tests do not establish GPU qualification.
