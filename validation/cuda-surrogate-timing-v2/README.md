# Additional uninstrumented timing evidence

Snapshot: **September 23, 2026, 02:28 PM CDT**. Four of six stages have completed;
below512 and mixed512 remain with the original serial controller. This snapshot
extends [the first-stage evidence](../cuda-surrogate-timing-v1/README.md).

| Fixture | Complete warm fits, scalar / coordinate | Scalar median | Coordinate median | Median paired ratio |
|---|---:|---:|---:|---:|
| below64 (prior archive) | 3 / 3 | 15.915 s | 15.702 s | 1.0136× |
| mixed64 | 3 / 3 | 83.798 s | 80.716 s | 1.0386× |
| below256 | 0 / 3 | Incomplete | 29.402 s | Not comparable |
| mixed256 | 3 / 3 | 56.052 s | 54.737 s | 1.0261× |

Each stage retains the initial pair separately and alternates the three warm
pairs. The below256 coordinate median uses its three complete warm fits; it is
not a paired speed statistic. Every scalar below256 warm search is incomplete.
No speed ratio is assigned to those pairs. The other modest observed differences
do not establish a general speedup or statistical significance.

The numerical-fit timer includes the normal audits and refits but excludes input,
upload, extra summaries, export, publication and file writes. No diagnostic
observer or candidate tracer is installed. Every new ZIP contains its original
frozen source, imported receipts, public outputs and scheduler evidence. These
are the same source-bound runs started before this review, not new executions
attributed to later source. Existing failed timing-canary evidence remains in
the prior archive.

```bash
python validation/cuda-surrogate-timing-v2/verify.py
```

This verifies integrity and timing arithmetic without running CUDA inference.
The live owner remains under `results/cuda-surrogate-timing-20260923-v1`.
Production scalar backtracking and frozen cohort runs are unchanged.
