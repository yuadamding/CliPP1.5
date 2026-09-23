# CUDA QP optimization evidence

This directory preserves exact bytes from the explicitly listed terminal attempts.
`FINAL_STUDY.json` reports each planned task, including failures, interruptions and unattempted work.
`SHA256.json` covers every published file except itself; `COPY_MAP.json` maps originals to copies.

Baseline: `daaf50ad5a2ae7301e9b54b9c31e87d87b24cfa6`; source `ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26`.
Current source: `726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88`.

Reproduce from the baseline commit plus each attempt's original `sealed/tracked-dirty.patch`.
The source reconstruction maps additionally retain exact changed production files and helper bytes,
including helpers absent from the tracked patch. Validate against the sealed inventories.
The original baseline and current full source trees/archives are not duplicated here.

Operational plans, scheduler receipts and all imported numerical receipts/inputs/path outputs are byte copies.
Transport payload wrappers, repeated observer stdout, compiler caches, binaries and profiler traces are excluded.
Original excluded-file hashes remain in FINAL_STUDY.json. No previous validation directory was changed.

An archived failure is not qualification. CPU tests, same-job QP throughput, instrumented attribution,
complete inference and cohort accuracy are separate scopes. See each exact receipt before using a result.

| Attempt | Job | Terminal / claims | Planned task results |
| --- | --- | --- | --- |
| qualification-a | 77334451 | passed / as_recorded | qualification-current: passed |
| attribution-b | 77334512 | passed / partially_invalidated | attribution-baseline: passed; attribution-current: passed |
| attribution-b2 | 77334577 | failed / as_recorded | attribution-baseline: failed; attribution-current: not_attempted |
| attribution-b3 | 77334593 | passed / as_recorded | attribution-baseline: passed; attribution-current: passed |
| mixed64-c | 77334538 | passed / as_recorded | mixed64-baseline: passed; mixed64-current: passed |
| below64-d | 77334573 | failed / as_recorded | below64-baseline: failed; below64-current: not_attempted |
| mixed256-e | 77334585 | failed / as_recorded | mixed256-current: failed |
| below64-f | 77334588 | failed / as_recorded | below64-current: failed |
| baseline256-g | 77334594 | failed / as_recorded | mixed256-baseline: failed |

| Mixed fixture | Nodes | Current complete path | Baseline parity |
| --- | --- | --- | --- |
| mixed_support | 64 | qualified | qualified |
| mixed_support | 256 | failed | not_requested_at_this_size |
| mixed_support | 512 | not_attempted | not_requested_at_this_size |
| below_one | 64 | failed | not_qualified |
| below_one | 256 | not_attempted | not_requested_at_this_size |
| below_one | 512 | not_attempted | not_requested_at_this_size |
