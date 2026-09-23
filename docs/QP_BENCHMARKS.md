# QP attribution and heterogeneous qualification

The production solver keeps the dense complete graph, observed likelihood,
adaptive weights, original boxes, raw-primary estimator, score and numerical
tolerances. This performance pass changes repeated QP work, not the statistical
problem. No clonal fitting constraint is introduced; public label zero still
designates the refitted cluster closest to one.

## QP measurements

`benchmarks/attribute_qp_cuda.py` runs the same literal QPs in separate baseline
and current-source processes on allocated CUDA. It distinguishes:

- Repeated synchronized wall latency without profiling or attribution hooks.
- A separate profiler pass measuring kernel dispatch counts.
- A separate attributed pass covering ADMM updates, equality proposals and
  preparation, dual-flow repair, and certificate/objective work.

Named host ranges do not overlap. Device kernel duration is reported separately
from host range duration, instrumented wall time and unattributed overhead.
Attribution schema v2 counts raw Kineto CUDA work and excludes mirrored user
annotation ranges. Linked CPU correlation IDs assign work to disjoint host stages;
duplicate IDs are accepted only when every possible owner agrees on the same
stage instance or is outside all stages. Conflicting ownership fails explicitly.
Uncorrelated work and discarded annotation counts remain visible. Recursive
PyTorch `device_time_total` is not used because it can include annotation spans.
These measurements must not be added together. Instrumented timing is not an
end-to-end speedup estimate. Compilation warmups are retained separately from
warmed latency samples, and no synchronization timers enter production loops.
The baseline flow range includes rebuilding geometry on every repair; the new
one-time preparation is charged to the equality-proposal range. Interpret both
ranges together when comparing costs. The literal host-prepared resource QPs
also differ in floating-point construction from the older CUDA-generated
resource probe; only identical input hashes support a matched timing ratio.

Every sample, including warmups and profiler runs, receives a fresh certificate
from the original QP and an input-integrity check. Paired comparison requires
identical input hashes, policy, controls, helper source and hardware/environment.
State and objective differences are checked against the original gap bounds.
All-start production work is reported as `qp_admm_iterations`; selected-start
`inner_iterations` remains separately available.

## Larger mixed-support fixtures

`benchmarks/qualify_mixed_cuda.py` accepts one of `mixed_support` or `below_one`
and one size, 64, 256 or 512. Canonical TSV input retains heterogeneous depths,
slopes, integer supports two through four, competing likelihood wells and
original clipping-sensitive boxes. Every row is distinct. The below-one family
has every original upper bound strictly below one.

The default complete production path is mandatory. Incremental artifacts retain
the planned penalties, active candidate and returned candidate diagnostics,
including failed starts. Per-start diagnostics are published when that candidate
returns; interruption within a candidate leaves its unfinished work explicitly
unqualified rather than recording every completed inner start. A pass additionally
requires independent final raw qualification, qualified membership refitting,
export and durable public-output validation. Larger sizes require a passed
same-source, same-family predecessor; a scheduler completion is insufficient.

`full` mode qualifies the current complete path without a baseline comparison.
`reference` additionally saves qualified baseline pilots, graph weights and
literal penalties. `paired` runs the new source's full path and independently
replays those exact fixed problems. Independently generated adaptive paths and
fixed-problem parity remain different claims.

These are synthetic execution and numerical checks. They do not establish
cohort accuracy, arbitrary-size capacity or global nonconvex optimality. Dense
storage remains quadratic; packed or tiled storage needs separate evidence.
