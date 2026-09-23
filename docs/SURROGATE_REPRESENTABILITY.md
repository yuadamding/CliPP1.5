# Binary64 representability and separately reported QP gates

The reviewed development baseline is `75c102f9dd1dafbe54010f1b96c1dfd551220341`.
Its fixed below256 quadratic surrogate remains unresolved. None of the work in
this document changes that saved problem, raises a gate, or claims complete
below256/below512 qualification without fresh allocated-CUDA evidence.

## Individual diagnostic gates

Positive-penalty raw diagnostics now distinguish:

- `inner_gap_pass`: finite, nonnegative gap and scale, with gap at most
  `inner_gap_allowed = inner_atol + inner_rtol * inner_gap_scale`.
- `inner_kkt_pass`: finite, nonnegative KKT residual at most `inner_kkt_allowed`.
- `inner_qp_qualified`: the actual combined QP solver result, including its
  validity and feasibility requirements. Two individually passing values do
  not override this result.

The old `inner_gap_qualified` and `inner_kkt_qualified` names remain aliases for
their corresponding individual test. Values, scale and thresholds remain in
the receipt. The allowance is computed with the same device arithmetic and
operation order as admission. These are reporting changes; admission is not
recomputed from the new flags.

`inner_certificate_present` distinguishes a measurement from an absent QP.
Before any surrogate is solved, values and gap allowance are null and no QP gate
is reported as passed. Lambda zero is qualified by its existing scalar-pilot
gates: its scope is `not_applicable_separable_lambda_zero`, with no invented
inner-QP certificate. Inner-QP qualification and the final observed-likelihood
raw audit remain separate claims.

## Permanent negative regression

`tests/cuda/test_literal_binary64_obstruction.py` reexecutes the committed
rational proof against the hash-checked below256 capture. It compares the full
result to the committed proof, including all exact-rational hashes. It also
locks the unchanged gap and KKT constants for this contract.

The proof reproduces the minimum incident edge-gap requirement of
`5.572854941897914e-6`, against a global allowance at most
`2.3085340981299313e-8`: a ratio greater than 241. The other adjacent binary64
coordinate is worse. The bound covers the entire mathematically admissible
region, including its preserved incident-edge signs and interior status.

This is a proof about the literal QP and exact mathematical gates with a
binary64 primal. It does not certify every CUDA reduction order, another outer
trajectory or the original observed-likelihood optimum. The saved arrays,
failed replay and historical source identities remain unchanged.

The new CUDA research driver additionally replays this literal QP from its
original initialization with the unchanged production QP solver. Both eager
and compiled certificates must retain the failed classification. An unexpected
pass stops qualification and requires an exact mathematical re-audit; it is not
silently treated as successful recovery of the represented-primal obstruction.
The driver saves exact problem/initialization and returned primal/dual NPZ arrays,
readback checks and hashes, plus both certificates, **before** applying this
expected-failure assertion. Contradictory measurements remain in the failed
receipt. Nonpositive wall budgets are rejected before creating run directories
or installing handlers; previous SIGTERM/SIGALRM handlers are restored on exit.

## Explicit experimental outer strategy

The production default remains `scalar_backtracking_v1`.
`coordinate_backtracking_v1` is an internal, explicitly selected research
strategy. It is not enabled by the public CLI or API. Its identifier is recorded
in every executed positive-penalty start and in the research output provenance.

Both strategies begin with the same repaired one-sided posterior curvature,
clamped below at 1. They use the same original likelihood, gradient, fixed
complete graph, weights, boxes, pilots, multistart construction and score. No
upper cap is imposed on curvature.

For a rejected trial, define each local quadratic prediction as

`major_i = loss_i + gradient_i * step_i + 0.5 * h_i * step_i**2`.

The experimental strategy doubles inflation only on coordinates whose trial
loss exceeds that prediction beyond a conservative arithmetic margin. The
margin includes absolute loss, linear and quadratic terms, so cancellation is
not mistaken for a reliable local violation. Nonfinite rows are flagged. If no
row explains an aggregate rejection, the strategy retains the existing global
doubling step. After an accepted trial it halves each inflation, bounded below
by 1; an observed-descent restart resets inflation to 1.

These row checks select the next proposal only. Every candidate must still pass
its original QP gap/KKT gates, aggregate likelihood majorization, surrogate
change and observed-objective checks. Final raw auditing, planned-start coverage,
independent membership refitting and output publication remain required. The
one-sided bound-curvature repair is retained in both strategies.

The hypothesis is that a difficult row need not multiply an unrelated already
stiff curvature. This may avoid creating some unattainable future surrogates.
It is not a representability detector or a guarantee that every new surrogate
can qualify. Any resulting QP differs explicitly from the saved literal QP;
that difference must never be presented as recovery of the old problem.

## CUDA qualification and repeated timing plan

`benchmarks/qualify_surrogate_cuda.py` compares both declared strategies on one
fixture and size per allocated job. It reuses the existing independent full-path
qualifier, including every planned start, final raw audit, refit and publication.
The pilot and weight hashes and initial penalty plans must match between the
strategies. Selection and allowed adaptive extensions may differ and are reported.
Version 2 receipts report both selected lambdas and exact-literal equality. The
selected `raw_objective_difference` is null when lambdas differ; the separately
selected objective difference is explicitly labeled and never an optimizer
quality measure. Candidate events retain raw objective, qualification, full-start
coverage and work at every literal lambda. Common-lambda comparisons use exact
equality (no nearest-penalty matching); objectives are compared only when both
returned candidates qualify. Unmatched penalties and unresolved starts remain
separate coverage evidence. Shared lambda/graph means the same fixed objective,
not necessarily the same continuation or optimization trajectory.

Research starts additionally retain bounded observations: the first 128 and last
16 outer trials per start, with explicit omitted counts, rejected-row masks,
scalar/vector inflation, curvature range and each original acceptance slack.
Every unresolved QP retains exact h/target/boxes/caps/initialization/returned
states and its original certificate in hash-bound artifacts. The production
default supplies no observer. Trace IDs are reserved on entry, independently
of returned calls. Attempted, returned, qualified and raised counts remain
separate even when a caller catches a start exception and continues.
Instrumented times include synchronization,
copies and artifact writes and do not establish uninstrumented throughput.

Qualify below64, then below256, then below512. Each size above 64 requires a
passed same-source, same-helper, same-family candidate receipt at the immediately
smaller size. An incomplete below256 candidate prevents below512 admission.
Mixed-support fixtures use the same 64/256/512 predecessor rule.

After coverage qualification, `benchmarks/time_surrogate_cuda.py` runs a separate
study without the observer or candidate tracer. One cold pair is retained
separately from three warm pairs with alternating order. Each fresh fit reuses
process/compiler caches, not another fit's solution. Synchronized inference
timing includes normal audits/refits, while input/upload, additional graph checks,
comparisons, export/publication and file writes remain outside it. Report each
fit time, QP work, completeness, selected raw and refit differences and scores.
Compute speed ratios only for complete, published warm pairs.

No CUDA results for this strategy are claimed by the local tests. The running
four-cohort attempt retains its frozen `a0e1406` source and production
`scalar_backtracking_v1` policy; it does not enable or qualify this experiment.

The completed L40 experiment qualified all six coordinate-backtracking stages:
below-one and mixed-support at 64, 256 and 512 mutations, each with 99 starts
across 26 penalties. Scalar below256 remained incomplete. All six selected
labels, raw/refitted CCFs, scores and lambdas matched; the original literal QP
still failed both eager and compiled KKT checks. The
[retrievable evidence bundle](../validation/cuda-surrogate-review-v4/README.md)
includes the original source, full artifacts and exact job receipts, plus actual
cohort staging adoption evidence. No production promotion is automatic.

The trace-attempt and separate timing-driver updates passed 1,126 local tests,
including the rational proof, plus lint. The supplied review ZIP's Git-verified
checks reproduced the conditional collision before its regression was fixed.
The uninstrumented timing study has its own frozen source and receipts; the
instrumented qualification results establish coverage, not throughput.
