# Binary64 representability and separately reported QP gates

The development baseline is `42f44d409b97b7762f393e1a7476f520b019be4c`.
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

Qualify below64, then below256, then below512. Each size above 64 requires a
passed same-source, same-helper, same-family candidate receipt at the immediately
smaller size. An incomplete below256 candidate prevents below512 admission.
Mixed-support fixtures use the same 64/256/512 predecessor rule.

After coverage qualification, three repeated paired trials distinguish complete
inference time from incomplete recovery time. Order alternates between strategies.
Repeat zero includes cold shape specialization; subsequent repeats reuse the
process/compiler cache. Report each fit time, QP work, completeness, selected raw
and refit differences and score difference. Do not turn incomplete-search runtime
or one observed pair into a general speedup claim.

No CUDA results for this strategy are claimed by the local tests. The latest
four-cohort run includes the diagnostic changes on top of `42f44d4`, with its
exact source and patch frozen in the run receipts. It retains the production
`scalar_backtracking_v1` policy. The coordinate strategy has not been allocated
a research GPU or qualified on CUDA; the cohort launch does not enable it or
provide evidence for it.
