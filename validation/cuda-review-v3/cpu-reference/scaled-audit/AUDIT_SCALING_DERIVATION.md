# Normalized directional-cut arithmetic

This is a CPU reference diagnosis, not CUDA qualification. The driver was saved
before execution; its hash, full source identity and exact input identity are in
`source-and-plan.json`. All 26 default penalties completed with all starts
qualified in 49.59 seconds. `terminal.json` records an unchanged source snapshot.

For the represented original cut coefficients, write

    F(t) = a^T t + sum_{i<j} c_ij |t_j-t_i|,  0 <= t <= allowed.

For a feasible skew dual q, `L = sum_i allowed_i min(a_i+(D'q)_i,0)`
is a lower bound. With `Gamma=8*(N+2)*eps`, the existing floating-point bound uses
the absolute summands before cancellation, not just the computed residual:

    Gamma * (1 + sum_i allowed_i (|a_i| + sum_j |q_ij|)).

After normalization by positive represented scale S, the same original-unit
error floor is

    Gamma * (1/S + sum_i allowed_i (|a_i/S| + sum_j |q_ij/S|)).

Retaining 1 instead of 1/S creates an artificial original-unit floor Gamma*S.
In the preserved H failure at N=256 and lambda=154555.71518032427, S is about
53251797.3. Even the constant term Gamma alone then exceeds the normalized
stationarity threshold. Iterating the cut cannot remove that error floor.

The implementation therefore passes `roundoff_unit=1/S` and the unchanged
`stationarity_tol/S`. Standalone, unnormalized callers retain `roundoff_unit=1`.
The same conversion applies to the independently computed primal-descent
margin, whose absolute terms are `|a_i*t_i|` and `c_ij*|t_j-t_i|`.

Normalization does not remove cancellation safeguards. Rounded division of
unary coefficients contributes a relative-error term proportional to |a/S|.
To interpret a dual feasible for rounded normalized capacities against exact
normalized capacities, clip it to the latter. Only a nearly cap-active edge can
change; its discrepancy is proportional to eps times that dual magnitude.
Allowed-node divergence errors are therefore bounded by the retained absolute
dual-row sums. Rounded primal capacities contribute proportionally to the
absolute edge terms. These terms, together with row/final reductions, are
covered by the existing conservative Gamma factor; threshold division error is
also below its scaled constant allowance. No gate, objective coefficient,
stationarity tolerance, solver iteration budget or grouping rule is enlarged.

Independent regressions enumerate small binary cuts using Decimal evaluation of
the original represented coefficients, exercise non-power-of-two normalization,
both direction signs and box masks, and preserve conservative refusal for large
canceling dual flows. The exact high-lambda N256 all-fused state now certifies
both signed cuts at iteration zero. The complete CPU path is additional causal
evidence; actual CUDA acceptance remains a separate fresh source-bound run.
