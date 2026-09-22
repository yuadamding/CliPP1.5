> Historical chain reference (through 0.4.1). The current production model and
> output contract are documented in [CUDA_FRAMEWORK.md](CUDA_FRAMEWORK.md).
> References to production below describe the historical revision.

# Statistical formulation

CliPP1D estimates mutation CCFs and clusters mutations from one tumor sample using
an observed-count likelihood and adaptive fusion along a fixed chain ordered by
unpenalized marginal CCF estimates. No cluster is required to have CCF one.

For mutation i, alternate/reference counts are a_i/r_i, purity is rho, and supplied
CN states are (A_ik, B_ik, s_ik), with A ≥ B. Let

```text
mean_cn_i = sum_k s_ik (A_ik + B_ik) / sum_k s_ik
eta_i = rho / ((1-rho) normal_cn_i + rho mean_cn_i)
c_i = min(4, max_k A_ik)
p_im(phi) = clip(eta_i m phi, eps, 1-eps), eps = 1e-6
f_i(phi) = -log sum_{m=1}^{c_i} exp(a_i log p_im + r_i log(1-p_im)) / c_i
u_i = clip(min(1, (1-eps)/max(eta_i c_i, eps)), eps, 1)
```

The binomial coefficient is omitted, as in upstream, because it is constant in
phi and the partition. Mixed CN uses bulk mean total CN; it does not reconstruct
cellular mutation occupancy. Inferior or impossible likelihood components are
not silently removed from the original support.

Each pilot minimizes f_i on [eps,u_i], without a clonal constraint. Sort once by
(pilot, mutation ID). With adjacent gaps delta, use

```text
floor = max(1e-8, 0.1 median(positive gaps))
w_j = (1 / max(delta_j, floor)) / mean(1 / max(delta, floor))
```

All-zero gaps give unit weights; M=1 has no edges. The chain and its weights are
immutable throughout the fit. No monotonicity constraint is imposed on fitted x.

At each lambda, minimize the attained nonconvex objective

```text
F_lambda(x) = sum_j f_order[j](x_j) + lambda sum_j w_j |x_{j+1}-x_j|
eps <= x_j <= u_order[j].
```

At lambda zero, use the qualified marginal pilots without projection to one.
At positive lambda, production tries at most four distinct boxed initial primal
vectors: preceding penalty, pilot, clipped pooled and alternative-well estimates.
For each outer iterate, form one quadratic surrogate and solve the bounded
weighted-TV problem on the full chain. Backtracking recomputes the messages.
Acceptance requires observed-likelihood majorization, surrogate descent and
true-objective descent. No witness is fixed, profiled or designated. A finite
start bank and qualified stationary candidates do not prove a nonconvex global
minimum. Historical constrained witness solvers remain offline references only.

The raw path is `{0} union {lambda_ref * 2**k: k=-12,...,12}`. The reference uses
the weighted pooled pilot quadratic and cumulative adjusted gradients, including
the common-bound box normal. A zero reference uses 1e-3. At most three doubling
extensions are evaluated if the selected candidate remains at the upper boundary.
M=1 is solved directly at zero. Cluster counts need not be monotone along the path.

Blocks must be contiguous. Extraction uses both adjacent differences and total
block range at tolerance 2e-5, treating exact-one and near-one values uniformly.
Numerical fusion polish is performed inside the raw solver and audited; reporting
never snaps a near-one value to one.

For each block C, qualify the scalar minimum L_C on the intersection of its
original mutation boxes. Blocks are refitted independently:

```text
L_refit = sum_C L_C
log P_part = lgamma(K) - lgamma(M+K) + sum_C lgamma(size_C+1) + lgamma(K+1)
score = 2 L_refit + K log M - 1.4 log P_part
```

This score preserves the pinned CliPP2 arithmetic. It is not a new Bayesian
derivation for ordered partitions. The reported estimator is the independent
refit of the selected chain partition, not the raw penalized iterate. Posterior
multiplicity is conditional on that refit; exact ties choose the smaller integer.
Labels only permute blocks and never merge them.

Version 0.4.0 changes the feasible set, not the score arithmetic. Public labels
assign the block closest to CCF one to clonal label zero, then order the other
blocks by descending CCF and chain position. This post-fit designation (0.4.1)
does not constrain or change any center; sMF counts mutations outside label zero.
