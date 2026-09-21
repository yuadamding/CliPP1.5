# Statistical formulation

CliPP1D estimates mutation CCFs and clusters mutations from one tumor sample using
an observed-count likelihood and adaptive fusion along a fixed chain ordered by
unpenalized marginal CCF estimates. At least one occupied cluster is constrained
to CCF exactly one.

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
eps <= x_j <= u_order[j], with at least one x_j exactly 1.
```

The occupied-clonal constraint is a union of branches, each fixing one
originally eligible witness to one. At lambda zero, profile the cost
f_j(1)-min f_j directly. At positive lambda, production tries at most four
clonal-feasible initial primal vectors: preceding penalty, pilot, clipped pooled
and alternative-well estimates. When a vector has no exact-one coordinate, impose
the eligible witness with smallest observed cost increase; deduplicate the
resulting primals. This initial witness is never permanent.

For each outer iterate, form one quadratic surrogate and minimize it over the
entire union using shared prefix/suffix values at one. Pass the original boxes,
reconstruct the selected witness once, and check its gap/KKT and predicted value.
Backtracking recomputes the shared messages. Acceptance requires observed
likelihood majorization, surrogate descent and true-objective descent. A finite
start bank and numerically qualified stationary candidates do not prove a
nonconvex global minimum. The former independent nonlinear witness enumeration,
including its safe scalar-lower-bound screening, is an offline validation
reference; the new search need not attain the same stationary points.

The raw path is `{0} union {lambda_ref * 2**k: k=-12,...,12}`. The reference uses
the weighted pooled pilot quadratic and cumulative adjusted gradients, including
the common-bound box normal. A zero reference uses 1e-3. At most three doubling
extensions are evaluated if the selected candidate remains at the upper boundary.
M=1 is solved directly at zero. Cluster counts need not be monotone along the path.

Blocks must be contiguous. Extraction uses both adjacent differences and total
block range at tolerance 2e-5, separating exact-one from merely near-one values.
Numerical fusion polish is performed inside the raw solver and audited; reporting
never snaps a near-one value to one.

For each block C, globally qualify the scalar minimum L_C and, when its original
domain contains one, compute L_C(1). The refit profiles the designated clonal block:

```text
L_refit = sum_C L_C + min_eligible_C (L_C(1) - L_C)
log P_part = lgamma(K) - lgamma(M+K) + sum_C lgamma(size_C+1) + lgamma(K+1)
score = 2 L_refit + K log M - 1.4 log P_part
```

This score preserves the pinned CliPP2 arithmetic. It is not a new Bayesian
derivation for ordered partitions. The reported estimator is the constrained
refit of the selected chain partition, not the raw penalized iterate. Posterior
multiplicity is conditional on that refit; exact ties choose the smaller integer.
Labels only permute blocks and never merge them.
