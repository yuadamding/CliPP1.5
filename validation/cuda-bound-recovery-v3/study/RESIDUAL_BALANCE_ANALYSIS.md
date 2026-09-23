# Residual-balancing analysis of the 13 captured QPs

This is read-only CPU array analysis of the exact failed CUDA surrogate states.
No QP iterations, full fits, recovery, or GPU qualification were performed.
The reproducible script is `residual_balance_analysis.py`. Its hash is recorded
in `residual-balance-diagnosis.json`, SHA-256
`06c4b426c77c75471b3fe413953aa3e06d1a45f1a32aced0e98573b233e1577b`.
That JSON binds both capture receipts and every constituent capture record.

## What the saved states establish

The source starts each surrogate with `base_rho = median(h)` (the lower median
for even N), then compares:

- `r = sqrt(0.5 * sum((D*x-z)^2))`, the norm over unique undirected edges;
- `s = rho * norm(D^T*(z-z_previous))`, a node-gradient norm.

These have different objective-scale behavior. The terminal captures retain
`x,z,v,q,rho,h`, but **not `z_previous`**. Therefore the table measures terminal
rho and primal feasibility only. It does not reconstruct any historical
balancing decision, exact final dual residual, or cause-and-effect trajectory.
The recorded terminal `q` is not assumed equal to `rho*v`, because rho/v may
have been rebalanced after the last clipped-dual checkpoint.

| Fixture/path/start | terminal rho / median(h) | N*rho / median(h) | terminal primal edge norm |
|---|---:|---:|---:|
| below64 / 1 / 1 | 1.525879e-5 | 0.0009765625 | 6.367250e-6 |
| below64 / 2 / 2 | 3.051758e-5 | 0.001953125 | 1.759611e-6 |
| mixed256 / 1 / 1 | 6.103516e-5 | 0.015625 | 1.310984e-7 |
| mixed256 / 2 / 2 | 0.0009765625 | 0.25 | 0.0001370819 |
| mixed256 / 3 / 2 | 3.051758e-5 | 0.0078125 | 2.038547e-6 |
| mixed256 / 4 / 2 | 3.051758e-5 | 0.0078125 | 3.391913e-6 |
| mixed256 / 5 / 2 | 0.00048828125 | 0.125 | 0.0007722566 |
| mixed256 / 6 / 2 | 0.000244140625 | 0.0625 | 0.0007086891 |
| mixed256 / 7 / 2 | 0.000244140625 | 0.0625 | 0.0005778535 |
| mixed256 / 8 / 2 | 0.000244140625 | 0.0625 | 0.02238646 |
| mixed256 / 9 / 2 | 0.001953125 | 0.5 | 0.02741739 |
| mixed256 / 10 / 1 | 0.03125 | 8 | 0.001842401 |
| mixed256 / 10 / 3 | 0.0625 | 16 | 0.002977603 |

For a complete graph, `D^T D = N*I - 11^T`. The third column compares the
consensus curvature on the contrast subspace with a typical node curvature.
It demonstrates substantial variation; it does not by itself prescribe an
optimal rho. Heterogeneous curvature and changing fused subgraphs still matter.

## Objective-scaling proof and minimal defensible correction

Multiply the entire QP objective by any positive constant c: replace
`h,caps,q,rho` by `c*h,c*caps,c*q,c*rho`, while keeping target, boxes, x, z and
scaled v unchanged. In exact arithmetic, both sides of the boxed x-update
scale by c, and the edge threshold `caps/rho` is unchanged. Thus the ADMM
iterates are identical until a controller makes a different decision.

The current primal norm r is unchanged, while s is multiplied by c. Comparing
r directly with s can therefore change the factor-10 balancing branch for the
same optimization problem and same iterates. This is an algebraic controller
defect independent of the unavailable historical `z_previous` values.

The minimal correction is to compare r against **s / base_rho**, retaining the
fixed positive initial `base_rho = median(h)`. Both quantities now have CCF
units and are unchanged by the objective scaling. The factor-10 comparisons,
update frequency, factor-two rho updates, relative rho bounds, state rescaling,
initial states, iteration budgets, objective, and certificate gates can remain
unchanged. The proof applies away from floating-point overflow/underflow and
activation of the tiny-value floor; it is exact-arithmetic invariance, not a
claim of bitwise GPU replay under arbitrary rescaling.

Optional edge/node RMS normalization would compare `r/sqrt(M)` with
`s/(base_rho*sqrt(N))`, where M=N(N-1)/2. This removes simple coordinate-count
factors while preserving objective-scale invariance, but introduces a further
algorithmic choice. It is not needed to correct the objective-scale mismatch
and should not be combined with the minimal repair without separate evidence.

No normalization proves convergence or speed. In particular, median curvature
does not remove node heterogeneity, inappropriate equality proposals, or
capacity-cut obstructions. Allocated-CUDA exact-QP replays and complete paths
must assess the numerical consequences. Keep the original absolute-plus-relative
gap and KKT gates unchanged: the absolute gap term is intentionally not
objective-scale invariant, and controller normalization is not permission to
rescale scientific admission.
