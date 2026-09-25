# Same-likelihood fixed-partition diagnostic

This is an **offline analysis of the frozen 193 finished CN-first4K cases** from
September 25, 2026, 1:12:36 PM CDT. It changes no source, live run, saved fit,
scientific tolerance, or qualification policy. At most four local analysis
processes were used. It never invokes the production fusion solver.

The main finding is a membership/candidate-search gap. Simple alternative
partitions can achieve lower scores with exactly the frozen CliPP1.5 likelihood
and allocation penalty. The production fixed-membership scalar refit itself
does not appear responsible: rerunning only those memberships with the
diagnostic scalar search did not improve any case's score by 0.0001.

## Results on all 193 cases

| Partition | Lower score than published | Mean ARI | Mean CCF MAE | sMF CCC | Mean K |
|---|---:|---:|---:|---:|---:|
| Published | — | 0.485004 | 0.088703 | 0.783055 | 6.5389 |
| Published, diagnostic scalar refit | 0 | 0.485004 | 0.088703 | 0.783055 | 6.5389 |
| Truth-free greedy merges | 134 | 0.516438 | 0.079979 | 0.819760 | 2.2539 |
| PyClone-VI memberships, CliPP1.5 scalar refit | 147 | 0.640890 | 0.055416 | 0.939642 | 2.8031 |
| Oracle truth memberships, CliPP1.5 scalar refit | 136 | 1.000000 | 0.005740 | 1.000000 | 2.5492 |

Greedy merging improved ARI in 119 cases, worsened it in 15, and left 59
unchanged. It is therefore a diagnostic of the search gap, not a validated
production repair. The oracle row uses true memberships and is not a deployable
method. The PyClone row transfers only memberships; all listed CCF and score
values use the CliPP1.5 observation model and newly computed diagnostic centers.
These native-population results use all 41,712 CliPP1.5 mutations, not the smaller
intersection with original CliPP.

## Method and evidence

`diagnose.py` imports the immutable source at
`/storage/CliPP2/CliPP1.5/results/cnfirst-pool14-20260925-v2/a100/payload/source`.
It checks each input SHA and PyClone assignment-file SHA, joins exact mutation
identities, and evaluates the frozen host likelihood. It reconstructs all
published `run.json` selection scores with maximum absolute error
**1.1642e-10**.

For each proposed membership, the diagnostic evaluates a 1,025-point CCF grid,
all published centers present in that membership, original feasible endpoints,
and bounded local minimization around every sampled local minimum. It always
retains the best feasible point. A dense grid is **not** the production interval
certificate; these are approximate scalar refits. Nevertheless, a feasible
point below the published score demonstrates a lower-score partition witness:
an exact scalar optimum for that membership can only improve it further.
Nothing here grants a raw KKT/stationarity certificate to a new partition.

The exact score is

`2*NLL + K*log(N) - 1.4*(lgamma(K)-lgamma(N+K)+sum(lgamma(n_k+1))+lgamma(K+1))`.

The greedy proposal considers all pairs of current groups, merges the pair with
the largest strict score decrease, and repeats until no pair lowers the score
by at least 0.0001. It begins with published memberships and uses no truth or
other method's assignments. Oracle and PyClone memberships are evaluated only
after the greedy search ends.

`verify_grid.py` reran the union of 68 cases in which oracle or PyClone
memberships did not lower the score, using 4,097 grid points. The maximum score
difference was 0.00009847 and no lower/higher classification changed. The
smallest losing oracle margin was 0.6325. Grid agreement remains a sensitivity
check, not a global scalar certificate.

## A separate scoring limitation

Excluding the 24 correctly recovered single-cluster cases, oracle memberships
had lower scores in **136/169** multi-cluster tumors and higher approximate
optimized scores in **33/169**. The latter were concentrated at lower depth:

| Depth | Multi-cluster tumors | Oracle lower | Oracle higher |
|---|---:|---:|---:|
| 100 | 65 | 38 | 27 |
| 200 | 51 | 45 | 6 |
| 500 | 53 | 53 | 0 |

Among the 33 oracle-losing cases, the published count was too small in 13, too
large in 12, and correct in 8. Their minimum true CCF gap had median 0.253 and
range 0.2004–0.5761. The score can therefore prefer erroneous memberships even
when it is given the true partition as a candidate; this secondary limitation
is distinct from candidate generation.

For `100_2_0.4_0.7_rep52`, a complete-search case, N=204 and the true CCFs are
0.7470466 and 1.0, with group sizes 144 and 60. Published K=1 is preferred:

| Partition | 2 NLL | K log N | Allocation penalty | Score |
|---|---:|---:|---:|---:|
| Published K=1 | 19815.858972 | 5.318120 | 0 | 19821.177092 |
| Oracle memberships, approximate refit | 19727.050943 | 10.636240 | 175.586651 | 19913.273833 |

The likelihood improves by 88.8080 in twice-NLL units, but the extra penalties
are larger, leaving the oracle partition 92.0967 score units worse. Evaluating
the **known true CCF values**, without optimization, gives the separate exact
feasible-point score 19914.639771. `score_limits.py` preserves this distinction
for every case rather than calling known-truth CCFs fitted optima.

## Files

- `all/SUMMARY.json`: aggregate evidence, source/analysis hashes, reconstruction error.
- `all/per_case_candidates.tsv`: five candidate families for all 193 cases.
- `all/<case>.json`: centers, sizes, mutation-index memberships, merge histories.
- `score_limits_per_case.tsv`, `score_limits_strata.tsv`, `score_limits_summary.json`:
  oracle comparison, decomposition, and depth/K/separation strata.
- `grid4097_verification/`: denser-grid sensitivity evidence.
- `pilot/`: the preliminary 12-case panel; the final all-case analysis additionally
  binds the collected `run.json` scores.

Reproduction uses the `ml1` interpreter with numerical-library threads set to
one: `diagnose.py --panel all`, then `verify_grid.py`, then `score_limits.py`.
The scripts write analysis artifacts only under this directory.
