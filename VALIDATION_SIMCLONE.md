# Matched SimClone diagnosis — September 21, 2026

The four-case comparison uses exact inputs from the latest completed main-source
CliPP2 Seadragon panel at `77525a6875e698f6834cb73e6d2a1c27af2af8bf` and
CliPP1.5 `72ba3edce3988e9281dd9dcbc856b791ffb163a3`. It is not a full-cohort
benchmark. The two true-K-one cases agree exactly; the two remaining cases
differ for distinct reasons. The [source-bound evidence](benchmarks/evidence/simclone-diagnosis-72ba3ed.json)
retains input hashes, common scores, metrics, source controls and limitations.

| Case | CliPP2 score | CliPP1.5 score | Best score over all contiguous partitions |
| --- | ---: | ---: | ---: |
| simnwk1j1, N=41 | 1314.126592 | 1311.949666 | 1311.949666, sizes 30/11 |
| sim1y9z5f, N=92 | 4837.816994 | 4852.031333 | 4837.816678, sizes 15/77 |

Scores use the same likelihood and fixed-partition score; lower is better.
All 5,139 contiguous block scalar fits qualify. All-K dynamic programming
agrees with independent exhaustive K<=3 enumeration and fixed-partition refits.
Numerical score bound gaps are 1.86e-8 and 1.46e-11. These are floating-point
certificates for a frozen chain, not formal interval proofs or unrestricted
partition optima.

For **sim1y9z5f**, the preferred split is contiguous but absent from the fusion
candidate path. 389 production/dense/cold fits preserve this miss. The boundary
after mutation 15 disappears around lambda 3.6313 while many blocks remain;
the two-block regime begins around lambda 1531.2986 with cut 48 instead.
Necessary weighted-TV stationarity conditions exclude an exact two-block
cut-15 state in both clonal orientations at every penalty. Independent numerical
dual checks across all 92 witness branches at four targeted penalties agree
with production attained objectives to arithmetic precision. This does not
prove an all-penalty statement about every approximately admitted raw state.
CliPP2 obtains the split through a recorded Ward/CEM direct candidate whose
raw parent has a one-block partition signature. Denser lambda sampling does
not address this candidate-generation mechanism.

For **simnwk1j1**, CliPP1.5 already reaches the best score on its frozen chain.
CliPP2's singleton clonal assignment lies inside tied pilot values and is not
a contiguous two-block partition. However, independently replaying CliPP2's
own finite-grid/local-refinement recipe on CliPP1.5's 30/11 labels gives score
1311.949761, better than its published 1314.126592. Refit precision cannot
explain this difference. Conditional on the saved selection contract, the
better partition was absent from the admitted candidate pool. The saved
CliPP2 search is explicitly provisional/unresolved and stopped at its lambda
refinement budget. Missing complete candidate records prevent distinguishing
never proposed from encountered only as an ineligible raw candidate.

Both cases have true K=3. The generating labels occupy 19 and 33 chain runs;
their true three-group assignments are excluded by chain contiguity. Even
unrestricted independent refits of the true groups score worse than the
two-group alternatives: 1357.397346 and 4909.995890. Fixing candidate coverage
does not alone recover biological truth. CCF MAE, clustering ARI and subclonal
mutation fraction must remain separate: all-one improves MAE but destroys
subclonal recovery in sim1y9z5f.

Both discordant cases have singleton multiplicity support, so mixture-mode
ambiguity is not the explanation. Fresh `ff5b5c3` replays produce byte-identical
public output tables on both inputs, excluding the latest performance reuse
changes as the source of these differences. Observed CPU and L40 runtimes
come from different computational workloads and are not controlled speedups.

The complete local report, plots, scripts, retained failures and immutable
inventory are in `results/simclone-investigation-20260921-v1/`; the preceding
matched inputs and production outputs are in `results/simclone-matched-20260921-v1/`.
The committed evidence binds these artifacts but does not embed the cohort.

This investigation motivates score-driven contiguous candidates in addition
to fusion candidates, preserving qualified refits and exact-one feasibility.
The diagnostic all-K algorithm has quadratic block storage and cubic transition
work, so it is a small-case oracle rather than an unqualified production
replacement. No candidate policy or numerical gate was changed for this study.
