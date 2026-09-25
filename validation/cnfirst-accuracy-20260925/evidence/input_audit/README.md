# Input and pilot-graph audit

The fixed 193-case export contains 41,712 mutations. This audit reads local,
saved data only; it does not fit or change any production model or running job.

Reproduce into a new output directory:

```bash
taskset -c 30,31 /home/yding1995/miniforge3/envs/ml1/bin/python -B audit.py --output /path/to/new/evidence
```

`evidence/summary.json` records definitions and source hashes;
`evidence/input_bindings.json` records all checked input hashes, and
`evidence/ARTIFACTS.json` binds the generated audit outputs.

All 193 cases pass raw-read, CN, purity, normal-CN, normalized-truth, conversion,
PyClone-VI input, PhylogicNDT input, multiplicity-support and CCF-bound checks.
Full-precision round-trip parsing confirms exact purity agreement. PhylogicNDT's
documented convention is a1=minor and a2=major. The generator's uniform
multiplicity prior agrees with CliPP1.5's likelihood. Standardized binomial
residuals have mean -0.00952 and mean square 1.0070, showing no evident count
generation or overdispersion discrepancy.

The truth-assisted assignment diagnostic fixes the true cluster centers and
evaluates the exact marginal likelihood. Its mean ARI is 0.63869 under equal
cluster probabilities and 0.70917 under the true cluster proportions. These
numbers are retrospective diagnostics, not deployable performance or an ARI
upper bound. They show that unavoidable read ambiguity alone is insufficient
to explain the observed mean ARI of 0.4850. Under the known-proportion rule,
errors affect 1,222/23,671 CN1/1 mutations (5.16%) and 2,951/18,041 CNA mutations
(16.36%).

Saved pilot values show a substantial multiplicity alias problem. For 10,876
of 11,321 loci with true multiplicity below major CN (96.07%), the pilot lies
closer to an alternate CCF/multiplicity explanation than to the true CCF.
Examples:

| Major CN / true multiplicity | Mutations | Lower-alias fraction | Mean pilot bias |
|---|---:|---:|---:|
| 2 / 1 | 1,883 | 97.66% | -0.3976 |
| 3 / 1 | 1,865 | 99.62% | -0.5218 |
| 4 / 1 | 1,908 | 99.79% | -0.5800 |
| 4 / 4 | 1,954 | 0% | -0.0051 |

The alias for true CCF `phi`, true multiplicity `m`, and an alternative
multiplicity `q` is `phi*m/q`. The graph then treats these scalar point modes as
the basis for inverse-gap weights. Reconstructing that rule from the saved
full-precision pilots yields these pooled mean normalized edge weights:

| Pair class | Mean weight |
|---|---:|
| Same truth cluster, diploid to diploid | 2.55127 |
| Same truth cluster, low-multiplicity CNA to diploid | 0.07782 |
| Different truth clusters, low-multiplicity CNA to diploid | 0.14811 |

Within 128/169 multi-cluster tumors, low-multiplicity CNA-to-diploid edges are
stronger on average across different truth clusters than within the same truth
cluster. The median within-tumor wrong/same weight ratio is 1.679. These
reconstructed weights are diagnostic source-equivalent arithmetic, not a
device-tensor-hash-certified replay. The evidence establishes misleading pilot
geometry; an intervention is still required to measure its causal contribution
to final clustering errors.

Detailed outputs are `per_case.tsv`, `pilot_by_major_multiplicity.tsv`,
`graph_edges_per_case.tsv`, and `graph_edges_summary.tsv` under `evidence/`.
