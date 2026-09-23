# Paired empirical evaluation of outer backtracking

The `af06b73` review recommends representative positive-penalty and larger-input
evaluation before considering a production-default change. The numerical
package remains `af06b7305b1928aeeabeac5ab5c5b454d2369c3f`, source SHA-256
`40c5304905d3e86832e22832e768bc9fe61cb291351358680d9053e93dd9505c`.
`evaluate_surrogate_cohort.py` is an explicit research driver. Scalar remains
the production default; the only intervention in a coordinate trial is the
already implemented `surrogate_policy` argument.

## Frozen panel

| Cohort | Case | Retained mutations | Retained truth K |
|---|---|---:|---:|
| CliPPSim4K | `100_2_0.6_0.1_rep26` | 200 | 2 |
| CliPPSim4K | `200_3_0.4_0.2_rep120` | 800 | 3 |
| SimClone1000 | `sim5ev6ub` | 802 | 6 |
| Regional-CN, single region | `1000_3_0.3_0.6_S1_Lm1000_M1000_rep0` | 1,000 | 3 |
| SimClone1000 | `sima29m0f` | 1,481 | 7 |
| PhylogicNDT500 | `Sim_500_374` | 1,715 | 3 |
| PhylogicNDT500 | `Sim_500_70` | 1,996 | 5 |
| Regional-CN, single region | `100_4_0.3_0.6_S1_Lm2000_M2002_rep0` | 2,000 | 4 |

The first case is the previously validated GPU canary with a positive selected
penalty. Other cases minimize distance to prespecified cohort size targets among
cases with at least two retained truth clusters, retained CNA mutations and at
least 99% truth coverage; case key breaks ties. Targets are 800 for the second
Sim4K case, 800/1,500 for SimClone, 1,000/2,000 for Regional-CN and 1,400/2,000 for
Phylogic. Cases run in increasing retained-size cost order. No coordinate result
was inspected when selecting the panel.

This is an exploratory eight-case panel that includes heterogeneous CN states,
not a prevalence-weighted accuracy estimate for the four full cohorts. Truth
contributes eligibility and evaluation only; fitting receives only the original
input. The frozen corrected SimClone normal-CN inputs are reused. The original
basename, byte hash, parsed tumor/sample IDs, retained IDs and truth hash are
validated before execution. The full cohort filters and score remain unchanged.

## Measurements and interpretation

Each strategy independently runs the default path on compiled CUDA float64,
including pilots, original graph recipe, all planned starts, qualifications,
unconstrained fitting, membership refits and score. Label 0 still designates the
refitted cluster closest to one. No positive penalty is forced. The trial order
alternates across panel cases; one pair per case supports numerical and empirical
comparison, not a warm throughput ratio.

The driver retains:

- Actual selected penalties, raw/refitted CCFs, labels, scores and certificates.
- Attempted, returned, qualified and raised starts, including thrown paths.
- Per-penalty objective, coverage, QP work and runtime records. Objective
  differences use exactly equal positive lambdas and independently qualified
  returned raw candidates. Lambda zero is excluded from this comparison.
- Pilot and graph equality checks before comparing the two strategies. Selected
  raw-objective differences are absent when selected penalties differ.
- Allocated/reserved GPU peaks and explicit numerical/publication time scopes.
- Truth-matched ARI, raw/refitted CCF error and CCC, designated-cluster sMF, and
  CNA-only multiplicity F1 with coverage. Constant-vector CCC and true-K-one ARI
  conventions remain explicit. Per-case sMF values are retained for later
  paired aggregation; they are not individual-case sMF CCC values.

An incomplete search can still publish a qualified winner and accuracy metrics,
but retains its incomplete status. A numerical failure is preserved as such and
does not produce invented objective/accuracy values. Unexpected execution or
output-validation failures halt further submissions. A top-level `status=passed`
means the paired observation was recorded; per-strategy completeness and
qualification are separate fields. No automatic promotion follows this study.

## Execution and evidence

The immutable local run is
`results/cuda-surrogate-empirical-20260923-v1`. `STUDY_PLAN.json` binds the panel,
source, payload and deadline; `SOURCE_INVENTORY.json` and
`CONTROLLER_INVENTORY.json` bind the complete driver and operations sources.
All eight original inputs passed local staging/identity validation, and the
immutable packager and existing terminal-import verifier were exercised before
launch. Fifty focused surrogate/timing tests passed, including six new empirical
comparison/failure tests. These checks are not empirical GPU results.

The detached controller was started on September 23, 2026, at 2:27 PM CDT,
PID 943828. It first requires exact terminal imports of all six existing timing
stages, then submits one tumor per scalar LSF job with one exclusive L40. It
retains a single nonterminal research-job slot, 8 GB host memory, two CPU slots
and a two-hour job limit. The empirical deadline is stored in UTC in the plan;
user-facing displays use America/Chicago. Pending or uncertain submissions keep
the slot. Each case has a fresh immutable remote attempt and separate empirical
output, with complete terminal artifact import before the next submission.

The existing 25-worker CPU pool and 15-job LSF cohort controller remain separate
and unchanged. Their outputs are not replaced by these intentional paired
research observations. `results/CURRENT_EMPIRICAL_STUDY.json` identifies the
owner and progress file; inspect that owner before any recovery. Never rerun its
controller entry point to check status.
