# Soft mixture and adaptive multiplicity experiment

This is an implemented, separately identified research estimator responding to
the [SimClone/Phylogic investigation](../RESEARCH_SIMCLONE_PHYLOGIC_PYCLONE_GAP.md).
It is **not the production default**. Its adoption requires the
[three-cohort acceptance contract](THREE_COHORT_ACCEPTANCE.md), including protection
of CN-first4K. A better likelihood, an EM fixed point, or a development-panel win
does not establish that contract.

The expanded study completed all **3,097 cases** on September 28, 2026. The
[complete performance report](../validation/experimental-pool-20260928/PERFORMANCE.md)
records the matched comparisons, individual losses and source identities.
Aggregate CN-first performance improves over the previous estimator, as do
SimClone and Phylogic headline metrics. PyClone-VI still leads sMF CCC in the
two latter cohorts. This is the complete development inventory, not independent
confirmation or the full 5,256 eligible tumors. Allocated A40, A100 and H100
qualification and the terminal recovery have their own
[execution records](../validation/experimental-pool-20260928/RECOVERY.md).

## What changes

The complete-graph solver, original raw outputs, and existing partition estimator
remain unchanged. The experiment consumes saved complete-graph partition centers
as one start alongside deterministic, observation-only starts. It writes separate
files. This changes the downstream statistical estimator, not the fusion graph.

The old hard-partition allocation score can reject accurate splits and favor
unbalanced memberships. The experiment instead maximizes the observed likelihood
by summing over uncertain cluster AND multiplicity assignments:

```text
L = sum_i log sum_c pi_c sum_m q_ic(m) Binomial(alt_i | depth_i, slope_im * phi_c)
q_ic(m) = (1 - w_c) / A_i + w_c * indicator(m = 1)
```

`A_i` is the original integer-support size. The fixed CN denominator, count data,
support cutoff, and probability clipping are unchanged. With `w=0`, each component
likelihood is exactly the existing uniform-multiplicity likelihood. Amplified
mutations can support enrichment near multiplicity one; no clonal label, tumor
name, cohort name, simulated multiplicity, or true CCF determines that enrichment.
The same prior applies to every latent component. Singleton supports do not
contribute artificial information about `w`.

The two model branches are fitted separately:

- Uniform: `w_c=0` for all components.
- Adaptive: independent `Beta(1, 2)` regularization on each `w_c` by default.
  This adds `-sum_c log(1-w_c)` to negative log likelihood. The uniform branch
  remains explicitly available and does not pay for unused enrichment parameters.

Diploid-only inputs instantiate only the uniform branch. The adaptive branch
still pays for every declared enrichment parameter, including estimates on the
boundary. This is a conservative, prespecified parameter-count convention; it
is not a fitted penalty chosen separately for any cohort.

## Inference and criterion

The code uses float64 PyTorch EM with a joint cluster/multiplicity posterior. The
binomial center M-step uses a bounded scalar derivative solve and compares the
represented clipped objective at its root and original endpoints. The mixing
masses and enrichment update from expected allocations. Identical represented
CN slopes are grouped once; each center update sums posterior reference counts
within those groups before its derivative iterations. This is an algebraic
reduction of the same M-step, not a changed likelihood or score. Every accepted step
independently recomputes the complete regularized observed objective and checks
that it did not increase beyond a floating-point margin. Failures are explicit.

Soft components assign positive probability across the tumor, so their shared
domain is the intersection of the original mutation bounds. No original bound
is enlarged and no component is fixed at CCF one. The shared interval is recorded;
an empty interval fails explicitly. Unlike a hard partition, a soft component
cannot use the looser bound of just its MAP-assigned subset. This distinction is
part of the experimental model, not a repair to the old solver's bounds.

The default search fits K=1 through K=8, with evenly spread and observed dosage
quantile starts, plus the preserved partition centers. All starts fit both model
branches when informative amplified loci exist. Seed centers initialize whole
components; no mutation is individually moved to a clipped pseudo-center. An
existing partition with K above the exploratory cap is still included up to the
explicit implementation limit of 20 components. All components may move away
from one. Identical initialization centers may remain identical, which is one
reason this is a bounded multistart search rather than a global guarantee.

Selection minimizes the **new** `observed_mixture_map_bic_v1` criterion:

```text
2 * regularized_negative_log_likelihood + d * log(N)
d = 2*K - 1                  (uniform)
d = 3*K - 1                  (adaptive)
```

This is a declared BIC-style heuristic. Finite mixtures have boundary and
nonidentifiability issues; this expression is not claimed to be a calibrated
Bayesian evidence calculation. Its values must not be compared to the old
hard-allocation score. Hard-empty components still count toward K and the penalty.
Omitted binomial coefficients are identical across candidates on a given input.

Convergence requires small absolute objective improvement per mutation AND a
small maximum parameter change. A 500-iteration cap is recorded independently
for every start. An unconverged best candidate can remain the provisional winner;
its selected status is `iteration_budget_exhausted`, never falsely `converged` or
`qualified`. All multistart results remain globally uncertified, even when every
individual EM run stops at a fixed point.

## Conservative structural selector

The first 12-case CN-first development screen exposed a tradeoff: the unrestricted
soft selector improved mean ARI and CCF MAE, but worsened sMF MAE and CCC. This
fails the user's non-regression requirement and is retained as negative evidence.
The exploratory soft fit therefore has an additional, separately versioned
[structural selector](../benchmarks/mixture_structural_guard.py):

1. Use the best recorded uniform soft-mixture fit at the original occupied K as
   the reference under the **new** criterion.
2. A replacement must have lower new-criterion score, have reached the declared
   EM fixed point, and either increase occupied K or select the adaptive
   multiplicity model. Occupied K may not decrease.
3. When these conditions are absent, preserve the original complete-graph
   memberships, CCFs and multiplicity calls together. In particular, a uniform
   fit that merely rearranges groups at the same K cannot replace the original.

This targets the documented missed populations and wrong multiplicity modes
without automatically replacing existing same-K partitions. It is one rule for
all cohorts, using observed inputs, baseline K and fitted objective values only.
It deliberately leaves some same-K membership improvements unexplored and cannot
repair every baseline overclustering error. The reference is the best *attempted*
fit, not a global optimum; a lower score is not a proof of higher accuracy.

The initial structural screen preserved all 12 CN-first results exactly. That
rule was formulated after the initial CN-first screen, so these results are
development evidence, **not independent confirmation** or a full-cohort pass.
No selector is promoted on that basis. Both selectors remain in the study report,
and all three cohorts retain separate acceptance checks.

[Apply the selector](../benchmarks/apply_mixture_structural_guard.py) only to a
complete, bound inference inventory. The resulting separate schema records
whether the old partition was preserved or a new mixture structure was selected,
the reference and candidate scores, source bindings and parent receipt hashes.
This neither overwrites the old outputs nor gives a replacement an old certificate.

## Outputs and scientific identity

[Numerical implementation](../src/clipp1d/cuda/mixture.py) exposes explicit compiled
CUDA inference and a separately named eager CPU component-reference adapter.
It is not called by `clipp1d fit`, and there is no automatic CPU fallback. The
reference adapter is for development diagnostics, not production CUDA evidence.

[Experiment runner](../benchmarks/run_mixture_experiment.py) requires a manifest
binding observed inputs and complete retained-population seed tables by SHA-256.
Its inference-case schema rejects extra fields such as truth paths. It records
the complete package source inventory, runner, configuration, environment,
manifest, inputs and output hashes. Output destinations must be absent; frozen
results cannot be overwritten. The runner completes every declared case and
records failures rather than silently creating a favorable successful subset.

Each case writes:

- `mixture_mutation_clusters.tsv`: occupied MAP cluster labels, component CCF,
  and multiplicity MAP conditional on the selected component.
- `mixture_cluster_centers.tsv`: occupied component centers, sizes and the
  corresponding latent component indices.
- `EXPERIMENT.json`: all latent centers/masses/enrichments, candidate histories,
  selected status, model/criterion identities, bounds and output bindings.

Public label 0 is the **occupied** group closest to one, with smallest member
mutation ID breaking exact ties. Labeling does not alter centers. CCFs are fitted
soft-mixture component modes with MAP memberships; they are neither the previous
hard-membership refit nor PyClone posterior means. Latent and occupied K are
separately identifiable because some positive-mass components can have no MAP
members. Their mixture likelihood and complexity cost are still retained.

The new model inherits **no raw KKT or scalar-refit certificate**. Its metadata
explicitly records that no production adoption, allocated-CUDA qualification or
global-optimality claim is supplied by this routine.

```bash
python benchmarks/run_mixture_experiment.py \
  --manifest INFERENCE.json --outdir NEW_OUTPUT --device cuda:0
```

For an explicitly authorized small CPU component diagnostic only:

```bash
python benchmarks/run_mixture_experiment.py \
  --manifest INFERENCE.json --outdir NEW_REFERENCE_OUTPUT \
  --device cpu --cpu-reference
```

The separate evaluator reads truth only after inference. It checks every frozen
case, input, output and mutation population; reports paired retained and matched
metrics, independent cohort non-regression checks, multiplicity diagnostics and
convergence limits; and always labels this small screen as development evidence.
Its summary cannot authorize adoption in place of the full three-cohort gate.

## Completed development screen

The [36-case validation report](../validation/soft-mixture-20260927/README.md)
contains the completed comparison and negative findings. The structural selector
preserves all 12 CN-first cases and improves mean ARI and CCF MAE in both other
panels. It nevertheless fails the strict development gate because SimClone sMF
CCC falls from 0.495489 to 0.494542. The three SimClone false K=1 collapses remain.
Neither this small decline nor the unrestricted selector's CN-first regressions
were hidden by rounding, pooled averages or changes to the acceptance rule.

The Phylogic mixed-CN truth repair described below is bound in a new evaluation
manifest. All other truth fields, inference inputs and evaluation populations
match the original frozen panel. CNA-only macro-F1 improves in the two older
cohorts and is unchanged in CN-first, with full diagnostic call coverage. The
numerical code has 22 focused CPU checks across its model and evaluation/preparation
interfaces; allocated-CUDA qualification and full-cohort acceptance remain absent.

## Required next evidence

The initial development screen was fixed at 36 cases: 12 per cohort, combining
previously reported mechanisms and hash-selected controls. These cases were
already available development data. There is no held-out accuracy claim. CN-first
inputs come from the existing prior-study discovery inventory, not a new random
sample of its entire cohort.

Independent CPU tests cover likelihood/posterior enumeration, exact nesting of
the uniform branch, center updates against a separate scalar optimizer, score
arithmetic, single-cluster controls, both multiplicity generators, convergence
exhaustion and fail-closed interfaces. Dedicated CUDA tests compare compiled
results with fresh references and check the complete fusion path before and
after downstream experimentation; skipped tests are not a CUDA pass. The
[qualification controller](../benchmarks/qualify_mixture_cuda.py) additionally
performs three matched observed-input CPU/CUDA component comparisons in fresh,
sequential child processes. It submits no scheduler jobs and rejects execution
outside its explicitly authorized LSF allocation.

Before adoption: qualify the final source on allocated CUDA, perform paired full
fits, freeze independent confirmation by simulation structure, and complete the
three-cohort comparison. Include CNA-only exact-class macro-F1 and coverage for
this changed multiplicity model. Phylogic's old normalized mixed-CN metadata is
known to be incomplete; supplied-CN diagnostics alone cannot satisfy its final
multiplicity gate. Preserve all negative development results and current runs.

## Phylogic truth preparation repair

[prepare_four_cohorts.py](../benchmarks/prepare_four_cohorts.py) now writes
`clipp1d.normalized_simulation_truth.v2`. For Phylogic it derives each `mixed_cn`
flag from the original simulator's two CN states and occupied fraction, rather
than writing zero for every mutation. Numerically identical states, including
different decimal formatting, are one state; a zero-weight secondary state is
not mixed. Supplied major/minor CN and original mixed-state truth are explicitly
distinguished in the preparation metadata.

New preparation refuses to reuse unversioned Phylogic truth through `--recover`
and requires a different output root from the recovered attempt. Existing inputs,
active campaigns and frozen truth files are preserved. The development check
reprepared all 12 Phylogic cases in a new directory: 664 of 23,799 retained
mutations acquired the correct mixed flag; all other normalized truth fields and
all input bytes were identical. Four targeted truth-join tests and the existing
SimClone repair test pass. The mixed flag does not itself resolve the mismatch
between original mixed CN and the converted single-state fitting inputs, so it
does not by itself complete the multiplicity acceptance contract.

## Expanded development test requested September 27

The follow-up request, “You should test on much more cases,” expands the frozen
candidate from 36 tumors to **all 3,097 tumors in the saved matched comparisons**:
2,661 CN-first4K, 385 SimClone1000 and 51 PhylogicNDT500. The scientific model,
starts, EM budgets, criterion and structural selector remain unchanged. The
inventory includes the original screen and every other saved matched case; it
is retrospective development evidence, not an independent confirmation set or
the complete 5,256-tumor benchmark.

Operational evidence is under `results/soft-mixture-expanded-20260927-v1`.
`SELECTION.json` binds the exact original comparison authorities before new
candidate execution. CN-first canonical inputs are reconstructed with the
original converter or imported from the receipt-bound original input tree;
both paths must match each original input SHA-256 exactly. Saved
seed partitions retain their original retained-row text; excluded output rows
are removed with explicit provenance. Phylogic mixed-CN truth is repaired in a
separate evaluation tree, preserving input bytes and all other truth fields.
Inference receives no truth table.

The original [isolated LSF driver](../benchmarks/mixture_lsf/controller.py) used one
additional slot across qualification and the rolling experiment. Existing
production and prior-proposal campaigns retained their allocations in that phase. An allocated
CUDA qualification must finish successfully before three publication canaries
run; those compare final guarded labels, CCFs, multiplicities and selection
status against the frozen CPU references. Only then can the remaining inventory
start. Each scalar submission fits one case and exits. Pending, held and unknown
jobs retain the slot. Ambiguous submission stops refills; no automatic retries
or cancellation occur. Case failures remain in the denominator and cannot be
discarded for an acceptance report.

On September 28 the user reassigned all prior CliPP1.5 resources to Experimental.
The [shared-pool driver](../benchmarks/mixture_pool/controller.py) and two
Kubernetes pools now cover the same frozen inventory, with 603 validated imports
and 2,494 remaining cases at handoff. The authorized ceilings are 22 scalar LSF
jobs, 10 A100 workers and 4 H100 workers. This changes execution only: the
scientific policy, original seed partitions, cohort selection and acceptance
rules remain unchanged. See the [resource-transfer handoff](../benchmarks/OPERATIONS.md#experimental-resource-transfer-september-28)
for exact ownership, hardware gates, historical failures and current pointers.

The qualification environment initially lacked pytest. Preserve failed attempts
`77453319` and `77453503`: the first lacked pytest, and the second test-only bundle
omitted pytest's `py.py` compatibility shim. The third bundle includes the full
pure-Python dependency distributions and verifies `python -S -m pytest --version`
locally and remotely before submission. Shared conda packages and numerical
source are unchanged. These setup failures are not numerical test failures or
CUDA passes. Current identities belong in `results/CURRENT_MIXTURE_STUDY.json`.

Attempt `77453550` then ran the CUDA tests: both mixture comparisons passed, and
the complete-graph fit in the third fixture passed. Its downstream mixture step
failed with a Triton “operand does not dominate this use” error for identical
slopes. The repair changes only compilation boundaries for that shape: six
compiled blocks of six bisection updates retain the original 36 updates and
endpoint checks. The original CPU reference and all scientific settings remain
unchanged. Fifteen reference/selector tests pass, including exact equality of
blocked and original center updates. The repair has a new source hash and a
fresh qualification attempt; the partial old GPU pass is not inherited.

For imported results, [the evaluator](../benchmarks/evaluate_mixture_experiment.py)
accepts `--artifact-map` with schema `clipp1d.artifact_tree_mapping.v1` and one
`remote_root`/`local_root` pair. It resolves artifact locations while retaining
the original input and output identities inside every receipt; hashes still
must match. Escaping paths and symlink substitutions are rejected. Matched-ID
authorities are loaded once and rehashed before publication, avoiding thousands
of repeated full-cohort JSON reads. A relocated replay of all original 36 cases
reproduces every headline metric and multiplicity result exactly. The full
expanded evaluation still requires every selected case and retains the strict
per-cohort non-regression rules, including the previously failed SimClone sMF
CCC check.
