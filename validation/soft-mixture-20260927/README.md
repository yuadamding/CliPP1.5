# Soft-mixture development revision — September 27, 2026

**Implemented and reference-tested; not adopted as the production default.**
The guarded estimator preserves all 12 CN-first development cases and improves
mean ARI, CCF error and sMF MAE in both other panels. It still fails the strict
three-cohort non-regression rule: SimClone sMF CCC declines from **0.495489 to
0.494542**. This failure is retained, without changing the gate or selecting
per-case exceptions from truth.

[Model and implementation](../../docs/SOFT_MIXTURE_EXPERIMENT.md) ·
[Bound development measurements](DEVELOPMENT.json) ·
[All per-case metrics](PER_CASE.tsv) ·
[Engineering checks](ENGINEERING.json) ·
[Remaining failures](REMAINING_FAILURES.json)

## What was implemented

- A float64 PyTorch soft cluster/multiplicity mixture, with explicit compiled
  CUDA inference and an explicit eager CPU component-reference adapter.
- A multiplicity-one enrichment model regularized toward the existing uniform
  multiplicity model. Every latent component follows the same rule; the eventual
  public clonal label does not determine its prior.
- A separately named observed-mixture, MAP/BIC-style criterion, deterministic
  multistart search and complete candidate/convergence records. It is not the
  previous hard-allocation score or a calibrated mixture evidence calculation.
- A conservative structural selector: compare under the new criterion,
  require the replacement's EM stopping condition, prohibit occupied-K decreases,
  and admit a replacement only for a cluster birth or adaptive-model change.
  Otherwise preserve all original memberships, CCFs and multiplicity calls.
- Explicit input/source/configuration/output bindings, separate inference and
  truth evaluation, no-clobber publication, complete-inventory checks, and
  independent provenance without inherited raw KKT or scalar-refit certificates.
- Correct Phylogic mixed-CN flags in future normalized truth preparation, with
  an explicit versioned schema and refusal to recover the defective legacy flags.

The complete-graph fit, original likelihood path and production defaults are
unchanged. The new estimator consumes saved complete-graph partition centers as
one initialization. Fitting remains unconstrained; the occupied component closest
to CCF one receives public label zero. The module does not run automatically from
the standard `clipp1d fit` command.

## Scope and results

The complete fixed development inventory contains **36 cases, 12 per cohort**:
reported failure mechanisms, hash-selected examples and single-cluster controls.
These are retrospectively inspected development data. CN-first cases come from
the existing prior-study discovery inventory. The structural rule was formulated
after the initial CN-first screen; this is **not independent confirmation**.

The table uses the exact common cross-method mutation population for each tumor.
Means weight tumors equally. ARI and CCC are higher-is-better; MAE is
lower-is-better. All 36 cases and the additional full-retained baseline comparison
are present; no failed, slow or unfavorable case was removed.

| Cohort | Method | Mean ARI | CCF MAE | sMF CCC | sMF MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| CN-first4K | Original CliPP1.5 | 0.736612 | 0.048522 | 0.981012 | 0.032747 |
| CN-first4K | Guarded experiment | 0.736612 | 0.048522 | 0.981012 | 0.032747 |
| CN-first4K | PyClone-VI | 0.723085 | 0.051106 | 0.964745 | 0.042999 |
| CN-first4K | CliPP | 0.597061 | 0.059508 | 0.922967 | 0.066089 |
| CN-first4K | PhylogicNDT | 0.706200 | 0.051618 | 0.844300 | 0.078670 |
| PhylogicNDT500 | Original CliPP1.5 | 0.353438 | 0.128928 | 0.233402 | 0.268935 |
| PhylogicNDT500 | Guarded experiment | 0.430169 | 0.099984 | 0.754629 | 0.097102 |
| PhylogicNDT500 | PyClone-VI | 0.425082 | 0.101832 | 0.689171 | 0.129336 |
| SimClone1000 | Original CliPP1.5 | 0.485728 | 0.202800 | 0.495489 | 0.218308 |
| SimClone1000 | Guarded experiment | 0.524233 | 0.095496 | 0.494542 | 0.204924 |
| SimClone1000 | PyClone-VI | 0.384187 | 0.121798 | 0.485026 | 0.229303 |

The guarded experiment leads the available external methods on these four
headline metrics in each small panel, but that does **not** establish full-cohort
superiority or erase its SimClone CCC regression against the preserved baseline.

The guard retains 21 original cases and replaces 15: eight Phylogic and seven
SimClone. Nine replacements use adaptive multiplicity; six use the uniform soft
mixture. All 12 CN-first cases preserve labels, CCFs and multiplicity calls. Their
full-retained metrics also remain identical.

Phylogic's false K=1 collapses fall from five to one. Its ARI improves in seven
cases and worsens in none. SimClone ARI improves in four cases and worsens in two.
None of the eight true-single-cluster controls is falsely split. SimClone's
three false collapses remain, including `sim3p5fhr`; its worst-case CCF MAE also
worsens slightly, from 0.461425 to 0.464647 in `simvo06wn`, despite the lower mean.
The bound report includes per-case wins/losses, error tails and conditional
paired bootstrap intervals. Those intervals ignore dependencies among related
simulation structures and are not population-wide uncertainty or acceptance.

The five named SimClone multiplicity/CCF failures improve substantially:

| Case | Original CCF MAE | Guarded CCF MAE |
| --- | ---: | ---: |
| simn0vt5y | 0.365282 | 0.011092 |
| simvf66y3 | 0.321725 | 0.041037 |
| sim2kigh0 | 0.293539 | 0.033179 |
| simbxvuhy | 0.238046 | 0.033451 |
| simkhuc9f | 0.214858 | 0.043960 |

The unrestricted soft selector is retained as negative evidence. In CN-first it
improves mean ARI to 0.744579 and CCF MAE to 0.045764, but worsens sMF CCC to
0.972429 and sMF MAE to 0.038420. It cannot be adopted under the user's requirement.
Preserving a baseline candidate in a search does not guarantee preserved accuracy.

## Multiplicity and truth checks

Eligibility here means supplied input CN other than 1/1, including balanced
amplification. The target is the original simulator's integer multiplicity,
not fractional effective dosage. These are diagnostics on the fixed development
cases, not full-cohort multiplicity acceptance. Original mixed-state flags are
recorded separately in the detailed tables; the converted input model is not
silently reinterpreted as the original mixed-CN generator.

| Cohort | Eligible mutations | Original macro-F1 | Guarded macro-F1 | Call coverage |
| --- | ---: | ---: | ---: | ---: |
| CN-first4K | 857 | 0.717996 | 0.717996 | 100% |
| PhylogicNDT500 | 1,658 | 0.457476 | 0.637628 | 100% |
| SimClone1000 | 5,253 | 0.523448 | 0.950968 | 100% |

Micro, weighted and per-class F1 are preserved in `DEVELOPMENT.json`. Correcting
Phylogic truth changes 664 mixed flags among 23,799 retained mutations in the
12 cases. Original input bytes and all other normalized truth fields are
unchanged. Existing campaign inputs and old truth artifacts remain intact.

## Verification and provenance

- **22 focused CPU tests passed**, across the recorded targeted invocations:
  independent likelihood enumeration, uniform nesting, independent scalar
  optimization, score arithmetic, both multiplicity mechanisms, single-cluster
  controls, iteration exhaustion, publication/identity checks and truth repair.
- Scoped Ruff and `git diff --check` pass. The complete 36-case publication,
  structural selection and paired evaluation pipelines finished successfully;
  numerical success is distinct from their negative accuracy-gate result.
- Grouping identical represented slopes accelerates the center step. All 20
  available earlier completed fits reproduce exactly: labels, CCFs,
  multiplicities, scores and selected statuses. This is CPU evidence only.
- All 36 selected unrestricted fits reach the declared EM stopping criterion.
  Some other starts exhaust their 500-iteration budget; this remains a bounded
  multistart search without a global-optimality claim.
- One interactive session interruption stopped a reference worker after 31
  total case outputs. Exact process absence and source/input/output hashes were
  checked; the 13 completed outputs of that interrupted shard were imported into
  a new receipt, and only its five unfinished cases were rerun. Original
  artifacts were preserved. Every final case is present exactly once.
- No production campaign or resource allocation was changed. CUDA qualification
  is prepared as a separate one-GPU LSF package, but no remote staging or
  submission occurred. The additional-slot approval is still pending.

The baseline commit is `da651879c2a33f481957ffae1d1c865d11ad39c8`; this revision is
dirty, uncommitted development source. Detailed source inventories, exact inputs,
policies, immutable parents and output hashes are under
`results/soft-mixture-development-20260927-v2/`. The qualification package is under
`results/soft-mixture-qualification-20260927-v1/`. Neither the source commit nor
the CPU passes certify this new estimator on CUDA.

Before adoption, address or quantify the remaining SimClone failure mechanisms,
qualify the final source on allocated CUDA, freeze independent confirmation, and
complete all three full-cohort comparisons with the unchanged gate. The evidence
supports further development of this common model; it does not authorize changing
the default or claiming that all requested performance targets have been achieved.
