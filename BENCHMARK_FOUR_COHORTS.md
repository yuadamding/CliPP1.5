# Full single-region evaluation of CliPP1.5 0.3.0

The production changes are integrated and the full evaluation is running.
No full-cohort accuracy claim is made until every planned tumor has a terminal
receipt and the final audit passes.

## Bound population

| Dataset | Tumors | Retained mutations | Median N | N range |
|---|---:|---:|---:|---:|
| CliPP2Sim1K_regionalCN_CNA0246_20260917 | 200 | 315,135 | 1,000.5 | 251–4,143 |
| SimClone1000_TSV | 756 | 4,152,931 | 2,054 | 4–28,749 |
| PhylogicNDT500_TSV | 500 | 2,025,729 | 4,004 | 1,383–6,924 |
| CliPPSim4K_tsv | 4,000 | 2,009,460 | 507 | 200–800 |

Only the regional-CN cohort's 200 original one-region tumors are included.
No multiregion tumor is projected onto a single region. All 5,456 cases passed
input/truth preparation; default major-CN>4, missing-count and zero-depth
eligibility rules apply. The folders' actual populations, rather than their
nominal names, define the requested scope.

SimClone is evaluated using separate corrected inputs with normal CN=2 on all
chromosomes, matching its original generative contract. This repairs 124,403
mutation rows in 338 tumors. Original canonical files remain unchanged. Six
retained mutations in three tumors have ambiguous coordinate-based truth and
are excluded from accuracy scoring, with coverage reported; they remain in the
fits. The repository's corrected legacy converter is
`benchmarks/convert_single_region_benchmarks.py`; the existing canonical TSVs
can be repaired without overwriting them using `repair_simclone_normal_cn.py`.

The PhylogicNDT TSVs all encode one CN state per mutation, while their original
truth contains 77,037 mutation rows with differing subclonal CN states. There
are also 171,006 rows whose first truth CN state differs from the supplied CN.
These counts describe a potential observation-model mismatch, not an inferred
solver defect. The requested PhylogicNDT TSVs are evaluated as supplied; integer
multiplicity truth comes from `integer_mult`, equal to `mult` throughout this
cohort. The SimClone normal-CN repair is not applied to PhylogicNDT.

## Execution and comparison

The immutable package source fingerprint is
`6f2b1b35672b9067ebac404774ed3c4905c74d4c59a24c4fc49f8745e8adea8d`.
This CPU-only NumPy/SciPy implementation runs under conda `ml1`, one fresh
process and one pinned CPU per tumor, with numerical libraries limited to one
thread. There are **25 CPU workers**, as requested. This is a shared host;
wall times are observed pool timings, not exclusive-resource measurements.

The initial 24-worker controller was retired after its admitted fits finished.
The replacement validated and imported all 364 successes, then continued at 25
workers with identical package, inputs, likelihood, chain, tolerances and score.
No completed fit was restarted. The original plan and capacity-change receipts
remain distinct. Each attempt has a six-hour time limit; a timeout is reported
as a timeout, not omitted or counted as numerical success.

Each fit retains the original fusion-path winner and the new selected partition.
Their paired accuracy comparison isolates proposal coverage using the same
pilot, chain and qualified raw search. It is not an independent rerun/timing
measurement of an earlier release. Report nontrivial (true-K>1) ARI separately
from true-K=1 false splits; final-refit CCF MAE/CCC; all-exact-one sMF error;
exact K; and pooled CNA-only exact-class macro/micro/weighted/per-class F1 and
coverage. CNA means anything other than 1/1, including equal amplified CN.

## Evidence and reproduction

Working evidence is in
`results/single-region-four-cohorts-20260921-v1/` (intentionally ignored by Git).
The full run is `full-run/`, with:

- `plan.json`: all case/input/truth identities, full frozen source hashes and policy.
- `frozen/`: exact inference package and worker/evaluation scripts.
- `capacity25-capacity.json`, `capacity25-imports.json`: active capacity and imports.
- `status.json`, `summary-current.json`: mutable operational progress, not completion.
- `runs/<dataset>/<tumor>/`: startup, published fit, paired metrics and terminal receipt.
- `COMPLETE.json`, `summary-final.json`: written only after all planned cases terminate.

Preparation uses `benchmarks/prepare_four_cohorts.py`; the initial regional
manifest-field parser error and four SimClone truth-join issues are retained
under the initial preparation directory. `preparation-recovery/` validates the
5,252 successful preparations and repairs the remaining 204 without changing
canonical data. They were preparation issues, not fit failures.

Use `benchmarks/run_four_cohorts.py freeze` with the prepared manifest and a
fresh result directory, then execute its frozen controller. Existing evidence
is never overwritten. The current run is already active: do not launch another
controller or duplicate its tumors.

## Integration qualification

364 tests passed before the source was frozen; one additional truth-join test
passed afterward. Ruff passed. A full-fit replay of the eight original cases
reproduced the frozen best-seed prototype's cuts and scores within 1e-8, recovering
all three known missing partitions and preserving the other five.

The full corrected SimClone fits for sim39w3sj, simqm2w9u and sim0zh5xa now all
select K=1, ARI=1 and zero CCF MAE, for both the fusion-only and integrated
estimators. This confirms the earlier fixed-partition normal-CN counterfactual
with full fits. These three checks do not establish cohort-wide accuracy.
