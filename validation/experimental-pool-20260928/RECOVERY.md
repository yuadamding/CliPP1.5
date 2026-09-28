# Experimental terminal-case recovery, September 28, 2026

All **3,097/3,097 planned Experimental cases are validated** as of
**September 28, 2026, 2:44:30 p.m. CDT**. Five failed cases were recovered.
There are no unresolved cases or active workers in this study.

| Cohort | Validated study cases |
|---|---:|
| CN-first4K | 2,661/2,661 |
| SimClone1000 | 385/385 |
| PhylogicNDT500 | 51/51 |

These counts cover the frozen development study, not the entire three source
cohorts. No new accuracy comparison or production-adoption claim follows from
operational completion.

## Failures and repairs

Two fits failed the independent EM monotonicity guard: A40 case `02593`
(`100_1_0.9_0.7_rep140`) and H100 case `02920` (`simu6nhvc`). The H100 Job then
stopped its other workers, interrupting `02915` (`sim0qindx`), `02921`
(`sim76ko0y`) and `02922` (`sim3hvs3a`). All five now have independently
validated replacement bundles in a fresh recovery directory. Original failed
claims, logs and partial outputs remain intact.

The original A40/H100 configurations lacked the compiler settings qualified
earlier on A100. The recovery applies those settings in a bound wrapper:
disable persistent reductions and epilogue fusion, set maximum fusion size to
one, and disable buffer reuse. Caches are separate by device family and
generation. This remains compiled float64 CUDA; no CPU fallback, candidate
discard, seed change or tolerance relaxation was introduced.

Three diagnostic replays per failed numerical case under the original
configuration did **not** reproduce either new failure. Their precise causes
therefore remain unconfirmed. The previous A100 posterior-corruption diagnosis
supports the compiler mitigation, but does not prove that these two failures
share the same cause or identify a specific upstream compiler defect. The
repaired component scores agree with the successful default replays exactly
for `02593`, and within `1.46e-11` for `02920`.

A separate original LSF Job, 77469627, published valid case `01059`
(`100_4_0.4_0.7_rep36`) in 20.5 seconds but retained its allocation until the
scheduler killed it at RUNLIMIT, after 4,223 seconds. Its logs do not identify
the surviving process. Code inspection and a real-process regression exposed a
cleanup gap: waiting for the Python leader did not terminate or reap compiler
descendants if that leader had already exited.

The repaired worker uses a scoped Linux child subreaper and bounded TERM/KILL
cleanup of its own child group on every exit path. It verifies group absence
before output validation. The separate replay, Job **77471986**, exited **DONE**
in 49 scheduler seconds. Both component and guarded output labels,
multiplicities and CCFs matched the original exactly. The component score was
identical; the preserved-baseline guarded result correctly has no mixture score.
The original scheduler failure remains recorded and is not relabeled DONE.

## Verification

- **22 operational tests passed**, including orphan cleanup after success or
  nonzero exit, a TERM-resistant child, timeout cleanup, changed predecessor
  rejection, explicit retry identity, concurrent claims and scheduler admission.
- **Three allocated-CUDA tests passed on each of A40 and H100**, with no skips.
- Each family passed **6,000 compiled posterior comparisons** at K=1, 2 and 8,
  plus 2,000 eager stability repetitions. Maximum posterior discrepancy was
  `8.55e-15`.
- Each family passed all **three paired component/publication cases** and the
  largest-input capacity case, N=2,180. Published CCFs matched exactly; the
  largest qualification score difference was `3.64e-12`.
- All **3,092 retained bundles** passed a source/input/schema/hash audit. Their
  population comprises 603 explicit imports and 2,489 original pool results.
- All **five replacements** passed complete output validation. Twelve child
  groups across the five recoveries and one shutdown replay were verified absent.
- Both scalar LSF jobs exited DONE. All four H100 recovery indexes succeeded;
  exact Job/Pod absence was verified after cleanup. No old owner was revived.

Ruff and whitespace checks passed. The scientific module remains byte-identical
at SHA-256 `8cdf12ed24410f2a4a3e2185cbb5fb874dc4b6d71fc71b0b46e358c5b8e897b7`.
The unconstrained fit and closest-to-one label 0 policy are unchanged.

## Reuse and evidence

Read `results/CURRENT_MIXTURE_STUDY.json` first. The resolved
`RECOVERY_RESULT_INDEX.json` maps every key to its scientific payload,
receipt directory and actual output directory. Use that mapping for future
evaluation: the original five failed directories remain failed, and the
shutdown replay must not add another case to coverage.

Remote evidence is under
`/rsrch8/home/bcb/yding4/clipp1d_experimental_pool36_20260928a/recoveries/terminal-five-v1`.
Local receipts are under `results/experimental-pool36-20260928-v1`.
The H100 Job was `clipp1d-experimental-h100-recovery-20260928a`, UID
`a2bab056-0452-4336-96bd-277a05dfec4d`. LSF numerical recovery was Job **77471879**.

| Binding | SHA-256 |
|---|---|
| Operational generation | `e4b712809974e74ae8bcac953eaa621d5837b7a79514a0d473b7218a71f80940` |
| Recovery plan | `62d818a1744306f96240da1bb4fa5c96bb7870a7da53e9ddb634ed2c9a8c04ca` |
| Preserved-result audit | `bdff54b4578df751747a3fb14c24fd74d81e0d6183d1dc6b7b36437d919c665e` |
| Resolved result index | `8db668bb23606313fe4b22b5dd97e003763ede398cf1b89f4075e5eb756340d7` |
| Completion receipt | `d074bd91250f313d669d904a24c4aff4ae59d371661ce23fda396b66f1bb4aa1` |
