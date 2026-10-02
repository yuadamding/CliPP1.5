# Output contract, schema 2

Only a fresh output directory is accepted. Work occurs in a sibling
`.NAME.inprogress.UUID` directory under an exclusive `.NAME.lock`. Success
verifies artifacts before directory publication and writes `COMPLETE.json` last.
A killed process, failed validation, or write failure cannot create a verified
success marker. Failed attempts/logs remain for diagnosis; they are never reused.
After a killed process, reconcile its PID before manually removing its stale
lock. There is no automatic lock stealing or overwrite flag.

| Path | Meaning |
| --- | --- |
| `inputs/` | Exact input copies, bound by hashes |
| `preprocess.stdout.log`, `preprocess.stderr.log` | Retained preprocessing output |
| `preprocess_result/` | Ledger, canonical data and native vectors |
| `preliminary_result/` | Frozen chain, pilot/initializer diagnostics, native K/replicate proposals |
| `final_result/bic_selection.tsv` | Capacity/replicate results and overall winner |
| `final_result/chain_candidates.tsv` | Proposal/refinement provenance, scores and scoped statuses |
| `final_result/*_K*.txt` | Three tables per capacity winner |
| `final_result/Best_K/` | Exact copies of the selected three tables |
| `manifest.json` | Input/configuration/source/build/environment/chain identities and artifact hashes |
| `COMPLETE.json` | Manifest hash, completion status, all stage timings, API wall time and sampled memory |

Assignments retain chromosome, position, `mutation_id`, `original_row` and
`cluster_index`. Cluster rows contain `num_SNV`, `cellular_prevalence` (CP),
`cancer_cell_fraction`, `purity`, `mixture_weight` and `assignment_proportion`.
Cluster labels descend by CP; label zero is not automatically clonal. CP is
bounded by purity; CCF is bounded by one. Multiple nonadjacent blocks with the
same center remain distinct labels.

Multiplicity tables contain the assigned block, major CN, posterior modal
multiplicity, its probability and posterior mean. They condition on fitted CP,
purity/CN and partition; they are not uncertainty over model selection.

Requested K is a budget; `num_clusters` is occupied q, and `num_parameters=2q-1`.
`conditional_log_likelihood` is the assigned-block objective;
`log_likelihood` is the observed mixture objective used in the BIC-form score.
These quantities must not be substituted for one another.

## Status scopes

- Initializer `grid_weight_optimum_certified` concerns its finite-grid weight
  problem only. An `iteration_budget` pilot can remain a valid proposal source.
- Native `penalty_stationary_constraint_tolerance` checks the recorded numerical
  tolerances. `penalty_numerical_limit_candidate`, `line_search_stalled_candidate`
  and `finite_budget_candidate` preserve incomplete-optimization evidence.
- Final `numerical_multimode_refit` describes conditional scalar search, not
  proven global optimization. `parent_raw_*` belongs to the exact raw source.
  A derived candidate's own `raw_status` is absent.
- Verifier success establishes identity, arithmetic and structural consistency,
  not biological correctness, global optimality or calibration.

`clipp verify RUN` recomputes all published per-K likelihoods, score dimensions,
weights, CP/CCF conversion, assignments, conditional multiplicity summaries,
chain contiguity, selected candidate/raw-parent identity and raw serialized-point
diagnostics. It also checks the complete artifact inventory and every hash.
It uses independent binomial enumeration, not the fitting model or optimizers.
Checksums are integrity evidence, not cryptographic authentication against an
adversary who rewrites every artifact and manifest.
