# CN-first4K accuracy investigation: fixed 193-case panel

See [the research report](../../RESEARCH_CNFIRST_FAILURES.md) for findings,
methods, source identity, limitations, and implementation recommendations.

This bundle preserves selected small evidence tables, summaries, diagnostic
script sources, and [the four-page figure PDF](evidence/error_patterns/CNfirst4K_CliPP15_error_diagnostics.pdf)
from the September 25, 2026 investigation. The evaluated snapshot is
September 25, 2026, **1:12:36 PM CDT**: 193 validated tumors and 41,712 mutations.
These are early-finished cases, not full-cohort accuracy or independent validation.

The unchanged-score, fixed-center membership diagnostic improves score and
ARI in 152 cases. Mean ARI changes from 0.4850 to 0.6397, CCF MAE from 0.08870
to 0.05930, and sMF CCC from 0.7831 to 0.8744. This is an offline experiment;
the suggested membership proposals are **not integrated into production**.
It conveys no new CUDA qualification or inherited raw-solver certificate.

`MANIFEST.json` binds every copied artifact to its SHA-256 and original path.
Original run receipts, mutation tables, and full investigation evidence remain
under `/storage/CliPP2/CNfirst4K_CliPP15_investigation_20260925`; they were not
moved, rewritten, or embedded in this compact bundle. The input cohort and
other methods' saved outputs are likewise external prerequisites.

`archived_scripts/` contains byte-identical source copies for review. Their
relative paths refer to the **original investigation layout**, not this archive.
Do not run them from this directory or interpret them as installed entry points.
Reproduction uses the original layout and `ml1` interpreter documented in the
research report and investigation README. The original evidence is bound to
the frozen run payload, not a moving checkout of `main`.

The separate one-pass frequency-prior reassignment experiment is retained as
supporting evidence. Headline results use `reassignment/summary.json` and
`reassignment/per_case.tsv`, which apply the exact existing partition score.
