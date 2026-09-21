# Actual failed production surrogates from df44e6a

These eight bounded NPZ fixtures were captured by running the archived 0.1.1
production solver on the seed-17, 1,000-mutation easy and mixture inputs generated
by `benchmarks/simulate.py`. They are actual failed surrogate calls, separate
from the independently generated wide-range stress suite. They are not the
reviewer's unavailable external stress fixtures.

`capture.json` preserves the original source-file fingerprint, input hashes,
penalty, witness, start, failure record and each NPZ hash. The archived source
matches the package fingerprint recorded in `VALIDATION_REVISION.md`. The old
solver returned no arithmetic exception; its gap gate rejected all eight while
KKT residuals were small. The old generic reason is retained as recorded.

Each NPZ contains `h`, `target`, `lower`, `upper`, `caps` and `start`. The original
frozen-witness boxes are retained exactly. Load with `numpy.load(...,
allow_pickle=False)`. `tests/test_production_qp_replay.py` checks every file hash
and replays every QP against the unchanged admission tolerances. Interior boxed
minimizers now contribute their mathematically zero normal exactly, repairing
the spurious negative-normal gap term without widening the gate.

New full-path benchmarks record up to eight failed surrogate fixtures per case
under `failed_surrogates/`, with precise arithmetic/gap/KKT/profile failure
reasons and original-versus-selected-witness box scope. Successful fit receipts
remain compact and do not embed numerical arrays.
