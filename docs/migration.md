# Migration

Install distribution `clipp15`; import `clipp`; invoke `clipp fit`. The legacy
`python run_clipp_main.py ...` wrapper is import-safe and forwards to that command.
It now **requires `--output`**. `-i/--sample-id` is an identifier, not a path.
Old `--preprocess`, `--final` directory overrides are retired so a run has one
transactional layout. Existing result directories are never overwritten.

Python implementation lives in `src/clipp/`; native implementation remains under
`src/`. The private native library is `clipp._native` and is loaded through ctypes,
not imported as a Python extension. Old loose `CliPP*.so` files are ignored. ABI 3
adds mandatory model/build identity and numerical test interfaces; ABI 2 wrappers
cannot silently load the current implementation.

`multiplicity.txt` column 4 has meant **major CN support bound**, not estimated
multiplicity, since the uniform mixture change. `Best_K` selects capacities,
not a legacy lambda penalty sweep. Center columns are CP, and schema 2 adds
explicit CCF/purity and assignment proportions. New mutation identity columns
are additive; original labels/CP/weights remain protected by baseline regression.

Old preprocessing silently chose the first overlapping segment and inconsistently
handled chromosomes. Such inputs now fail and require an explicit upstream data
resolution. Valid, unambiguous autosomal inputs preserve canonical row order.
This input-contract change is intentional and distinct from changing the model.

Historical fixed-multiplicity/lambda filtering is archived in
`legacy/postprocess.R`, with its current-mixture rejection guard intact. New runs
cannot invoke it. For a historical reproduction, use the exact old commit and
old inputs/intermediates, in a separate directory; do not feed current outputs to
that script or imply that a historical result validates this estimator.
