# Contributing

Preserve AGPLv3 attribution. Keep scientific model/search/score changes separate
from engineering changes. Model and output contracts have explicit identities.

```bash
python -m pip install '.[dev]'
python -m ruff check src/clipp setup.py
python -m compileall -q src/clipp
git diff --check
CLIPP_USE_CUDA=0 python -m build
```

Rebuild after package/native source changes. Install wheels and source distributions
in separate clean environments, then run `clipp doctor`, `clipp validate`,
`clipp fit --device cpu` and `clipp verify` on the supplied sample outside the
checkout, with a new output directory. Use the input paths and `--output` option
shown by `clipp fit --help` and the README. For source development, rebuild with
`CLIPP_USE_CUDA=0 python setup.py build_ext --inplace`.

## Validation scope

Current CI checks lint, compilation, wheel/sdist installation and sample fitting
and verification on Python 3.12 and 3.13. It does not run a numerical regression
suite: the docs, tests and benchmark directories were deliberately removed after
archiving their evidence. The manual artifact workflow also uses these smoke gates
before uploading artifacts; it does not publish a package or tag.

The archived pre-removal validation passed 180 tests with six CUDA skips, installed
wheel suites on Python 3.12/3.13, and 240 exact capacity comparisons across 24 small
engineering cases. The numerical source remains unchanged by the directory removal.
These are historical engineering results, not broad-cohort accuracy claims.

The local, unpublished evidence bundle is
`/storage/CliPP2/clipp15_readiness_19f310b_20261002/validation-capsule.tar.gz`, SHA-256
`db1036435bfc4d40d49d0e4ecbb12dcaa9d73fe0e8c4fb50a038db8517c929b9`.
It contains the original regression fixture, Python/R migration evidence,
source-bound outputs, failures and reproduction instructions. This machine-local
archive is not a public download or the current reduced source distribution.
Do not regenerate a golden fixture to conceal a regression.

Python 3.10 with SciPy 1.15.3 failed candidate-partition equivalence; version 1.6.0
requires Python 3.12+ and SciPy 1.18.x. CUDA and container execution remain
unqualified. A sample fit cannot establish global optimization, calibrated
uncertainty or biological accuracy.

Publish only immutable source-bound artifacts after their exact gates pass; keep
previous artifacts and failures. Report bugs with exact source/native build IDs,
configuration, `doctor` output and a minimal non-sensitive input.
