# Contributing and release gates

Changes remain under the existing AGPLv3 license. Preserve author attribution.
Separate scientific model/search/score changes from engineering repairs. Do not
update golden outputs merely to make a regression disappear. Explain any changed
input or output contract and increment its scoped identity.

## Local development

Use an isolated environment with Python, a C++17 compiler, R/data.table, and the
project's test dependencies. Editable installation is not required; the following
source-test route rebuilds identity after Python/R/native changes:

```bash
python -m pip install '.[test]'
CLIPP_USE_CUDA=0 python setup.py build_ext --inplace
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest -q
python -m ruff check src/clipp tests benchmarks setup.py run_clipp_main.py
python -m compileall -q src/clipp tests benchmarks run_clipp_main.py
git diff --check
CLIPP_USE_CUDA=0 python -m build
```

`setup.py build_ext --inplace` is a development rebuild, not an installation
recipe. Production users install wheels or use `pip install .`. Check wheels and
sdists in fresh environments outside the checkout, including read-only installed
package files and new external output directories. CPU CI runs these contracts;
check its real status before claiming the Python matrix passed.

CUDA tests require an authorized allocated device and a prepared CUDA build:

```bash
CLIPP_USE_CUDA=1 python -m pip install --no-build-isolation .
CLIPP_TEST_CUDA=1 python -m pytest -q tests/test_cuda.py
```

CPU skips are not CUDA qualification. Retain build/device/runtime identity and
paired full-fit outputs. Container qualification must build from a complete
source archive, execute a real fit as a non-root user with a writable external
volume, then verify the result. Save image ID/digest and dependencies. The current
Docker recipe has not passed that gate.

## Scientific evaluation

See `benchmarks/README.md`. Freeze the source/build/protocol before protected
validation. Retain failed attempts in the denominator. A development smoke panel
or improvement in score alone does not qualify accuracy on the three historical
cohorts. A new scientific policy needs its own identity, an independently
specified comparison and a protected comparison against the prior estimator.

## Release process

1. Complete CPU regression, fresh wheel and sdist tests; retain logs and artifact
   SHA-256 values. Confirm source-only archive contents and AGPL license.
2. Record which Python/platform/backend/container combinations actually passed.
   Keep any unqualified combinations plainly experimental in the README.
3. Review model/input/output identities, documentation, changelog and citation.
4. Build from an immutable source commit. The manual artifact workflow produces
   wheels/sdist, checksums and dependency records; it does not publish to PyPI,
   push tags, or certify a release by itself.
5. An authorized maintainer may publish immutable artifacts after checking the
   evidence. Never replace an artifact under the same release identity.

Report defects with exact software/native build IDs, command/configuration,
`doctor` output, failure status and a minimal non-sensitive reproducer. Do not
attach private tumor data to public issues.
