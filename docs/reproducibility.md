# Reproducibility, configuration and timing

Package version, model, initialization, candidate search, scoring, input/output
schema, subsampling and native ABI have separate identities. Native build identity
binds the compiled sources, Python/R sources, compiler command, flags, Python
version and backend choice. The exact library filename/hash is package-owned;
mtime and unrelated libraries cannot influence loading. A modified Python/R
source requires rebuilding. A source archive without Git metadata reports a null
commit and retains exact source hashes; it never invents a commit.

Build CPU explicitly with `CLIPP_USE_CUDA=0 python -m pip install .` (CPU is also
the deterministic build default). CUDA requires explicit `CLIPP_USE_CUDA=1`,
CUDA 12 runtime/NVRTC headers/libraries and a linkable driver library in the build
environment; use `--no-build-isolation` only in a prepared CUDA build environment.
It does not require PyTorch. This NVRTC backend is retained from the reviewed
source and remains unqualified for this revised package.

Boolean variables accept case-insensitive `1/0`, `true/false`, `yes/no`, `on/off`.
Missing or empty means unspecified; whitespace/unknown values are errors. Python,
build configuration and native dispatch share these semantics. Runtime
`CLIPP_FORCE_CPU=true` and `CLIPP_REQUIRE_CUDA=true` conflict. Explicit `--device`
conflicts fail before fitting. A CUDA request can never silently become CPU.
Auto mode uses CPU for <=1,000 retained SNVs; otherwise it probes CUDA. Unavailable
CUDA can fall back before substantive work; a runtime/compilation failure cannot.
GPU manifests retain actual device, driver/runtime/NVRTC and compute capability.

Subsampling uses largest-remainder VAF-bin quotas, no replacement, the first
covering window and recorded indices. `--seed 0` preserves the old replicate-j
seed `j`; general seed = base seed + 1-based replicate, using NumPy RandomState
MT19937. Seed + replicate must fit uint32. Full-data pooled initialization happens
**before** subsampling. Rank midpoints expand boundaries; every candidate is
refitted and scored on all retained rows, with the same full-data N.

Example:

```bash
clipp fit snv.tsv cna.tsv purity.txt --output new-subsampled-run \
  --device cpu --subsample-size 500 --replicates 5 --seed 0
```

Record thread settings when comparing times:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 clipp fit \
  snv.tsv cna.tsv purity.txt --device cpu --output new-timed-run
```

The native CPU likelihood loop itself is serial. NumPy/SciPy library threading
can still affect runtime. The API serializes simultaneous callers in one process
because native environment flags are process-global; use separate processes for
independent fits. External mutation of those environment flags during a call is
unsupported.

`COMPLETE.json` measures API entry through verified directory publication,
excluding the final completion-record write and Python import/startup time.
Stage fields include source validation, input copies, preprocessing, warmup,
initialization, native proposals, refitting/selection, verification and publication.
Report this total instead of the old native-stage timer. Process-tree RSS is
sampled every 20 ms, includes preprocessing children and is a sampled peak, not a
precise allocation bound. GPU peak VRAM is **null/unmeasured**, not zero. CUDA
still compiles NVRTC programs for warmup and fitting and performs synchronous
host/device transfers; stage time is not a kernel-only speed claim.
