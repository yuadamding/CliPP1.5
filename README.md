# CliPP1.5

CliPP1.5 clusters mutations from a single tumor sample and estimates cancer cell
fractions (CCFs) and mutation multiplicities.

## Install

Requires Python 3.12+ and a C++17 compiler. Run from the repository root.

**CPU:**

```bash
CLIPP_USE_CUDA=0 python -m pip install .
```

**GPU:** requires an NVIDIA GPU and driver providing `libcuda.so`.

```bash
python -m pip install 'setuptools>=77' wheel nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
CLIPP_USE_CUDA=1 python -m pip install --no-build-isolation .
```

## Run

```bash
# CPU
clipp fit snv.txt cna.txt purity.txt --device cpu --output ../clipp-cpu

# GPU (requires the GPU build)
clipp fit snv.txt cna.txt purity.txt --device cuda --output ../clipp-gpu
```

Use a new output directory for each run. `--max-clusters` defaults to **10**;
use a smaller value to limit the search.

## Input

Provide two tab-separated tables with headers and one purity file:

| File | Required contents |
| --- | --- |
| `snv.txt` | `chromosome_index`, `position`, `ref_count`, `alt_count`; optional unique `mutation_id` |
| `cna.txt` | `chromosome_index`, `start_position`, `end_position`, `major_cn`, `minor_cn`, `total_cn` |
| `purity.txt` | One number in `(0, 1]`, without a header |

Use autosomes 1–22, positive 1-based coordinates and inclusive, nonoverlapping
CNA intervals. Require `major_cn >= 1`, `0 <= minor_cn <= major_cn`, and
`total_cn = major_cn + minor_cn`. See [example inputs](sample/).

## Output

The output directory contains these tables under `final_result/`:

| File | Contents |
| --- | --- |
| `mutations.tsv` | Mutation cluster assignments and multiplicity estimates |
| `clusters.tsv` | Cluster sizes and CCFs |

Cluster 0 has the highest estimated CCF. Input exclusions are recorded in
`preprocess_result/input_ledger.tsv`.

## Contact

For questions or concerns, please submit an issue or email yding4@mdanderson.org.
