# CliPP1.5
Clonal structure identification through penalizing pairwise differences

This repository contains the current chain-based CliPP1.5 implementation.
The native library remains named `CliPP`, and the entry point is
`run_clipp_main.py`.

## Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setting up CliPP](#setting-up-clipp)
- [Structure of CliPP implementation](#structure-of-clipp-implementation)
- [Input data sample](#input-data-sample)
- [Running CliPP with one-step implementation](#running-clipp-with-one-step-implementation)
- [The CliPP Outputs](#the-clipp-outputs)
- [Citation](#citation)

## Introduction
Subpopulations of tumor cells characterized by mutation profiles may confer differential fitness and consequently influence prognosis of cancers. Understanding subclonal architecture has the potential to provide biological insight in tumor evolution and advance precision cancer treatment. Recent methods comprehensively integrate single nucleotide variants (SNVs) and copy number aberrations (CNAs) to reconstruct subclonal architecture using whole-genome or whole-exome sequencing (WGS, WES) data from bulk tumor samples. However, the commonly used Bayesian methods require a large amount of computational resources, a prior knowledge of the number of subclones, and extensive post-processing. Regularized likelihood modeling approach, never explored for subclonal reconstruction, can inherently address these drawbacks. We therefore propose a model-based method, Clonal structure identification through Pair-wise Penalization, or CliPP, for clustering subclonal mutations without prior knowledge or post-processing. The CliPP model is applicable to genomic regions with or without CNAs. CliPP demonstrates high accuracy in subclonal reconstruction through extensive simulation studies. A penalized likelihood framework for subclonal reconstruction will help address intrinsic drawbacks of existing methods and expand the scope of computational analysis for cancer evolution in large cancer genomic studies. Also see our paper: https://www.biorxiv.org/content/10.1101/2021.03.31.437383v2.


### Multiplicity, chain penalty and BIC

Multiplicity is marginalized over every integer `m = 1, ..., major_cn`, with
uniform prior `1/major_cn`; preprocessing no longer rounds VAF into one call.
For cellular prevalence `CP` in `[0, purity]`, the mutation likelihood is
`L_i(CP) = sum_m Binomial(alt_i; depth_i, CP*m/D_i) / major_cn_i`, where
`D_i = purity*total_cn_i + 2*(1-purity)`.

The active fitter replaces the complete graph and SCAD penalty with a fixed
chain and a squared distance-to-set penalty. Estimate a pooled CP distribution
from the same multiplicity-marginal likelihood using a fixed grid and an
unsupervised, safeguarded accelerated EM initializer. Grid-weight convergence
requires a concave-simplex duality gap; a small likelihood change alone is not
convergence, and budget exhaustion remains explicit. Sort mutations once by posterior-mean pilot CP,
breaking ties by chromosome and position strings. This initialization resolves
multiplicity aliases using the tumor's observations; it uses neither simulation
truth nor a known cluster count. Its weights do not enter the chain objective.
For the adjacent-difference operator `D` and discrete cluster-count
hyperparameter `K`, hold `K` fixed during each fit and minimize

```text
C_K = {z : number of nonzero entries in z <= K-1}
F(CP) = -sum_i log L_i(CP_i) + rho_CP/2 * ||D CP - projection_C_K(D CP)||^2
```

`K` defines the allowed number of chain breaks; it is selected in an outer
comparison over `K=1,...,min(10,N)`. The numerical continuation parameter
`rho_CP` strengthens enforcement of the distance-to-set constraint during each
fixed-`K` fit. These roles are separate. Because `C_K` allows **at most** `K-1`
breaks, the fitted number of occupied blocks `q` can be smaller than `K`.
Diagnostics therefore retain both the requested hyperparameter `K` and the
fitted count `q`. Unsupported mixture components require explicit adjacent
merges and fresh fits within the same at-most-`K` budget before publication.
Selecting `K` does not add an extra continuous parameter to the BIC dimension.

The projection retains the `K-1` largest absolute adjacent differences, with
chain-edge order breaking ties. The graph has `N-1` edges and uses linear graph
storage. Internal calculations can use `CCF = CP/purity`; then
`rho_CP = rho_CCF/purity^2` gives the same objective. The chain is never resorted
during optimization. The MM update holds the projected edge support fixed and
penalizes only the other edges. This is an upper bound touching the same
distance-to-set objective, without artificial coupling across free edges.
No monotonicity or required clonal cluster is imposed.

By default, capacities `K=1,...,min(10,N)` are compared. Projection/MM and a
finite penalty continuation propose contiguous blocks; a finite penalty fit
is not an exact constrained or global optimum. Raw optimization uses an explicit
numerical CCF box `[1e-8, 1-1e-8]`; final block refits include the exact physical
endpoints 0 and 1. Each proposed partition is refitted without the penalty.
Exactly equal neighboring refitted CPs coalesce; equal centers separated along
the chain remain distinct blocks. Continuation checks both the adjacent-edge
residual and the maximum within-block CCF range, preventing long shallow ramps
from masquerading as fused blocks. A cancellation-resistant tridiagonal solve
and objective-checked feasible warm starts improve enforcement. The finite
iteration/penalty ceiling and floating-point limit remain explicit candidate
statuses; raw CPs and their diagnostics always describe the same point.
Before selection, the proposal bank also
includes every adjacent-block coarsening of the native partitions. A coarsening
with `q` proposed blocks is evaluated only when `K=q` was requested. Native
proposals are retained as diagnostics. The native partition and best bank
partitions additionally seed alternating conditional block refits and exact
fixed-center boundary optimization over all chain positions, in `O(Nq)` per
boundary update. Component visit order is preserved, including nonmonotone CP
sequences. This minimizes the same zero-distance conditional objective; it
does not replace it with unrestricted mixture optimization. The best admissible
baseline BIC is retained even if conditional polishing worsens that score.
No nonadjacent merges, truth labels, or mixture-based mutation reassignment
enter this search. Candidates are ranked
by the existing observed-data BIC-form score:

```text
logL_observed = sum_i log(sum_k weight_k * L_i(CP_k))
BIC = -2*logL_observed + (2*occupied_clusters - 1)*log(N)
```

At the candidate's refitted centers, cluster weights maximize the observed
likelihood on the probability simplex. The parameter count includes the CP
centers and `q-1` free weights. Weight fitting checks both the simplex duality
gap and positive-coordinate stationarity. An exactly zero fitted weight makes
that occupied-component candidate ineligible for publication; adjacent merges
touching unsupported blocks are actually refitted and rescored, without a
minimum cluster size or positive-weight floor. Rejected candidates remain in
`chain_candidates.tsv`. Weights never reassign mutation memberships.
The old fixed-label likelihood is retained as a diagnostic, not used for BIC:
scoring noisy, data-selected labels as known assignments rewards overclustering.
Here `N` counts retained positive-depth mutations, not reads or multiplicity
states. A capacity may yield fewer occupied blocks; BIC counts the blocks
actually published. This scores the chain candidates; their conditional CP
refits are not a joint maximum of the observed mixture likelihood.
Consequently this is a BIC-form selection criterion for the constrained
estimator, not a claim of ordinary mixture maximum-likelihood fitting or a
globally optimal constrained solution. Nonadjacent repeated centers are
reported by `distinct_centers`; their nominal block parameter count is retained,
and regular-mixture BIC asymptotics do not apply at that singular fit.
Independent scalar refits search a dense grid plus component modes and refine
all bracketed maxima and both endpoint-adjacent intervals, while retaining exact
physical endpoints. This is a numerical multimode search, not a certificate
of global optimality. Diagnostics distinguish the raw penalty fit from the
published fixed-partition refit.

Conditional block refitting is the default and only supported center estimator.
The optional `--center-refit conditional` argument is retained for explicit
invocations. Joint observed-mixture center refitting has been removed.
Observed-mixture likelihood and fitted cluster weights remain the BIC scoring
rule described above; they do not reassign mutation memberships.

At most `2^(K-1)` cut subsets arise from one native `K`-block proposal (512 at
K=10); boundary polishing adds new cut locations and has its own finite budget
and convergence diagnostics. Unsupported-component repairs may reduce `q`
within a requested `K`, including when only one `K` was requested. Interval center
refits are cached with compact interval keys, and a 32 MiB LRU bounds cached
likelihood columns. This does not bound the likelihood model's own memory or
make refitting inexpensive for very large inputs or high copy number. Numerical
failures of additional coarsenings are recorded while viable native proposals
are preserved; native refit failures still fail the fit. Better BIC or cluster
count does not guarantee better mutation membership.

For subsampling, the full chain remains fixed. Each sampled-block boundary
extends to the midpoint between neighboring sampled ranks on that chain.
The resulting full-data blocks are refitted and scored on the same `N`
mutations. Sample size must support every requested cluster capacity.

Rebuild the shared library and rerun preprocessing after updating. `major.txt`
contains multiplicity support bounds and `multiplicity_model.txt` identifies
that likelihood. The fourth column of the historical `multiplicity.txt`
coordinate index now contains major CN, not a multiplicity call. Old native
libraries cannot run the new chain interface. Use the one-step runner for
refitting and selection; `src/postprocess.R` is historical fixed-multiplicity
code and rejects new intermediate files.

## Prerequisites
- MacOS or Linux
  - If MacOS, CliPP only runs with one core due to a lack of support of OpenMP on this OS.
  - If Linux, CliPP requires OpenMP for paralell computing. For instruction on how to install OpenMP, please check the instruction at http://bioinformatics.mdanderson.org/Software/DeMixT/HowtoinstallOpenMP.docx.
- R [>3.3.1]
- R package `data.table` (`install.packages("data.table")`)
- python [>3.5.1]
- NumPy
- SciPy
- pandas
- Optional CUDA acceleration on Linux requires an NVIDIA GPU and driver (`libcuda`), plus the `nvidia-cuda-runtime-cu12` and `nvidia-cuda-nvrtc-cu12` Python packages.

## Setting up CliPP

### Manual Install
```
git clone https://github.com/yuadamding/CliPP1.5.git
cd CliPP1.5

python setup.py build
```

For optional CUDA acceleration on Linux:
```bash
python -m pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
CLIPP_USE_CUDA=1 python setup.py build
```

CUDA is otherwise detected automatically at build time. Use `CLIPP_USE_CUDA=0 python setup.py build` to force a CPU-only build.


### Docker container
We also include a `Dockerfile` in the repository. The user can build and run a Docker contrainer of CliPP to avoid any issues caused by package dependencies. The following provides a tutorial about this.

**CUDA note**: The provided `Dockerfile` builds the CPU backend; use the manual installation steps above for CUDA.

- Download and install `Docker` (https://docs.docker.com/get-docker/).
- Download the `Dockerfile` from this repository. Alternatively, you can clone the whole repository to your machine.
- Open a terminal on your machine and change directory to the folder of `Dockerfile`. Create a Docker container for CliPP using the command:

  ```
  docker build -t clipp .
  ```

- Use the sample data in the repository as an example to show how the run the docker container:

  ```
  git clone https://github.com/yuadamding/CliPP1.5.git
  cd CliPP1.5
  docker run -v $(pwd):/Sample clipp python3 /CliPP/run_clipp_main.py -i /Sample/test /Sample/sample/sample.snv.txt /Sample/sample/sample.cna.txt /Sample/sample/sample.purity.txt
  ```

**Note**: `-v $(pwd):/Sample` is to mount the the current directory in the host (the machine itself) to the container's directory of `/Sample`, so the input data in the current directory `sample/sample.cna.txt`, `sample/sample.cna.txt` and `sample/sample.purity.txt` can be seen by the container through `/Sample/sample/sample.snv.txt`, `/Sample/sample/sample.cna.txt`, `/Sample/sample/sample.purity.txt`. The output folder is located at `/Sample/test` in the container, which can be accessed at `$(pwd)/test` in the host machine.

<ins>The installation normally takes a few seconds to 1 minute.</ins>

## Structure of CliPP implementation
The one-step runner combines R input preprocessing, Python marginal pilots and
BIC refitting, and the native chain distance-to-set fitter. CPU and optional
CUDA modes share the chain projection and bounded tridiagonal updates; CUDA
accelerates batch likelihood evaluations. Source files are under `src/`.
Subsampling is optional and preserves the fixed full-chain ordering.

For the one-step runner, a CUDA-enabled build uses CUDA automatically when more than 1,000 SNVs remain after preprocessing and a CUDA device is available; otherwise, CliPP uses the CPU backend. Set `CLIPP_FORCE_CPU=1` to force the CPU backend at runtime.

## Input data sample
There are three required input files:

1. ```sample.snv.txt```: A tab separated file which stores a data matrix with the following named columns:
* ```chromosome_index```: The chromosomal location of the SNV.
* ```position```: the single-nucleotide position of the SNV on the corresponding chromosome.
* ```ref_count```: The number of reads covering the locus and containing the reference allele.
* ```alt_count```: The number of reads covering the locus and containing the alternative allele.

2. ```sample.cna.txt```: A tab separated file which stores a data matrix with the following named columns:
* ```chromosome_index```: The chromosome location of the CNA.
* ```start_position```: the start position of the CNA segment on the corresponding chromosome.
* ```end_position```: the end position of the CNA segment on the corresponding chromosome.
* ```major_cn```: The copy number of the major allele in tumor cells. This should be greater than equal to the value in the minor_cn column and greater than 0.
* ```minor_cn```: The copy number of the minor allele. This must be less than equal the value in the major_cn column.
* ```total_cn```: The sum of major_cn and minor_cn.

3. ```sample.purity.txt```: A file storing a scalar purity value between 0 and 1.

A simulated sample input data is under `sample/`.

## Running CliPP with one-step implementation

The caller function `run_clipp_main.py` wraps up the CliPP pipeline and enables users to implement subclonal reconstruction in one-step. To try CliPP with our sample input, you may run:
```
python run_clipp_main.py sample/sample.snv.txt sample/sample.cna.txt sample/sample.purity.txt
```

A full manual is as follows:

```
usage: run_clipp_main.py [-h] [-i SAMPLE_ID] [-p PREPROCESS] [-b] [-f FINAL] [--clusters K | --max-clusters K]
                        [-s SUBSAMPLE_SIZE] [-n REP_NUM] [-w WINDOW_SIZE] [-o OVERLAP_SIZE]
                        snv_input cn_input purity_input

positional arguments:
  snv_input             Path/Filename of the snv input.
  cn_input              Path/Filename of the copy number input.
  purity_input          Path/Filename of the purity input.


optional arguments:
  -h, --help            show this help message and exit
  -i SAMPLE_ID, --sample_id SAMPLE_ID
                        Name of the sample being processed. Default is 'sample_id'.
  --clusters K, --K K   Fit only this fixed cluster capacity (1..10, K <= N).
  --max-clusters K      Compare capacities 1 through min(K,N) by the BIC-form score.
                        Default and maximum are 10. This replaces lambda selection.
  -b, --subsampling     Whether doing subsampling or not. Default is not doing the subsampling,
                        and a flag -b is needed for subsampling.
  -p PREPROCESS, --preprocess PREPROCESS
                        Directory that stores the preprocess results. Default name is 'preprocess_result/'.
  -f FINAL, --final FINAL
                        Directory that stores the final results after postprocessing. Default name
                        is 'final_result/'.
```

The followings parameters are only needed when doing subsampling. We take partitions from 0 to 1 (determined by the `WINDOW_SIZE` parameter, default at `0.05`), then stratify SNVs by observed VAF and sample within those windows without duplicate observations. VAF is used only for sampling, not as a multiplicity or CP call. The number of sampled SNV is proportional to number of total SNVs belonging to each window.
```
  -s SUBSAMPLE_SIZE, --subsample_size SUBSAMPLE_SIZE
                        (Required if doing subsampling) The number of SNVs you want to include in
                        each subsamples. We use 40,000 for 256GB memory.
  -n REP_NUM, --rep_num REP_NUM
                        (Required if doing subsampling) The number of random subsamples needed.
  -w WINDOW_SIZE, --window_size WINDOW_SIZE
                        Controls the length of the window. Takes value between 0 and 1. Default
                        is 0.05.
  -o OVERLAP_SIZE, --overlap_size OVERLAP_SIZE
                        Controls the overlapped length of two consecutive windows. Takes value
                        between 0 and 1. Default is 0.
```


## The CliPP Outputs
By default, all outputs will be stored in the folder named `sample_id`, and this name can be changed with the `-i` or `--sample_id ` option.

Clusters are numbered in descending refitted cellular prevalence. Cluster 0 has the highest CP; it is not forced to be clonal. CP retains the original convention: cancer cell fraction (CCF) is `CP / purity`.

The final result for CliPP is two-fold:
* The subclonal structure, i.e., clustering results: cluster number, the total number of SNVs in each cluster, and the estimated CP for each cluster.
* The mutation assignment, i.e., cluster id for each mutation. This output can then serve as the basis for inference of phylogenetic trees.

`final_result/bic_selection.tsv` records each requested candidate's likelihood,
conditional likelihood, cluster count, parameter count, BIC definition, weight
optimality gap and selection status. It retains the native proposal's BIC next
to the selected proposal's score and identifies native versus derived candidates.
`chain_candidates.tsv` records the complete scored proposal bank, failures,
source capacity/replicate, and proposal/final partition hashes. For a derived
candidate, raw solver diagnostics are explicitly marked as `parent_raw_*`;
they do not certify the derived partition. Cluster tables include `mixture_weight` so
the observed-data score can be independently reconstructed. `Best_K/` contains the two
traditional result tables for the lowest-BIC candidate, plus
`posterior_multiplicity_KX.txt` with each mutation's posterior MAP multiplicity,
its probability and posterior mean, conditional on the refitted cluster CP.
Single-mutation runs compare K=1 and also receive a BIC score and a `Best_K/` result. Preprocessing excludes invalid or zero-depth read
counts, while retaining zero-alt and high-VAF observations with valid inputs.

To fit a single hyperparameter value, use `--clusters 4` (or `--K 4`). This runs
only the native K=4 fit; it does not silently search K=1..3 or force four
distinct blocks when centers coincide. Use `--max-clusters 10`, or omit both
options, for the outer K search. The two options are mutually exclusive.

### Demo output
By default, CliPP compares capacities 1 through 10, capped by mutation count.
Output filenames retain the requested capacity, for example
`subclonal_structure_K3.txt` and `mutation_assignments_K3.txt` when the capacity-3
candidate wins. The images below illustrate the traditional table formats;
they are not expected results of the new likelihood and chain penalty.

The first file outputs an overview of subclonal reconstruction result for this simulation sample, which is shown below. An index is assigned to each cluster, ranging from 0 to n, with clusters sorted in descending order based on their Cellular Prevalence (CP), such that the cluster with the highest CP is assigned an index of 0, the second highest an index of 1, and so on.
<img width="583" alt="subclonal_structure" src="https://github.com/wwylab/CliPP/assets/14543452/d10294a4-b76a-49ae-91bf-afa8860e164a">

The second file outputs a detailed list of mutation assignment result, i.e., for each mutation, which cluster is it assigned to. The figure below shows the top 10 rows of this mutation assignment table.
<img width="583" alt="mutation_assignments" src="https://github.com/wwylab/CliPP/assets/14543452/f61a0ba7-429f-4adf-911f-a722fdca8b1c">

Likelihood refitting adds computation beyond the penalized fit; runtime depends on mutation count, copy-number support and the candidate partitions.

## Citation
If you are using this framework, please cite our methods description on bioRxiv:
```
@article{CliPPmethod,
    title = {Scalable subclonal reconstruction of cancer cells in DNA sequencing data using a penalized likelihood model},
    author = {Yujie Jiang,  Matthew D Montierth, Yu Ding, Kaixian Yu, Quang Tran, Aaron Wu, Ruonan Li, Shuangxi Ji, Xiaoqian Liu, Seung Jun Shin, Shaolong Cao, Yuxin Tang, Tom Lesluyes, Marek Kimmel, Jennifer R. Wang, Maxime Tarabichi, Hongtu Zhu,  Peter Van Loo,  Wenyi Wang},
    journal = {bioRxiv},
    year = {2026},
    publisher = {Cold Spring Harbor Laboratory}
}
```
