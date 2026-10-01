'''----------------------------------------------------------------------
This script takes care of the running of CliPP in case subsampling is needed
Usually you will need to run CliPP on HPC;
the preprocess script makes the input anonymous, so that you can up load them to HPC
Authors: Kaixian Yu, Yujie Jiang
Date: 04/02/2021
Email: yujiejiang679@gmail.com
----------------------------------------------------------------------
This script takes the following argument: path_to_input path_to_output path_to_clipp No_subsampling Rep_num window_size overlap_size lam
-----------------------------------------------------------------------
Debug use
sys.argv = ['/Users/kaixiany/Working/CliPP/Sample_data/intermediate/', '/Users/kaixiany/Working/CliPP/Sample_data/results/', '/Users/kaixiany/Working/CliPP/', '1.5', '1500', '1', '0.05', '0']
'''
import os
import numpy as np
from run_kernel_nosub import _prepare_chain, _run_kernel


def run_clipp_sub(prefix, preliminary_result, cluster_list, No_subsampling, rep, window_size, overlap):
    if No_subsampling < 1 or rep < 1:
        raise ValueError("Subsample size and replicate count must be positive.")
    if (not np.isfinite(window_size) or not np.isfinite(overlap) or
            not 0 < window_size <= 1 or not 0 <= overlap < window_size):
        raise ValueError("Require 0 <= overlap < window_size <= 1.")
    (r_all, n_all, major_all, total_all, purity), pilot, order = _prepare_chain(prefix, preliminary_result)
    count = len(r_all)
    size = min(No_subsampling, count)
    if max(cluster_list) > size:
        raise ValueError("Subsample size must support every requested cluster capacity.")
    rank = np.argsort(order)
    # Stratify by observed VAF; no multiplicity call is needed for sampling.
    # Each observation belongs to its first covering window, so overlap cannot
    # duplicate observations and zero-alt rows are retained.
    ending = np.minimum(np.arange(window_size, 1 + window_size - overlap,
                                  window_size - overlap), 1.0)
    bins = np.searchsorted(ending, r_all / n_all, side="left")
    groups = [np.flatnonzero(bins == i) for i in np.unique(bins)]
    quotas = np.array([len(group) * size / count for group in groups])
    take = np.floor(quotas).astype(int)
    remainder = size - int(take.sum())
    quota_order = np.argsort(-(quotas - take), kind="stable")
    take[quota_order[:remainder]] += 1
    os.makedirs(preliminary_result, exist_ok=True)
    for j in range(1, rep + 1):
        rng = np.random.RandomState(j)
        sample_index = np.concatenate([
            rng.choice(group, amount, replace=False)
            for group, amount in zip(groups, take) if amount
        ])
        sample_index = sample_index[np.argsort(rank[sample_index])]
        np.savetxt(os.path.join(preliminary_result, "sample_indices_rep%d.txt" % j),
                   sample_index, fmt="%d")
        _run_kernel(r_all[sample_index], n_all[sample_index], major_all[sample_index],
                    total_all[sample_index], purity, pilot[sample_index], preliminary_result, cluster_list)
        for capacity in cluster_list:
            for suffix, extension in (("phi", "txt"), ("label", "txt"), ("fit", "tsv")):
                old = os.path.join(preliminary_result, "K%d_%s.%s" % (capacity, suffix, extension))
                new = os.path.join(preliminary_result, "K%d_%s_rep%d.%s" % (capacity, suffix, j, extension))
                os.replace(old, new)
