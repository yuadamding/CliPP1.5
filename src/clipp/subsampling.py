import os
import numpy as np
from .kernel import _run_kernel


def run_clipp_sub(
    prepared,
    preliminary_result,
    cluster_list,
    No_subsampling,
    rep,
    window_size,
    overlap,
    seed=0,
    *,
    library=None,
):
    if No_subsampling < 1 or rep < 1:
        raise ValueError("Subsample size and replicate count must be positive.")
    if (
        not np.isfinite(window_size)
        or not np.isfinite(overlap)
        or not 0 < window_size <= 1
        or not 0 <= overlap < window_size
    ):
        raise ValueError("Require 0 <= overlap < window_size <= 1.")
    (r_all, n_all, major_all, total_all, purity), pilot, order = prepared
    count = len(r_all)
    size = min(No_subsampling, count)
    if max(cluster_list) > size:
        raise ValueError("Subsample size must support every requested cluster capacity.")
    rank = np.argsort(order)
    # Stratify by observed VAF; no multiplicity call is needed for sampling.
    # Each observation belongs to its first covering window, so overlap cannot
    # duplicate observations and zero-alt rows are retained.
    ending = np.minimum(np.arange(window_size, 1 + window_size - overlap, window_size - overlap), 1.0)
    bins = np.searchsorted(ending, r_all / n_all, side="left")
    groups = [np.flatnonzero(bins == i) for i in np.unique(bins)]
    quotas = np.array([len(group) * size / count for group in groups])
    take = np.floor(quotas).astype(int)
    remainder = size - int(take.sum())
    quota_order = np.argsort(-(quotas - take), kind="stable")
    take[quota_order[:remainder]] += 1
    os.makedirs(preliminary_result, exist_ok=True)
    for j in range(1, rep + 1):
        rng = np.random.RandomState(seed + j)
        sample_index = np.concatenate(
            [rng.choice(group, amount, replace=False) for group, amount in zip(groups, take) if amount]
        )
        sample_index = sample_index[np.argsort(rank[sample_index])]
        np.savetxt(os.path.join(preliminary_result, "sample_indices_rep%d.txt" % j), sample_index, fmt="%d")
        _run_kernel(
            r_all[sample_index],
            n_all[sample_index],
            major_all[sample_index],
            total_all[sample_index],
            purity,
            pilot[sample_index],
            preliminary_result,
            cluster_list,
            library=library,
        )
        for capacity in cluster_list:
            for suffix, extension in (("phi", "txt"), ("label", "txt"), ("fit", "tsv")):
                old = os.path.join(preliminary_result, "K%d_%s.%s" % (capacity, suffix, extension))
                new = os.path.join(preliminary_result, "K%d_%s_rep%d.%s" % (capacity, suffix, j, extension))
                os.replace(old, new)
