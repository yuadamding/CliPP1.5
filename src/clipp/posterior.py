"""Plug-in allocation diagnostics, without changing a fitted state or its labels.

Multiplicity support is evaluated in the model's bounded, equal-CN blocks.
The optional membership matrix is N by K; no N by K by maximum-CN tensor is
constructed. These are latent-state probabilities at fixed fitted parameters,
not uncertainty intervals over those parameters, purity, CN, or component count.
"""

import numpy as np
from scipy.special import logsumexp, xlogy

from ._reduction import grouped_logsumexp


def _parameters(model, centers, weights):
    centers = np.asarray(centers, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        centers.ndim != 1
        or not centers.size
        or np.any(~np.isfinite(centers))
        or np.any(centers < 0)
        or np.any(centers > model.purity)
        or weights.shape != centers.shape
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, rtol=0, atol=5e-12)
    ):
        raise ValueError("Expected centers in [0, purity] and nonnegative simplex weights")
    return centers, weights


def _blocks(model, centers, weights):
    # The model reserves sixteen state arrays. Only one component's state
    # matrix is live at a time; bound the additional row-by-component arrays.
    model._batch_size(1)
    positive = weights > 0
    log_weights = np.full(len(weights), -np.inf)
    log_weights[positive] = np.log(weights[positive])
    row_limit = max(1, model.workspace_bytes // (16 * 8 * len(centers)))
    for all_rows, all_scale in model._chunks(1):
        for start in range(0, len(all_rows), row_limit):
            rows = all_rows[start : start + row_limit]
            scale = all_scale[start : start + row_limit]
            log_kernel = np.column_stack([
                grouped_logsumexp(model._kernel(cp, rows, scale), len(model.m))
                for cp in centers
            ])
            log_joint = log_kernel + log_weights
            log_q = logsumexp(log_joint, axis=1)
            if np.any(~np.isfinite(log_q)):
                raise ValueError("Cluster posterior is undefined at zero mixture likelihood")
            membership = np.exp(log_joint - log_q[:, None])
            if not np.allclose(membership.sum(axis=1), 1.0, rtol=0, atol=5e-12):
                raise RuntimeError("Cluster posterior failed normalization")
            yield rows, scale, log_kernel, log_q, membership, log_weights


def posterior_diagnostics(
    model, centers, weights, chain_labels, *, marginal_multiplicity=False, membership_matrix=False,
    chain_to_component=None,
):
    """Return separate hard, expected, MAP and fitted-mass allocation summaries.

    ``chain_labels`` remains the proposal/production assignment and is never
    overwritten. Boundary weights are supported for experimental fits. Every
    row must have positive mixture likelihood. ``marginal_multiplicity`` also
    sums joint cluster--multiplicity responsibilities across components; it
    never substitutes a MAP multiplicity in the likelihood.
    """
    centers, weights = _parameters(model, centers, weights)
    labels = np.asarray(chain_labels)
    n, k = len(model), len(centers)
    mapping = np.arange(k) if chain_to_component is None else np.asarray(chain_to_component)
    if (mapping.ndim != 1 or not mapping.size or not np.issubdtype(mapping.dtype, np.integer)
            or np.any(mapping < -1) or np.any(mapping >= k)):
        raise ValueError("Proposal mapping must contain final component indices or -1 for removed components")
    if (
        labels.shape != (n,)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= len(mapping))
    ):
        raise ValueError("Chain labels must be integer component indices for every mutation")
    mapped_labels = mapping[labels]
    mapped = mapped_labels >= 0
    orphan_count = int(np.count_nonzero(~mapped))
    result = {
        "chain_num_SNV": np.bincount(mapped_labels[mapped], minlength=k),
        "posterior_expected_num_SNV": np.zeros(k),
        "posterior_map_num_SNV": np.zeros(k, dtype=int),
        "mixture_weight": weights.copy(),
        "fitted_mixture_num_SNV": n * weights,
        "posterior_map_cluster_index": np.empty(n, dtype=int),
        "chain_label_posterior": np.empty(n),
        "max_cluster_posterior": np.empty(n),
        "membership_entropy": np.empty(n),
        "expected_cancer_cell_fraction": np.empty(n),
        "allocation_flow": np.zeros((len(mapping), k)),
    }
    if chain_to_component is not None:
        result["mapped_proposal_cluster_index"] = mapped_labels.copy()
    if membership_matrix:
        result["membership_matrix"] = np.empty((n, k))
    if marginal_multiplicity:
        result["marginal_multiplicity"] = np.empty(n, dtype=int)
        result["marginal_expected_multiplicity"] = np.empty(n)
        result["posterior_map_conditional_multiplicity"] = np.empty(n, dtype=int)
        result["posterior_map_conditional_expected_multiplicity"] = np.empty(n)
    score = np.zeros(k)
    log_likelihood = 0.0
    for rows, scale, log_kernel, log_q, membership, log_weights in _blocks(model, centers, weights):
        map_labels = np.argmax(membership, axis=1)
        result["posterior_expected_num_SNV"] += membership.sum(axis=0)
        result["posterior_map_num_SNV"] += np.bincount(map_labels, minlength=k)
        result["posterior_map_cluster_index"][rows] = map_labels
        chain_probability = np.zeros(len(rows))
        present = mapped[rows]
        chain_probability[present] = membership[np.flatnonzero(present), mapped_labels[rows[present]]]
        result["chain_label_posterior"][rows] = chain_probability
        result["max_cluster_posterior"][rows] = membership.max(axis=1)
        result["membership_entropy"][rows] = -xlogy(membership, membership).sum(axis=1)
        result["expected_cancer_cell_fraction"][rows] = membership @ (centers / model.purity)
        np.add.at(result["allocation_flow"], labels[rows], membership)
        if membership_matrix:
            result["membership_matrix"][rows] = membership
        # This derivative includes zero-weight components, which can still
        # violate the simplex KKT condition even though their T column is zero.
        with np.errstate(over="ignore"):
            score += np.exp(log_kernel - log_q[:, None]).sum(axis=0) / n
        log_likelihood += float(log_q.sum())
        if marginal_multiplicity:
            marginal = np.zeros_like(scale)
            for component, cp in enumerate(centers):
                if weights[component] > 0:
                    joint = np.exp(
                        model._kernel(cp, rows, scale) + log_weights[component] - log_q[:, None]
                    )
                    marginal += joint
                    selected = map_labels == component
                    if np.any(selected):
                        conditional = joint[selected] / membership[selected, component, None]
                        result["posterior_map_conditional_multiplicity"][rows[selected]] = (
                            np.argmax(conditional, axis=1) + 1
                        )
                        result["posterior_map_conditional_expected_multiplicity"][rows[selected]] = (
                            conditional @ model.m[: scale.shape[1]]
                        )
            if not np.allclose(marginal.sum(axis=1), 1.0, rtol=0, atol=5e-12):
                raise RuntimeError("Marginal multiplicity posterior failed normalization")
            result["marginal_multiplicity"][rows] = np.argmax(marginal, axis=1) + 1
            result["marginal_expected_multiplicity"][rows] = marginal @ model.m[: scale.shape[1]]
    if np.any(~np.isfinite(score)):
        raise ValueError("Weight derivatives are nonfinite at the supplied mixture parameters")
    expected = result["posterior_expected_num_SNV"]
    difference = expected - result["fitted_mixture_num_SNV"]
    result["posterior_count_minus_mixture_count"] = difference
    global_gap = n * max(0.0, float(score.max() - 1.0))
    active_gap = float(score.max() - score[weights > 0].min())
    tolerance = 1e-8
    identity_tolerance = n * tolerance + 1e-12
    ccf = centers / model.purity
    result["summary"] = {
        "posterior_schema_version": 1,
        "num_mutations": n,
        "num_components": k,
        "log_likelihood": log_likelihood,
        "allocation_total_variation": float((np.abs(result["chain_num_SNV"] - expected).sum() + orphan_count) / (2 * n)),
        "posterior_map_disagreement_fraction": float(np.mean(result["posterior_map_cluster_index"] != mapped_labels)),
        "mean_membership_entropy": float(result["membership_entropy"].mean()),
        "weight_optimality_gap": global_gap,
        "weight_active_score_gap": active_gap,
        "weight_stationarity_passed": bool(global_gap <= n * tolerance and active_gap <= tolerance),
        "maximum_expected_count_identity_error": float(np.abs(difference).max()),
        "expected_count_identity_tolerance": identity_tolerance,
        "expected_count_identity_passed": bool(np.abs(difference).max() <= identity_tolerance),
        "non_top_cluster_fraction": float(1 - result["chain_num_SNV"][int(np.argmax(centers))] / n),
        "clonality_definition": "none; highest-CCF component is a ranking, not biological clonality",
        "uncertainty_scope": "plug-in latent-state posteriors conditional on fitted parameters",
        "mixture_weight_meaning": "mass among analyzed mutations, not exclusive cancer-cell population fraction",
        # The strict inequality is intentional: S(t) = sum(T * [CCF < t])/N.
        "ccf_cumulative_mass": [
            {"ccf_threshold": float(threshold), "posterior_mass_below": float(expected[ccf < threshold].sum() / n)}
            for threshold in np.unique(ccf)
        ],
    }
    if chain_to_component is not None:
        result["summary"].update({
            "chain_to_component": mapping.tolist(),
            "num_proposal_chain_clusters": len(mapping),
            "unmapped_proposal_num_SNV": orphan_count,
            "proposal_chain_num_SNV": np.bincount(labels, minlength=len(mapping)).tolist(),
        })
    return result
