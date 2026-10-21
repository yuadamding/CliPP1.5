"""Independent replay of published allocation diagnostics; no fitting calls."""

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, xlogy
from scipy.stats import binom

from ._io import read_table


def _require(condition, name):
    if not condition:
        raise ValueError("Verification failed: posterior " + name)


def _close(actual, expected, name, *, atol=2e-7, rtol=2e-9):
    try:
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        matches = actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=rtol)
    except (TypeError, ValueError):
        matches = False
    _require(matches, name)


def verify_posterior(
    data, purity, labels, centers, weights, directory, *, chain_to_component=None, marginal_multiplicity=False
):
    """Replay schema-1 production diagnostics from retained observations.

    The enclosing run verifier binds the input and selected fitted state. This
    function independently checks the four diagnostic artifacts, including
    row identities and reporting definitions. It uses scipy.stats binomial
    probabilities, never the model, posterior reporter, or an optimizer.
    """
    directory = Path(directory)
    labels, centers, weights = np.asarray(labels), np.asarray(centers), np.asarray(weights)
    n, k = len(data), len(centers)
    mapping = np.arange(k) if chain_to_component is None else np.asarray(chain_to_component)
    _require(mapping.ndim == 1 and mapping.size > 0 and np.issubdtype(mapping.dtype, np.integer)
             and np.all((mapping >= -1) & (mapping < k)), "proposal mapping")
    _require(
        n > 0 and centers.ndim == 1 and k > 0 and np.isfinite(centers).all()
        and 0 < purity <= 1 and np.all((centers >= 0) & (centers <= purity))
        and weights.shape == (k,) and np.isfinite(weights).all() and np.all(weights >= 0)
        and np.isclose(weights.sum(), 1, rtol=0, atol=5e-12)
        and labels.shape == (n,) and np.issubdtype(labels.dtype, np.integer)
        and np.all((labels >= 0) & (labels < len(mapping))),
        "selected parameters",
    )
    identities = ["chromosome_index", "position", "mutation_id", "original_row"]
    mutation_columns = [
        *identities, "proposal_chain_cluster_index", "posterior_map_cluster_index",
        "chain_label_posterior", "max_cluster_posterior", "membership_entropy",
        "expected_cancer_cell_fraction",
    ]
    if marginal_multiplicity:
        mutation_columns.extend([
            "marginal_multiplicity", "marginal_expected_multiplicity",
            "posterior_map_conditional_multiplicity", "posterior_map_conditional_expected_multiplicity",
        ])
    if chain_to_component is not None:
        mutation_columns.append("mapped_proposal_cluster_index")
    cluster_columns = [
        "cluster_index", "cellular_prevalence", "cancer_cell_fraction", "chain_num_SNV",
        "posterior_expected_num_SNV", "posterior_map_num_SNV", "mixture_weight",
        "fitted_mixture_num_SNV", "posterior_count_minus_mixture_count",
    ]
    flow_columns = ["chain_cluster_index", "posterior_cluster_index", "posterior_expected_num_SNV"]
    mutations = read_table(directory / "posterior_mutations.tsv")
    clusters = read_table(directory / "posterior_clusters.tsv")
    flow = read_table(directory / "posterior_flow.tsv")
    summary = json.loads((directory / "posterior_summary.json").read_text())
    _require(mutations.columns.tolist() == mutation_columns and len(mutations) == n, "mutation columns/rows")
    _require(clusters.columns.tolist() == cluster_columns and len(clusters) == k, "cluster columns/rows")
    _require(flow.columns.tolist() == flow_columns and len(flow) == len(mapping) * k, "flow columns/rows")
    for name in identities:
        _require(np.array_equal(mutations[name], data[name]), "mutation identity " + name)
    _require(np.array_equal(mutations.proposal_chain_cluster_index, labels), "proposal labels")
    mapped_labels = mapping[labels]
    mapped = mapped_labels >= 0
    orphan_count = int(np.count_nonzero(~mapped))
    if chain_to_component is not None:
        _require(np.array_equal(mutations.mapped_proposal_cluster_index, mapped_labels), "mapped proposal labels")
    _require(np.array_equal(clusters.cluster_index, np.arange(k)), "component indices")
    _require(np.array_equal(clusters.cellular_prevalence, centers), "fitted centers")
    _require(np.array_equal(clusters.mixture_weight, weights), "fitted weights")
    map_labels = mutations.posterior_map_cluster_index.to_numpy()
    _require(np.issubdtype(map_labels.dtype, np.integer) and np.all((map_labels >= 0) & (map_labels < k)),
             "MAP label support")
    _require(np.array_equal(flow.chain_cluster_index, np.repeat(np.arange(len(mapping)), k))
             and np.array_equal(flow.posterior_cluster_index, np.tile(np.arange(k), len(mapping))),
             "flow component indices")
    alt, depth, major, total = (data[name].to_numpy() for name in ("alt_count", "depth", "major_cn", "total_cn"))
    _require(
        all(np.isfinite(values).all() and np.all(values == np.floor(values)) for values in (alt, depth, major, total))
        and np.all((alt >= 0) & (depth > 0) & (alt <= depth) & (major >= 1) & (total >= major)),
        "observations",
    )
    log_weights = np.full(k, -np.inf)
    log_weights[weights > 0] = np.log(weights[weights > 0])
    expected, scores, computed_flow = np.zeros(k), np.zeros(k), np.zeros((len(mapping), k))
    if marginal_multiplicity:
        for name in ("marginal_multiplicity", "posterior_map_conditional_multiplicity"):
            calls = mutations[name].to_numpy()
            _require(np.issubdtype(calls.dtype, np.integer) and np.all((calls >= 1) & (calls <= major)),
                     "multiplicity support " + name)
    log_likelihood, entropy_sum = 0.0, 0.0
    # Limit temporary multiplicity arrays and component matrices independently.
    # Only the current component's valid-support binomial matrix is live.
    for cn in np.unique(major):
        group = np.flatnonzero(major == cn)
        support = np.arange(1, int(cn) + 1)
        size = max(1, 1_000_000 // max(int(cn), k))
        for first in range(0, len(group), size):
            rows = group[first : first + size]
            denominator = 2 * (1 - purity) + purity * total[rows]
            log_kernel = np.empty((len(rows), k))
            for component, cp in enumerate(centers):
                p = np.minimum(1, cp * support[None, :] / denominator[:, None])
                kernel = binom.logpmf(alt[rows, None], depth[rows, None], p) - np.log(cn)
                log_kernel[:, component] = logsumexp(kernel, axis=1)
            joint = log_kernel + log_weights
            normalizer = logsumexp(joint, axis=1)
            _require(np.isfinite(normalizer).all(), "positive mixture support")
            membership = np.exp(joint - normalizer[:, None])
            _close(membership.sum(axis=1), np.ones(len(rows)), "normalization", atol=5e-12, rtol=0)
            # Independently tied MAP choices may differ by final-bit arithmetic.
            chosen = membership[np.arange(len(rows)), map_labels[rows]]
            _require(np.all(membership.max(axis=1) - chosen <= 2e-12), "MAP assignments")
            entropy = -xlogy(membership, membership).sum(axis=1)
            chain_probability = np.zeros(len(rows))
            present = mapped[rows]
            chain_probability[present] = membership[np.flatnonzero(present), mapped_labels[rows[present]]]
            for name, values in (
                ("chain_label_posterior", chain_probability),
                ("max_cluster_posterior", membership.max(axis=1)),
                ("membership_entropy", entropy),
                ("expected_cancer_cell_fraction", membership @ (centers / purity)),
            ):
                _close(mutations[name].to_numpy()[rows], values, name, atol=2e-10)
            expected += membership.sum(axis=0)
            np.add.at(computed_flow, labels[rows], membership)
            with np.errstate(over="ignore"):
                scores += np.exp(log_kernel - normalizer[:, None]).sum(axis=0) / n
            log_likelihood += float(normalizer.sum())
            entropy_sum += float(entropy.sum())
            if marginal_multiplicity:
                marginal = np.zeros((len(rows), len(support)))
                for component, cp in enumerate(centers):
                    if weights[component] == 0:
                        continue
                    p = np.minimum(1, cp * support[None, :] / denominator[:, None])
                    kernel = binom.logpmf(alt[rows, None], depth[rows, None], p) - np.log(cn)
                    responsibility = np.exp(kernel + log_weights[component] - normalizer[:, None])
                    marginal += responsibility
                    selected = map_labels[rows] == component
                    if np.any(selected):
                        conditional = responsibility[selected] / membership[selected, component, None]
                        calls = mutations.posterior_map_conditional_multiplicity.to_numpy()[rows[selected]] - 1
                        _require(np.all(conditional.max(axis=1) - conditional[np.arange(len(calls)), calls] <= 2e-12),
                                 "conditional multiplicity MAP")
                        _close(mutations.posterior_map_conditional_expected_multiplicity.to_numpy()[rows[selected]],
                               conditional @ support, "conditional multiplicity expectation")
                _close(marginal.sum(axis=1), np.ones(len(rows)), "marginal normalization", atol=5e-12, rtol=0)
                calls = mutations.marginal_multiplicity.to_numpy()[rows] - 1
                _require(np.all(marginal.max(axis=1) - marginal[np.arange(len(rows)), calls] <= 2e-12),
                         "marginal multiplicity MAP")
                _close(mutations.marginal_expected_multiplicity.to_numpy()[rows], marginal @ support,
                       "marginal multiplicity expectation")
    _require(np.isfinite(scores).all(), "finite weight derivatives")
    counts = np.bincount(mapped_labels[mapped], minlength=k)
    proposal_counts, map_counts = np.bincount(labels, minlength=len(mapping)), np.bincount(map_labels, minlength=k)
    _require(np.array_equal(clusters.chain_num_SNV, counts), "hard counts")
    _require(np.array_equal(clusters.posterior_map_num_SNV, map_counts), "MAP counts")
    difference = expected - n * weights
    for name, values in (
        ("cancer_cell_fraction", centers / purity), ("posterior_expected_num_SNV", expected),
        ("fitted_mixture_num_SNV", n * weights), ("posterior_count_minus_mixture_count", difference),
    ):
        _close(clusters[name], values, name)
    _close(flow.posterior_expected_num_SNV, computed_flow.ravel(), "flow values")
    _close(computed_flow.sum(axis=0), expected, "flow column totals")
    _close(computed_flow.sum(axis=1), proposal_counts, "flow row totals")
    tolerance = 1e-8
    global_gap = n * max(0.0, float(scores.max() - 1))
    active_gap = float(scores.max() - scores[weights > 0].min())
    identity_error = float(np.abs(difference).max())
    identity_tolerance = n * tolerance + 1e-12
    ccf = centers / purity
    expected_summary = {
        "posterior_schema_version": 1, "num_mutations": n, "num_components": k,
        "log_likelihood": log_likelihood,
        "allocation_total_variation": float((np.abs(counts - expected).sum() + orphan_count) / (2 * n)),
        "posterior_map_disagreement_fraction": float(np.mean(map_labels != mapped_labels)),
        "mean_membership_entropy": entropy_sum / n,
        "weight_optimality_gap": global_gap, "weight_active_score_gap": active_gap,
        "weight_stationarity_passed": bool(global_gap <= n * tolerance and active_gap <= tolerance),
        "maximum_expected_count_identity_error": identity_error,
        "expected_count_identity_tolerance": identity_tolerance,
        "expected_count_identity_passed": bool(identity_error <= identity_tolerance),
        "non_top_cluster_fraction": float(1 - counts[int(np.argmax(centers))] / n),
        "clonality_definition": "none; highest-CCF component is a ranking, not biological clonality",
        "uncertainty_scope": "plug-in latent-state posteriors conditional on fitted parameters",
        "mixture_weight_meaning": "mass among analyzed mutations, not exclusive cancer-cell population fraction",
        "ccf_cumulative_mass": [
            {"ccf_threshold": float(threshold), "posterior_mass_below": float(expected[ccf < threshold].sum() / n)}
            for threshold in np.unique(ccf)
        ],
    }
    if chain_to_component is not None:
        expected_summary.update({
            "chain_to_component": mapping.tolist(), "num_proposal_chain_clusters": len(mapping),
            "unmapped_proposal_num_SNV": orphan_count, "proposal_chain_num_SNV": proposal_counts.tolist(),
        })
    _require(isinstance(summary, dict) and summary.keys() == expected_summary.keys(), "summary fields")
    for name, value in expected_summary.items():
        actual = summary[name]
        if isinstance(value, str) or type(value) is int:
            _require(type(actual) is type(value) and actual == value, "summary " + name)
        elif type(value) is bool:
            _require(type(actual) is bool, "summary boolean " + name)
        elif isinstance(value, list):
            if name != "ccf_cumulative_mass":
                _require(isinstance(actual, list) and all(type(item) is int for item in actual)
                         and actual == value, "summary " + name)
                continue
            _require(isinstance(actual, list) and len(actual) == len(value), "CCF sensitivity shape")
            for observed, calculated in zip(actual, value):
                _require(isinstance(observed, dict) and observed.keys() == calculated.keys(), "CCF sensitivity fields")
                for field in calculated:
                    _close(observed[field], calculated[field], "CCF sensitivity " + field, atol=2e-10)
        else:
            _require(type(actual) in (int, float), "summary numeric " + name)
            if name == "expected_count_identity_tolerance":
                _require(actual == value, "count identity tolerance")
            arithmetic = max(2e-10, 2e-12 * n) if name in (
                "weight_optimality_gap", "maximum_expected_count_identity_error",
            ) else 2e-10
            _close(actual, value, "summary " + name, atol=2e-7 if name == "log_likelihood" else arithmetic)
    # Flags must exactly describe their stored diagnostics. Independent replay
    # above permits only arithmetic tolerance, not a different qualification rule.
    _require(summary["weight_stationarity_passed"] == (
        summary["weight_optimality_gap"] <= n * tolerance and summary["weight_active_score_gap"] <= tolerance
    ), "weight stationarity flag")
    _require(summary["expected_count_identity_passed"] == (
        summary["maximum_expected_count_identity_error"] <= summary["expected_count_identity_tolerance"]
    ), "expected count identity flag")
    return expected_summary
