"""Versioned scientific and serialization contracts, independent of packaging."""

from . import __version__

IDENTITIES = {
    "software_version": __version__,
    "model_version": "uniform_1_to_major_v1",
    "estimator_version": "conditional_fixed_chain_distance_to_set_v1",
    "initialization_version": "pooled_grid_posterior_mean_v1",
    "candidate_search_version": "native_chain_boundary_polish_supported_weights_v2",
    "scoring_version": "observed_cluster_multiplicity_2K_minus_1_v1",
    "output_schema_version": 2,
    "native_abi_version": 3,
    "input_schema_version": "autosomal_inclusive_intervals_v1",
    "subsampling_version": "vaf_largest_remainder_mt19937_seed_plus_rep_v1",
}

# These are the existing constants, not new fitting controls.
NUMERICS = {
    "native": {
        "levels": 20,
        "iterations_per_level": 300,
        "stationarity_tolerance": 1e-6,
        "constraint_tolerance": 1e-6,
        "maximum_rho": 1e12,
        "ccf_bounds": [1e-8, 1 - 1e-8],
    },
    "scalar_refit": {"grid_size": 513, "xatol": 1e-12, "maxiter": 500},
    "weights": {"mean_gap_tolerance": 1e-8},
    "boundary_refinement": {"max_iterations": 100, "mean_loglik_tolerance": 1e-10},
}
