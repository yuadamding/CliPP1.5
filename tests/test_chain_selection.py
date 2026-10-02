import itertools
import numpy as np
import pytest

from clipp.model import MultiplicityModel
from clipp.initialization import pooled_cp_initialization, _initialization_grid
from clipp.refinement import fixed_center_chain_partition, polish_chain_partition
from clipp.selection import (
    refit_partition,
    expand_subsample_partition,
    refine_chain_search,
    search_chain_coarsenings,
    _ChainRefitCache,
)
from clipp.scoring import fit_cluster_weights


@pytest.mark.parametrize("n", range(1, 8))
def test_dynamic_program_against_all_cuts(n):
    rng = np.random.default_rng(14 + n)
    for q in range(1, min(n, 4) + 1):
        for repeat in range(16):
            matrix = rng.integers(-5, 1, (n, q)).astype(float)
            if repeat % 2:
                matrix[rng.random((n, q)) < 0.25] = -np.inf
            scores = []
            for cuts in itertools.combinations(range(1, n), q - 1):
                labels = np.searchsorted(cuts, np.arange(n), side="right")
                scores.append(np.sum(matrix[np.arange(n), labels]))
            optimum = max(scores)
            if np.isfinite(optimum):
                result = fixed_center_chain_partition(matrix)
                assert result["conditional_log_likelihood"] == optimum
                assert len(result["cuts"]) == q - 1
            else:
                with pytest.raises(RuntimeError, match="No finite partition"):
                    fixed_center_chain_partition(matrix)
            tied = fixed_center_chain_partition(np.zeros((n, q)))
            np.testing.assert_array_equal(tied["cuts"], np.arange(1, q))


def test_nonmonotone_visits_and_polish_incumbent():
    matrix = np.array([[0.0, -10.0, -5.0], [-4.0, 0.0, -5.0], [-2.0, -10.0, 0.0]])
    np.testing.assert_array_equal(fixed_center_chain_partition(matrix)["labels"], [0, 1, 2])
    model = MultiplicityModel([5, 7, 90], [100] * 3, [1] * 3, [1] * 3, 1.0)
    initial = {"labels": np.array([0, 1, 1]), "centers": np.array([0.1, 0.9])}

    def worse(labels):
        return {"labels": labels, "centers": np.array([0.49, 0.5])}

    result = polish_chain_partition(model, initial, np.arange(3), worse)
    assert result["result"] is initial
    assert result["diagnostics"]["status"] == "refit_conditional_decrease"


def test_initialization_permutation_duplicates_and_budget(monkeypatch):
    import clipp.initialization as initialization

    model = MultiplicityModel([20, 65, 20, 32], [100] * 4, [2, 4, 2, 1], [3, 5, 3, 2], 0.9)
    pilot, diagnostics = pooled_cp_initialization(model)
    order = np.array([2, 3, 1, 0])
    other, other_diag = pooled_cp_initialization(model.subset(order))
    np.testing.assert_array_equal(other, pilot[order])
    assert pilot[0] == pilot[2]
    assert diagnostics["num_unique_observations"] == 3
    assert diagnostics["log_likelihood"] >= diagnostics["initial_log_likelihood"] - 1e-8
    assert diagnostics["grid_weight_optimum_certified"] == (diagnostics["grid_score_excess"] <= 1e-8)
    assert other_diag["grid_score_excess"] == diagnostics["grid_score_excess"]
    monkeypatch.setattr(initialization, "MAX_ITERATIONS", 0)
    budget_pilot, budget = pooled_cp_initialization(model)
    assert np.isfinite(budget_pilot).all()
    assert budget["termination_reason"] == "iteration_budget"
    assert not budget["grid_weight_optimum_certified"]
    grid, compressed = _initialization_grid(np.tile([100000, 10000000, 100, 102], (100, 1)), 0.9)
    assert compressed["mode_grid_compressed"]
    assert len(grid) <= 257 + 2048


def test_adjacent_merges_nonadjacent_repeat_and_parameters():
    model = MultiplicityModel([10, 10, 90, 10], [100] * 4, [1] * 4, [1] * 4, 1.0)
    result = refit_partition(model, np.arange(4), chain_order=np.arange(4))
    assert result["num_clusters"] == 3
    labels = result["labels"]
    assert labels[0] == labels[1] and labels[0] != labels[3]
    assert result["num_parameters"] == 5
    np.testing.assert_allclose(result["bic"], -2 * result["log_likelihood"] + 5 * np.log(4))
    assert result["log_likelihood"] != result["conditional_log_likelihood"]


def test_zero_weight_and_actual_adjacent_repair():
    model = MultiplicityModel([10, 10, 10, 11], [100] * 4, [1] * 4, [1] * 4, 1.0)
    weights = fit_cluster_weights(model, [0.1, 0.9])
    assert weights["cluster_weights"][1] == 0
    assert weights["weight_optimality_gap"] <= 4e-8
    seeds = [{"requested_k": 2, "replicate": 1, "labels": np.array([0, 0, 1, 1])}]
    cache = _ChainRefitCache(model, np.arange(4))
    search = search_chain_coarsenings(model, seeds, np.arange(4), [1, 2], cache)
    polished = refine_chain_search(model, search, seeds, np.arange(4), cache)
    assert all((v["result"]["cluster_weights"] > 0).all() for v in polished["winners"].values())
    assert any(r["candidate_kind"] == "unsupported_component_coarsening" for r in polished["candidates"])


def test_subsample_midpoints():
    order = np.array([5, 4, 3, 2, 1, 0])
    labels = expand_subsample_partition([0, 1], [5, 0], order, 2)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        expand_subsample_partition([0, 1], [5, 5], order, 2)
