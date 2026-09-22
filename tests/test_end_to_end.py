"""Historical chain-reference regression; public CUDA contracts live in tests/cuda."""
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from legacy_chain_api import fit
from clipp1d.types import InputError, NoEligibleMutationsError


def test_single_mutation_and_cli(make_input, tmp_path):
    path = make_input([{"mutation_id": "0001", "alt_count": 10}])
    out = tmp_path / "one"
    result = fit(path, out)
    assert_allclose([result.raw_phi[0], result.refitted_phi[0], result.cluster_centers[0]], .25)
    assert result.cluster_labels.tolist() == [0]
    assert {p.name for p in out.iterdir()} == {"mutation_clusters.tsv", "cluster_centers.tsv",
                                             "mutation_multiplicity.tsv", "run.json"}
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "success" and run["provenance"]["backend"] == "cpu"
    assert run["search_status"] == result.search_status == "complete"
    assert run["raw_objective"] == result.raw_objective
    assert run["raw_witness_mutation_id"] is result.raw_witness_mutation_id is None
    assert run["raw_witness_index"] is result.raw_witness_index is None
    assert run["designated_clonal_block"] == 0
    assert run["schema"] == "clipp1d.run.v6"
    assert run["provenance"]["clonal_constraint"] is False
    assert not run["raw_diagnostics"]["global_optimality_proven"]
    before = (out / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        fit(path, out)
    assert (out / "run.json").read_bytes() == before


def test_two_mutations_permutation_and_exclusions(make_input, tmp_path):
    rows = [{"mutation_id": "a", "alt_count": 10, "ref_count": 90},
            {"mutation_id": "b", "alt_count": 40, "ref_count": 60},
            {"mutation_id": "missing", "count_observed": 0, "alt_count": ".", "ref_count": "."}]
    first = fit(make_input(rows), tmp_path / "first")
    second = fit(make_input(rows[::-1], name="reverse.tsv"))
    assert np.array_equal(first.cluster_labels, second.cluster_labels)
    assert_allclose(first.raw_phi, second.raw_phi, atol=0, rtol=0)
    assert_allclose(first.refitted_phi, second.refitted_phi, atol=0, rtol=0)
    assert first.frozen_chain.fingerprint == second.frozen_chain.fingerprint
    assert first.exclusion_reasons[-1] == "MISSING_COUNTS"
    assert first.search_diagnostics["path_candidates"] >= 26
    assert first.raw_diagnostics["raw_branch_stationarity_qualified"]
    assert first.raw_witness_mutation_id is first.raw_witness_index is None
    assert first.refitted_phi[np.flatnonzero(first.cluster_labels == 0)[0]] == max(first.cluster_centers)


def test_all_clonal(make_input):
    result = fit(make_input([{"alt_count": 40, "ref_count": 60}] * 2))
    assert_allclose(result.refitted_phi, 1)
    assert len(result.cluster_centers) == 1


def test_close_centers_unequal_depth_and_distinct_raw_refit(make_input):
    result = fit(make_input([{"alt_count": 10, "ref_count": 90},
                            {"alt_count": 22, "ref_count": 178},
                            {"alt_count": 40, "ref_count": 60}]))
    assert len(result.cluster_centers) == 2
    assert result.cluster_labels[0] == result.cluster_labels[1] != result.cluster_labels[2]
    assert np.max(np.abs(result.raw_phi - result.refitted_phi)) > 1e-6
    assert_allclose(result.refitted_phi[:2], (32 / 300) / .4, atol=2e-6)


def test_ambiguous_multiplicity(make_input):
    result = fit(make_input([{"alt_count": 35, "ref_count": 65, "allele_a_cn": 2},
                            {"alt_count": 40, "ref_count": 60}]))
    assert np.any(result.pilot.alternative_phi != result.pilot.phi)
    assert np.all(np.isfinite(result.refitted_phi))
    assert result.multiplicity_calls[0] in (1, 2)
    assert np.all(np.diff(result.cluster_centers) <= 0)


@pytest.mark.parametrize("rows,error", [
    ([{"count_observed": 0}], NoEligibleMutationsError),
    ([{}, {"sample_id": "two"}], InputError),
])
def test_failures_only_publish_diagnostics(make_input, tmp_path, rows, error):
    out = tmp_path / "failed"
    with pytest.raises(error):
        fit(make_input(rows), out)
    assert [p.name for p in out.iterdir()] == ["run.json"]
    assert json.loads((out / "run.json").read_text())["status"] == "failure"


def test_invalid_policy_diagnostic(make_input, tmp_path):
    out = tmp_path / "invalid_policy"
    with pytest.raises(ValueError):
        fit(make_input([{}]), out, max_major_cn=0)
    assert json.loads((out / "run.json").read_text())["status"] == "failure"
