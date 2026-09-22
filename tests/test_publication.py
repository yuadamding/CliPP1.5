# Historical chain-reference regression; public CUDA contracts live in tests/cuda.
"""Publication must bind every table to the selected partition and raw reference."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from legacy_chain_api import fit
from clipp1d.report import write_json, write_result


@pytest.fixture
def fitted(make_input):
    return fit(make_input([dict(alt_count=10, ref_count=90),
                           dict(alt_count=22, ref_count=178),
                           dict(alt_count=40, ref_count=60)]))


@pytest.mark.parametrize("fault", ["center", "labels", "partition", "mask", "multiplicity", "pilot",
                                   "raw_phi", "reference_chain", "reference_partition", "raw_certificate",
                                   "candidate_family", "lambda", "seed", "raw_parent", "search_status", "score"])
def test_inconsistent_results_fail_before_writing(fitted, tmp_path, fault):
    result = deepcopy(fitted)
    candidate = result.candidate_provenance
    reference = candidate["raw_reference"]
    if fault == "center":
        result.cluster_centers[0] -= .01
    elif fault == "labels":
        result.cluster_labels[0] = result.cluster_labels[-1]
    elif fault == "partition":
        result.partition = (0, 1, len(result.raw_phi))
    elif fault == "mask":
        result.retained_mask[0] = False
    elif fault == "multiplicity":
        result.multiplicity_calls[0] = 0
    elif fault == "pilot":
        result.pilot = replace(result.pilot, mutation_ids=tuple(reversed(result.pilot.mutation_ids)))
    elif fault == "raw_phi":
        result.raw_phi[0] += .01
    elif fault == "reference_chain":
        reference["chain_sha256"] = "unrelated"
    elif fault == "reference_partition":
        reference["partition_sha256"] = "unrelated"
    elif fault == "raw_certificate":
        candidate["selected_raw_certificate"] = None
    elif fault == "candidate_family":
        candidate["candidate_family"] = "unknown"
    elif fault == "lambda":
        result.selected_lambda = None
    elif fault == "seed":
        candidate["seed_origin"] = "unknown"
    elif fault == "raw_parent":
        candidate["raw_parent"] = None
    elif fault == "score":
        result.selection_score += 1
    else:
        result.search_status = "unknown"
    out = tmp_path / fault
    with pytest.raises(ValueError, match="Refusing to publish"):
        write_result(result, out)
    assert not out.exists()


def test_closest_cluster_is_designated_without_changing_its_ccf(make_input, tmp_path):
    import csv
    out = tmp_path / "subclonal"
    result = fit(make_input([dict(alt_count=10, ref_count=90)] * 8 +
                            [dict(alt_count=30, ref_count=70)] * 8), out)
    np.testing.assert_allclose(result.cluster_centers, [.75, .25], atol=1e-12)
    assert result.designated_clonal_block == 1
    with (out / "cluster_centers.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["cluster_label"] for row in rows] == ["0", "1"]
    assert [row["designated_clonal"] for row in rows] == ["1", "0"]


@pytest.mark.parametrize("seed", ["production_path", "adjacent_ward_selected_raw", "adjacent_ward_pilot"])
def test_direct_candidate_raw_parent_tracks_seed(make_input, tmp_path, monkeypatch, seed):
    import legacy_chain_api as api
    from clipp1d.selection import refit_partition
    original = api.propose_partitions

    def forced(model, pilot, raw, baseline, policy):
        _, diagnostics = original(model, pilot, raw, baseline, policy)
        cuts = (0, len(model)) if len(baseline.cuts) > 2 else tuple(range(len(model) + 1))
        direct = refit_partition(model, cuts, policy)
        diagnostics.update(selected_origin="boundary_refinement", selected_seed_origin=seed)
        return direct, diagnostics

    monkeypatch.setattr(api, "propose_partitions", forced)
    result = fit(make_input([dict(alt_count=4), dict(alt_count=40)]), tmp_path / seed)
    selected = result.candidate_provenance
    assert selected["candidate_family"] == "direct_chain_partition"
    assert selected["selected_raw_certificate"] is None
    assert selected["selected_partition_certified"] is False
    assert (selected["raw_parent"] is None) == (seed == "adjacent_ward_pilot")
    # Forging an inherited raw certificate remains invalid even when the seed
    # was itself a qualified raw state.
    selected["selected_raw_certificate"] = dict(result.raw_diagnostics)
    with pytest.raises(ValueError, match="inherited raw certificate"):
        write_result(result, tmp_path / "forged")


def test_json_serialization_errors_never_create_a_marker(tmp_path):
    path = tmp_path / "run.json"
    with pytest.raises(TypeError):
        write_json(path, {"status": "success", "bad": object()})
    assert list(tmp_path.iterdir()) == []
    write_json(path, {"status": "failure", "scalar": np.array(2), "gap": np.inf})
    before = path.read_bytes()
    assert json.loads(before) == {"status": "failure", "scalar": 2, "gap": None}
    with pytest.raises(FileExistsError):
        write_json(path, {"status": "success"})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("field", ["provenance", "search_diagnostics"])
def test_full_receipt_serialization_precedes_table_creation(fitted, tmp_path, field):
    getattr(fitted, field)["unsupported_metadata"] = object()
    out = tmp_path / "unsupported"
    with pytest.raises(TypeError, match="Not JSON serializable"):
        write_result(fitted, out)
    assert not out.exists()


def test_json_publication_failure_never_leaves_partial_marker(tmp_path, monkeypatch):
    import clipp1d.report as report
    def failed_link(*args):
        raise OSError("publication unavailable")
    monkeypatch.setattr(report.os, "link", failed_link)
    with pytest.raises(OSError, match="publication unavailable"):
        write_json(tmp_path / "run.json", {"status": "success"})
    assert list(tmp_path.iterdir()) == []
