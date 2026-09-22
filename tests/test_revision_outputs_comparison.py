"""Read-only output comparisons reject identity drift and retain changed fits."""

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("compare_revision_outputs", directory / "compare_revision_outputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case():
    rows = {"a": dict(pilot_ccf=".2", raw_ccf=".25", refitted_ccf=".2", cluster_label="1"),
            "b": dict(pilot_ccf="1", raw_ccf="1", refitted_ccf="1", cluster_label="0")}
    truth = {"a": dict(true_ccf=".2", true_cluster="1", true_multiplicity="1", major_cn="1", minor_cn="1"),
             "b": dict(true_ccf="1", true_cluster="0", true_multiplicity="2", major_cn="2", minor_cn="1")}
    run = dict(selected_lambda=1., selection_score=12., raw_objective=6., search_status="complete",
               table_sha256=dict(mutations="same"), search=dict(path=[dict(
                   **{"lambda": 1.}, raw_status="qualified", refit_status="qualified", raw_objective=6.,
                   starts_attempted=2, starts_qualified=2, starts_unresolved=0, raw_witness_mutation_id="b",
                   search_profile_calls=8, search_inner_iterations=8, score=12.)]))
    return dict(directory="fixture", source_sha256="source", input_sha256="input", truth_sha256="truth",
                policy_sha256="policy", benchmark_sha256="benchmark", status="success", comparable=True,
                wall_seconds=1., model_sha256="model", chain_sha256="chain", retained_ids=["a", "b"],
                run_sha256="receipt", rows=rows, truth=truth, calls={"a": dict(multiplicity_call="1"),
                                                                   "b": dict(multiplicity_call="2")}, run=run)


def test_identical_paired_outputs_include_truth_and_cna_coverage(comparison, case):
    result = comparison.compare_case(case, copy.deepcopy(case))
    assert result["comparable"] and not result["warnings"]
    assert result["selected_comparison"]["raw_ccf"]["exactly_equal"]
    assert result["selected_comparison"]["label_ari"] == 1
    metrics = result["baseline_truth_metrics"]
    assert metrics["ari"] == 1 and metrics["rmse"] == 0
    cna = metrics["cna_only_multiplicity"]
    assert cna["eligible"] == cna["called"] == 1
    assert cna["macro_f1"] == cna["micro_f1"] == cna["weighted_f1"] == cna["coverage"] == 1
    assert cna["per_class"]["2"]["support"] == 1


@pytest.mark.parametrize("field", ["input_sha256", "truth_sha256", "policy_sha256", "model_sha256", "chain_sha256", "retained_ids"])
def test_paired_identity_mismatches_are_errors(comparison, case, field):
    changed = copy.deepcopy(case)
    changed[field] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        comparison.compare_case(case, changed)


def test_changed_search_and_outputs_are_reported_without_equivalence_claim(comparison, case):
    changed = copy.deepcopy(case)
    changed["rows"]["a"]["raw_ccf"] = ".3"
    changed["run"]["search"]["path"][0]["search_profile_calls"] = 9
    changed["run"]["search"]["path"][0]["raw_objective"] = 6.01
    result = comparison.compare_case(case, changed)
    assert len(result["warnings"]) == 2
    assert result["selected_comparison"]["raw_ccf"]["max_abs_difference"] == pytest.approx(.05)
    assert result["paths"][0]["differing_fields"] == ["search_profile_calls"]
    assert result["paths"][0]["raw_objective"]["difference"] == pytest.approx(.01)


def test_source_manifest_and_aggregate_hashes_are_verified(comparison, tmp_path):
    package = tmp_path / "clipp1d"
    package.mkdir()
    (package / "model.py").write_text("source fixture\n")
    files = {"model.py": comparison.file_hash(package / "model.py")}
    digest = hashlib.sha256()
    digest.update(f"model.py\0{files['model.py']}\n".encode())
    provenance = dict(source_files=files, source_sha256=digest.hexdigest())
    assert comparison.verify_source(provenance, tmp_path) == digest.hexdigest()
    with pytest.raises(ValueError, match="aggregate"):
        comparison.verify_source(dict(provenance, source_sha256="wrong"), tmp_path)
    (package / "model.py").write_text("changed source fixture\n")
    with pytest.raises(ValueError, match="manifest"):
        comparison.verify_source(provenance, tmp_path)


def test_nonfinite_output_vectors_are_rejected(comparison):
    with pytest.raises(ValueError, match="finite"):
        comparison.numeric_change([.1], [float("nan")])


def test_direct_current_outputs_compare_independent_reference_without_selected_raw_claim(comparison, case):
    for row in case["rows"].values():
        row["raw_reference_ccf"] = row.pop("raw_ccf")
    case["run"].update(schema="clipp1d.run.v5", selected_lambda=None, raw_objective=None,
                       raw_reference_objective=6.)
    changed = copy.deepcopy(case)
    changed["rows"]["a"]["raw_reference_ccf"] = ".3"
    changed["run"]["raw_reference_objective"] = 6.1
    result = comparison.compare_case(case, changed)
    observed = result["selected_comparison"]
    assert observed["raw_reference_ccf"]["max_abs_difference"] == pytest.approx(.05)
    assert "raw_ccf" not in observed
    assert observed["selected_raw_objective_difference"] is None
    assert observed["raw_reference_objective_difference"] == pytest.approx(.1)
    assert result["baseline_truth_metrics"]["designated_clonal_fraction"] is None


def test_cross_schema_raw_meanings_are_explicit(comparison, case):
    changed = copy.deepcopy(case)
    for row in changed["rows"].values():
        row["raw_reference_ccf"] = row.pop("raw_ccf")
    result = comparison.compare_case(case, changed)
    assert any("different semantics" in warning for warning in result["warnings"])
