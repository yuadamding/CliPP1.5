"""The joint gate cannot trade one cohort or mutation population for another."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "joint_cohort_gate", Path(__file__).parents[1] / "benchmarks/evaluate_joint_cohort_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

BASE_SOURCE = gate.ACCEPTED_BASELINE_SOURCE
CANDIDATE_SOURCE = "b" * 64
CONFIG = gate.ACCEPTED_BASELINE_CONFIG


def _artifact(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_table(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _panel(tmp_path, *, full=False, matched_n=4, protection=False, self_check=False):
    """Real required cohort sizes in inventories, a small discovery panel by default."""
    metrics = []
    entries = []
    metrics_path = tmp_path / "metrics.tsv"
    for cohort, count in gate.EXPECTED_COHORT_SIZES.items():
        folder = tmp_path / cohort
        folder.mkdir()
        inventory = []
        for index in range(count):
            ids = [f"m{i}" for i in range(4)]
            inventory.append(dict(case_id=f"case{index}", input_sha256="d" * 64,
                truth_sha256="e" * 64, n_retained_mutations=4,
                retained_mutation_ids_sha256=gate.mutation_ids_sha256(ids),
                n_scorable_retained_mutations=4,
                scorable_retained_mutation_ids_sha256=gate.mutation_ids_sha256(ids)))
        inventory_path = folder / "inventory.tsv"
        _write_table(inventory_path, inventory)
        exclusions = folder / "truth_exclusions.json"
        _write_json(exclusions, {})
        evaluated = count if full else 3

        def population(name, n):
            ids = {f"case{i}": [f"m{j}" for j in range(n)] for i in range(evaluated)}
            rows = []
            for index, (case_id, mutation_ids) in enumerate(ids.items()):
                k = index % 3 + 1
                smf = 0 if k == 1 else (n - 1) / n
                authority = dict(case_id=case_id, n_mutations=n,
                    mutation_ids_sha256=gate.mutation_ids_sha256(mutation_ids),
                    true_k=k, true_smf=smf)
                rows.append(authority)
                for method in ("baseline", "candidate", *sorted(gate.REQUIRED_COMPARATORS[cohort])):
                    source = (CANDIDATE_SOURCE if method == "candidate" and not self_check
                              else BASE_SOURCE)
                    metrics.append(dict(cohort_id=cohort, population_id=name, method_id=method,
                        **authority, selected_k=k, ari=1 if k == 1 else .5,
                        ccf_mae=.1, estimated_smf=smf, status="validated_complete",
                        source_sha256=source, scientific_config_sha256=CONFIG))
            table_path = folder / f"{name}.tsv"
            ids_path = folder / f"{name}.json"
            _write_table(table_path, rows)
            _write_json(ids_path, ids)
            return dict(population_id=name, table=_artifact(table_path), mutation_ids=_artifact(ids_path))

        def binding(method):
            result = dict(method_id=method,
                source_sha256=CANDIDATE_SOURCE if method == "candidate" and not self_check else BASE_SOURCE,
                scientific_config_sha256=CONFIG, metrics={"path": str(metrics_path), "sha256": "0" * 64})
            if method == "candidate":
                result["cohort_specific_settings"] = False
                result["runtime"] = {"gpu_type": "H100" if cohort == "CN-first4K" else "L40"}
            return result

        entry = dict(cohort_id=cohort, scope="full_cohort" if full else "development_snapshot",
            full_inventory=_artifact(inventory_path), truth_exclusions=_artifact(exclusions),
            population=population("matched", matched_n), baseline=binding("baseline"),
            candidate=binding("candidate"), comparators=[binding(name) for name in sorted(gate.REQUIRED_COMPARATORS[cohort])], protection_panels=[])
        if protection:
            entry["protection_panels"].append(dict(population=population("retained", 4),
                baseline=binding("baseline"), candidate=binding("candidate")))
        entries.append(entry)
    _write_table(metrics_path, metrics)
    manifest = dict(schema_version=1, cohorts=entries)
    path = tmp_path / "manifest.json"
    _refresh_metrics(manifest, metrics_path)
    _write_json(path, manifest)
    return path, manifest, metrics_path


def _refresh_metrics(manifest, metrics_path):
    artifact = _artifact(metrics_path)
    for entry in manifest["cohorts"]:
        for binding in [entry["baseline"], entry["candidate"], *entry["comparators"]]:
            if binding is not None:
                binding["metrics"] = artifact.copy()
        for panel in entry["protection_panels"]:
            for role in ("baseline", "candidate"):
                panel[role]["metrics"] = artifact.copy()


def _edit_metrics(path, manifest, metrics_path, edit):
    with metrics_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = edit(rows)
    _write_table(metrics_path, rows)
    _refresh_metrics(manifest, metrics_path)
    _write_json(path, manifest)


def test_full_cohort_observed_pass_is_not_release_or_statistical_proof(tmp_path):
    path, _, metrics = _panel(tmp_path, full=True)
    before = metrics.read_bytes()
    report = gate.evaluate_manifest(path)
    assert report["gate_passed"]
    assert report["full_cohort_coverage"]
    assert report["observed_nonregression"] and report["best_among_declared_comparators"]
    assert not report["statistical_superiority_proven"] and not report["release_ready"]
    assert not report["baseline_self_check"]
    assert metrics.read_bytes() == before
    assert report["cohorts"]["CN-first4K"]["candidate_summary"]["cases"] == 4000


def test_development_and_baseline_self_checks_cannot_accept_algorithm(tmp_path):
    path, _, _ = _panel(tmp_path, self_check=True)
    report = gate.evaluate_manifest(path)
    assert report["observed_nonregression"] and report["best_among_declared_comparators"]
    assert report["baseline_self_check"]
    assert not report["gate_passed"] and not report["full_cohort_coverage"]
    assert report["assessment"] == "engineering_self_check_only"


def test_large_gains_elsewhere_cannot_offset_tiny_cnfirst_regression(tmp_path):
    path, manifest, metrics = _panel(tmp_path)

    def edit(rows):
        for row in rows:
            if row["method_id"] == "candidate":
                row["ccf_mae"] = .10000001 if row["cohort_id"] == "CN-first4K" else 0
                if row["cohort_id"] != "CN-first4K":
                    row["ari"] = 1
        return rows

    _edit_metrics(path, manifest, metrics, edit)
    report = gate.evaluate_manifest(path)
    assert not report["observed_nonregression"] and not report["gate_passed"]
    assert not report["cohorts"]["CN-first4K"]["baseline_comparison"]["checks"]["mean_ccf_mae"]["passed"]
    assert report["cohorts"]["SimClone1000"]["observed_nonregression"]


@pytest.mark.parametrize("method", ["candidate", "clipp_original"])
def test_missing_case_fails_closed(tmp_path, method):
    path, manifest, metrics = _panel(tmp_path)
    _edit_metrics(path, manifest, metrics, lambda rows: [r for r in rows if not
        (r["cohort_id"] == "CN-first4K" and r["method_id"] == method and r["case_id"] == "case0")])
    report = gate.evaluate_manifest(path)
    assert not report["gate_passed"] and report["integrity_errors"]
    assert "coverage mismatch" in report["integrity_errors"][0]


def test_missing_candidate_preserves_inspectable_baseline_summary(tmp_path):
    path, manifest, _ = _panel(tmp_path)
    for entry in manifest["cohorts"]:
        entry["candidate"] = None
    _write_json(path, manifest)
    report = gate.evaluate_manifest(path)
    assert not report["gate_passed"]
    assert all(v["baseline_summary"]["cases"] == 3 for v in report["cohorts"].values())


def test_artifact_tampering_fails_closed(tmp_path):
    path, _, metrics = _panel(tmp_path)
    metrics.write_text(metrics.read_text() + "\n")
    report = gate.evaluate_manifest(path)
    assert not report["observed_nonregression"]
    assert "hash mismatch" in report["integrity_errors"][0]


@pytest.mark.parametrize("field,value", [("mutation_ids_sha256", "f" * 64),
    ("n_mutations", "3"), ("true_k", "2"), ("ccf_mae", "nan"), ("status", "failed")])
def test_bound_but_invalid_candidate_metrics_fail_closed(tmp_path, field, value):
    path, manifest, metrics = _panel(tmp_path)

    def edit(rows):
        for row in rows:
            if row["cohort_id"] == "CN-first4K" and row["method_id"] == "candidate" and row["case_id"] == "case0":
                row[field] = value
        return rows

    _edit_metrics(path, manifest, metrics, edit)
    assert gate.evaluate_manifest(path)["integrity_errors"]


def test_different_candidate_scientific_config_across_cohorts_rejected(tmp_path):
    path, manifest, metrics = _panel(tmp_path)
    manifest["cohorts"][0]["candidate"]["scientific_config_sha256"] = "f" * 64

    def edit(rows):
        for row in rows:
            if row["cohort_id"] == "CN-first4K" and row["method_id"] == "candidate":
                row["scientific_config_sha256"] = "f" * 64
        return rows

    _edit_metrics(path, manifest, metrics, edit)
    report = gate.evaluate_manifest(path)
    assert not report["observed_nonregression"]
    assert any("differs across cohorts" in v for v in report["integrity_errors"])


def test_cohort_specific_switch_attestation_rejected(tmp_path):
    path, manifest, _ = _panel(tmp_path)
    manifest["cohorts"][0]["candidate"]["cohort_specific_settings"] = True
    _write_json(path, manifest)
    assert gate.evaluate_manifest(path)["integrity_errors"]


@pytest.mark.parametrize("protection", [False, True])
def test_narrow_comparison_requires_full_retained_protection(tmp_path, protection):
    path, _, _ = _panel(tmp_path, matched_n=3, protection=protection)
    report = gate.evaluate_manifest(path)
    assert report["retained_population_protected"] is protection
    assert report["observed_nonregression"] is protection


def test_native_population_regression_cannot_hide_behind_matched_panel(tmp_path):
    path, manifest, metrics = _panel(tmp_path, matched_n=3, protection=True)

    def edit(rows):
        for row in rows:
            if row["cohort_id"] == "CN-first4K" and row["method_id"] == "candidate" and row["population_id"] == "retained":
                row["ccf_mae"] = .11
        return rows

    _edit_metrics(path, manifest, metrics, edit)
    report = gate.evaluate_manifest(path)
    result = report["cohorts"]["CN-first4K"]
    assert result["baseline_comparison"]["passed"]
    assert not result["protection_panels"]["retained"]["passed"]
    assert not report["observed_nonregression"]


def test_truth_exclusions_are_explicit_and_union_recovers_retained_ids(tmp_path):
    path, manifest, _ = _panel(tmp_path, matched_n=3)
    for entry in manifest["cohorts"]:
        inventory_path = Path(entry["full_inventory"]["path"])
        with inventory_path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows[:3]:
            row["n_scorable_retained_mutations"] = 3
            row["scorable_retained_mutation_ids_sha256"] = gate.mutation_ids_sha256(["m0", "m1", "m2"])
        _write_table(inventory_path, rows)
        entry["full_inventory"] = _artifact(inventory_path)
        exclusions_path = Path(entry["truth_exclusions"]["path"])
        _write_json(exclusions_path, {f"case{i}": ["m3"] for i in range(3)})
        entry["truth_exclusions"] = _artifact(exclusions_path)
    _write_json(path, manifest)
    report = gate.evaluate_manifest(path)
    assert report["retained_population_protected"] and report["observed_nonregression"]
    assert report["cohorts"]["CN-first4K"]["truth_unmatched_exclusions"]["planned_mutations"] == 3
    # Count-correct but wrong excluded IDs must not silently redefine retained truth.
    entry = manifest["cohorts"][0]
    exclusions_path = Path(entry["truth_exclusions"]["path"])
    _write_json(exclusions_path, {f"case{i}": ["wrong"] for i in range(3)})
    entry["truth_exclusions"] = _artifact(exclusions_path)
    _write_json(path, manifest)
    assert gate.evaluate_manifest(path)["integrity_errors"]


def test_multi_cluster_ari_is_separate_from_single_cluster_success(tmp_path):
    path, manifest, metrics = _panel(tmp_path)

    def edit(rows):
        for row in rows:
            if row["method_id"] == "baseline" and row["case_id"] == "case0":
                row.update(selected_k=2, ari=0, estimated_smf=.5)
            if row["method_id"] == "candidate" and row["case_id"] == "case1":
                row["ari"] = .49
        return rows

    _edit_metrics(path, manifest, metrics, edit)
    report = gate.evaluate_manifest(path)
    check = report["cohorts"]["CN-first4K"]["baseline_comparison"]["checks"]
    assert check["mean_ari"]["passed"]
    assert not check["multi_cluster_mean_ari"]["passed"]
    assert not report["observed_nonregression"]


def test_safeguards_are_required_against_baseline_not_external_comparators(tmp_path):
    path, manifest, metrics = _panel(tmp_path)

    def edit(rows):
        for row in rows:
            if row["case_id"] == "case1" and row["method_id"] in {"baseline", "candidate"}:
                row.update(selected_k=1, ari=0, estimated_smf=0)
            if row["method_id"] not in {"baseline", "candidate"}:
                row.update(ari=0, ccf_mae=.5, estimated_smf=0)
        return rows

    _edit_metrics(path, manifest, metrics, edit)
    report = gate.evaluate_manifest(path)
    assert report["observed_nonregression"]
    comparison = report["cohorts"]["CN-first4K"]["comparators"]["clipp_original"]
    assert comparison["passed"]
    assert not comparison["checks"]["false_k_one_collapses"]["passed"]
    assert "false_k_one_collapses" not in comparison["required_metrics"]


def test_numeric_epsilon_is_fixed_not_a_scientific_regression_margin(tmp_path):
    path, manifest, _ = _panel(tmp_path)
    manifest["numeric_epsilon"] = .1
    _write_json(path, manifest)
    report = gate.evaluate_manifest(path)
    assert report["numeric_epsilon"] == 1e-12
    assert report["integrity_errors"]


def test_existing_constant_vector_ccc_conventions():
    assert gate.concordance([0, 0], [0, 0]) == 1
    assert gate.concordance([.1, .1], [.2, .2]) == 0
    assert gate.concordance([0, 1], [.5, .5]) == 0
    assert gate.concordance([0, 1], [0, 1]) == 1
    with pytest.raises(gate.GateInputError):
        gate.concordance([0], [0])


def test_weaker_baseline_identity_cannot_replace_accepted_campaign(tmp_path):
    path, manifest, _ = _panel(tmp_path)
    manifest["cohorts"][0]["baseline"]["source_sha256"] = "f" * 64
    _write_json(path, manifest)
    report = gate.evaluate_manifest(path)
    assert not report["observed_nonregression"]
    assert "accepted campaign contract" in report["integrity_errors"][0]


def test_known_comparator_cannot_be_omitted_for_favorable_comparison(tmp_path):
    path, manifest, _ = _panel(tmp_path)
    manifest["cohorts"][0]["comparators"].pop()
    _write_json(path, manifest)
    report = gate.evaluate_manifest(path)
    assert not report["best_among_declared_comparators"]
    assert "Missing required campaign comparators" in report["integrity_errors"][0]


def test_non_object_manifest_returns_fail_closed_report(tmp_path):
    path = tmp_path / "manifest.json"
    _write_json(path, [])
    report = gate.evaluate_manifest(path)
    assert not report["gate_passed"]
    assert report["integrity_errors"] == ["Manifest must be a JSON object"]
