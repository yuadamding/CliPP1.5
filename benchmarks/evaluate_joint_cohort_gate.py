"""Strict, read-only observed-performance gate across the three simulation cohorts.

Usage: python benchmarks/evaluate_joint_cohort_gate.py MANIFEST.json --output NEW.json
       python benchmarks/evaluate_joint_cohort_gate.py --example-manifest NEW.json

Manifest schema v1 is illustrated by ``example_manifest()``. Every artifact has
``path`` (relative to the manifest, or absolute) and ``sha256``. Inventory TSVs
contain case_id,input_sha256,truth_sha256,n_retained_mutations,
retained_mutation_ids_sha256,n_scorable_retained_mutations,
scorable_retained_mutation_ids_sha256. Each cohort also binds a truth_exclusions
JSON mapping cases with ambiguous truth to their exact unmatched retained IDs.
Protection covers all scorable retained IDs; their union with declared exclusions
must recover the original retained identity. Population TSVs contain case_id,n_mutations,
mutation_ids_sha256,true_k,true_smf; their companion JSON maps each case to its
exact mutation IDs. The mutation hash is SHA256 of
json.dumps(sorted(ids), ensure_ascii=True).encode('utf-8').

Metric TSVs contain cohort_id,population_id,method_id,case_id,n_mutations,
mutation_ids_sha256,true_k,true_smf,selected_k,ari,ccf_mae,estimated_smf,status,
source_sha256,scientific_config_sha256. Other columns are ignored. A shared TSV
may contain several explicitly bound cohorts, populations and methods. Validated
incomplete searches remain publishable and are reported, never called complete.

Scientific identities must identify actual fit settings, excluding runtime-only
metadata. Candidate identity and the explicit no-cohort-switch attestation must
agree everywhere. This checks bound reported identities; it does not replace
source review, verify how an upstream producer created metrics, certify CUDA,
prove statistical superiority, or establish independent held-out qualification.
No model fitting, remote access, mutable current pointers or baseline writes.
The versioned contract pins the September 27 accepted baseline identity, current
eligible cohort sizes and already available comparator sets. A future accepted
baseline requires an explicit reviewed contract revision, not a manifest override.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
NUMERIC_EPSILON = 1e-12
CONTRACT_ID = "accepted_three_cohort_baseline_20260927_v1"
ACCEPTED_BASELINE_SOURCE = "55d6e45949e4dee2e8798ac81fb49510baffca9959104355db87d45acca1a523"
ACCEPTED_BASELINE_CONFIG = "45593932d7effb9d97469b38db489661b61fa9fcd4badb0b2c1cba6291402ad3"
EXPECTED_COHORT_SIZES = {
    "CN-first4K": 4000,
    "SimClone1000": 756,
    "PhylogicNDT500": 500,
}
REQUIRED_COMPARATORS = {
    "CN-first4K": {"clipp_original", "pyclone_vi_cnfirst", "phylogicndt"},
    "SimClone1000": {"pyclone_vi_twocohort"},
    "PhylogicNDT500": {"pyclone_vi_twocohort"},
}
VALID_STATUSES = {"validated_complete", "validated_incomplete"}
METRIC_DIRECTIONS = {
    "mean_ari": 1,
    "multi_cluster_mean_ari": 1,
    "mean_ccf_mae": -1,
    "smf_ccc": 1,
    "mean_smf_mae": -1,
    "true_k_one_false_splits": -1,
    "false_k_one_collapses": -1,
}
HEADLINE_METRICS = tuple(list(METRIC_DIRECTIONS)[:5])
IDENTITY_FIELDS = ("method_id", "source_sha256", "scientific_config_sha256")


class GateInputError(ValueError):
    """A missing or inconsistent bound input; never a performance pass."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateInputError(message)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def mutation_ids_sha256(ids: list[str]) -> str:
    return _hash_bytes(json.dumps(sorted(ids), ensure_ascii=True).encode("utf-8"))


def _sha(value: Any, field: str) -> str:
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"Invalid {field} SHA256")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(value: bytes) -> Any:
    return json.loads(value, object_pairs_hook=_unique_object)


class _Reader:
    def __init__(self, parent: Path):
        self.parent = parent
        self.artifacts: dict[str, str] = {}
        self.cache: dict[tuple[str, str], bytes] = {}

    def artifact(self, binding: dict[str, Any]) -> bytes:
        _require(isinstance(binding, dict), "Missing artifact binding")
        _require(set(binding) == {"path", "sha256"}, "Artifact needs exactly path and sha256")
        _require(isinstance(binding["path"], str) and binding["path"], "Invalid artifact path")
        expected = _sha(binding["sha256"], "artifact")
        path = (self.parent / binding["path"]).resolve()
        key = (str(path), expected)
        if key not in self.cache:
            value = path.read_bytes()
            _require(_hash_bytes(value) == expected, f"Artifact hash mismatch: {path}")
            _require(str(path) not in self.artifacts or self.artifacts[str(path)] == expected,
                     f"Conflicting artifact identities: {path}")
            self.cache[key] = value
            self.artifacts[str(path)] = expected
        return self.cache[key]

    def table(self, binding: dict[str, Any], columns: set[str]) -> list[dict[str, str]]:
        reader = csv.DictReader(io.StringIO(self.artifact(binding).decode("utf-8")), delimiter="\t")
        header = reader.fieldnames or []
        _require(len(header) == len(set(header)), "Duplicate TSV column")
        _require(columns <= set(header), f"Missing TSV columns: {sorted(columns - set(header))}")
        rows = list(reader)
        _require(all(None not in row and all(value is not None for value in row.values())
                     for row in rows), "Malformed TSV row")
        return rows


def _integer(value: str, field: str, minimum: int = 1) -> int:
    _require(isinstance(value, str) and value.isascii() and value.isdigit(), f"Invalid {field}")
    result = int(value)
    _require(result >= minimum, f"Invalid {field}")
    return result


def _number(value: str, field: str, lower: float, upper: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GateInputError(f"Invalid {field}") from exc
    _require(math.isfinite(result) and lower <= result <= upper, f"Invalid {field}")
    return result


def _by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        case_id = row["case_id"]
        _require(bool(case_id) and case_id not in result, f"Empty or duplicate case_id: {case_id}")
        result[case_id] = row
    return result


def _inventory(reader: _Reader, entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = reader.table(entry["full_inventory"], {
        "case_id", "input_sha256", "truth_sha256", "n_retained_mutations",
        "retained_mutation_ids_sha256",
        "n_scorable_retained_mutations", "scorable_retained_mutation_ids_sha256",
    })
    inventory = _by_case(rows)
    expected = EXPECTED_COHORT_SIZES[entry["cohort_id"]]
    _require(len(inventory) == expected, f"Full inventory needs {expected} eligible cases")
    exclusions = _json(reader.artifact(entry["truth_exclusions"]))
    _require(isinstance(exclusions, dict) and set(exclusions) <= set(inventory),
             "Truth-exclusion cases differ from full inventory")
    for case_id, row in inventory.items():
        for key in ("input_sha256", "truth_sha256", "retained_mutation_ids_sha256",
                    "scorable_retained_mutation_ids_sha256"):
            _sha(row[key], key)
        n = _integer(row["n_retained_mutations"], "n_retained_mutations")
        scored = _integer(row["n_scorable_retained_mutations"], "n_scorable_retained_mutations")
        ids = exclusions.get(case_id, [])
        _require(isinstance(ids, list) and all(isinstance(v, str) and v for v in ids)
                 and len(ids) == len(set(ids)), f"Invalid truth exclusions: {case_id}")
        _require(scored + len(ids) == n, f"Scorable/excluded count mismatch: {case_id}")
        _require(bool(ids) or row["scorable_retained_mutation_ids_sha256"] ==
                 row["retained_mutation_ids_sha256"], f"Scorable identity mismatch: {case_id}")
        row["_truth_unmatched_ids"] = set(ids)
    return inventory


def _population(reader: _Reader, spec: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(spec, dict), "Missing population binding")
    _require(isinstance(spec.get("population_id"), str) and bool(spec["population_id"]),
             "Missing population_id")
    rows = _by_case(reader.table(spec["table"], {
        "case_id", "n_mutations", "mutation_ids_sha256", "true_k", "true_smf",
    }))
    _require(len(rows) >= 2, "sMF CCC requires at least two bound cases")
    _require(set(rows) <= set(inventory), "Population case absent from full inventory")
    ids = _json(reader.artifact(spec["mutation_ids"]))
    _require(isinstance(ids, dict) and set(ids) == set(rows), "Mutation-ID case coverage mismatch")
    result = {}
    for case_id, row in rows.items():
        values = ids[case_id]
        _require(isinstance(values, list) and all(isinstance(v, str) and v for v in values),
                 f"Invalid mutation IDs: {case_id}")
        _require(len(values) == len(set(values)), f"Duplicate mutation IDs: {case_id}")
        _require(not set(values) & inventory[case_id]["_truth_unmatched_ids"],
                 f"Population contains declared truth-unmatched IDs: {case_id}")
        n = _integer(row["n_mutations"], "n_mutations")
        _require(len(values) == n, f"Mutation count mismatch: {case_id}")
        digest = _sha(row["mutation_ids_sha256"], "mutation IDs")
        _require(mutation_ids_sha256(values) == digest, f"Mutation identity mismatch: {case_id}")
        k = _integer(row["true_k"], "true_k")
        _require(k <= n, f"true_k exceeds population: {case_id}")
        result[case_id] = {"n_mutations": n, "mutation_ids_sha256": digest, "true_k": k,
                           "true_smf": _number(row["true_smf"], "true_smf", 0, 1),
                           "mutation_ids": set(values)}
    return result


def _method(reader: _Reader, binding: dict[str, Any], cohort: str, population_id: str,
            population: dict[str, Any], *, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(isinstance(binding, dict), f"Missing {role} binding")
    allowed = {*IDENTITY_FIELDS, "metrics", "runtime", "cohort_specific_settings", "identity_note"}
    _require(set(binding) <= allowed, f"Unknown {role} binding fields: {sorted(set(binding) - allowed)}")
    _require(isinstance(binding.get("method_id"), str) and bool(binding["method_id"]),
             f"Missing {role} method_id")
    identity = {key: binding.get(key) for key in IDENTITY_FIELDS}
    if role in {"baseline", "candidate"}:
        for key in IDENTITY_FIELDS[1:]:
            _sha(identity[key], key)
    else:
        for key in IDENTITY_FIELDS[1:]:
            if identity[key] is not None:
                _sha(identity[key], key)
    if role == "candidate":
        _require(binding.get("cohort_specific_settings") is False,
                 "Candidate must explicitly declare cohort_specific_settings=false")
    if role == "baseline":
        _require(identity["source_sha256"] == ACCEPTED_BASELINE_SOURCE and
                 identity["scientific_config_sha256"] == ACCEPTED_BASELINE_CONFIG,
                 "Baseline source/config differs from the accepted campaign contract")
    columns = {"cohort_id", "population_id", "method_id", "case_id", "n_mutations",
               "mutation_ids_sha256", "true_k", "true_smf", "selected_k", "ari", "ccf_mae",
               "estimated_smf", "status", "source_sha256", "scientific_config_sha256"}
    selected = [row for row in reader.table(binding["metrics"], columns)
                if row["cohort_id"] == cohort and row["population_id"] == population_id
                and row["method_id"] == identity["method_id"]]
    rows = _by_case(selected)
    _require(set(rows) == set(population),
             f"{role} case coverage mismatch: missing={len(set(population) - set(rows))}, "
             f"extra={len(set(rows) - set(population))}")
    result = {}
    for case_id, row in rows.items():
        authority = population[case_id]
        _require(row["status"] in VALID_STATUSES, f"Unpublishable {role} case: {case_id}")
        for key in IDENTITY_FIELDS[1:]:
            if identity[key] is not None:
                _require(row[key] == identity[key], f"{role} {key} mismatch: {case_id}")
        n = _integer(row["n_mutations"], "n_mutations")
        k = _integer(row["true_k"], "true_k")
        smf = _number(row["true_smf"], "true_smf", 0, 1)
        _require(n == authority["n_mutations"] and row["mutation_ids_sha256"] == authority["mutation_ids_sha256"]
                 and k == authority["true_k"] and abs(smf - authority["true_smf"]) <= NUMERIC_EPSILON,
                 f"{role} population/truth mismatch: {case_id}")
        selected_k = _integer(row["selected_k"], "selected_k")
        _require(selected_k <= n, f"selected_k exceeds population: {case_id}")
        estimate = _number(row["estimated_smf"], "estimated_smf", 0, 1)
        _require(selected_k != 1 or abs(estimate) <= NUMERIC_EPSILON,
                 f"K=1 requires sMF=0: {case_id}")
        result[case_id] = dict(n_mutations=n, true_k=k, true_smf=smf, selected_k=selected_k,
                              estimated_smf=estimate, status=row["status"],
                              ari=_number(row["ari"], "ari", -1, 1),
                              ccf_mae=_number(row["ccf_mae"], "ccf_mae", 0, 1))
    return result, identity


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def concordance(x: list[float], y: list[float]) -> float:
    """Population-moment Lin CCC; identical constants=1, other constants=0."""
    _require(len(x) == len(y) and len(x) >= 2, "CCC needs two or more matched pairs")
    if min(x) == max(x) or min(y) == max(y):
        return float(x == y)
    mx, my = _mean(x), _mean(y)
    vx = _mean([(v - mx) ** 2 for v in x])
    vy = _mean([(v - my) ** 2 for v in y])
    covariance = _mean([(a - mx) * (b - my) for a, b in zip(x, y)])
    return 2 * covariance / (vx + vy + (mx - my) ** 2)


def summarize(rows: dict[str, Any]) -> dict[str, Any]:
    values = [rows[key] for key in sorted(rows)]
    multi = [v for v in values if v["true_k"] > 1]
    single = [v for v in values if v["true_k"] == 1]
    return dict(cases=len(values), mutations=sum(v["n_mutations"] for v in values),
                mean_ari=_mean([v["ari"] for v in values]),
                multi_cluster_mean_ari=_mean([v["ari"] for v in multi]) if multi else None,
                mean_ccf_mae=_mean([v["ccf_mae"] for v in values]),
                smf_ccc=concordance([v["true_smf"] for v in values], [v["estimated_smf"] for v in values]),
                mean_smf_mae=_mean([abs(v["estimated_smf"] - v["true_smf"]) for v in values]),
                true_k_one_cases=len(single), multi_cluster_cases=len(multi),
                true_k_one_false_splits=sum(v["selected_k"] > 1 for v in single),
                false_k_one_collapses=sum(v["selected_k"] == 1 for v in multi),
                incomplete_search_cases=sum(v["status"] == "validated_incomplete" for v in values))


def compare(candidate: dict[str, Any], reference: dict[str, Any], *, safeguards: bool = True) -> dict[str, Any]:
    current, prior = summarize(candidate), summarize(reference)
    checks = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        a, b = current[metric], prior[metric]
        if a is None and b is None:
            checks[metric] = dict(passed=True, candidate=None, reference=None, delta=None,
                                  applicable=False)
        else:
            delta = a - b
            checks[metric] = dict(passed=direction * delta >= -NUMERIC_EPSILON,
                                  candidate=a, reference=b, delta=delta, applicable=True)
    deltas = []
    for case_id in sorted(candidate):
        a, b = candidate[case_id], reference[case_id]
        deltas.append(dict(case_id=case_id, ari_delta=a["ari"] - b["ari"],
                           ccf_mae_delta=a["ccf_mae"] - b["ccf_mae"],
                           smf_mae_delta=abs(a["estimated_smf"] - a["true_smf"]) -
                           abs(b["estimated_smf"] - b["true_smf"])))
    wins = sum(v["ari_delta"] > NUMERIC_EPSILON for v in deltas)
    losses = sum(v["ari_delta"] < -NUMERIC_EPSILON for v in deltas)
    required = tuple(METRIC_DIRECTIONS) if safeguards else HEADLINE_METRICS
    return dict(passed=all(checks[key]["passed"] for key in required), checks=checks,
                required_metrics=list(required),
                ari_wins=wins, ari_losses=losses, ari_ties=len(deltas) - wins - losses,
                worst_ari_cases=sorted(deltas, key=lambda v: (v["ari_delta"], v["case_id"]))[:5],
                worst_ccf_cases=sorted(deltas, key=lambda v: (-v["ccf_mae_delta"], v["case_id"]))[:5],
                worst_smf_cases=sorted(deltas, key=lambda v: (-v["smf_mae_delta"], v["case_id"]))[:5])


def _is_retained(population: dict[str, Any], inventory: dict[str, Any]) -> bool:
    protected = True
    for key, row in population.items():
        authority = inventory[key]
        full = (row["n_mutations"] == int(authority["n_scorable_retained_mutations"])
                and row["mutation_ids_sha256"] == authority["scorable_retained_mutation_ids_sha256"])
        if full:
            union = row["mutation_ids"] | authority["_truth_unmatched_ids"]
            _require(mutation_ids_sha256(list(union)) == authority["retained_mutation_ids_sha256"],
                     f"Scorable plus excluded IDs do not recover retained identity: {key}")
        protected &= full
    return protected


def evaluate_manifest(path: str | Path) -> dict[str, Any]:
    """Return a fail-closed JSON-compatible report; never run fits or write inputs."""
    path = Path(path).resolve()
    report: dict[str, Any] = dict(schema_version=SCHEMA_VERSION, gate_passed=False,
        contract_id=CONTRACT_ID, evaluator_sha256=_hash_bytes(Path(__file__).read_bytes()),
        observed_nonregression=False, best_among_declared_comparators=False,
        full_cohort_coverage=False, retained_population_protected=False,
        candidate_identity=None, baseline_self_check=False, statistical_superiority_proven=False,
        release_ready=False, numeric_epsilon=NUMERIC_EPSILON, cohorts={}, integrity_errors=[],
        retained_population_definition="All source-retained mutations with unambiguous bound truth; exclusions bound separately.",
        interpretation="Observed cohort-specific comparisons only; no statistical, CUDA, held-out or release claim.")
    reader = _Reader(path.parent)
    identities = []
    baseline_identities = []
    try:
        content = path.read_bytes()
        report["manifest_sha256"] = _hash_bytes(content)
        manifest = _json(content)
        _require(isinstance(manifest, dict), "Manifest must be a JSON object")
        _require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported schema_version")
        _require(set(manifest) == {"schema_version", "cohorts"}, "Unknown manifest-level fields")
        entries = manifest["cohorts"]
        _require(isinstance(entries, list) and len(entries) == len(EXPECTED_COHORT_SIZES),
                 "All three cohorts are required")
        _require({v["cohort_id"] for v in entries} == set(EXPECTED_COHORT_SIZES),
                 "Duplicate, missing or unknown cohort")
        for entry in entries:
            cohort = entry["cohort_id"]
            result: dict[str, Any] = dict(valid=False, observed_nonregression=False,
                                         best_among_declared_comparators=False,
                                         retained_population_protected=False, full_cohort_coverage=False)
            report["cohorts"][cohort] = result
            try:
                _require(entry["scope"] in {"development_snapshot", "full_cohort"}, "Invalid scope")
                inventory = _inventory(reader, entry)
                population = _population(reader, entry["population"], inventory)
                full = set(population) == set(inventory)
                _require(entry["scope"] != "full_cohort" or full,
                         "full_cohort scope requires every eligible case")
                population_id = entry["population"]["population_id"]
                result.update(scope=entry["scope"], planned_cases=len(inventory),
                              evaluated_cases=len(population), full_cohort_coverage=full and entry["scope"] == "full_cohort")
                result["truth_unmatched_exclusions"] = {
                    "planned_mutations": sum(len(v["_truth_unmatched_ids"]) for v in inventory.values()),
                    "evaluated_cases": {key: sorted(inventory[key]["_truth_unmatched_ids"])
                                        for key in population if inventory[key]["_truth_unmatched_ids"]},
                }
                baseline, baseline_identity = _method(reader, entry["baseline"], cohort,
                    population_id, population, role="baseline")
                result["baseline_summary"] = summarize(baseline)
                candidate, identity = _method(reader, entry["candidate"], cohort,
                    population_id, population, role="candidate")
                baseline_identities.append(baseline_identity)
                identities.append(identity)
                result["candidate_identity"] = identity
                result["candidate_summary"] = summarize(candidate)
                result["baseline_comparison"] = compare(candidate, baseline)
                comparisons = {}
                specs = entry["comparators"]
                _require(isinstance(specs, list) and len(specs) > 0, "At least one declared comparator is required")
                _require(REQUIRED_COMPARATORS[cohort] <= {v["method_id"] for v in specs},
                         f"Missing required campaign comparators: {sorted(REQUIRED_COMPARATORS[cohort] - {v['method_id'] for v in specs})}")
                for binding in specs:
                    comparator, comparator_identity = _method(reader, binding, cohort,
                        population_id, population, role="comparator")
                    method_id = comparator_identity["method_id"]
                    _require(method_id not in comparisons and method_id not in
                             {identity["method_id"], baseline_identity["method_id"]},
                             "Comparator must be unique and distinct from candidate/baseline")
                    comparisons[method_id] = dict(identity=comparator_identity,
                        summary=summarize(comparator), **compare(candidate, comparator, safeguards=False))
                result["comparators"] = comparisons
                protected = _is_retained(population, inventory)
                protections = {}
                for panel in entry.get("protection_panels", []):
                    pp = _population(reader, panel["population"], inventory)
                    panel_id = panel["population"]["population_id"]
                    _require(panel_id != population_id and panel_id not in protections,
                             "Protection population IDs must be distinct")
                    _require(set(pp) == set(population), "Protection case coverage differs from comparison")
                    _require(all(population[k]["mutation_ids"] <= pp[k]["mutation_ids"] for k in pp),
                             "Protection population does not contain comparison mutation IDs")
                    pb, bi = _method(reader, panel["baseline"], cohort, panel_id, pp, role="baseline")
                    pc, ci = _method(reader, panel["candidate"], cohort, panel_id, pp, role="candidate")
                    _require(ci == identity and bi == baseline_identity, "Protection scientific identity mismatch")
                    verdict = compare(pc, pb)
                    protections[panel_id] = verdict
                    if _is_retained(pp, inventory):
                        protected = True
                result["protection_panels"] = protections
                result["retained_population_protected"] = protected
                result["observed_nonregression"] = (result["baseline_comparison"]["passed"] and
                    all(v["passed"] for v in protections.values()) and protected)
                result["best_among_declared_comparators"] = all(v["passed"] for v in comparisons.values())
                result["valid"] = True
            except (GateInputError, KeyError, TypeError, ValueError, OSError) as exc:
                result["error"] = str(exc)
                report["integrity_errors"].append(f"{cohort}: {exc}")
        valid = len(report["cohorts"]) == 3 and all(v["valid"] for v in report["cohorts"].values())
        unique = {tuple(identity[k] for k in IDENTITY_FIELDS) for identity in identities}
        if len(unique) > 1:
            report["integrity_errors"].append("Candidate source/scientific config/method differs across cohorts")
            valid = False
        if valid:
            report["candidate_identity"] = identities[0]
            report["baseline_self_check"] = all(
                all(a[k] == b[k] for k in IDENTITY_FIELDS[1:])
                for a, b in zip(identities, baseline_identities))
            report["retained_population_protected"] = all(v["retained_population_protected"] for v in report["cohorts"].values())
            report["observed_nonregression"] = all(v["observed_nonregression"] for v in report["cohorts"].values())
            report["best_among_declared_comparators"] = all(v["best_among_declared_comparators"] for v in report["cohorts"].values())
            report["full_cohort_coverage"] = all(v["full_cohort_coverage"] for v in report["cohorts"].values())
            report["gate_passed"] = (report["observed_nonregression"] and
                report["best_among_declared_comparators"] and report["full_cohort_coverage"]
                and not report["baseline_self_check"])
        report["assessment"] = ("engineering_self_check_only" if report["baseline_self_check"] else
            "full_cohort_observed_gate_passed" if report["gate_passed"] else
            "failed_or_not_ready")
    except (GateInputError, KeyError, TypeError, ValueError, OSError) as exc:
        report["integrity_errors"].append(str(exc))
        report["assessment"] = "failed_or_not_ready"
    report["bound_artifacts"] = reader.artifacts
    return report


def example_manifest() -> dict[str, Any]:
    """Return an intentionally not-ready scaffold; zero hashes are placeholders."""
    def artifact(path: str) -> dict[str, str]:
        return {"path": path, "sha256": "0" * 64}

    cohorts = []
    for cohort in EXPECTED_COHORT_SIZES:
        cohorts.append(dict(cohort_id=cohort, scope="development_snapshot",
            full_inventory=artifact(f"{cohort}/full_inventory.tsv"),
            truth_exclusions=artifact(f"{cohort}/truth_exclusions.json"),
            population=dict(population_id="matched", table=artifact(f"{cohort}/population.tsv"),
                            mutation_ids=artifact(f"{cohort}/matched_ids.json")),
            baseline=dict(method_id="clipp15_baseline", source_sha256=ACCEPTED_BASELINE_SOURCE,
                          scientific_config_sha256=ACCEPTED_BASELINE_CONFIG, metrics=artifact("metrics.tsv")),
            candidate=None,
            comparators=[dict(method_id=name, metrics=artifact("metrics.tsv"))
                         for name in sorted(REQUIRED_COMPARATORS[cohort])],
            protection_panels=[]))
    return dict(schema_version=SCHEMA_VERSION, cohorts=cohorts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-manifest", type=Path)
    args = parser.parse_args()
    if args.example_manifest:
        if args.manifest or args.output:
            parser.error("--example-manifest cannot accompany a manifest or --output")
        with args.example_manifest.open("x", encoding="utf-8") as handle:
            json.dump(example_manifest(), handle, indent=2, allow_nan=False)
            handle.write("\n")
        return 0
    if not args.manifest:
        parser.error("manifest is required")
    report = evaluate_manifest(args.manifest)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
