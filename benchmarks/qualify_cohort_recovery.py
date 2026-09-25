"""Compare a repaired-source full fit with a preserved successful canary."""

import csv
import io
import json
import math


def compare_canary(output, baseline):
    """Require identity, labels, fitted CCFs, score, and qualification semantics."""
    current = json.loads((output / "run.json").read_text())
    prior = json.loads(baseline["run.json"])
    for name in ("schema", "status", "graph_sha256", "partition_labels", "search_status"):
        if current[name] != prior[name]:
            raise ValueError(f"Paired full-fit mismatch: {name}")
    for name in ("raw_objective", "selected_lambda", "selection_score"):
        if not math.isclose(current[name], prior[name], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"Paired full-fit mismatch: {name}")

    def rows(text):
        return {r["mutation_id"]: r for r in csv.DictReader(io.StringIO(text), delimiter="\t")}

    old = rows(baseline["mutation_clusters.tsv"])
    new = rows((output / "mutation_clusters.tsv").read_text())
    if set(old) != set(new):
        raise ValueError("Paired retained mutation set changed")
    max_ccf_error = 0.0
    for key, row in new.items():
        for name in ("tumor_id", "sample_id", "status", "cluster_label", "designated_clonal"):
            if row[name] != old[key][name]:
                raise ValueError(f"Paired mutation identity/label mismatch: {key}, {name}")
        for name in ("pilot_ccf", "raw_ccf", "refitted_ccf"):
            delta = abs(float(row[name]) - float(old[key][name]))
            max_ccf_error = max(max_ccf_error, delta)
            if delta > 1e-9:
                raise ValueError(f"Paired CCF mismatch: {key}, {name}")
    for name in ("raw_qualified", "refit_qualified", "candidate_family"):
        if current["candidate_provenance"][name] != prior["candidate_provenance"][name]:
            raise ValueError(f"Paired certificate semantics mismatch: {name}")
    old_calls = rows(baseline["mutation_multiplicity.tsv"])
    new_calls = rows((output / "mutation_multiplicity.tsv").read_text())
    if set(old_calls) != set(new_calls):
        raise ValueError("Paired multiplicity coverage changed")
    for key in new_calls:
        for name in ("raw_multiplicity_call", "refitted_multiplicity_call"):
            if new_calls[key][name] != old_calls[key][name]:
                raise ValueError(f"Paired multiplicity mismatch: {key}, {name}")
    return dict(
        passed=True,
        mutations=len(new),
        max_ccf_error=max_ccf_error,
        labels_identical=True,
        graph_identical=True,
        before_source=prior["provenance"]["source_sha256"],
        after_source=current["provenance"]["source_sha256"],
        score_delta=current["selection_score"] - prior["selection_score"],
    )
