import json

import pytest

from benchmarks.qualify_cohort_recovery import compare_canary


def fixture(tmp_path):
    run = dict(
        schema="test",
        status="success",
        graph_sha256="graph",
        partition_labels=[0],
        search_status="complete",
        raw_objective=2.0,
        selected_lambda=1.0,
        selection_score=3.0,
        candidate_provenance=dict(
            raw_qualified=True, refit_qualified=True, candidate_family="complete_graph_path"
        ),
        provenance=dict(source_sha256="old"),
    )
    baseline = {
        "run.json": json.dumps(run),
        "mutation_clusters.tsv": "mutation_id\ttumor_id\tsample_id\tstatus\tcluster_label\tdesignated_clonal\tpilot_ccf\traw_ccf\trefitted_ccf\na\tt\ts\tretained\t0\t1\t0.7\t0.7\t0.7\n",
        "mutation_multiplicity.tsv": "mutation_id\traw_multiplicity_call\trefitted_multiplicity_call\na\t1\t2\n",
    }
    for name, text in baseline.items():
        (tmp_path / name).write_text(text)
    run["provenance"]["source_sha256"] = "new"
    (tmp_path / "run.json").write_text(json.dumps(run))
    return baseline


def test_source_changes_are_explicit_while_outputs_must_match(tmp_path):
    result = compare_canary(tmp_path, fixture(tmp_path))
    assert result["passed"] and result["before_source"] == "old" and result["after_source"] == "new"
    assert result["max_ccf_error"] == 0 and result["score_delta"] == 0


@pytest.mark.parametrize(
    "file,old,new",
    [
        ("mutation_clusters.tsv", "0.7", "0.71"),
        ("mutation_clusters.tsv", "retained\t0", "retained\t1"),
        ("mutation_multiplicity.tsv", "a\t1\t2", "a\t1\t1"),
        ("run.json", '"score": 3.0', '"score": 4.0'),
    ],
)
def test_changed_results_cannot_pass_pair_gate(tmp_path, file, old, new):
    baseline = fixture(tmp_path)
    if file == "run.json":
        value = json.loads((tmp_path / file).read_text())
        value["selection_score"] = 4.0
        (tmp_path / file).write_text(json.dumps(value))
    else:
        text = (tmp_path / file).read_text()
        assert old in text
        (tmp_path / file).write_text(text.replace(old, new))
    with pytest.raises(ValueError, match="Paired"):
        compare_canary(tmp_path, baseline)
