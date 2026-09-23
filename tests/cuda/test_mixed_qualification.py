"""Fixture/schema CPU smoke checks; no large fit or allocated-CUDA claim."""

from copy import deepcopy
from dataclasses import asdict
import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import mixed_fixtures as fixtures
from benchmarks import qualify_cuda as common
from benchmarks import qualify_mixed_cuda as qualification
from clipp1d.cuda.graph import build_graph
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.scalar import pilot


@pytest.mark.parametrize("family", fixtures.FAMILIES)
def test_heterogeneous_fixture_prefixes_and_difficulty_are_preserved(tmp_path, family):
    previous = None
    for n in fixtures.SIZES:
        path = tmp_path / f"{family}-{n}.tsv"
        data, host = fixtures.write_fixture(path, family, n)
        assert path.read_bytes() == fixtures.fixture_text(family, n)
        assert host.mutation_ids == tuple(f"m{i:06d}" for i in range(n))
        assert len(data.mutations) == n and len(data.retained) == n
        assert np.all(host.valid.sum(1) >= 2) and host.valid.sum(1).max() == 4
        assert np.all(host.lower == 1e-6) and np.all(host.upper <= 1)
        if previous is not None:
            for name in ("alt", "ref", "slope", "lower", "upper", "log_prior", "valid"):
                np.testing.assert_array_equal(
                    getattr(host, name)[: len(previous)], getattr(previous, name)
                )
        summary = fixtures.difficulty_summary(host, data)
        fixtures.validate_difficulty(summary, family)
        assert summary["distinct_likelihood_rows"] == summary["distinct_depths"] == n
        assert summary["rows_with_multiple_competitive_wells"] > n * 0.8
        assert summary["purity"] == data.purity
        if family == "below_one":
            assert summary["original_upper_below_one"] == n
            assert np.all(host.upper < 1)
        previous = host


@pytest.mark.parametrize(
    "family,n", [("unknown", 64), ("mixed_support", 16), ("below_one", 1024), ("below_one", True)]
)
def test_fixture_cannot_silently_change_family_or_size(family, n):
    with pytest.raises(ValueError):
        fixtures.fixture_text(family, n)


def test_difficulty_gate_rejects_easy_repeated_or_unimodal_recipe(tmp_path):
    data, host = fixtures.write_fixture(tmp_path / "input.tsv", "mixed_support", 64)
    summary = fixtures.difficulty_summary(host, data)
    for key in (
        "distinct_likelihood_rows",
        "rows_with_multiple_competitive_wells",
        "upper_clipping_rows",
    ):
        changed = deepcopy(summary)
        changed[key] = 0
        with pytest.raises(AssertionError, match="lost"):
            fixtures.validate_difficulty(changed, "mixed_support")


def write_predecessor(root, *, source="a" * 64, nodes=64, family="mixed_support", status="passed"):
    root.mkdir()
    events = root / "events.jsonl"
    events.write_text("{}\n")
    artifact = root / "full.json"
    artifact.write_text("{}\n")
    receipt = dict(
        schema=qualification.SCHEMA,
        status=status,
        mode="full",
        fixture_family=family,
        fixture_recipe=fixtures.RECIPE,
        nodes=nodes,
        source=dict(source_sha256=source),
        helpers=qualification.helper_hashes(),
        cuda_available=True,
        device="cuda:0",
        policy=asdict(CudaPolicy()),
        elapsed_seconds=1.0,
        full_path=dict(search_status="complete", final_export_and_publication_qualified=True),
        artifacts={"full.json": common.sha(artifact)},
        events_file="events.jsonl",
        artifact_directory=".",
        events_sha256=common.sha(events),
    )
    path = root / "qualification.json"
    common.write_json(path, receipt)
    return path, common.sha(path)


def test_larger_size_requires_bound_same_source_smaller_qualified_fixture(tmp_path):
    source = dict(source_sha256="a" * 64)
    path, digest = write_predecessor(tmp_path / "previous")
    args = SimpleNamespace(
        nodes=256, fixture="mixed_support", predecessor=path, predecessor_sha256=digest
    )
    assert qualification.check_predecessor(args, source)["nodes"] == 64
    for key, value in [
        ("nodes", 512),
        ("fixture", "below_one"),
        ("predecessor_sha256", "b" * 64),
        ("predecessor", None),
    ]:
        modified = deepcopy(args)
        setattr(modified, key, value)
        with pytest.raises(AssertionError):
            qualification.check_predecessor(modified, source)
    with pytest.raises(AssertionError, match="same family/source"):
        qualification.check_predecessor(args, dict(source_sha256="b" * 64))
    original = json.loads(path.read_text())
    changed = deepcopy(original)
    changed["helpers"]["fixtures"] = "b" * 64
    path.write_text(json.dumps(changed))
    modified = deepcopy(args)
    modified.predecessor_sha256 = common.sha(path)
    with pytest.raises(AssertionError, match="same family/source/helpers"):
        qualification.check_predecessor(modified, source)
    path.write_text(json.dumps(original))
    args.predecessor_sha256 = common.sha(path)
    (path.parent / "events.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(AssertionError, match="event journal"):
        qualification.check_predecessor(args, source)


def test_failed_or_cpu_predecessor_is_not_admitted(tmp_path):
    path, digest = write_predecessor(tmp_path / "failed", status="failed")
    with pytest.raises(AssertionError, match="did not pass"):
        qualification.load_receipt(path, digest)
    path, _ = write_predecessor(tmp_path / "cpu")
    value = json.loads(path.read_text())
    value["device"] = "cpu"
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError, match="actual CUDA"):
        qualification.load_receipt(path, common.sha(path))


def tiny_shared_problem():
    # A three-row analytical scalar smoke only, not a complete fitting path.
    model = TensorModel.from_host(common.host_fixture("single_support"), "cpu", False)
    scalar = pilot(model)
    graph = build_graph(scalar.phi, model.mutation_ids)
    fitted = SimpleNamespace(
        model=model,
        pilot=scalar,
        graph=graph,
        lambda_value=torch.tensor(0.5),
        records=[dict(lambda_value=0.0), dict(lambda_value=0.1)],
        timings=dict(lambda_reference=1.0),
    )
    return model, qualification.shared_problem(fitted, "input-sha")


def test_shared_problem_restores_literal_pilot_graph_and_rejects_changes():
    model, payload = tiny_shared_problem()
    graph, scalar = qualification.restore_shared(model, payload, "input-sha")
    assert torch.equal(graph.pilot, scalar.phi)
    np.testing.assert_array_equal(graph.weights.numpy(), payload["graph"]["weights"])
    with pytest.raises(AssertionError, match="input or canonical"):
        qualification.restore_shared(model, payload, "different")
    changed = deepcopy(payload)
    changed["graph"]["weights"][0][1] *= 1.01
    changed["graph"]["weights"][1][0] *= 1.01
    with pytest.raises(AssertionError, match="exact pilot recipe"):
        qualification.restore_shared(model, changed, "input-sha")
    changed = deepcopy(payload)
    changed["pilot"]["loss"][0] += 1
    with pytest.raises(ValueError, match="qualified scalar"):
        qualification.restore_shared(model, changed, "input-sha")


def test_fixed_problem_does_not_allow_lambda_or_label_drift():
    reference = dict(
        lambda_value=0.25,
        raw_ccf=[0.2, 0.8],
        refitted_ccf=[0.2, 0.8],
        centers=[0.8, 0.2],
        score=15.0,
        raw_objective=10.0,
        labels=[1, 0],
        raw_multiplicity=[1, 2],
        refitted_multiplicity=[1, 2],
    )
    actual = deepcopy(reference)
    actual["lambda_value"] += 1e-12
    fixed = qualification.compare_snapshots(actual, reference, fixed_problem=True)
    assert not fixed["within_existing_parity_gates"] and fixed["differing_fields"] == [
        "literal_lambda"
    ]
    independent = qualification.compare_snapshots(actual, reference, fixed_problem=False)
    assert not independent["literal_lambda_equal"] and independent["within_existing_parity_gates"]
    actual["labels"] = [0, 1]
    assert (
        "labels"
        in qualification.compare_snapshots(actual, reference, fixed_problem=True)[
            "differing_fields"
        ]
    )


def test_adaptive_graph_difference_is_separate_and_bounds_feasible_penalty_changes():
    class Host(SimpleNamespace):
        def __len__(self):
            return 2

    host = Host(lower=np.array([0.1, 0.2]), upper=np.array([0.6, 0.8]))
    reference = dict(
        pilot_and_graph=dict(weights=[[0, 1], [1, 0]], pilot_phi=[0.2, 0.7]),
        selected_lambda=0.5,
        timings=dict(lambda_reference=2.0),
    )
    actual = deepcopy(reference)
    actual["pilot_and_graph"]["weights"] = [[0, 1.1], [1.1, 0]]
    actual["selected_lambda"] = 0.6
    report = qualification.graph_differences(actual, reference, host)
    assert report["graph_weights_identical"] is False
    assert report["selected_caps_max_absolute_difference"] == pytest.approx(0.16)
    assert report["selected_objective_perturbation_envelope"] == pytest.approx(0.16 * 0.7)
    assert "within_existing_parity_gates" not in report


def test_candidate_trace_persists_unresolved_raw_and_refit_failure(tmp_path, monkeypatch):
    def raw(*a, **k):
        return SimpleNamespace(
            qualified=False, diagnostics={"search_complete": False}, x=torch.tensor([0.2]),
            objective=torch.tensor(1.)
        )

    def refit(*a, **k):
        raise RuntimeError("refit failure")

    monkeypatch.setattr(qualification.selection, "fit_lambda", raw)
    monkeypatch.setattr(qualification.selection, "refit", refit)
    journal = qualification.Journal(tmp_path / "trace.json")
    with pytest.raises(RuntimeError, match="refit failure"):
        with qualification.trace_candidates(journal):
            result = qualification.selection.fit_lambda(None, None, None, torch.tensor(0.25))
            assert result.qualified is False
            qualification.selection.refit(None, None)
    assert qualification.selection.fit_lambda is raw and qualification.selection.refit is refit
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == [
        "candidate_started",
        "candidate_raw",
        "candidate_refit_failed",
    ]
    assert rows[1]["qualified"] is False and rows[2]["lambda_value"] == 0.25


def test_path_plan_survives_interruption_before_first_raw_fit(tmp_path, monkeypatch):
    value = torch.tensor(0.1, dtype=torch.float64)

    def reference(*args, **kwargs):
        return value

    monkeypatch.setattr(qualification.selection, "penalty_reference", reference)
    journal = qualification.Journal(tmp_path / "planned.json")
    with pytest.raises(TimeoutError):
        with qualification.trace_candidates(journal):
            assert qualification.selection.penalty_reference(SimpleNamespace(n=64)) is value
            raise TimeoutError("before first candidate")
    assert qualification.selection.penalty_reference is reference
    plan = json.loads((journal.root / "initial-path-plan.json").read_text())
    candidates = plan["initial_candidates"]
    assert len(candidates) == 26
    assert candidates[0] == dict(index=0, lambda_value=0.0, status="not_attempted")
    assert candidates[-1]["lambda_value"] == float(torch.ldexp(value, torch.tensor(12)))
    assert all(row["status"] == "not_attempted" for row in candidates)


def test_cli_failure_writes_terminal_receipt_without_skipped_success(tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify_mixed_cuda",
            "--fixture",
            "mixed_support",
            "--nodes",
            "64",
            "--mode",
            "full",
            "--expected-source-sha256",
            "a" * 64,
            "--out",
            str(out),
        ],
    )

    def fail(args, receipt, journal):
        journal.artifact("partial-path.json", dict(search_status="incomplete"))
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(qualification, "execute", fail)
    assert qualification.main() == 1
    saved = json.loads(out.read_text())
    assert saved["status"] == "failed" and saved["error"] == "deliberate failure"
    assert len(saved["source"]["source_sha256"]) == 64
    assert saved["artifacts"]["run.artifacts/partial-path.json"] == common.sha(
        tmp_path / "run.artifacts/partial-path.json"
    )
    assert saved["events_sha256"] == common.sha(tmp_path / "run.events.jsonl")
    with pytest.raises(FileExistsError):
        qualification.main()


def test_wall_timeout_preserves_failed_receipt_and_partial_artifact(tmp_path, monkeypatch):
    output = tmp_path / "timeout.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "qualify_mixed_cuda",
            "--fixture",
            "below_one",
            "--nodes",
            "64",
            "--mode",
            "full",
            "--expected-source-sha256",
            "a" * 64,
            "--timeout-seconds",
            "3000",
            "--out",
            str(output),
        ],
    )
    original_term, original_alarm = (
        signal.getsignal(signal.SIGTERM),
        signal.getsignal(signal.SIGALRM),
    )

    def interrupted(args, receipt, journal):
        journal.artifact("partial.json", {"planned_penalty": 1.0, "search_status": "incomplete"})
        signal.raise_signal(signal.SIGALRM)

    monkeypatch.setattr(qualification, "execute", interrupted)
    assert qualification.main() == 1
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "TimeoutError"
    assert receipt["timeout_seconds"] == 3000 and receipt["source"]["source_sha256"]
    assert "timeout.artifacts/partial.json" in receipt["artifacts"]
    assert signal.getsignal(signal.SIGTERM) is original_term
    assert signal.getsignal(signal.SIGALRM) is original_alarm
