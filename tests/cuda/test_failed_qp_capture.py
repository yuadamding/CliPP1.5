"""Diagnostic capture CPU smoke tests, not allocated-CUDA qualification."""

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import sys

import numpy as np
import pytest
import torch

from benchmarks import capture_failed_qp as capture
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.policy import CudaPolicy


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def collector(tmp_path):
    journal = capture.mixed.Journal(tmp_path / "capture.json")
    result = capture.Capture(journal, {"source_sha256": capture.FROZEN_SOURCE})
    result.context = dict(path_index=1, start_index=0, lambda_value=1.0)
    return result


def receipt_for(result):
    return dict(
        _receipt_path=result.journal.receipt_path,
        artifacts=result.journal.artifacts,
        source=result.source,
        policy=asdict(CudaPolicy(inner_max_iterations=1)),
    )


def test_gap_components_expose_normal_allocation_failure():
    h = tensor([128] * 3)
    target = tensor([37 / 64, 1 / 2, 127 / 256])
    lower, upper = tensor([1e-6] * 3), tensor([0.5, 0.5, 0.9])
    caps = tensor([[0, 0.125, 1], [0.125, 0, 0.125], [1, 0.125, 0]])
    x = tensor([0.5] * 3)
    good = tensor([[0, 0, -0.5], [0, 0, 0], [0.5, 0, 0]])
    bad = tensor([[0, -0.125, -1], [0.125, 0, 0.125], [1, -0.125, 0]])
    valid = capture.decompose_gap(x, good, h, target, lower, upper, caps)
    invalid = capture.decompose_gap(x, bad, h, target, lower, upper, caps)
    assert valid["total"] == valid["certificate"]["gap"] == 0
    assert invalid["node_quadratic"] == 0.00054931640625
    assert invalid["box_normal"] == invalid["edge"] == 0
    assert invalid["total"] == invalid["certificate"]["gap"]
    assert invalid["certificate"]["kkt"] == pytest.approx(0.15789473684210525)
    assert invalid["bounds"] == dict(lower_active=0, upper_active=2, fixed=0, interior=1)
    assert invalid["group_sizes"] == [3]


@pytest.mark.parametrize(
    "bits",
    [
        [0, 1 << 63, 1, 0x3FF0000000000001],
        [0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000123],
    ],
)
def test_tensor_arrays_roundtrip_literal_float64_bits_and_deduplicate(tmp_path, bits):
    result = collector(tmp_path)
    array = np.asarray(bits, dtype=np.uint64).view(np.float64)
    value = torch.from_numpy(array.copy())
    descriptor = result._array(value)
    assert result._array(value.clone()) == descriptor
    assert len(result.journal.artifacts) == 1
    loaded = capture.load_tensor(receipt_for(result), descriptor, "cpu")
    np.testing.assert_array_equal(loaded.numpy().view(np.uint64), bits)
    assert capture.load_tensor(receipt_for(result), None, "cpu") is None
    with pytest.raises(AssertionError, match="literal float64"):
        result._array(value.float())
    path = tmp_path / descriptor["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(AssertionError, match="JSON hash"):
        capture.load_tensor(receipt_for(result), descriptor, "cpu")


def test_tensor_hash_rejects_changed_values_even_when_file_hash_rebound(tmp_path):
    result = collector(tmp_path)
    descriptor = result._array(tensor([0.1, 0.2]))
    path = tmp_path / descriptor["path"]
    value = json.loads(path.read_text())
    value["values"][0] = 0.3
    path.write_text(json.dumps(value))
    descriptor["sha256"] = result.journal.artifacts[descriptor["path"]] = capture.common.sha(path)
    with pytest.raises(AssertionError, match="roundtrip exactly"):
        capture.load_tensor(receipt_for(result), descriptor, "cpu")


def test_literal_small_qp_return_is_unchanged_and_failed_states_are_preserved(tmp_path):
    result = collector(tmp_path)
    h = torch.arange(1.0, 6.0, dtype=torch.float64)
    target = tensor([0.2, 0.21, 0.8, 0.81, 0.5])
    lower, upper = torch.full_like(h, 1e-6), torch.ones_like(h)
    caps = torch.full((5, 5), 0.5, dtype=torch.float64)
    caps.fill_diagonal_(0)
    kernels = Kernels("cpu", compiled=False)
    policy = CudaPolicy(inner_max_iterations=1)
    original = capture.qp.solve_qp
    expected = original(h, target, lower, upper, caps, kernels, policy)
    assert not expected.qualified
    with result.instrument():
        actual = capture.qp.solve_qp(h, target, lower, upper, caps, kernels, policy)
    assert capture.qp.solve_qp is original
    for name in ("x", "dual", "gap", "scale", "kkt"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    assert actual.qualified is expected.qualified
    saved = result.pending
    assert saved["problem"]["start"] is saved["problem"]["dual"] is None
    assert all(saved["terminal_admm"][name] is not None for name in ("x", "z", "v", "q", "rho"))
    literal_target = saved["problem"]["target"].clone()
    target.add_(0.05)
    assert torch.equal(saved["problem"]["target"], literal_target)
    result.flush()
    assert len(result.entries) == 1
    record = capture.load_record(receipt_for(result), result.entries[0])
    assert record["original_compiled_certificate_replay"]["exact_bits_equal"]
    assert not record["returned"]["qualified"]
    for key in ("x", "z", "v", "q", "rho"):
        restored = capture.load_tensor(receipt_for(result), record["terminal_admm"][key], "cpu")
        assert torch.equal(restored, saved["terminal_admm"][key])
    if record["last_refined_state"] is not None:
        assert record["last_refined_state"]["input_q"] is not None


def test_rejected_attempt_and_final_refinement_keep_distinct_incoming_duals(tmp_path):
    result = collector(tmp_path)
    incoming = tensor([[0, 0.25], [-0.25, 0]])
    final = incoming * 0.5
    result.prepared(None, dict(candidate=tensor([0.2, 0.2]), q=incoming, tolerance=1e-5))
    result.refined(
        (None, 9),
        dict(
            candidate=tensor([0.3, 0.3]),
            candidate_q=final,
            q=incoming,
            stats=tensor([1, 1, 1]),
            steps=9,
        ),
    )
    assert result.latest_attempt["objective_admitted"] is False
    assert result.latest_attempt["q"] is None
    assert result.latest_refined["returned_qualified_candidate"] is False
    assert result.latest_refined["input_q"] is incoming
    assert result.latest_refined["q"] is final
    assert not torch.equal(result.latest_refined["x"], result.latest_attempt["x"])


def test_instrumentation_restores_all_entrypoints_after_exception(tmp_path):
    result = collector(tmp_path)
    before = (
        capture.selection.fit_lambda,
        capture.solver._solve_start,
        capture.solver.solve_qp,
        capture.qp.solve_qp,
    )
    with pytest.raises(RuntimeError, match="intentional"):
        with result.instrument():
            raise RuntimeError("intentional")
    assert before == (
        capture.selection.fit_lambda,
        capture.solver._solve_start,
        capture.solver.solve_qp,
        capture.qp.solve_qp,
    )


def test_capture_loader_rejects_direct_receipt_symlink(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(AssertionError, match="nonsymlink"):
        capture.load_capture(link, capture.common.sha(path))


def profile_receipt(name, family_size=None):
    profile = capture.PROFILES[name]
    family, nodes = next(iter(profile["failures"])) if family_size is None else family_size
    context = dict(profile["expected_contexts"][(family, nodes)][0],
                   source_sha256=profile["source_sha256"],
                   input_sha256=capture.INPUTS[(family, nodes)],
                   policy=asdict(CudaPolicy()))
    return dict(schema=capture.PROFILE_SCHEMA, source_profile=name,
                source=dict(source_sha256=profile["source_sha256"],
                            source_files=dict(profile["source_files"])),
                fixture_family=family, nodes=nodes, expected_failures=1,
                captured_failures=1, input_sha256=capture.INPUTS[(family, nodes)],
                policy=asdict(CudaPolicy()),
                captures=[dict(context=context)])


@pytest.mark.parametrize("profile_name", ["normalized276da", "bound8627"])
def test_named_current_capture_profile_and_archived_v1_are_distinct(profile_name):
    current = profile_receipt(profile_name)
    name, profile = capture.check_capture_profile(current)
    assert name == profile_name and profile["commit"] is None
    assert capture.profile_for_source(profile["source_sha256"])[0] == name
    legacy = dict(schema=capture.SCHEMA, source=dict(source_sha256=capture.FROZEN_SOURCE))
    assert capture.check_capture_profile(legacy)[0] == "baseline430"
    current["schema"] = capture.SCHEMA
    with pytest.raises(AssertionError, match="v1"):
        capture.check_capture_profile(current)


@pytest.mark.parametrize("fault", ["profile", "source", "file", "count", "input", "context",
                                   "context_source", "context_input", "context_policy", "policy"])
@pytest.mark.parametrize("profile_name", ["normalized276da", "bound8627"])
def test_current_capture_profile_rejects_identity_inventory_drift(fault, profile_name):
    value = deepcopy(profile_receipt(profile_name))
    if fault == "profile":
        value["source_profile"] = "baseline430"
    elif fault == "source":
        value["source"]["source_sha256"] = "0" * 64
    elif fault == "file":
        value["source"]["source_files"]["cuda/qp.py"] = "0" * 64
    elif fault == "count":
        value["captured_failures"] = 2
    elif fault == "input":
        value["input_sha256"] = "0" * 64
    elif fault == "policy":
        value["policy"]["inner_rtol"] *= 2
    elif fault == "context_source":
        value["captures"][0]["context"]["source_sha256"] = capture.FROZEN_SOURCE
    elif fault == "context_input":
        value["captures"][0]["context"]["input_sha256"] = "0" * 64
    elif fault == "context_policy":
        value["captures"][0]["context"]["policy"]["inner_rtol"] *= 2
    else:
        value["captures"][0]["context"]["qp_ordinal_within_start"] += 1
    with pytest.raises(AssertionError):
        capture.check_capture_profile(value)


@pytest.mark.parametrize("profile_name", ["normalized276da", "bound8627"])
def test_current_capture_cli_requires_named_profile_and_exact_failure_count(profile_name):
    family, nodes = next(iter(capture.PROFILES[profile_name]["failures"]))
    args = capture.parser().parse_args([
        "capture", "--fixture", family, "--nodes", str(nodes), "--expected-failures", "1",
        "--source-profile", profile_name, "--expected-source-sha256",
        capture.PROFILES[profile_name]["source_sha256"], "--out", "capture.json",
    ])
    assert args.source_profile == profile_name and args.expected_failures == 1


@pytest.mark.parametrize("field", ["path_index", "start_index", "outer_iteration", "backtrack_index",
                                   "backtracks", "qp_ordinal_within_start", "inflation", "lambda_value"])
def test_bound_profile_rejects_each_changed_failed_surrogate_context(field):
    value = profile_receipt("bound8627")
    value["captures"][0]["context"][field] += 1
    with pytest.raises(AssertionError, match="failure identities"):
        capture.check_capture_profile(value)


def test_bound_profile_captures_below256_without_inventing_backtrack_allocation():
    value = profile_receipt("bound8627", ("below_one", 256))
    assert capture.check_capture_profile(value)[0] == "bound8627"
    expected = capture.PROFILES["bound8627"]["expected_contexts"][("below_one", 256)][0]
    assert expected == dict(path_index=1, start_index=1, outer_iteration=1,
                           backtracks=2, qp_ordinal_within_start=4,
                           lambda_value=137.6911252669794)
    assert value["input_sha256"] == capture.INPUTS[("below_one", 256)]
    for key in expected:
        changed = deepcopy(value)
        changed["captures"][0]["context"][key] += 1
        with pytest.raises(AssertionError, match="failure identities"):
            capture.check_capture_profile(changed)
    for key in ("fixture_family", "input_sha256"):
        changed = deepcopy(value)
        changed[key] = "mixed_support" if key == "fixture_family" else capture.INPUTS[("below_one", 64)]
        with pytest.raises(AssertionError):
            capture.check_capture_profile(changed)


@pytest.mark.parametrize("family,nodes", capture.INPUTS)
def test_input_hashes_match_archived_failed_fixtures(family, nodes):
    actual = hashlib.sha256(capture.fixtures.fixture_text(family, nodes)).hexdigest()
    assert actual == capture.INPUTS[family, nodes]


@pytest.mark.parametrize("failure", ["timeout", "helper_mutation"])
def test_terminal_failure_keeps_partial_artifacts_and_never_claims_success(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "terminal.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_failed_qp.py",
            "capture",
            "--fixture",
            "below_one",
            "--nodes",
            "64",
            "--expected-failures",
            "2",
            "--expected-source-sha256",
            capture.FROZEN_SOURCE,
            "--out",
            str(path),
        ],
    )
    calls = 0

    def helpers():
        nonlocal calls
        calls += 1
        return dict(
            driver=capture.common.sha(capture.Path(capture.__file__)),
            fixtures="a" if calls == 1 else "b",
        )

    def execute(args, receipt, journal):
        journal.artifact("partial.json", {"completed": False})
        if failure == "timeout":
            raise TimeoutError("diagnostic deadline")

    monkeypatch.setattr(capture, "execute", execute)
    monkeypatch.setattr(capture, "helper_hashes", helpers)
    assert capture.main() == 1
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "failed"
    assert receipt.get("diagnostic_capture_qualified") is not True
    assert receipt.get("scientific_fit_qualified") is not True
    assert receipt["source"]["source_sha256"]
    assert len(receipt["artifacts"]) == 1
    assert receipt["events_sha256"] == capture.common.sha(tmp_path / receipt["events_file"])
