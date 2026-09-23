"""Qualification harness regression checks; no local CUDA claim or large CPU fit."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import qualify_cuda as qualification
from benchmarks import profile_scalar_cuda as profiling
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.policy import CudaPolicy


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def cpu_reference_measurements(monkeypatch):
    monkeypatch.setattr(qualification, "timed", lambda device, fn: (fn(), 0.0))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)


def test_scaling_inputs_are_exact_nondegenerate_and_canonical():
    for n in (16, 64, 256):
        host = qualification.scaling_fixture(n)
        i = np.arange(n)
        np.testing.assert_array_equal(host.alt, np.array([100, 250, 400])[i % 3] + (i // 3) % 41)
        np.testing.assert_array_equal(host.alt + host.ref, np.full(n, 1000))
        assert host.slope.shape == (n, 1)
        assert np.all(host.slope == 0.5) and np.all(host.upper == 1)
        assert len(np.unique(host.alt)) > 3
        assert tuple(sorted(host.mutation_ids)) == host.mutation_ids
        assert qualification.host_identity(host) == qualification.host_identity(qualification.scaling_fixture(n))


def test_reduced_output_checks_cover_all_original_probes(monkeypatch):
    cpu_reference_measurements(monkeypatch)
    events = []
    device = torch.device("cpu")
    qualification.kernel_cases(device, Kernels(device), Kernels(device),
                               lambda kind, detail: events.append((kind, detail)))
    likelihood = [detail for kind, detail in events if kind == "likelihood_parity"]
    assert len(likelihood) == 7
    for row in likelihood:
        for mode in ("eager", "compiled"):
            assert set(row[mode]["reduced_output_errors"]) == {
                "loss_only", "loss_gradient_loss", "loss_gradient_gradient"}


def test_warm_qp_comparisons_recompute_current_problem_certificates(monkeypatch):
    cpu_reference_measurements(monkeypatch)
    events = []
    qualification.warm_qp_cases(torch.device("cpu"), Kernels("cpu"), Kernels("cpu"),
                                lambda kind, detail: events.append((kind, detail)))
    attempts = [detail for kind, detail in events if kind == "qp_warm_attempt"]
    assert len(attempts) == 8 and all(row["qualified"] for row in attempts)
    assert {row["changed_problem"] for row in attempts} == {True, False}
    assert any(row["iterations"] == 0 and row["warm"] and not row["changed_problem"] for row in attempts)


def test_resource_probe_requires_production_budget_and_two_qualified_repeats(monkeypatch, tmp_path):
    cpu_reference_measurements(monkeypatch)
    events = []
    def recorder(kind, detail):
        events.append((kind, detail))
    with pytest.raises(ValueError, match="production QP budget"):
        qualification.resource_probe(torch.device("cpu"), Kernels("cpu"), 4, 256, recorder)
    qualification.resource_probe(torch.device("cpu"), Kernels("cpu"), 4,
                                  CudaPolicy().inner_max_iterations, recorder)
    final = [row for kind, row in events if kind == "resource_probe"]
    assert len(final) == 1 and final[0]["status"] == "qualified"
    assert len(final[0]["repeats"]) == 2
    original = qualification.solve_qp
    monkeypatch.setattr(qualification, "solve_qp", lambda *a, **k: replace(original(*a, **k), qualified=False))
    events.clear()
    with pytest.raises(AssertionError, match="production gap/KKT"):
        qualification.resource_probe(torch.device("cpu"), Kernels("cpu"), 4,
                                      CudaPolicy().inner_max_iterations, recorder)
    assert [kind for kind, _ in events] == ["resource_qp_attempt"]
    events.clear()
    with pytest.raises(AssertionError, match="production gap/KKT"):
        qualification.resource_probe(torch.device("cpu"), Kernels("cpu"), 4,
                                      CudaPolicy().inner_max_iterations, recorder, tmp_path)
    assert [kind for kind, _ in events] == ["resource_qp_attempt", "resource_failure_state"]
    saved = json.loads((tmp_path / "resource-4-repeat0-unqualified.json").read_text())
    assert saved["certificate"]["qp_qualified"] is False
    assert set(saved["state"]) == {"h", "target", "lower", "upper", "caps", "x", "dual"}
    assert saved["state"]["x"] == original(
        torch.tensor(saved["state"]["h"], dtype=torch.float64),
        torch.tensor(saved["state"]["target"], dtype=torch.float64),
        torch.tensor(saved["state"]["lower"], dtype=torch.float64),
        torch.tensor(saved["state"]["upper"], dtype=torch.float64),
        torch.tensor(saved["state"]["caps"], dtype=torch.float64), Kernels("cpu")
    ).x.tolist()


def public_fixture():
    phases = {key: 0.1 for key in (
        "input_preparation_seconds", "device_upload_and_compile_seconds", "pilot_seconds",
        "graph_build_seconds", "path_seconds", "refit_seconds", "final_device_qualification_seconds",
        "device_export_seconds", "output_preparation_seconds")}
    source = dict(phase_seconds=phases, numerical_completed_utc="2026-09-22T18:00:00+00:00",
                  output_prepared_utc="2026-09-22T18:00:01+00:00",
                  device_measurement_scope="upload through final qualification and completed device export",
                  elapsed_scope="through output preparation; excludes receipt serialization, durable publication and return bookkeeping",
                  publication_completion_scope="return-only operation_metrics; external caller receipt required for durable completion timing",
                  elapsed_seconds=1.0, through_device_export_seconds=0.9,
                  peak_allocated_bytes=1000, peak_reserved_bytes=2000)
    metrics = dict(publication_completed_utc="2026-09-22T18:00:02+00:00",
                   publication_seconds=0.2, elapsed_seconds=1.2)
    return SimpleNamespace(operation_metrics=metrics), dict(provenance=source)


def test_public_phase_scope_and_return_completion_are_separate():
    result, receipt = public_fixture()
    expected = deepcopy(receipt)
    detail = qualification.validate_public_measurements(result, receipt)
    assert receipt == expected
    assert detail["return_only_operation_metrics"] == result.operation_metrics
    assert "operation_metrics" not in receipt


@pytest.mark.parametrize("defect", ["negative", "missing", "nonfinite", "earlier_publication", "self_receipt", "early_peak"])
def test_public_phase_validation_rejects_invalid_scope(defect):
    result, receipt = public_fixture()
    source = receipt["provenance"]
    if defect in ("negative", "nonfinite"):
        source["phase_seconds"]["device_export_seconds"] = -1 if defect == "negative" else float("nan")
    elif defect == "missing":
        del source["phase_seconds"]["final_device_qualification_seconds"]
    elif defect == "earlier_publication":
        result.operation_metrics["publication_completed_utc"] = "2026-09-22T17:00:00+00:00"
    elif defect == "self_receipt":
        receipt["operation_metrics"] = result.operation_metrics
    else:
        source["device_measurement_scope"] = "before export"
    with pytest.raises(AssertionError):
        qualification.validate_public_measurements(result, receipt)


def test_common_profiler_fixture_identity_and_optional_counter_compatibility():
    mixed = profiling.fixture("mixed6")
    analytical = profiling.fixture("analytical32")
    assert profiling.identity(mixed) == profiling.identity(profiling.fixture("mixed6"))
    assert np.array_equal(mixed.slope, qualification.host_fixture("mixed_multiplicity").slope)
    assert np.array_equal(analytical.alt, qualification.scaling_fixture(32).alt)
    assert profiling.counters(SimpleNamespace()) == dict(integrity_counters={}, scalar_work_counters={})
    before = dict(integrity_counters={"full_checks": 4}, scalar_work_counters={})
    after = dict(integrity_counters={"full_checks": 6}, scalar_work_counters={"loss_only_calls": 5})
    assert profiling.difference(after, before) == {
        "integrity_counters": {"full_checks": 2}, "scalar_work_counters": {"loss_only_calls": 5}}


def test_common_profiler_preserves_failure_receipt_without_cuda(tmp_path, monkeypatch):
    out = tmp_path / "failed-profile.json"

    def unavailable(device):
        raise RuntimeError("CUDA unavailable test sentinel")

    monkeypatch.setattr(profiling, "require_cuda", unavailable)
    monkeypatch.setattr(profiling.sys, "argv", ["profile_scalar_cuda.py", "--out", str(out)])
    monkeypatch.setattr(profiling.signal, "signal", lambda *args: None)
    monkeypatch.setattr(profiling.signal, "alarm", lambda *args: None)
    assert profiling.main() == 1
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"] == "CUDA unavailable test sentinel"
    assert receipt["events_sha256"] == profiling.digest(out.with_suffix(".events.jsonl"))
    assert out.with_suffix(".artifacts").joinpath("source-and-plan.json").exists()
