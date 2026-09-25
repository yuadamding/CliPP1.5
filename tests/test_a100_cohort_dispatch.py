"""Failure injection through the real dispatcher, without scheduler mutations."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

from benchmarks import cohort_failures


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def worker(tmp_path, monkeypatch):
    pay, out = tmp_path / "payload", tmp_path / "output"
    for folder in (pay, out / "workers", out / "cases", out / "logs"):
        folder.mkdir(parents=True)
    common = types.SimpleNamespace(read=read, save=save, digest=digest, now=lambda: "test")
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "cohort_failures", cohort_failures)
    staging = types.SimpleNamespace(stage_case_input=lambda *args: None)
    monkeypatch.setitem(sys.modules, "cohort_staging", staging)
    validator = types.SimpleNamespace(
        validate=lambda *args: dict(
            metrics={}, output_sha256={}, source_sha256="source", search_status="complete"
        )
    )
    monkeypatch.setitem(sys.modules, "validator", validator)
    original_path = sys.path[:]
    spec = importlib.util.spec_from_file_location(
        "test_a100_worker", Path(__file__).parents[1] / "benchmarks/run_a100_cohort.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path[:] = original_path
    monkeypatch.setattr(module, "PAY", pay)
    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "hardware", lambda: {"gpu": "fake"})
    monkeypatch.setenv("POD_UID", "owned-pod")
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "1")
    save(out / "pod-owned-pod.admitted.json", {})
    save(out / "CAPACITY.json", {"passed": True})
    save(out / "CANARY.json", {"passed": True})
    keys = ["000001", "000002", "000003", "000004"]
    cases = [
        dict(
            key=k,
            truth_member=k,
            truth_sha256=hashlib.sha256(b"truth").hexdigest(),
            input_sha256="input",
            wall_minutes=1,
        )
        for k in keys
    ]
    plan = dict(cases=cases, queues=[[], keys], parallelism=2, source_commit="source")
    save(pay / "plan.json", plan)
    with zipfile.ZipFile(pay / "payload.zip", "w") as archive:
        for k in keys:
            archive.writestr(k, b"truth")
    monkeypatch.setattr(module, "verify", lambda: plan)
    return module, out, validator


def simulate(worker, monkeypatch, statuses):
    module, out, validator = worker
    calls = []

    def child(action, key, *args):
        assert action == "fit"
        calls.append(key)
        status = statuses[len(calls) - 1]
        if status == "validated_complete":
            save(out / "cases" / key / "validated.json", validator.validate())
            return 0
        save(out / "cases" / key / "failure.json", dict(status=status, error="injected"))
        return 20

    monkeypatch.setattr(module, "run_child", child)
    module.dispatch()
    return calls, read(out / "workers/01/terminal.json")


def test_isolated_solver_exception_does_not_cancel_following_cases(worker, monkeypatch):
    calls, terminal = simulate(
        worker, monkeypatch, ["execution_failure"] + ["validated_complete"] * 3
    )
    assert calls == ["000001", "000002", "000003", "000004"]
    assert terminal["complete"] and not terminal["drained"]
    out = worker[1]
    assert not (out / "HALT.json").exists()
    assert read(out / "cases/000001/terminal.json")["status"] == "execution_failure"
    assert read(out / "cases/000004/terminal.json")["status"] == "validated_complete"


@pytest.mark.parametrize(
    "statuses,count",
    [
        (["execution_failure"] * 3, 3),
        (["output_validation_failure"], 1),
        (["setup_failure"], 1),
    ],
)
def test_integrity_or_repeated_errors_drain_without_starting_more_cases(
    worker, monkeypatch, statuses, count
):
    calls, terminal = simulate(worker, monkeypatch, statuses)
    assert len(calls) == count
    assert terminal["drained"] and not terminal["complete"]
    assert read(worker[1] / "HALT.json")["drain"]
    assert not (worker[1] / "cases/000004").exists()


def test_existing_halt_allows_other_workers_to_exit_without_fit(worker, monkeypatch):
    save(worker[1] / "HALT.json", {"drain": True})
    calls, terminal = simulate(worker, monkeypatch, [])
    assert not calls
    assert terminal["drained"] and not terminal["complete"]


def test_device_failure_stops_affected_queue_without_halting_other_gpus(worker, monkeypatch):
    calls, terminal = simulate(worker, monkeypatch, ["infrastructure_failure"])
    assert calls == ["000001"]
    assert terminal["infrastructure_failure"] and not terminal["complete"]
    assert not terminal["drained"]
    assert not (worker[1] / "HALT.json").exists()
    assert not (worker[1] / "cases/000002").exists()


def test_hardware_initialization_error_publishes_classified_case_failure(worker, monkeypatch):
    module, out, _ = worker
    (out / "cases/000001").mkdir()
    staging = sys.modules["cohort_staging"]
    monkeypatch.setattr(staging, "staged_input_path", lambda *args: None, raising=False)
    monkeypatch.setattr(staging, "verify_staged_input", lambda *args: None, raising=False)

    def failed_gpu():
        raise RuntimeError("CUDA unavailable during worker hardware check")

    monkeypatch.setattr(module, "hardware", failed_gpu)
    assert module.fit_case("000001") == 20
    failure = read(out / "cases/000001/failure.json")
    assert failure["status"] == "infrastructure_failure"
    assert "failed_gpu" in failure["traceback"]
