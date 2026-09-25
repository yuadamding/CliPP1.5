"""Exercise terminal and cleanup decisions against retained failure conditions."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from benchmarks import cohort_failures


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    def save(path, value):
        with path.open("x") as stream:
            json.dump(value, stream)

    def read(path):
        return json.loads(path.read_text())

    pay, out, evidence = tmp_path / "payload", tmp_path / "output", tmp_path / "evidence/gpu"
    for folder in (pay, out, evidence):
        folder.mkdir(parents=True)
    save(pay / "plan.json", {"cases": [{"key": "first"}, {"key": "second"}]})
    save(pay / "gpu.manifest.json", {"metadata": {"name": "owned"}})
    common = types.SimpleNamespace(
        ROOT=tmp_path,
        PAY=pay,
        OUT=out,
        save=save,
        read=read,
        digest=lambda p: "digest",
        now=lambda: "now",
        canonical=json.dumps,
        identity=lambda pid: {},
    )
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "cohort_failures", cohort_failures)
    monkeypatch.setattr(sys, "argv", ["lifecycle", "supervise", "gpu"])
    old_path = sys.path[:]
    spec = importlib.util.spec_from_file_location(
        "test_lifecycle", Path(__file__).parents[1] / "benchmarks/supervise_a100_cohort.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path[:] = old_path
    return module


def test_failed_job_keeps_condition_without_success_or_assertion(lifecycle):
    current = {
        "status": {
            "conditions": [{"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded"}]
        }
    }
    assert lifecycle.record_terminal(current, [], {"first": "infrastructure_failure"}, True)
    failure = lifecycle.read(lifecycle.E / "FAILED.json")
    assert not failure["passed"]
    assert failure["conditions"][0]["reason"] == "BackoffLimitExceeded"
    assert not (lifecycle.E / "COMPLETE.json").exists()


def test_completed_dispatchers_with_unstarted_cases_are_partial(lifecycle):
    current = {"status": {"conditions": [{"type": "Complete", "status": "True"}]}}
    assert lifecycle.record_terminal(current, [], {"first": "infrastructure_failure"}, True)
    assert lifecycle.read(lifecycle.E / "PARTIAL.json")["unfinished"] == ["second"]
    assert not (lifecycle.E / "COMPLETE.json").exists()


def test_unconfirmed_delete_is_not_reissued_or_reported_clean(lifecycle, monkeypatch):
    current = {"metadata": {"uid": "bound", "resourceVersion": "fresh"}}
    lifecycle.save(lifecycle.E / "delete-intent.json", {"preconditions": {"uid": "bound"}})
    monkeypatch.setattr(lifecycle, "job", lambda: current)
    monkeypatch.setattr(lifecycle, "pods", lambda uid: [])
    monkeypatch.setattr(lifecycle, "validate", lambda *args: current)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda delay: None)

    def no_mutation(*args):
        pytest.fail("repeated an uncertain delete")

    monkeypatch.setattr(lifecycle, "kube", no_mutation)
    assert not lifecycle.cleanup(current)
    assert not lifecycle.read(lifecycle.E / "cleanup-pending.json")["absence_verified"]
    assert not (lifecycle.E / "cleanup.json").exists()


def test_unreachable_pod_logs_do_not_hide_terminal_pod(lifecycle, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("kubectl logs", 40)

    monkeypatch.setattr(lifecycle.subprocess, "run", timeout)
    pod = {"metadata": {"name": "owned-pod", "uid": "bound"}}
    lifecycle.capture(pod)
    record = lifecycle.read(lifecycle.E / "pod-bound.terminal.json")
    assert record["pod"] == pod and record["logs_unavailable"] == "timeout"
