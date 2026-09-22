"""Diagnostic clocks must not double-count nested likelihood work."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def diagnosis(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("diagnose_outer_progress", directory / "diagnose_outer_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_audit_clocks_account_for_exclusive_substages(monkeypatch, diagnosis):
    times = iter([0., 2., 5., 7.])
    monkeypatch.setattr(diagnosis, "perf_counter", lambda: next(times))
    timers = diagnosis.Timers()
    with timers.measure("anchor_audit"):
        with timers.measure("likelihood"):
            pass
    assert timers.values["anchor_audit"] == dict(calls=1, inclusive_seconds=7., exclusive_seconds=4.)
    assert timers.values["likelihood"] == dict(calls=1, inclusive_seconds=3., exclusive_seconds=3.)
    assert sum(row["exclusive_seconds"] for row in timers.values.values()) == 7.
    assert not timers.stack


def test_diagnostic_timer_retains_exception_and_releases_stack(monkeypatch, diagnosis):
    times = iter([3., 8.])
    monkeypatch.setattr(diagnosis, "perf_counter", lambda: next(times))
    timers = diagnosis.Timers()
    with pytest.raises(ArithmeticError, match="original failure"):
        with timers.measure("audit"):
            raise ArithmeticError("original failure")
    assert timers.values["audit"] == dict(calls=1, inclusive_seconds=5., exclusive_seconds=5.)
    assert not timers.stack
