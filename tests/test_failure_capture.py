"""Failure fixtures and bounded full-path accounting must preserve numerical scope."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_scaling.py"
    spec = importlib.util.spec_from_file_location("failure_capture_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_numerics()
    return module


def quadratic():
    return (np.array([1., 2., 3.]), np.array([.1, .6, 1.]), np.zeros(3),
            np.ones(3), np.array([.2, .4]))


def test_actual_failure_captures_frozen_boxes_and_exact_reason(tmp_path, benchmark):
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "source-test"},
                                        {"lambda_value": 4., "start_index": 2}, limit=2)
    args = quadratic()

    def failed_qp(*unused):
        return SimpleNamespace(qualified=False, work={"failure_reason": "No feasible chain dual interval"},
                               gap=np.inf, gap_scale=np.longdouble(.125), kkt_residual=np.inf)

    wrapped_qp = recorder.wrap_finalizer(failed_qp)

    def profile(*arrays):
        h, target, lower, upper, caps = arrays
        fixed_lower = lower.copy()
        fixed_lower[1] = 1.
        fit = wrapped_qp(h, target, fixed_lower, upper, caps, lower)
        return SimpleNamespace(qualified=False, fit=fit)

    result = recorder.wrap_profile(profile)(*args)
    assert not result.qualified
    records = [json.loads(line) for line in (tmp_path / "failed/manifest.jsonl").read_text().splitlines()]
    assert len(records) == 1  # A failed selected QP is not captured twice by its profile.
    record = records[0]
    assert record["failure_reason"] == "No feasible chain dual interval"
    assert record["box_scope"] == "selected_witness_frozen_boxes"
    assert record["context"]["selected_witness"] == 1
    assert record["source_sha256"] == "source-test"
    path = tmp_path / "failed" / record["fixture"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["fixture_sha256"]
    with np.load(path) as captured:
        assert set(captured.files) == {"h", "target", "lower", "upper", "caps"}
        assert_array_equal(captured["lower"], [0., 1., 0.])
        assert_array_equal(captured["target"], args[1])
        assert_array_equal(captured["caps"], args[4])
    assert record["diagnostics"]["gap"] is None
    assert record["diagnostics"]["gap_scale"] == .125
    assert not any(name in record for name in recorder.names)
    assert recorder.work_snapshot()["finalization_calls_completed"] == 1
    assert recorder.work_snapshot()["quadratic_calls_completed"] == 0


def test_standalone_finalization_failure_counted_and_captured_once(tmp_path, benchmark):
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "test"}, {}, limit=2)

    def failed_finalization(*unused):
        return SimpleNamespace(qualified=False, work={"failure_reason": "quadratic_kkt_gate"},
                               gap=.01, gap_scale=1., kkt_residual=.1)

    finalizer = recorder.wrap_finalizer(failed_finalization)
    result = recorder.wrap_quadratic(lambda *arrays: finalizer(*arrays))(*quadratic())
    assert not result.qualified
    assert recorder.snapshot()["failed_calls"] == recorder.snapshot()["captured"] == 1
    assert recorder.snapshot()["duplicates_not_recaptured"] == 0
    work = recorder.work_snapshot()
    assert work["quadratic_calls_completed"] == work["finalization_calls_completed"] == 1


def test_direct_failure_before_finalization_is_still_captured(tmp_path, benchmark):
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "test"}, {}, limit=2)

    def failed_reconstruction(*unused):
        return SimpleNamespace(qualified=False, work={"failure_reason": "message reconstruction failure"},
                               gap=np.inf, gap_scale=0., kkt_residual=np.inf)

    result = recorder.wrap_quadratic(failed_reconstruction)(*quadratic())
    assert not result.qualified
    assert recorder.snapshot()["captured"] == 1
    assert recorder.work_snapshot()["finalization_calls_started"] == 0


def test_profile_arithmetic_and_value_gate_capture_original_boxes(tmp_path, benchmark):
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "test"}, {}, limit=2)
    args = quadratic()

    def arithmetic(*unused):
        raise ArithmeticError("prefix cancellation fixture")

    with pytest.raises(ArithmeticError, match="prefix cancellation"):
        recorder.wrap_profile(arithmetic)(*args)
    assert recorder.profile_boxes is None

    def value_gate(*unused):
        return SimpleNamespace(qualified=False, fit=SimpleNamespace(qualified=True), witness=2,
                               diagnostics={"prefix_value_error": np.longdouble(.002)},
                               surrogate_sha256="original-surrogate")

    recorder.wrap_profile(value_gate)(*args)
    records = [json.loads(line) for line in (tmp_path / "failed/manifest.jsonl").read_text().splitlines()]
    assert [row["failure_reason"] for row in records] == ["prefix cancellation fixture", "witness_profile_value_gate"]
    assert all(row["box_scope"] == "original_profile_boxes" for row in records)
    for record in records:
        with np.load(tmp_path / "failed" / record["fixture"]) as captured:
            assert_array_equal(captured["lower"], args[2])
            assert_array_equal(captured["upper"], args[3])
    recorder.wrap_profile(value_gate)(*args)
    assert recorder.snapshot()["captured"] == 2
    assert recorder.snapshot()["over_limit"] == 1
    assert len(list((tmp_path / "failed").glob("*.npz"))) == 2


def test_capture_deduplicates_within_bound_and_rejects_unbounded_limit(tmp_path, benchmark):
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "test"}, {}, limit=2)
    recorder.capture(quadratic(), "reason", "original_profile_boxes")
    recorder.capture(quadratic(), "reason", "original_profile_boxes")
    assert recorder.snapshot()["captured"] == 1
    assert recorder.snapshot()["duplicates_not_recaptured"] == 1
    with pytest.raises(ValueError, match="eight"):
        benchmark.FailureCapture(tmp_path / "invalid", {}, {}, limit=9)


def test_clean_longdouble_uses_json_floats(benchmark):
    cleaned = benchmark._clean({"x": np.longdouble(.3), "nested": np.array([np.longdouble(1)]),
                                "infinite": np.longdouble("inf"), "integer": np.int64(2)})
    assert isinstance(cleaned["x"], float)
    assert cleaned["infinite"] is None
    assert json.loads(json.dumps(cleaned, allow_nan=False))["integer"] == 2


def test_surrogate_progress_is_rate_limited_and_counts_nested_completions(tmp_path, benchmark, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(benchmark, "perf_counter", lambda: clock[0])
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "test"}, {}, limit=0)
    progress = []
    recorder.on_progress = lambda kind: progress.append((clock[0], kind, recorder.work_snapshot()))

    def quadratic_call(*unused):
        clock[0] += .4
        return SimpleNamespace(qualified=True)

    wrapped_quadratic = recorder.wrap_finalizer(quadratic_call)

    def profile_call(*arrays):
        fit = wrapped_quadratic(*arrays)
        clock[0] += .4
        return SimpleNamespace(qualified=True, fit=fit)

    wrapped_profile = recorder.wrap_profile(profile_call)
    for _ in range(3):
        wrapped_profile(*quadratic())
    assert len(progress) == 2
    assert all(after[0] - before[0] >= 1. for before, after in zip(progress, progress[1:]))
    assert progress[0][1] == "finalization"
    assert progress[0][2]["finalization_calls_completed"] == 2
    assert progress[0][2]["profile_calls_started"] == 2
    assert progress[0][2]["profile_calls_completed"] == 1
    work = recorder.work_snapshot()
    assert work["finalization_calls_completed"] == work["profile_calls_completed"] == 3
    assert work["quadratic_calls_completed"] == 0
    assert work["finalization_completed_seconds"] == pytest.approx(1.2)
    assert work["profile_completed_seconds"] == pytest.approx(2.4)
    assert work["surrogate_completed_seconds"] == pytest.approx(2.4)
    assert "must not be added" in work["scope"]


def test_legacy_profile_quadratic_timing_is_not_added_twice(tmp_path, benchmark, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(benchmark, "perf_counter", lambda: clock[0])
    recorder = benchmark.FailureCapture(tmp_path / "failed", {"source_sha256": "legacy"}, {},
                                        limit=0, finalizer_instrumented=False)

    def quadratic_call(*unused):
        clock[0] += .4
        return SimpleNamespace(qualified=True)

    wrapped_quadratic = recorder.wrap_quadratic(quadratic_call)

    def profile_call(*arrays):
        fit = wrapped_quadratic(*arrays)
        clock[0] += .6
        return SimpleNamespace(qualified=True, fit=fit)

    recorder.wrap_profile(profile_call)(*quadratic())
    work = recorder.work_snapshot()
    assert not work["finalizer_instrumented"]
    assert work["finalization_calls_completed"] == 0
    assert work["quadratic_calls_completed"] == work["profile_calls_completed"] == 1
    assert work["profile_nested_quadratic_calls_completed"] == 1
    assert work["quadratic_completed_seconds"] == pytest.approx(.4)
    assert work["profile_completed_seconds"] == pytest.approx(1.)
    assert work["surrogate_completed_seconds"] == pytest.approx(1.)


def test_completed_start_aggregates_sum_work_and_ignore_no_restart_minimum(benchmark):
    totals = {}
    benchmark.accumulate_start_metrics(totals, {
        "audit_context_count": 2, "audit_scan_count": 3, "audit_scan_seconds": .1,
        "interval_restart_count": 2, "interval_restart_length_sum": .3,
        "interval_restart_length_min": .1, "interval_restart_length_max": .2,
        "objective_decrease": 5., "largest_curvature_scale": 32., "outer_iterations": 4,
    })
    benchmark.accumulate_start_metrics(totals, {
        "audit_context_count": 5, "audit_scan_count": 6, "audit_scan_seconds": .2,
        "interval_restart_count": 0, "interval_restart_length_sum": 0.,
        "interval_restart_length_min": 0., "interval_restart_length_max": 0.,
        "objective_decrease": 2., "largest_curvature_scale": 4., "outer_iterations": 6,
    })
    assert totals["audit_context_count"] == 7
    assert totals["audit_scan_count"] == 9
    assert totals["audit_scan_seconds"] == pytest.approx(.3)
    assert totals["interval_restart_length_min"] == .1
    assert totals["interval_restart_length_max"] == .2
    assert totals["interval_restart_length_sum"] == .3
    assert totals["interval_restart_count"] == 2
    assert totals["objective_decrease"] == 7.
    assert totals["largest_curvature_scale"] == 32.
    assert totals["outer_iterations"] == 10


def test_partial_summary_prefers_current_search_complete(tmp_path, benchmark):
    common = {"peak_process_rss_kib": 100, "totals": {"starts_completed": 3},
              "failure_capture": {"captured": 1}, "start_failure_counts": {"specific failure": 1}}
    events = [dict(common, stage="start", search_policy="common_surrogate_multistart_v1"),
              dict(common, stage="raw_start"),
              dict(common, stage="raw_complete", seconds=.1, diagnostics={"search_complete": True}),
              dict(common, stage="raw_start")]
    (tmp_path / "progress.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events) + '{"stage":')
    result = benchmark.summarize(tmp_path, 3, "easy", 2., True, None)
    assert result["status"] == "timeout"
    assert result["search_status"] == "not_completed"
    assert result["raw_penalties_started"] == 2
    assert result["raw_penalties_finished"] == 1
    assert result["raw_penalties_interrupted"] == 1
    assert result["raw_penalties_incomplete_search"] == 0
    assert result["start_failure_counts"] == {"specific failure": 1}


@pytest.mark.parametrize("reference", [False, True])
def test_small_full_path_instruments_correct_policy(tmp_path, benchmark, reference):
    outdir = tmp_path / "path"
    package_source = tmp_path / "pinned-source"
    shutil.copytree(Path(benchmark.api.__file__).parent, package_source / "clipp1d",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    command = [sys.executable, str(Path(benchmark.__file__)), "--outdir", str(outdir),
               "--sizes", "3", "--scenarios", "easy", "--timeout-seconds", "15",
               "--max-failure-captures", "2", "--threads", "1", "--cpu-count", "1",
               "--package-source", str(package_source)]
    if reference:
        command.append("--reference-enumeration")
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    record = json.loads((outdir / "easy-n3/benchmark.json").read_text())
    assert record["status"] == "success", (outdir / "easy-n3/worker.log").read_text()
    assert record["setup"]["reference_enumeration"] is reference
    assert record["setup"]["execution"]["controls_set_before_numpy_import"]
    assert record["setup"]["package_source"] == str(package_source)
    assert record["setup"]["runtime_precision"]["floating_point_formats"]["longdouble"]["significand_bits"] == np.finfo(np.longdouble).nmant + 1
    assert record["failure_capture"]["limit"] == 2
    assert record["partial_totals"]["starts_completed"] > 0
    assert record["partial_totals"]["inner_iterations"] > 0
    assert record["partial_totals"]["outer_iterations"] > 0
    assert record["partial_totals"]["audit_context_count"] > 0
    assert record["surrogate_work"]["finalization_calls_completed"] > 0
    assert (record["surrogate_work"]["quadratic_calls_completed"] > 0) is reference
    assert (record["surrogate_work"]["profile_calls_completed"] == 0) is reference
    assert (record["partial_totals"]["profile_calls"] == 0) is reference
    assert record["raw_penalties_finished"] == 26
    assert record["raw_penalties_interrupted"] == 0
