"""Check benchmark controls and evidence, without running a performance study."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_chain.py"
    spec = importlib.util.spec_from_file_location("controlled_chain_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_numerics()
    return module


def test_caps_separate_topology_and_strength(benchmark):
    n = 12
    complete, chain = benchmark.operators(n, True), benchmark.operators(n, False)
    for regime in benchmark.REGIMES:
        base = benchmark.REGIMES[regime][0]
        per_edge_complete = benchmark.penalty_caps(n, complete, regime, "per_edge")
        per_edge_chain = benchmark.penalty_caps(n, chain, regime, "per_edge")
        assert per_edge_complete.mean() == pytest.approx(base)
        assert per_edge_chain.mean() == pytest.approx(base)
        assert per_edge_complete.sum() / per_edge_chain.sum() == pytest.approx(n / 2)
        total_complete = benchmark.penalty_caps(n, complete, regime, "total_matched")
        total_chain = benchmark.penalty_caps(n, chain, regime, "total_matched")
        assert total_complete.sum() == pytest.approx(total_chain.sum())
        if benchmark.REGIMES[regime][1]:
            assert np.ptp(per_edge_chain) > 0
            assert np.ptp(per_edge_complete) > 0


def test_timing_retains_dispersion_and_samples(benchmark):
    result = benchmark.timing_summary([1., 2., 4.])
    assert result["samples_seconds"] == [1., 2., 4.]
    assert result["repetitions"] == 3
    assert result["median_seconds"] == 2.
    assert result["minimum_seconds"] == 1.
    assert result["maximum_seconds"] == 4.
    assert result["population_standard_deviation_seconds"] > 0


def test_memory_phase_counts_allocations_and_labels_process_peak(benchmark):
    result, memory = benchmark.measured_memory(lambda: bytearray(65_536))
    assert len(result) == 65_536
    assert memory["traced_new_peak_bytes"] >= 65_536
    assert memory["traced_new_live_bytes_at_return"] >= 65_536
    assert memory["process_lifetime_peak_rss_bytes_after"] >= memory["process_lifetime_peak_rss_bytes_before"]
    assert "not an isolated solver peak" in memory["scope"]
    assert "excluded from timing" in memory["phase"]


def test_controls_reject_configuration_after_numpy_import(benchmark):
    with pytest.raises(RuntimeError, match="before NumPy"):
        benchmark.configure_execution()


def test_fresh_cli_records_controls_qualifications_and_measured_memory(tmp_path, benchmark):
    outdir = tmp_path / "controls"
    command = [sys.executable, str(Path(benchmark.__file__)), "--outdir", str(outdir),
               "--sizes", "6", "--kernel-sizes", "6", "--profile-sizes", "6",
               "--regimes", "moderate_heterogeneous", "--strength-modes", "total_matched",
               "--repetitions", "2", "--kernel-repetitions", "3", "--threads", "1",
               "--first-order-seconds", ".5", "--profile-seconds", ".5"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    setup = json.loads((outdir / "setup.json").read_text())
    execution = setup["execution"]
    assert execution["controls_set_before_numpy_import"]
    assert set(execution["thread_environment"].values()) == {"1"}
    if execution["affinity_supported"]:
        assert len(execution["selected_cpus"]) == 1
        assert set(execution["selected_cpus"]) <= set(execution["allowed_cpus_before"])
    assert "no exclusivity claim" in execution["resource_scope"]
    kernel = json.loads((outdir / "kernel-6.json").read_text())
    assert all(len(arm["samples_seconds"]) == 3 for arm in kernel["arms"])
    qp = json.loads((outdir / "quadratic-6-moderate_heterogeneous-total_matched.json").read_text())
    assert qp["schema"] == "clipp1d.quadratic_benchmark.v2"
    assert qp["arms"][0]["caps_sum"] == pytest.approx(qp["arms"][1]["caps_sum"])
    for arm in qp["arms"]:
        assert len(arm["samples"]) == 2
        assert arm["qualified_samples"] + arm["unqualified_samples"] == 2
        assert arm["measured_solver_memory"]["traced_new_peak_bytes"] > 0
        assert arm["existing_input_array_bytes"] > 0
    comparison = qp["arms"][-1]["comparison_to_last_chain_first_order"]
    assert comparison["both_qualified"]
    assert comparison["max_coordinate_difference"] < 1e-5
    profile = json.loads((outdir / "profile-6.json").read_text())
    assert profile["all_witnesses_qualified"]
    assert len(profile["samples"]) == 2
    assert all(sample["independent_completed"] == 6 for sample in profile["samples"])


def test_chain_reference_certificate_uses_same_stable_gap(benchmark):
    from clipp1d.solver import quadratic_gap, quadratic_kkt
    h = np.array([98.3, 134.91])
    target = np.array([.225, .765])
    caps, lower, upper = np.array([.05123456]), np.zeros(2), np.ones(2)
    q = caps.copy()
    x = target - np.array([-q[0], q[0]]) / h
    gap, scale, kkt = benchmark.reference_certificate(x, q, h, target, lower, upper, caps,
                                                      benchmark.operators(2, False))
    expected_gap, expected_scale = quadratic_gap(x, q, h, target, lower, upper, caps)
    assert gap == expected_gap and scale == expected_scale
    assert kkt == quadratic_kkt(x, q, h, target, lower, upper, caps)
