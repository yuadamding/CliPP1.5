"""Replay bounded actual production failures, independently of generated stress."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from clipp1d.policy import Policy
from clipp1d.solver import solve_quadratic

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks/fixtures/df44e6a_failed_qps"
CAPTURE = json.loads((FIXTURES / "capture.json").read_text())


@pytest.mark.parametrize("case", CAPTURE["cases"], ids=lambda case: case["file"])
def test_actual_failed_qp_preserves_arrays_and_qualifies(case):
    file = FIXTURES / case["file"]
    assert hashlib.sha256(file.read_bytes()).hexdigest() == case["sha256"]
    data = np.load(file, allow_pickle=False)
    result = solve_quadratic(*(data[key] for key in ("h", "target", "lower", "upper", "caps")),
                             data["lower"])
    assert result.qualified, result.work
    assert result.gap <= Policy().inner_atol + Policy().inner_rtol * result.gap_scale
    assert result.kkt_residual <= Policy().inner_kkt_tol
