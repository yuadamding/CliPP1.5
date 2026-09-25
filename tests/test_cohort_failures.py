import pytest
import torch

from benchmarks.cohort_failures import (
    CASE_STATUSES,
    ISOLATED_FAILURES,
    failure_record,
    requires_drain,
)
from clipp1d.cuda.policy import QualificationError


@pytest.mark.parametrize(
    "error,status",
    [
        (QualificationError("unqualified"), "scientific_failure"),
        (MemoryError("host allocation"), "resource_failure"),
        (torch.OutOfMemoryError("device allocation"), "resource_failure"),
        (ValueError("Invalid finite symmetric capacities"), "execution_failure"),
        (RuntimeError("unexpected fitter failure"), "execution_failure"),
        (RuntimeError("CUDA error: unspecified launch failure"), "infrastructure_failure"),
        (RuntimeError("Error 802: system not yet initialized"), "infrastructure_failure"),
    ],
)
def test_fit_failures_are_retained_and_isolated(error, status):
    record = failure_record(error, "fit")
    assert record["status"] == status
    assert record["error"] == str(error)
    assert record["error_type"] == type(error).__name__
    assert type(error).__name__ in record["traceback"]
    assert not requires_drain(record["status"])
    assert record["status"] in ISOLATED_FAILURES
    assert not record["status"].startswith("validated_")


@pytest.mark.parametrize(
    "phase,status",
    [
        ("setup", "setup_failure"),
        ("validation", "output_validation_failure"),
    ],
)
@pytest.mark.parametrize("error", [ValueError("hash"), MemoryError("memory")])
def test_integrity_phase_failure_stops_admission_even_for_memory_errors(phase, status, error):
    record = failure_record(error, phase)
    assert record["status"] == status
    assert requires_drain(record["status"])


def test_unknown_status_and_phase_fail_closed():
    assert "execution_or_validation_failure" not in CASE_STATUSES
    with pytest.raises(ValueError, match="Unknown"):
        requires_drain("execution_or_validation_failure")
    with pytest.raises(ValueError, match="Unknown"):
        failure_record(ValueError("broken"), "unknown")


def test_hardware_failure_is_separate_from_setup_integrity_failure():
    error = RuntimeError("CUDA unavailable during worker hardware check")
    assert failure_record(error, "hardware")["status"] == "infrastructure_failure"
    assert failure_record(error, "setup")["status"] == "setup_failure"
