"""Per-case failure classification shared by frozen cohort launchers.

A fitting failure has no validated result. It can be isolated from independent
cases; input/setup or output-integrity failures instead stop new admissions.
Already admitted healthy fits should drain before lifecycle cleanup.
"""

import traceback


VALIDATED_STATUSES = frozenset({"validated_complete", "validated_incomplete"})
ISOLATED_FAILURES = frozenset(
    {
        "scientific_failure",
        "resource_failure",
        "resource_timeout",
        "execution_failure",
        "infrastructure_failure",
    }
)
DRAIN_FAILURES = frozenset({"setup_failure", "output_validation_failure"})
CASE_STATUSES = VALIDATED_STATUSES | ISOLATED_FAILURES | DRAIN_FAILURES


def failure_record(error, phase):
    """Preserve the error while separating fit failure from integrity failure."""
    import torch
    from clipp1d.types import NumericalQualificationError

    if phase not in ("hardware", "setup", "fit", "validation"):
        raise ValueError("Unknown case execution phase")
    if phase == "validation":
        status = "output_validation_failure"
    elif phase == "setup":
        status = "setup_failure"
    elif phase == "hardware":
        status = "infrastructure_failure"
    elif isinstance(error, (MemoryError, torch.OutOfMemoryError)):
        status = "resource_failure"
    elif isinstance(error, RuntimeError) and any(
        marker in str(error).lower()
        for marker in (
            "cuda error: unspecified launch failure",
            "cuda error: system not yet initialized",
            "error 802: system not yet initialized",
            "cuda error: device is lost",
            "cuda error: unknown error",
        )
    ):
        status = "infrastructure_failure"
    elif isinstance(error, NumericalQualificationError):
        status = "scientific_failure"
    else:
        status = "execution_failure"
    return dict(
        status=status,
        error_type=type(error).__name__,
        error=str(error),
        traceback="".join(traceback.format_exception(error)),
    )


def requires_drain(status):
    """Fail closed on unknown terminal states, without calling them success."""
    if status not in CASE_STATUSES:
        raise ValueError(f"Unknown cohort terminal status: {status}")
    return status in DRAIN_FAILURES
