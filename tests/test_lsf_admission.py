import pytest

from benchmarks.lsf_admission import memory_contract, verify_admission


def report(memory=8, state="PSUSP", owner="yding4", reservation=8):
    return (
        f"Job <123>, Job Name <test>, User <{owner}>, Status <{state}>, "
        "Queue <egpu>, Command <python worker.py>\n"
        f", 2 Task(s), Requested Resources <rusage[mem={reservation}] span[hosts=1]>, "
        "Requested GPU <num=1:mode=exclusive_process:gmodel=NVIDIAL40>\n"
        f" RUNLIMIT\n 60.0 min\n MEMLIMIT\n {memory} G\n"
    )


def verify(raw):
    return verify_admission(raw, {"job_id": "123", "job_name": "test"},
                            "python worker.py", 8, 60)


def test_matching_admission():
    assert verify(report()) == "PSUSP"


def test_lower_hard_limit_rejected_with_actual_values():
    with pytest.raises(ValueError, match="MEMLIMIT GB: expected 8, admitted 4"):
        verify(report(memory=4))


def test_under_reservation_rejected_before_submission():
    with pytest.raises(ValueError, match="reservation must equal"):
        memory_contract(8, 4)


@pytest.mark.parametrize("change", [{"state": "RUN"}, {"owner": "foreign"},
                                    {"reservation": 4}])
def test_foreign_or_unexpected_admission_rejected(change):
    with pytest.raises(ValueError):
        verify(report(**change))


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError, match="Expected one"):
        verify(report() + "Job <123>\n")
