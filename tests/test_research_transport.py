import pytest

from benchmarks.research_transport import (
    job_missing, retryable_read_failure, terminal_state, verify_accepted,
    validate_import_paths,
)

ACCEPTED = {"job_id": "123", "job_name": "study"}
LOG = ("Subject: Job 123: <study>\n"
       "Job <study> was submitted from host <ldragon5> by user <yding4>\n"
       "Terminated at Thu Sep 24\nResults reported at Thu Sep 24\n"
       "Successfully completed.\n")


def test_retained_log_proves_aged_out_terminal():
    assert terminal_state(LOG, ACCEPTED, log=True) == "DONE"


@pytest.mark.parametrize("before,after", [("123", "456"), ("yding4", "other"),
                                         ("Terminated at", "Started at")])
def test_wrong_or_incomplete_log_is_rejected(before, after):
    with pytest.raises(ValueError):
        terminal_state(LOG.replace(before, after), ACCEPTED, log=True)


def test_missing_job_is_not_completion():
    with pytest.raises(ValueError):
        terminal_state("Job <123> is not found", ACCEPTED)


def test_running_is_not_completion():
    assert terminal_state("Job <123>, Job Name <study>, User <yding4>, "
                          "Status <RUN>", ACCEPTED) is None


def test_retry_policy_is_transport_only():
    assert retryable_read_failure(255)
    assert retryable_read_failure(None, timed_out=True)
    assert not retryable_read_failure(1)
    assert not retryable_read_failure(0)


def test_launch_acknowledgement_extends_accepted_schema():
    accepted = dict(ACCEPTED, plan_sha256="a" * 64, initially_held=True)
    verify_accepted(accepted, dict(accepted, released_acknowledged=True))
    with pytest.raises(ValueError):
        verify_accepted(accepted, dict(accepted, job_id="456"))


def test_zero_exit_with_missing_message_requires_log_proof():
    assert job_missing("", "Job <123> is not found\n", "123")
    assert not job_missing("", "Job <456> is not found\n", "123")
    assert not job_missing("Job <123>", "Job <123> is not found\n", "123")


def test_original_input_extension_requires_exact_path_and_hash(tmp_path):
    import hashlib

    path = tmp_path / "results/case.trials/input/tumor.clipp2.txt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original canonical input")
    bound = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_import_paths(tmp_path, [path], bound)
    with pytest.raises(ValueError, match="Unbound artifact"):
        validate_import_paths(tmp_path, [path], {})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        validate_import_paths(tmp_path, [path], bound)


def test_artifact_import_rejects_duplicate_traversal_and_symlink(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}")
    validate_import_paths(tmp_path, [path], {})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_import_paths(tmp_path, [path, path], {})
    with pytest.raises(ValueError, match="Unsafe"):
        validate_import_paths(tmp_path, [tmp_path / "x/../data.json"], {})
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="Symlink"):
        validate_import_paths(tmp_path, [link], {})
