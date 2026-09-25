import hashlib
import json

import pytest

from benchmarks.lsf_retry_lineage import validate_retry_lineage


@pytest.fixture
def records(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "receipts/000001").mkdir(parents=True)
    (new / "receipts/deferred-retries").mkdir(parents=True)
    path = old / "receipts/000001/reconciled.json"
    path.write_text(json.dumps({"accepted": {"job_id": "123", "key": "000001"},
                                "outcome": {"status": "resource_timeout"},
                                "scheduler_state": "DONE"}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    proof = {"root": str(old), "job_id": "123", "reconciled_sha256": digest}
    previous = {"source_root": str(old), "status": "resource_timeout"}
    return old, new, path, proof, previous


def test_legacy_and_new_generations(records):
    old, new, _, proof, previous = records
    legacy = {"timeout_retry_root": str(new), "retry_parents": {"000001": proof}}
    assert validate_retry_lineage(legacy, new, "000001", previous) == proof["reconciled_sha256"]
    modern = {"timeout_retry_root": "other", "retry_parents": {},
              "retry_generations": [{"root": str(new), "retry_parents": {"000001": proof}}]}
    assert validate_retry_lineage(modern, new, "000001", previous) == proof["reconciled_sha256"]


def test_conditional_retry(records):
    _, new, _, proof, previous = records
    amendment = {"key": "000001", "parent_job_id": "123", "retry_limit": 1,
                 "parent_reconciled_sha256": proof["reconciled_sha256"]}
    (new / "receipts/deferred-retries/000001.json").write_text(json.dumps(amendment))
    config = {"retry_generations": [{"root": str(new), "conditional_receipt":
              "receipts/deferred-retries/{key}.json", "retry_parents":
              {"000001": dict(proof, conditional=True)}}]}
    assert validate_retry_lineage(config, new, "000001", previous)
    amendment["parent_job_id"] = "999"
    (new / "receipts/deferred-retries/000001.json").write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match="identity"):
        validate_retry_lineage(config, new, "000001", previous)


def test_never_replace_success(records):
    _, new, _, proof, previous = records
    config = {"timeout_retry_root": str(new), "retry_parents": {"000001": proof}}
    with pytest.raises(ValueError, match="timeout parent"):
        validate_retry_lineage(config, new, "000001", dict(previous, status="validated_complete"))


def test_changed_parent_bytes(records):
    _, new, path, proof, previous = records
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt changed"):
        validate_retry_lineage({"timeout_retry_root": str(new), "retry_parents": {"000001": proof}},
                               new, "000001", previous)


def test_unbound_generation(records):
    _, new, _, _, previous = records
    with pytest.raises(ValueError, match="generation"):
        validate_retry_lineage({}, new, "000001", previous)


def test_wrong_parent_job(records):
    _, new, _, proof, previous = records
    proof["job_id"] = "999"
    with pytest.raises(ValueError, match="identity"):
        validate_retry_lineage({"timeout_retry_root": str(new), "retry_parents": {"000001": proof}},
                               new, "000001", previous)
