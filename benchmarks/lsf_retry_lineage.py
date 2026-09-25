"""Validate retry lineage across multiple immutable LSF generations."""
import hashlib
import json
from pathlib import Path


def validate_retry_lineage(config, current_root, key, previous):
    """Reject duplicate successes and require the exact terminal timeout parent."""
    generations = list(config.get("retry_generations", []))
    if config.get("timeout_retry_root"):
        generations.append({
            "root": config["timeout_retry_root"],
            "retry_parents": config.get("retry_parents", {}),
            "conditional_receipt": "receipts/deferred-retry-authorized.json",
        })
    matches = [g for g in generations if g["root"] == str(current_root)
               and key in g["retry_parents"]]
    if len(matches) != 1:
        raise ValueError("Retry generation is missing or ambiguous")
    generation = matches[0]
    proof = generation["retry_parents"][key]
    if previous["status"] != "resource_timeout" or previous["source_root"] != proof["root"]:
        raise ValueError("Retry must replace its exact timeout parent")
    if proof.get("conditional"):
        relative = Path(generation["conditional_receipt"].format(key=key))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe conditional receipt path")
        amendment = json.loads((Path(current_root) / relative).read_text())
        if (amendment["key"] != key or amendment["parent_job_id"] != proof["job_id"]
                or amendment["retry_limit"] != 1):
            raise ValueError("Conditional retry identity or allowance differs")
        expected = amendment["parent_reconciled_sha256"]
    else:
        expected = proof["reconciled_sha256"]
    path = Path(proof["root"]) / "receipts" / key / "reconciled.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Parent terminal receipt changed")
    parent = json.loads(raw)
    if (parent["accepted"]["job_id"] != proof["job_id"]
            or parent["accepted"]["key"] != key
            or parent["outcome"]["status"] != "resource_timeout"
            or parent["scheduler_state"] not in ("DONE", "EXIT")):
        raise ValueError("Parent identity or terminal timeout differs")
    return expected
