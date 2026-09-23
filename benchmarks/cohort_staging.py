"""Preserve canonical input identity when unpacking a cohort GPU task.

The reader uses the filename when ##tumor_id metadata is absent. Renaming an
input to input.tsv changes that identity even when its bytes are unchanged.
"""
import hashlib
import json
from pathlib import Path

from clipp1d.io import read_tumor
from clipp1d.policy import Policy


def staged_input_path(task, case):
    name = case["input_filename"]
    if (not isinstance(name, str) or not name or name in (".", "..")
            or Path(name).name != name or "/" in name or "\\" in name or "\0" in name):
        raise ValueError("Input filename must be a single safe basename")
    return Path(task) / "input" / name


def verify_staged_input(task, case):
    path = staged_input_path(task, case)
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular staged input")
    # Hash the archive bytes independently of the reader (including gzip).
    if hashlib.sha256(path.read_bytes()).hexdigest() != case["input_sha256"]:
        raise ValueError("Staged input hash differs from the manifest")
    tumor = read_tumor(path, Policy(max_major_cn=4))
    if (tumor.tumor_id, tumor.sample_id) != (case["tumor_id"], case["sample_id"]):
        raise ValueError(f"Staged tumor/sample IDs differ: {(tumor.tumor_id, tumor.sample_id)!r}")
    ids = sorted(m.mutation_id for m in tumor.retained)
    if (len(ids) != case["retained_mutations"] or
            hashlib.sha256(json.dumps(ids).encode()).hexdigest() != case["retained_ids_sha256"]):
        raise ValueError("Staged retained mutation population differs from the manifest")
    return tumor


def stage_case_input(task, case, bundle):
    path = staged_input_path(task, case)
    data = bundle.read(case["input_member"])
    if hashlib.sha256(data).hexdigest() != case["input_sha256"]:
        raise ValueError("Archived input hash differs from the manifest")
    path.parent.mkdir()
    with path.open("xb") as stream:
        stream.write(data)
    verify_staged_input(task, case)
    return path
