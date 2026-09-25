"""Read-only research transport and terminal evidence validation."""

import re
import hashlib
from pathlib import Path


def validate_import_paths(root, paths, bound_inputs):
    """Allow original input basenames only through exact path/hash bindings.

    Research artifacts include staged inputs. Their extension is not an input
    contract: Regional-CN inputs legitimately end in .clipp2.txt. Other files
    retain the existing restricted artifact suffixes.
    """
    root = Path(root)
    paths = [Path(path) for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact path")
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError("Artifact is outside its bound root") from error
        if ".." in relative.parts or not relative.parts:
            raise ValueError("Unsafe artifact path")
        if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
            raise ValueError("Symlink in artifact path")
        if not path.is_file():
            raise ValueError("Artifact is not a regular file")
        expected = bound_inputs.get(relative.as_posix())
        if expected is not None:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Bound input artifact hash differs")
        elif path.suffix not in (".json", ".jsonl", ".npz", ".tsv", ".err"):
            raise ValueError("Unbound artifact filename")


def verify_accepted(accepted, launch):
    """Launch acknowledgement adds fields; compare the complete accepted schema."""
    keys = ("job_id", "job_name", "plan_sha256", "initially_held")
    if set(accepted) != set(keys) or accepted != {key: launch[key] for key in keys}:
        raise ValueError("Accepted identity differs from launch receipt")
    if accepted["initially_held"] is not True:
        raise ValueError("Expected initially held submission")


def job_missing(stdout, stderr, job_id):
    # bjobs -UF can return zero with only a not-found message on stderr.
    return not stdout.strip() and stderr.strip() == f"Job <{job_id}> is not found"


def terminal_state(raw, accepted, *, log=False):
    """Require owned scheduler evidence; an aged-out job alone proves nothing."""
    job, name = accepted["job_id"], accepted["job_name"]
    if log:
        required = (
            f"Subject: Job {job}: <{name}>",
            "Results reported at ", "Terminated at ",
        )
        if not all(value in raw for value in required):
            raise ValueError("Missing exact terminal log identity or timestamps")
        if not re.search(r"Job <" + re.escape(name)
                         + r"> was submitted .*by user <yding4>", raw):
            raise ValueError("Wrong terminal log owner")
        if "Successfully completed." in raw:
            return "DONE"
        if re.search(r"Exited with exit code|Exited with signal termination|"
                     r"Exited by signal|TERM_(?:MEMLIMIT|RUNLIMIT)", raw):
            return "EXIT"
        raise ValueError("No recognized terminal log outcome")
    if (re.findall(r"^Job <(\d+)>", raw, re.M) != [job]
            or re.findall(r"Job Name <([^>]+)>", raw) != [name]
            or re.findall(r"User <([^>]+)>", raw) != ["yding4"]):
        raise ValueError("Scheduler identity differs from accepted job")
    states = re.findall(r"Status <([^>]+)>", raw)
    if len(states) != 1:
        raise ValueError("Ambiguous scheduler state")
    return states[0] if states[0] in ("DONE", "EXIT") else None


def retryable_read_failure(returncode, timed_out=False):
    """Only read-only transport failure is retryable, never remote code errors."""
    return timed_out or returncode == 255
