"""Strict admission checks for scalar Seadragon research jobs."""

import re


def field(pattern, raw):
    values = re.findall(pattern, raw, re.M)
    if len(values) != 1:
        raise ValueError(f"Expected one {pattern!r}; observed {values!r}")
    return values[0]


def memory_contract(memory, reservation):
    # This site's submission policy rewrites MEMLIMIT to the reservation.
    # A smaller reservation therefore cannot preserve a larger hard limit.
    if memory <= 0 or reservation != memory:
        raise ValueError("LSF memory reservation must equal the intended hard limit")


def verify_admission(raw, accepted, worker_command, memory, minutes, reservation=None):
    reservation = memory if reservation is None else reservation
    memory_contract(memory, reservation)
    checks = (
        (r"^Job <(\d+)>", accepted["job_id"], "job ID"),
        (r"Job Name <([^>]+)>", accepted["job_name"], "job name"),
        (r"User <([^>]+)>", "yding4", "owner"),
        (r"Queue <([^>]+)>", "egpu", "queue"),
        (r"Command <([^>]+)>", worker_command, "command"),
        (r", (\d+) Task\(s\), Requested Resources", "2", "CPU slots"),
        (r"Requested GPU <([^>]+)>",
         "num=1:mode=exclusive_process:gmodel=NVIDIAL40", "GPU request"),
        (r"Requested Resources <([^>]+)>",
         f"rusage[mem={reservation}] span[hosts=1]", "reservation"),
    )
    for pattern, expected, label in checks:
        observed = field(pattern, raw)
        if observed != expected:
            raise ValueError(f"{label}: expected {expected!r}, admitted {observed!r}")
    for pattern, expected, label in (
        (r"^ MEMLIMIT\s*\n\s*(\d+(?:\.\d+)?) G\b", memory, "MEMLIMIT GB"),
        (r"^ RUNLIMIT\s*\n\s*(\d+(?:\.\d+)?) min\b", minutes, "RUNLIMIT minutes"),
    ):
        observed = float(field(pattern, raw))
        if observed != expected:
            raise ValueError(f"{label}: expected {expected}, admitted {observed}")
    state = field(r"Status <([^>]+)>", raw)
    if state not in ("PEND", "PSUSP"):
        raise ValueError(f"Unexpected admission state {state}; leave unreleased")
    return state
