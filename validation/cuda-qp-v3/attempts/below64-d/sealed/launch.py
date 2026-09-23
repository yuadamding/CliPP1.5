"""Admit exactly one held scalar LSF qualification, then release it once."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from worker import digest, write


def command(argv, timeout=60):
    try:
        run = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(code=run.returncode, stdout=run.stdout, stderr=run.stderr)
    except subprocess.TimeoutExpired as error:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        return dict(code=None, stdout=text(error.stdout), stderr=text(error.stderr), error="timeout")


def field(pattern, raw):
    values = re.findall(pattern, raw, re.M)
    assert len(values) == 1, (pattern, values)
    return values[0]


def main(root):
    assert root.resolve() == root and root.stat().st_uid == os.getuid() == 307469
    assert re.fullmatch(r"clipp2_clipp1d_qp_20260923[a-z][a-z0-9]*", root.name)
    plan = json.loads((root / "PREPARED.json").read_bytes())
    assert str(root) == plan["remote_root"] and root.name == plan["run_id"]
    assert sys.executable == plan["python"] and plan["max_submitted"] == 1
    inventory = json.loads((root / "inventory.json").read_bytes())
    for name, expected in inventory.items():
        assert digest(root / name) == expected, name
    units = subprocess.check_output(["lsadmin", "showconf", "lim"], text=True, stderr=subprocess.STDOUT)
    assert re.findall(r"^[ \t]*LSF_UNIT_FOR_LIMITS[ \t]*=[ \t]*(\S+)", units, re.M) == ["GB"]
    queue = subprocess.check_output(["bqueues", "-l", "egpu"], text=True)
    maximum = queue.split("MAXIMUM LIMITS:", 1)[1]
    hard_minutes = float(field(r"RUNLIMIT[^\n]*\n\s*(\d+(?:\.\d+)?) min", maximum))
    wall_minutes = plan["resource_contract"]["wall_minutes"]
    assert isinstance(wall_minutes, int) and 10 <= wall_minutes <= hard_minutes
    params = subprocess.check_output(["bparams", "-a"], text=True)
    write(root / "receipts/resource-contract.json", dict(
        memory_unit="GB", queue=queue, units=units,
        reservation_semantics=[line for line in params.splitlines() if "RESOURCE_RESERVE" in line],
        hard_minutes=wall_minutes, memory_gb=8, cpu_slots=2, gpu_count=1,
        conservative_reservation_upper_gb=16,
    ))
    worker_command = f'{plan["python"]} -B {root}/worker.py {root}'
    argv = ["bsub", "-H", "-q", "egpu", "-J", plan["job_name"], "-n", "2",
            "-M", "8", "-R", "rusage[mem=8] span[hosts=1]",
            "-gpu", "num=1:mode=exclusive_process:gmodel=NVIDIAL40", "-W", str(wall_minutes),
            "-cwd", str(root), "-oo", str(root / "lsf-logs/qualification.%J.out"),
            "-eo", str(root / "lsf-logs/qualification.%J.err"), worker_command]
    write(root / "receipts/submit-intent.json", dict(argv=argv, plan_sha256=digest(root / "PREPARED.json")))
    response = command(argv, 120)
    write(root / "receipts/submit-response.json", response)
    jobs = re.findall(r"Job <(\d+)> is submitted", response["stdout"])
    if len(jobs) == 1:
        accepted = dict(job_id=jobs[0], job_name=plan["job_name"],
                        plan_sha256=digest(root / "PREPARED.json"), initially_held=True)
        write(root / "receipts/accepted.json", accepted)
    assert response["code"] == 0 and len(jobs) == 1, "Uncertain submission; reconcile, never repeat."
    observations = []
    deadline = time.monotonic() + 300
    try:
        while True:
            raw = subprocess.check_output(["bjobs", "-UF", accepted["job_id"]], text=True, timeout=30)
            assert field(r"^Job <(\d+)>", raw) == accepted["job_id"]
            assert field(r"Job Name <([^>]+)>", raw) == plan["job_name"]
            assert field(r"User <([^>]+)>", raw) == "yding4"
            assert field(r"Queue <([^>]+)>", raw) == "egpu"
            assert field(r"Command <([^>]+)>", raw) == worker_command
            assert field(r", (\d+) Task\(s\), Requested Resources", raw) == "2"
            assert field(r"Requested GPU <([^>]+)>", raw) == "num=1:mode=exclusive_process:gmodel=NVIDIAL40"
            assert float(field(r"^ MEMLIMIT\s*\n\s*(\d+(?:\.\d+)?) G\b", raw)) == 8
            assert float(field(r"^ RUNLIMIT\s*\n\s*(\d+(?:\.\d+)?) min\b", raw)) == wall_minutes
            assert field(r"Requested Resources <([^>]+)>", raw) == "rusage[mem=8] span[hosts=1]"
            state = field(r"Status <([^>]+)>", raw)
            observed = dict(job_id=accepted["job_id"], state=state, raw=raw)
            observations.append(observed)
            assert state in ("PEND", "PSUSP"), "Unexpected state: leave unreleased."
            if state == "PSUSP":
                write(root / "receipts/admitted.json", observed)
                break
            assert time.monotonic() < deadline, "Leave held; do not resubmit."
            time.sleep(5)
    finally:
        write(root / "receipts/admission-observations.json", observations)
    argv = ["bresume", accepted["job_id"]]
    write(root / "receipts/release-intent.json", dict(argv=argv))
    response = command(argv)
    write(root / "receipts/release-response.json", response)
    assert response["code"] == 0 and accepted["job_id"] in response["stdout"], "Uncertain release; never repeat."
    print("RECEIPT=" + json.dumps(dict(**accepted, released_acknowledged=True)))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
