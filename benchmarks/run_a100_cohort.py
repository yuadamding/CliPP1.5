"""Frozen A100 dispatcher: disjoint queues, fresh CUDA process per tumor."""

from pathlib import Path
import sys
import os
import hashlib
import subprocess
import time
import signal
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PAY = ROOT / "payload"
OUT = ROOT / "output"
sys.path[:0] = [str(PAY / "source/src"), str(PAY / "source/benchmarks"), str(PAY)]
from common import read, digest, save, now  # noqa: E402
from cohort_failures import failure_record, requires_drain  # noqa: E402


def verify():
    p = read(PAY / "plan.json")
    assert digest(PAY / "plan.json") == os.environ["CLIPP1D_PLAN_SHA256"]
    for name, sha in read(PAY / "INVENTORY.json").items():
        if name == "payload.zip":
            continue
        f = PAY / name
        assert f.is_file() and not f.is_symlink() and digest(f) == sha, name
    assert sys.executable == p["python"] and os.getuid() == 307469 and os.getgid() == 600815
    freeze = (
        "\n".join(
            sorted(
                subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
                    text=True,
                ).splitlines()
            )
        )
        + "\n"
    )
    assert hashlib.sha256(freeze.encode()).hexdigest() == p["environment_sha256"]
    assert digest(p["compiler"]["path"]) == p["compiler"]["sha256"]
    from clipp1d.api import source_provenance

    source = source_provenance()
    assert source["source_sha256"] == p["source_sha256"]
    import clipp1d

    assert Path(clipp1d.__file__).resolve().parent == PAY / "source/src/clipp1d"
    return p


def hardware():
    import torch

    assert os.environ.get("NVIDIA_VISIBLE_DEVICES") != "none"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable during worker hardware check")
    assert torch.cuda.device_count() == 1
    d = torch.cuda.get_device_properties(0)
    assert (
        "A100" in d.name
        and (d.major, d.minor) == (8, 0)
        and 38 * 1024**3 <= d.total_memory <= 41 * 1024**3
    )
    assert torch.ones(4, device="cuda:0", dtype=torch.float64).sum().item() == 4
    return dict(
        gpu=d.name,
        memory_bytes=d.total_memory,
        capability=[d.major, d.minor],
        visible_devices=1,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )


def static():
    runtime = hardware()
    p = verify()
    f = OUT / "static-canary"
    f.write_bytes(b"CliPP1.5 write/read/remove")
    assert f.stat().st_uid == os.getuid() and f.read_bytes() == b"CliPP1.5 write/read/remove"
    f.unlink()
    save(
        OUT / "STATIC.json",
        dict(
            passed=True,
            plan_sha256=digest(PAY / "plan.json"),
            source_sha256=p["source_sha256"],
            gpu_requests=1,
            runtime=runtime,
            kind="bounded_gpu_static_qualification_v1",
            uid=os.getuid(),
            gid=os.getgid(),
            pod_uid=os.environ["POD_UID"],
            utc=now(),
        ),
    )


def capacity():
    # Same compiled kernels, FP64 policy, largest cohort N; independent of fit accuracy.
    runtime = hardware()
    p = verify()
    import torch
    from qualify_cuda import resource_probe
    from clipp1d.cuda.kernels import Kernels
    from clipp1d.cuda.policy import CudaPolicy

    torch.cuda.set_per_process_memory_fraction(CudaPolicy().memory_fraction, 0)
    from qualify_audit_recovery import qualify

    recovery = qualify("cuda:0")
    save(
        OUT / "AUDIT_RECOVERY.json",
        dict(recovery, plan_sha256=digest(PAY / "plan.json"), utc=now()),
    )
    records = []
    resource_probe(
        torch.device("cuda:0"),
        Kernels(torch.device("cuda:0"), compiled=True),
        p["maximum_n"],
        CudaPolicy().inner_max_iterations,
        lambda k, v: records.append(dict(kind=k, detail=v)),
    )
    peak = max(x["detail"].get("peak_reserved_bytes", 0) for x in records)
    assert peak < 0.7 * runtime["memory_bytes"] and runtime["memory_bytes"] - peak >= 8 * 1024**3
    save(
        OUT / "CAPACITY.json",
        dict(
            passed=True,
            runtime=runtime,
            maximum_n=p["maximum_n"],
            records=records,
            peak_reserved_bytes=peak,
            plan_sha256=digest(PAY / "plan.json"),
            pod_uid=os.environ["POD_UID"],
            utc=now(),
        ),
    )


def fit_case(key):
    p = verify()
    case = next(c for c in p["cases"] + p.get("qualification_cases", []) if c["key"] == key)
    task = OUT / ("qualification" if case.get("qualification_repeat") else "cases") / key
    from cohort_staging import staged_input_path, verify_staged_input
    from clipp1d.cuda_api import fit
    from validator import validate

    phase = "hardware"
    try:
        hardware()
        phase = "setup"
        verify_staged_input(task, case)
        phase = "fit"
        result = fit(
            staged_input_path(task, case), task / "output", device="cuda:0", max_major_cn=4
        )
        phase = "validation"
        v = validate(PAY, case, task / "output", result)
        save(
            task / "validated.json",
            dict(
                v,
                source_commit=p["source_commit"],
                input_sha256=case["input_sha256"],
                truth_sha256=case["truth_sha256"],
                operation_metrics=result.operation_metrics,
                utc=now(),
            ),
        )
    except Exception as error:
        save(task / "failure.json", dict(failure_record(error, phase), utc=now()))
        return 20
    return 0


def run_child(action, key, timeout, logstem):
    argv = [sys.executable, "-B", __file__, action] + ([key] if key else [])
    with (
        (OUT / "logs" / (logstem + ".out")).open("xb") as out,
        (OUT / "logs" / (logstem + ".err")).open("xb") as err,
    ):
        child = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return None


def paired_qualification():
    p = verify()
    (case,) = p["qualification_cases"]
    key = case["key"]
    task = OUT / "qualification" / key
    task.mkdir(parents=True)
    from cohort_staging import stage_case_input
    from qualify_cohort_recovery import compare_canary

    with zipfile.ZipFile(PAY / "payload.zip") as archive:
        stage_case_input(task, case, archive)
        truth = archive.read(case["truth_member"])
        assert hashlib.sha256(truth).hexdigest() == case["truth_sha256"]
        (task / "truth.tsv").write_bytes(truth)
    assert run_child("fit", key, case["wall_minutes"] * 60, "paired-canary") == 0
    result = compare_canary(task / "output", read(PAY / "canary-baseline.json"))
    save(
        OUT / "CANARY.json",
        dict(
            result,
            key=key,
            qualification_repeat=True,
            validated_sha256=digest(task / "validated.json"),
            plan_sha256=digest(PAY / "plan.json"),
            utc=now(),
        ),
    )


def dispatch():
    runtime = hardware()
    p = verify()
    idx = int(os.environ["JOB_COMPLETION_INDEX"])
    assert 0 <= idx < p["parallelism"]
    Path("/tmp/cache/torch/kernels").mkdir(parents=True, exist_ok=True)
    worker = OUT / "workers" / f"{idx:02d}"
    worker.mkdir()
    uid = os.environ["POD_UID"]
    save(
        worker / "startup.json",
        dict(
            runtime=runtime,
            pod_uid=uid,
            index=idx,
            plan_sha256=digest(PAY / "plan.json"),
            utc=now(),
        ),
    )
    # Supervisor verifies the admitted image identity before allowing scientific work.
    deadline = time.monotonic() + 600
    while not (OUT / f"pod-{uid}.admitted.json").exists():
        assert time.monotonic() < deadline
        time.sleep(2)
    if idx == 0:
        assert run_child("capacity", None, 3600, "capacity") == 0, (
            "A100 capacity qualification failed"
        )
        if p.get("qualification_cases"):
            paired_qualification()
    elif not (OUT / "HALT.json").exists():
        assert read(OUT / "CAPACITY.json")["passed"] and read(OUT / "CANARY.json")["passed"]
    finished = []
    consecutive_errors = 0
    for position, key in enumerate(p["queues"][idx]):
        if (OUT / "HALT.json").exists():
            break
        case = next(c for c in p["cases"] if c["key"] == key)
        task = OUT / "cases" / key
        task.mkdir()
        began = time.monotonic()
        from cohort_staging import stage_case_input

        with zipfile.ZipFile(PAY / "payload.zip") as archive:
            stage_case_input(task, case, archive)
            truth = archive.read(case["truth_member"])
            assert hashlib.sha256(truth).hexdigest() == case["truth_sha256"]
            (task / "truth.tsv").write_bytes(truth)
        save(
            task / "startup.json",
            dict(
                key=key,
                pod_uid=uid,
                worker_index=idx,
                source_commit=p["source_commit"],
                input_sha256=case["input_sha256"],
                plan_sha256=digest(PAY / "plan.json"),
                utc=now(),
            ),
        )
        code = run_child("fit", key, case["wall_minutes"] * 60, key)
        terminal = dict(
            key=key,
            pod_uid=uid,
            worker_index=idx,
            returncode=code,
            elapsed_seconds=time.monotonic() - began,
            plan_sha256=digest(PAY / "plan.json"),
            utc=now(),
        )
        if code == 0:
            try:
                v = read(task / "validated.json")
                from validator import validate

                assert validate(PAY, case, task / "output") == {
                    k: v[k] for k in ("metrics", "output_sha256", "source_sha256", "search_status")
                }
                terminal.update(
                    status="validated_" + v["search_status"],
                    validated_sha256=digest(task / "validated.json"),
                )
            except Exception as error:
                save(task / "failure.json", dict(failure_record(error, "validation"), utc=now()))
                terminal.update(
                    status="output_validation_failure", failure_sha256=digest(task / "failure.json")
                )
        elif code == 20:
            terminal.update(
                status=read(task / "failure.json")["status"],
                failure_sha256=digest(task / "failure.json"),
            )
        elif code is None:
            terminal["status"] = "resource_timeout"
        else:
            terminal.update(
                status="execution_failure",
                error="Child exited without a classified failure receipt",
            )
        save(task / "terminal.json", terminal)
        finished.append(key)
        if terminal["status"] == "infrastructure_failure":
            # The device cannot serve another fit. Leave other GPUs running;
            # this queue's remaining keys require a fresh, disjoint attempt.
            save(worker / "INFRASTRUCTURE_FAILURE.json", terminal)
            break
        consecutive_errors = (
            consecutive_errors + 1 if terminal["status"] == "execution_failure" else 0
        )
        if requires_drain(terminal["status"]) or consecutive_errors >= 3:
            try:
                save(
                    OUT / "HALT.json",
                    dict(
                        terminal,
                        drain=True,
                        reason="integrity failure or three consecutive execution failures",
                    ),
                )
            except FileExistsError:
                pass
            break
        if idx == 0 and position == 0 and not p.get("qualification_cases"):
            if not terminal["status"].startswith("validated_"):
                try:
                    save(
                        OUT / "HALT.json",
                        dict(terminal, drain=True, reason="Qualification canary failed"),
                    )
                except FileExistsError:
                    pass
                break
            save(
                OUT / "CANARY.json",
                dict(
                    passed=True,
                    key=key,
                    terminal_sha256=digest(task / "terminal.json"),
                    plan_sha256=digest(PAY / "plan.json"),
                    utc=now(),
                ),
            )
    save(
        worker / "terminal.json",
        dict(
            index=idx,
            pod_uid=uid,
            keys=finished,
            planned_keys=p["queues"][idx],
            complete=len(finished) == len(p["queues"][idx]),
            drained=(OUT / "HALT.json").exists(),
            infrastructure_failure=(worker / "INFRASTRUCTURE_FAILURE.json").exists(),
            utc=now(),
        ),
    )


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "static":
        static()
    elif action == "capacity":
        capacity()
    elif action == "fit":
        sys.exit(fit_case(sys.argv[2]))
    elif action == "dispatch":
        dispatch()
    else:
        raise ValueError(action)
