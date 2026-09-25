"""Suspended creation, exact UID binding, and attach-only lifecycle supervision."""

import os
import sys
import json
import subprocess
import time
import fcntl
import traceback
import hashlib
from common import ROOT, PAY, OUT, read, save, digest, now, canonical, identity

MODE = sys.argv[2]
assert MODE in ("static", "gpu")
E = ROOT / "evidence" / MODE
P = read(PAY / "plan.json")
sys.path.insert(0, str(PAY / "source/benchmarks"))
from cohort_failures import CASE_STATUSES  # noqa: E402

M = read(PAY / (MODE + ".manifest.json"))
NAME = M["metadata"]["name"]
K = [
    "/risapps/noarch/kubectl/1.28.4/bin/kubectl",
    "--request-timeout=20s",
    "--context",
    "yding4_yn-gpu-workload@kubernetes-admin@kubernetes",
    "-n",
    "yn-gpu-workload",
]
ENV = dict(
    os.environ, KUBECONFIG="/rsrch8/home/bcb/yding4/.kube/config", PYTHONDONTWRITEBYTECODE="1"
)


def kube(args, value=None):
    assert (
        args[0] in ("get", "patch", "logs", "delete")
        or args[0] == "create"
        and sys.argv[1] == "launch"
    )
    r = subprocess.run(
        K + args,
        input=None if value is None else canonical(value),
        capture_output=True,
        env=ENV,
        timeout=40,
    )
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-4000:])
    return json.loads(r.stdout) if r.stdout.strip() else None


def job():
    return kube(["get", "job", NAME, "--ignore-not-found", "-o", "json"])


def pods(uid):
    items = kube(["get", "pods", "-l", f"job-name={NAME},controller-uid={uid}", "-o", "json"])[
        "items"
    ]
    assert all(
        any(
            o.get("uid") == uid and o.get("controller")
            for o in x["metadata"].get("ownerReferences", [])
        )
        for x in items
    )
    return items


def audit(a):
    assert (
        a["metadata"]["name"] == NAME
        and a["metadata"]["namespace"] == "yn-gpu-workload"
        and a["metadata"]["annotations"]["seadragon_run_id"] == NAME
    )
    for k in (
        "completions",
        "parallelism",
        "backoffLimit",
        "activeDeadlineSeconds",
        "ttlSecondsAfterFinished",
        "suspend",
    ):
        assert a["spec"][k] == M["spec"][k], k
    actual = a["spec"]["template"]["spec"]
    expected = M["spec"]["template"]["spec"]
    for k in ("volumes", "securityContext", "automountServiceAccountToken", "restartPolicy"):
        assert actual[k] == expected[k], k
    assert len(actual["containers"]) == 1
    ac = actual["containers"][0]
    ec = expected["containers"][0]
    assert set(ac) == set(ec)
    for k in ec:
        assert ac[k] == ec[k], k
    for k in ("hostNetwork", "hostPID", "hostIPC", "initContainers", "ephemeralContainers"):
        assert not actual.get(k), k
    assert (
        actual["schedulerName"] == "kai-scheduler"
        and actual["priorityClassName"] == "high-nonpreempting"
    )
    assert actual.get("affinity") == expected.get("affinity")
    if MODE == "gpu":
        assert actual["nodeSelector"] == dict(expected["nodeSelector"], **{"gpu-type": "A100"})
    else:
        assert actual.get("nodeSelector") == expected.get("nodeSelector")
    assert (
        a["spec"]["template"]["metadata"]["labels"]["runai/queue"] == "yding4-yn-gpu-workload-queue"
    )


def validate(current, bound):
    assert (
        current
        and current["metadata"]["uid"] == bound["metadata"]["uid"]
        and current["metadata"]["annotations"]["seadragon_run_id"] == NAME
    )
    for k in (
        "selector",
        "template",
        "completions",
        "completionMode",
        "backoffLimit",
        "activeDeadlineSeconds",
        "ttlSecondsAfterFinished",
    ):
        assert current["spec"].get(k) == bound["spec"].get(k), k
    assert current["spec"]["parallelism"] in ((1, 10) if MODE == "gpu" else (1,))
    return current


def patch(bound, changes):
    cur = validate(job(), bound)
    ops = [
        dict(op="test", path=k, value=v)
        for k, v in [
            ("/metadata/uid", bound["metadata"]["uid"]),
            ("/metadata/resourceVersion", cur["metadata"]["resourceVersion"]),
            ("/metadata/annotations/seadragon_run_id", NAME),
            ("/spec/suspend", cur["spec"]["suspend"]),
            ("/spec/parallelism", cur["spec"]["parallelism"]),
            ("/spec/selector", bound["spec"]["selector"]),
            ("/spec/template", bound["spec"]["template"]),
        ]
    ]
    ops.extend(dict(op="replace", path="/spec/" + k, value=v) for k, v in changes.items())
    return kube(["patch", "job", NAME, "--type=json", "-p", canonical(ops).decode(), "-o", "json"])


def capture(pod):
    path = E / ("pod-" + pod["metadata"]["uid"] + ".terminal.json")
    if path.exists():
        return
    try:
        r = subprocess.run(
            K
            + ["logs", pod["metadata"]["name"], "-c", "main", "--tail=150", "--limit-bytes=40000"],
            env=ENV,
            capture_output=True,
            timeout=40,
        )
    except subprocess.TimeoutExpired as error:
        save(path, dict(pod=pod, logs_unavailable="timeout", error=str(error), utc=now()))
        return
    save(
        path,
        dict(
            pod=pod,
            log_returncode=r.returncode,
            stdout=r.stdout.decode(errors="replace"),
            stderr=r.stderr.decode(errors="replace"),
            utc=now(),
        ),
    )


def cleanup(bound):
    cur = job()
    uid = bound["metadata"]["uid"]
    if cur:
        validate(cur, bound)
        for pod in pods(uid):
            capture(pod)
        options = dict(
            apiVersion="v1",
            kind="DeleteOptions",
            propagationPolicy="Foreground",
            preconditions=dict(uid=uid, resourceVersion=cur["metadata"]["resourceVersion"]),
        )
        if not (E / "delete-intent.json").exists():
            save(E / "delete-intent.json", options)
            kube(
                [
                    "delete",
                    "--raw",
                    "/apis/batch/v1/namespaces/yn-gpu-workload/jobs/" + NAME,
                    "-f",
                    "-",
                ],
                options,
            )
        else:
            assert read(E / "delete-intent.json")["preconditions"]["uid"] == uid
            # A prior delete may still be propagating. Never repeat it blindly.
    for _ in range(30):
        if job() is None and not pods(uid):
            save(E / "cleanup.json", dict(uid=uid, job_absent=True, pods_absent=True, utc=now()))
            return True
        time.sleep(2)
    save(
        E / "cleanup-pending.json",
        dict(uid=uid, absence_verified=False, job=job(), pods=pods(uid), utc=now()),
    )
    return False


def record_terminal(cur, owned, verified, scaled):
    conditions = [
        c["type"]
        for c in cur.get("status", {}).get("conditions", [])
        if c.get("status") == "True" and c["type"] in ("Complete", "Failed")
    ]
    if not conditions:
        return False
    save(E / "terminal.json", dict(job=cur, pods=owned, results=verified, utc=now()))
    if "Failed" in conditions:
        save(
            E / "FAILED.json",
            dict(
                passed=False,
                results=verified,
                conditions=cur["status"]["conditions"],
                utc=now(),
            ),
        )
        return True
    if MODE == "static":
        assert read(OUT / "STATIC.json")["passed"]
    elif (OUT / "HALT.json").exists():
        save(
            E / "DRAINED.json",
            dict(results=verified, halt=read(OUT / "HALT.json"), utc=now()),
        )
        return True
    else:
        if len(verified) != len(P["cases"]):
            save(
                E / "PARTIAL.json",
                dict(
                    passed=False,
                    results=verified,
                    unfinished=[c["key"] for c in P["cases"] if c["key"] not in verified],
                    utc=now(),
                ),
            )
            return True
        assert scaled
    save(E / "COMPLETE.json", dict(passed=True, results=verified, utc=now()))
    return True


def supervise():
    lock = (E / "supervisor.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bound = read(E / "binding.json")["observations"][-1]
    for _ in range(30):
        if (E / "process.json").exists():
            break
        time.sleep(1)
    assert identity(os.getpid()) == read(E / "process.json")["identity"]
    assert digest(__file__) == read(E / "spawn-intent.json")["source_sha256"]
    validate(job(), bound)
    assert not pods(bound["metadata"]["uid"])
    save(E / "activation.json", patch(bound, dict(suspend=False)))
    save(
        E / "launch-success.json",
        dict(
            uid=bound["metadata"]["uid"],
            name=NAME,
            initial_parallelism=1,
            authorized_parallelism=10 if MODE == "gpu" else 1,
            utc=now(),
        ),
    )
    scaled = False
    seen = set()
    verified = {}
    began = time.monotonic()
    errors = 0
    try:
        with (E / "progress.jsonl").open("x") as progress:
            while time.monotonic() - began < M["spec"]["activeDeadlineSeconds"]:
                try:
                    cur = validate(job(), bound)
                    owned = pods(bound["metadata"]["uid"])
                    errors = 0
                except (RuntimeError, TimeoutError) as exc:
                    errors += 1
                    if errors >= 10:
                        raise
                    progress.write(canonical(dict(api_error=str(exc), utc=now())).decode())
                    progress.flush()
                    time.sleep(30)
                    continue
                for pod in owned:
                    uid = pod["metadata"]["uid"]
                    spec = pod["spec"]
                    expected = bound["spec"]["template"]["spec"]
                    for k in (
                        "containers",
                        "volumes",
                        "securityContext",
                        "nodeSelector",
                        "affinity",
                        "automountServiceAccountToken",
                        "restartPolicy",
                        "schedulerName",
                    ):
                        assert spec.get(k) == expected.get(k), k
                    if uid not in seen:
                        save(E / f"pod-{uid}.admitted.json", pod)
                        seen.add(uid)
                    statuses = pod.get("status", {}).get("containerStatuses", [])
                    if statuses and statuses[0].get("imageID"):
                        assert statuses[0]["imageID"].endswith(P["image"].split("@")[1])
                        path = OUT / f"pod-{uid}.admitted.json"
                        if not path.exists():
                            save(
                                path,
                                dict(
                                    pod_uid=uid,
                                    job_uid=bound["metadata"]["uid"],
                                    image=P["image"],
                                    image_id=statuses[0]["imageID"],
                                    utc=now(),
                                ),
                            )
                    if pod.get("status", {}).get("phase") in ("Failed", "Succeeded"):
                        capture(pod)
                if MODE == "gpu":
                    for c in P["cases"]:
                        key = c["key"]
                        f = OUT / "cases" / key / "terminal.json"
                        if key not in verified and f.exists():
                            t = read(f)
                            assert t["key"] == key and t["plan_sha256"] == digest(PAY / "plan.json")
                            assert t["status"] in CASE_STATUSES, t
                            if not t["status"].startswith("validated_") and "failure_sha256" in t:
                                assert digest(f.parent / "failure.json") == t["failure_sha256"]
                            if t["status"].startswith("validated_"):
                                assert digest(f.parent / "validated.json") == t["validated_sha256"]
                                for name, sha in read(f.parent / "validated.json")[
                                    "output_sha256"
                                ].items():
                                    assert digest(f.parent / "output" / name) == sha
                            verified[key] = t["status"]
                    if (
                        not scaled
                        and not (OUT / "HALT.json").exists()
                        and (OUT / "CAPACITY.json").exists()
                        and (OUT / "CANARY.json").exists()
                    ):
                        for f in ("CAPACITY.json", "CANARY.json"):
                            assert read(OUT / f)["passed"] and read(OUT / f)[
                                "plan_sha256"
                            ] == digest(PAY / "plan.json")
                        save(
                            E / "scale.json",
                            dict(
                                job=patch(bound, dict(parallelism=10)),
                                capacity_sha256=digest(OUT / "CAPACITY.json"),
                                canary_sha256=digest(OUT / "CANARY.json"),
                                utc=now(),
                            ),
                        )
                        scaled = True
                phases = [p.get("status", {}).get("phase") for p in owned]
                item = dict(
                    utc=now(),
                    name=NAME,
                    uid=bound["metadata"]["uid"],
                    parallelism=cur["spec"]["parallelism"],
                    running=phases.count("Running"),
                    pending=phases.count("Pending"),
                    ready=sum(
                        x.get("status", {}).get("phase") == "Running"
                        and any(
                            c.get("type") == "Ready" and c.get("status") == "True"
                            for c in x.get("status", {}).get("conditions", [])
                        )
                        for x in owned
                    ),
                    validated=sum(s.startswith("validated_") for s in verified.values()),
                    finished=len(verified),
                    planned=len(P["cases"]),
                    scaled=scaled,
                )
                progress.write(canonical(item).decode())
                progress.flush()
                os.fsync(progress.fileno())
                if record_terminal(cur, owned, verified, scaled):
                    break
                time.sleep(15 if MODE == "static" else 30)
            else:
                raise TimeoutError("Bounded lifecycle deadline")
    except BaseException:
        save(E / "failure.json", dict(traceback=traceback.format_exc(), utc=now()))
        raise
    finally:
        cleanup(bound)


def launch():
    assert sys.executable == P["python"]
    assert read(ROOT / "evidence/publication.json")["all_hashes_verified"]
    for name, sha in read(PAY / "INVENTORY.json").items():
        assert digest(PAY / name) == sha, name
    if MODE == "gpu":
        assert (
            read(ROOT / "evidence/static/COMPLETE.json")["passed"]
            and read(ROOT / "evidence/static/cleanup.json")["job_absent"]
        )
    assert job() is None
    dry = kube(["create", "-f", "-", "--dry-run=server", "-o", "json"], M)
    save(E / "dryrun.json", dry)
    audit(dry)
    save(
        E / "arm.json",
        dict(
            manifest_sha256=digest(PAY / (MODE + ".manifest.json")),
            plan_sha256=digest(PAY / "plan.json"),
            utc=now(),
        ),
    )
    created = kube(["create", "-f", "-", "-o", "json"], M)
    save(E / "create.json", created)
    child = None
    try:
        audit(created)
        assert created["spec"]["template"]["spec"] == dry["spec"]["template"]["spec"]
        observations = []
        for _ in range(2):
            cur = validate(job(), created)
            assert cur["spec"]["suspend"] is True and not pods(created["metadata"]["uid"])
            observations.append(cur)
        assert observations[0]["spec"] == observations[1]["spec"]
        save(
            E / "binding.json",
            dict(
                observations=observations,
                admitted_spec_sha256=hashlib.sha256(
                    canonical(created["spec"]["template"]["spec"])
                ).hexdigest(),
            ),
        )
        save(
            E / "spawn-intent.json",
            dict(source_sha256=digest(__file__), plan_sha256=digest(PAY / "plan.json"), utc=now()),
        )
        with (E / "supervisor.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, "-B", __file__, "supervise", MODE],
                cwd=ROOT,
                env=ENV,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        time.sleep(0.2)
        save(E / "process.json", dict(identity=identity(child.pid), utc=now()))
        for _ in range(50):
            if (E / "launch-success.json").exists():
                print("RECEIPT=" + json.dumps(read(E / "launch-success.json")))
                return
            assert child.poll() is None, "Supervisor exited; preserve evidence"
            time.sleep(1)
        raise TimeoutError("Launch handshake pending; reconcile exact child")
    except BaseException:
        if child is None:
            cleanup(created)
        else:
            save(
                E / "attach-required.json",
                dict(uid=created["metadata"]["uid"], pid=child.pid, utc=now()),
            )
        raise


if __name__ == "__main__":
    if sys.argv[1] == "launch":
        launch()
    elif sys.argv[1] == "supervise":
        supervise()
    else:
        raise ValueError(sys.argv[1])
