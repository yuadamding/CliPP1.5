"""Seal the exact current source and 430db26 comparison source; no submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


BASELINE = "430db26cf07466e88e53c6e1a8fbe2be7b7b25e9"
STUDY = Path(__file__).resolve().parent
REPO = STUDY.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def source_identity(root):
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py"))}
    h = hashlib.sha256()
    for name, sha in files.items():
        h.update(f"{name}\0{sha}\n".encode())
    return dict(source_sha256=h.hexdigest(), source_files=files)


def prepare(parent, tasks, wall_minutes, source_template=None, diagnostic_script=None):
    parent = parent.resolve()
    assert parent.parent == STUDY and parent.is_dir()
    suffix = parent.name.rsplit("-", 1)[1]
    assert suffix.isalnum()
    run_id = "clipp2_clipp1d_timing_20260923" + suffix
    remote_root = "/rsrch8/scratch/bcb/yding4/" + run_id
    root = parent / "sealed" / run_id
    root.mkdir(parents=True)
    current = root / "source"
    baseline = root / "baseline_source"
    current.mkdir()
    baseline.mkdir()
    baseline_only = all(task["source_role"] == "baseline" for task in tasks)
    if source_template is not None:
        source_template = source_template.resolve()
        assert source_template.parent == STUDY
        old_template_plan = json.loads((source_template / "PREPARED.json").read_bytes())
        template_root = source_template / "sealed" / old_template_plan["run_id"]
        template_inventory = json.loads((template_root / "inventory.json").read_bytes())
        for name, sha in template_inventory.items():
            assert digest(template_root / name) == sha
        assert not baseline_only
    names = subprocess.check_output(["git", "ls-files", "-z", "--", "src", "pyproject.toml"], cwd=REPO).decode().rstrip("\0").split("\0")
    benchmark_names = {task["script"] for task in tasks} | {"qualify_cuda.py"}
    if "capture_outer_failure.py" in benchmark_names:
        benchmark_names.update({"capture_failed_qp.py", "mixed_fixtures.py", "qualify_mixed_cuda.py"})
    if benchmark_names & {"qualify_mixed_cuda.py", "capture_failed_qp.py", "replay_failed_qp.py"}:
        benchmark_names.update({"mixed_fixtures.py", "qualify_mixed_cuda.py"})
    if "time_surrogate_cuda.py" in benchmark_names:
        benchmark_names.add("qualify_surrogate_cuda.py")
    if "qualify_surrogate_cuda.py" in benchmark_names:
        benchmark_names.update({"mixed_fixtures.py", "qualify_mixed_cuda.py", "replay_failed_qp.py", "surrogate_trace.py"})
    if "replay_failed_qp.py" in benchmark_names:
        benchmark_names.add("capture_failed_qp.py")
    names += ["benchmarks/" + name for name in sorted(benchmark_names)]
    assert len(names) == len(set(names))
    for name in names:
        path = REPO / name
        if diagnostic_script is not None and name == "benchmarks/" + diagnostic_script.name:
            path = diagnostic_script
        if source_template is not None and (name.startswith("src/") or name.startswith("benchmarks/") or name == "pyproject.toml"):
            path = template_root / "source" / name
        assert path.is_file() and not path.is_symlink()
        destination = current / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if baseline_only and (name.startswith("src/") or name == "pyproject.toml"):
            destination.write_bytes(subprocess.check_output(["git", "show", BASELINE + ":" + name], cwd=REPO))
        else:
            shutil.copyfile(path, destination)
    archive = subprocess.check_output(["git", "archive", BASELINE, "src", "pyproject.toml", "benchmarks/qualify_cuda.py"], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar.getmembers():
            assert not entry.name.startswith("/") and ".." not in Path(entry.name).parts
            assert entry.isdir() or entry.isfile()
        tar.extractall(baseline, filter="data")
    for name in ("worker.py", "launch.py"):
        shutil.copyfile(STUDY / name, root / name)
    patch = (template_root / "tracked-dirty.patch").read_bytes() if source_template is not None else (
        b"" if baseline_only else subprocess.check_output(["git", "diff", "75c102f9dd1dafbe54010f1b96c1dfd551220341", "--", "src", "pyproject.toml", "benchmarks"], cwd=REPO))
    (root / "tracked-dirty.patch").write_bytes(patch)
    preflight = json.loads((parent / "transfer-targets.json").read_bytes())["remote"]
    old = json.loads((REPO / "validation/cuda-review-v3/operational/PREPARED.json").read_bytes())
    identities = dict(current=source_identity(current / "src/clipp1d"), baseline=source_identity(baseline / "src/clipp1d"))
    if source_template is not None:
        assert identities["current"] == old_template_plan["source_identities"]["current"]
    assert identities["baseline"]["source_sha256"] == "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88"
    plan = dict(schema="clipp1d.bound.recovery.study.v1", run_id=run_id,
        job_name="clipp1d_timing_20260923"+suffix, remote_root=remote_root,
        source_commit="75c102f9dd1dafbe54010f1b96c1dfd551220341", baseline_commit=BASELINE,
        source_is_dirty_overlay=not baseline_only, max_submitted=1, python=old["python"],
        compiler=old["compiler"], environment_sha256=preflight["environment_sha256"],
        source_sha256={role: identity["source_sha256"] for role, identity in identities.items()},
        source_identities=identities, tasks=tasks,
        resource_contract=dict(queue="egpu", cpus=2, memory_gb=8, memory_unit="GB",
            gpu_count=1, gpu_request="num=1:mode=exclusive_process:gmodel=NVIDIAL40", wall_minutes=wall_minutes),
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="isolated synthetic CUDA bound recovery study; capture/replay diagnostic completion is not full-fit qualification; no cohorts")
    if source_template is not None:
        plan["frozen_source_template"] = dict(attempt=source_template.name,
            inventory_sha256=digest(template_root / "inventory.json"),
            source_sha256=identities["current"]["source_sha256"],
            scope="package and tracked patch only; new diagnostic helpers separately inventoried")
    write(root / "PREPARED.json", plan)
    inventory = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write(root / "inventory.json", inventory)
    archive = parent / (run_id + ".tar.gz")
    with tarfile.open(archive, "x:gz") as tar:
        tar.add(root, arcname=run_id)
    archive.chmod(0o600)
    write(parent / "PREPARED.json", dict(plan, archive=str(archive), archive_sha256=digest(archive),
        inventory_sha256=digest(root / "inventory.json")))
    print(json.dumps(dict(run_id=run_id, files=len(inventory), archive_bytes=archive.stat().st_size,
                         source_sha256=plan["source_sha256"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", type=Path)
    p.add_argument("tasks", type=Path)
    p.add_argument("--wall-minutes", type=int, default=60)
    p.add_argument("--source-template", type=Path)
    p.add_argument("--diagnostic-script", type=Path)
    a = p.parse_args()
    prepare(a.parent, json.loads(a.tasks.read_bytes()), a.wall_minutes, a.source_template, a.diagnostic_script)
