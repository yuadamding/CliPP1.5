"""Seal the exact current source and daaf50a comparison source; no submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


BASELINE = "daaf50ad5a2ae7301e9b54b9c31e87d87b24cfa6"
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


def prepare(parent, tasks, wall_minutes):
    parent = parent.resolve()
    assert parent.parent == STUDY and parent.is_dir()
    suffix = parent.name.rsplit("-", 1)[1]
    assert suffix.isalnum()
    run_id = "clipp2_clipp1d_qp_20260923" + suffix
    remote_root = "/rsrch8/scratch/bcb/yding4/" + run_id
    root = parent / "sealed" / run_id
    root.mkdir(parents=True)
    current = root / "source"
    baseline = root / "baseline_source"
    current.mkdir()
    baseline.mkdir()
    names = subprocess.check_output(["git", "ls-files", "-z", "--", "src", "pyproject.toml"], cwd=REPO).decode().rstrip("\0").split("\0")
    benchmark_names = {task["script"] for task in tasks} | {"qualify_cuda.py"}
    if "qualify_mixed_cuda.py" in benchmark_names:
        benchmark_names.add("mixed_fixtures.py")
    names += ["benchmarks/" + name for name in sorted(benchmark_names)]
    assert len(names) == len(set(names))
    for name in names:
        path = REPO / name
        assert path.is_file() and not path.is_symlink()
        destination = current / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    archive = subprocess.check_output(["git", "archive", BASELINE, "src", "pyproject.toml", "benchmarks/qualify_cuda.py"], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in tar.getmembers():
            assert not entry.name.startswith("/") and ".." not in Path(entry.name).parts
            assert entry.isdir() or entry.isfile()
        tar.extractall(baseline, filter="data")
    for name in ("worker.py", "launch.py"):
        shutil.copyfile(STUDY / name, root / name)
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "pyproject.toml", "benchmarks"], cwd=REPO)
    (root / "tracked-dirty.patch").write_bytes(patch)
    preflight = json.loads((parent / "transfer-targets.json").read_bytes())["remote"]
    old = json.loads((REPO / "validation/cuda-review-v3/operational/PREPARED.json").read_bytes())
    identities = dict(current=source_identity(current / "src/clipp1d"), baseline=source_identity(baseline / "src/clipp1d"))
    assert identities["baseline"]["source_sha256"] == "ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26"
    plan = dict(schema="clipp1d.qp.optimization.qualification.v1", run_id=run_id,
        job_name="clipp1d_qp_20260923"+suffix, remote_root=remote_root,
        source_commit=BASELINE, baseline_commit=BASELINE,
        source_is_dirty_overlay=True, max_submitted=1, python=old["python"],
        compiler=old["compiler"], environment_sha256=preflight["environment_sha256"],
        source_sha256={role: identity["source_sha256"] for role, identity in identities.items()},
        source_identities=identities, tasks=tasks,
        resource_contract=dict(queue="egpu", cpus=2, memory_gb=8, memory_unit="GB",
            gpu_count=1, gpu_request="num=1:mode=exclusive_process:gmodel=NVIDIAL40", wall_minutes=wall_minutes),
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="isolated paired synthetic CUDA optimization qualification; no cohort launch")
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
    a = p.parse_args()
    prepare(a.parent, json.loads(a.tasks.read_bytes()), a.wall_minutes)
