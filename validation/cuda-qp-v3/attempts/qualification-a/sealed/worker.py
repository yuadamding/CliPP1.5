"""Execute one immutable CUDA qualification task bundle on its assigned LSF GPU."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def main(root):
    assert root.resolve() == root and root.stat().st_uid == os.getuid() == 307469
    assert re.fullmatch(r"clipp2_clipp1d_qp_20260923[a-z][a-z0-9]*", root.name)
    plan = json.loads((root / "PREPARED.json").read_bytes())
    assert str(root) == plan["remote_root"] and root.name == plan["run_id"]
    assert sys.executable == plan["python"]
    accepted = json.loads((root / "receipts/accepted.json").read_bytes())
    admitted = json.loads((root / "receipts/admitted.json").read_bytes())
    assert accepted["job_id"] == admitted["job_id"] == os.environ["LSB_JOBID"]
    assert accepted["job_name"] == plan["job_name"]
    assert accepted["plan_sha256"] == digest(root / "PREPARED.json")
    began = time.time()
    write(root / "receipts/execution.json", dict(job_id=accepted["job_id"],
          host=os.uname().nodename, pid=os.getpid(), started_unix=began))
    result = dict(passed=False, job_id=accepted["job_id"],
                  plan_sha256=digest(root / "PREPARED.json"), tasks=[])
    try:
        inventory = json.loads((root / "inventory.json").read_bytes())
        for name, expected in inventory.items():
            path = root / name
            assert path.is_file() and not path.is_symlink() and digest(path) == expected, name
        packages = "\n".join(sorted(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"], text=True,
        ).splitlines())) + "\n"
        assert hashlib.sha256(packages.encode()).hexdigest() == plan["environment_sha256"]
        compiler = Path(plan["compiler"]["path"])
        assert digest(compiler) == plan["compiler"]["sha256"]
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CC=str(compiler),
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1")
        probe = '''import json,sys
from pathlib import Path
import torch,clipp1d
from clipp1d.api import source_provenance
assert Path(clipp1d.__file__).resolve().parent==Path(sys.argv[1])/'clipp1d'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'L40' in torch.cuda.get_device_name(0)
assert torch.ones(4,device='cuda',dtype=torch.float64).sum().item()==4
print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,source=source_provenance())))
'''
        runtimes = {}
        for role in sorted({task["source_role"] for task in plan["tasks"]}):
            source = root / ("baseline_source" if role == "baseline" else "source") / "src"
            runtime = json.loads(subprocess.check_output([sys.executable, "-B", "-c", probe, str(source)],
                env=dict(environment, PYTHONPATH=str(source)), text=True, timeout=120))
            assert runtime["source"]["source_sha256"] == plan["source_sha256"][role]
            runtimes[role] = runtime
        write(root / "receipts/startup.json", dict(runtimes=runtimes,
              job_id=accepted["job_id"], environment_sha256=plan["environment_sha256"],
              compiler_sha256=digest(compiler), inventory_sha256=digest(root / "inventory.json")))
        for task in plan["tasks"]:
            name, role = task["name"], task["source_role"]
            assert re.fullmatch(r"[a-zA-Z0-9_-]+", name)
            tree = root / ("baseline_source" if role == "baseline" else "source")
            cache = root / ("results/compiler-cache-" + name)
            cache.mkdir()
            env = dict(environment, PYTHONPATH=str(tree / "src"),
                TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
            out = root / "results" / (name + ".json")
            arguments = [str(root / arg[7:]) if arg.startswith("@ROOT@/") else arg for arg in task["args"]]
            command = [sys.executable, "-B", str(root / "source/benchmarks" / task["script"]),
                       "--device", "cuda:0", "--out", str(out), *arguments]
            write(root / f"receipts/command-{name}.json", dict(argv=command,
                  source_role=role, timeout_seconds=task["timeout_seconds"]))
            with (root / f"case-logs/{name}.out").open("xb") as stdout, (root / f"case-logs/{name}.err").open("xb") as stderr:
                completed = subprocess.run(command, cwd=tree, env=env, stdout=stdout, stderr=stderr,
                                           timeout=task["timeout_seconds"])
            entry = dict(name=name, source_role=role, returncode=completed.returncode)
            result["tasks"].append(entry)
            if out.is_file():
                receipt = json.loads(out.read_bytes())
                entry.update(receipt_sha256=digest(out), status=receipt["status"])
                assert receipt["source"]["source_sha256"] == plan["source_sha256"][role]
            assert completed.returncode == 0 and entry.get("status") == "passed", entry
        for name, expected in inventory.items():
            assert digest(root / name) == expected, name
        result["passed"] = True
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        result.update(elapsed_seconds=time.time()-began,
                      scope="source-bound synthetic CUDA qualification, not cohort accuracy")
        write(root / "receipts/terminal.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
