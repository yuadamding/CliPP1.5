"""Reconcile an explicit terminal evidence set; dry-run unless --publish is set.

No numerical code is imported or run. Files above 10,000,000 bytes are rejected.
The caller must supply a reviewed total-byte ceiling and exact current source.
Diagnostic completion is recorded separately from original-QP/full-fit success.
"""

import argparse
import ast
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tempfile


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DESTINATION = REPO / "validation/cuda-bound-recovery-v3"
BASELINE_COMMIT = "430db26cf07466e88e53c6e1a8fbe2be7b7b25e9"
BASELINE_SHA = "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88"
FILE_LIMIT = 10_000_000
SUFFIXES = {".json", ".jsonl", ".tsv", ".err", ".py", ".patch", ".toml", ".log", ".md", ".txt"}
OPERATIONS = (
    "PREPARED.json", "tasks.json", "IMPORT_MANIFEST.json", "transfer-targets.json",
    "remote-preflight.json", "mac-preflight.json", "publication.json", "mac-upload.json",
    "remote-upload.json", "launch.json", "FAILURE_SUMMARY.json", "MEASUREMENT_INVALIDATION.json",
    "cancel-intent.json", "cancel-result.json", "stop-intent.json", "stop-result.json",
)
HELPERS = {"fixtures": "mixed_fixtures.py", "common_qualifier": "qualify_cuda.py",
           "mixed_qualifier": "qualify_mixed_cuda.py"}
CAPTURE_PROFILES = {
    "baseline430": dict(
        source_sha256=BASELINE_SHA, commit=BASELINE_COMMIT,
        source_files={
            "cuda/qp.py": "925e1e5ee67f7630f0d229a560f2af8958fffd1b01ac7609f868a85aa01a74b0",
            "cuda/solver.py": "80a0b34cd856698efd7034a523e6d850c21ba21e3805f38928897db1f2d6091b",
        }, failures={("below_one", 64): 2, ("mixed_support", 256): 11}),
    "normalized276da": dict(
        source_sha256="276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5", commit=None,
        source_files={
            "cuda/qp.py": "07bacefb982c75418e24936d30de04edd70d5ffc7f577f7dab4befd77171e695",
            "cuda/solver.py": "80a0b34cd856698efd7034a523e6d850c21ba21e3805f38928897db1f2d6091b",
        }, failures={("below_one", 64): 1},
        expected_contexts={("below_one", 64): [dict(
            path_index=2, start_index=2, outer_iteration=1, backtracks=20,
            qp_ordinal_within_start=22, lambda_value=553.4678435191591)]}),
    "bound8627": dict(
        source_sha256="8627daf96b3b72be689a756445dfb60a7798a03d8bdd8d896f837eb1f11184c2", commit=None,
        source_files={
            "cuda/qp.py": "c97f7e827fceb032d9fc494885583e0900e83fd1e0f23240df5b6f8ce87d7d7d",
            "cuda/solver.py": "2f9f9e494a1c10b6818508c08dc239c273ace0f41df06eee46074b4b2cc4781a",
        }, failures={("mixed_support", 256): 1, ("below_one", 256): 1},
        expected_contexts={("mixed_support", 256): [dict(
            path_index=25, start_index=2, outer_iteration=0, backtrack_index=0,
            backtracks=0, qp_ordinal_within_start=1, inflation=1.0,
            lambda_value=3631358527.4736257)],
            ("below_one", 256): [dict(
                path_index=1, start_index=1, outer_iteration=1, backtracks=2,
                qp_ordinal_within_start=4, lambda_value=137.6911252669794)]}),
}
CAPTURE_INPUTS = {
    ("below_one", 64): "ae53764990bc252cb7f6a2a14a8eb21a57e3ec46fef28423c08354575ef8a5db",
    ("below_one", 256): "e40d233273f308283ee71ab67e4c240688a48b39254419d3f6e3eb2497ab2fb7",
    ("mixed_support", 256): "6b5cb4f2cfce4cb6fbd81eff8a6ab8dca4e3c810dcfdf4e1ec75fa76fa6b2428",
}
POOLED_SCHEMA = "clipp1d.cuda.pooled_outer_diagnostic.v1"
POOLED_SOURCE = CAPTURE_PROFILES["normalized276da"]["source_sha256"]
POOLED_REFERENCE = "a910183f847f0ce715b7a476b1d244796f6d94736b4ad3975229c91b78a745ca"
POOLED_SCRIPT = "6948917beecd8cb3753431a0d82ea5b8e107828e592261dc9ce52713ed1f8521"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def relative(name):
    value = PurePosixPath(name)
    require(not value.is_absolute() and value.parts and all(p not in (".", "..") for p in value.parts),
            f"Unsafe relative path: {name}")
    require(str(value) == name, f"Noncanonical relative path: {name}")
    return value


def under(root, name):
    value = root / str(relative(name))
    for part in (value, *value.parents):
        require(not part.is_symlink(), f"Symlink evidence path: {part}")
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    require(path.is_file() and not path.is_symlink(), f"Missing regular file: {path}")
    for part in path.parents:
        require(not part.is_symlink(), f"Symlink evidence ancestor: {part}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, parse_constant=invalid)


def load(path):
    file_sha(path)
    return parse(path.read_bytes())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def fingerprint(files):
    return sha("".join(f"{name}\0{digest}\n" for name, digest in sorted(files.items())).encode())


class Collection:
    def __init__(self):
        self.files = {}
        self.mapping = []

    def add(self, path, name, expected=None):
        digest, size = file_sha(path), path.stat().st_size
        require(expected is None or digest == expected, f"Changed input: {path}")
        require(path.suffix in SUFFIXES and size <= FILE_LIMIT, f"Unreviewed file type/size: {path} ({size})")
        data = path.read_bytes()
        require(b"\0" not in data, f"Binary evidence file: {path}")
        data.decode("utf-8")
        self._put(name, path, digest, size)
        self.mapping.append(dict(original=str(path.relative_to(REPO)), published=name, sha256=digest, bytes=size))

    def _put(self, name, value, digest, size):
        relative(name)
        if name in self.files:
            require(self.files[name][1:] == (digest, size), f"Archive path collision: {name}")
        else:
            self.files[name] = (value, digest, size)

    def generated(self, name, value):
        data = value if isinstance(value, bytes) else encoded(value)
        require(len(data) <= FILE_LIMIT and name not in self.files, f"Generated file collision/size: {name}")
        self._put(name, data, sha(data), len(data))

    def inventory(self):
        return {name: value[1] for name, value in sorted(self.files.items())}


def validate_local(collection, receipt_name, log_name, current):
    path, log = under(BASE, receipt_name), under(BASE, log_name)
    receipt = load(path)
    source = receipt["source"]
    require(source["source_sha256"] == current == fingerprint(source["source_files"]), "Local source identity differs")
    actual = {str(p.relative_to(REPO / "src/clipp1d")): file_sha(p)
              for p in sorted((REPO / "src/clipp1d").rglob("*.py"))}
    require(actual == source["source_files"], "Current source changed since final local tests")
    for key, directory in (("test_files", "tests"), ("benchmark_files", "benchmarks")):
        files = {str(p.relative_to(REPO)): file_sha(p) for p in sorted((REPO / directory).rglob("*.py"))}
        require(files == receipt[key], f"Final {directory} validation does not cover current files")
    tests = receipt["tests"]
    require(tests["passed"] > 0 and tests["failed"] == tests["skipped"] == 0, "Incomplete final local test result")
    require(receipt["static_checks"] and all(v == "passed" for v in receipt["static_checks"].values()),
            "Final static checks failed")
    require(file_sha(log) == tests["log_sha256"], "Final suite log hash differs")
    summaries = re.findall(rb"(?m)^(\d+) passed(?:, (\d+) warnings?)? in ([0-9.]+)s", log.read_bytes())
    require(len(summaries) == 1 and int(summaries[0][0]) == tests["passed"] and
            int(summaries[0][1] or b"0") == tests.get("warnings", 0) and float(summaries[0][2]) == tests["seconds"],
            "Final suite summary differs from receipt")
    collection.add(path, "local/LOCAL_VALIDATION.json")
    collection.add(log, "local/pytest-observed-summary.txt")
    return dict(receipt="local/LOCAL_VALIDATION.json", sha256=file_sha(path), tests=tests,
                scope="Local CPU unit/reference and static checks, not CUDA qualification")


def source_classification(source_sha256, current, historical):
    if source_sha256 == current:
        return "current"
    if source_sha256 == BASELINE_SHA:
        return "baseline"
    require(source_sha256 in historical, "Unadmitted historical production source")
    return "historical"


def seal(collection, parent, current, historical):
    local = load(parent / "PREPARED.json")
    run_id = local["run_id"]
    require(re.fullmatch(r"clipp2_clipp1d_bound_20260923[a-z][a-z0-9]*", run_id), "Unexpected run identity")
    root = under(parent, "sealed/" + run_id)
    plan, inventory = load(root / "PREPARED.json"), load(root / "inventory.json")
    require(all(local.get(k) == v for k, v in plan.items()), "Local/sealed prepared plans differ")
    require(file_sha(root / "inventory.json") == local["inventory_sha256"], "Changed frozen inventory")
    archive = parent / (run_id + ".tar.gz")
    require(file_sha(archive) == local["archive_sha256"], "Changed original source archive")
    require(plan["baseline_commit"] == plan["source_commit"] == BASELINE_COMMIT, "Unexpected baseline commit")
    require(plan["source_sha256"]["baseline"] == BASELINE_SHA, "Unadmitted baseline production source")
    source_classification(plan["source_sha256"]["current"], current, historical)
    require(plan["remote_root"] == "/rsrch8/scratch/bcb/yding4/" + run_id, "Unexpected remote root")
    require(load(parent / "tasks.json") == plan["tasks"], "Local tasks differ from frozen tasks")
    require(inventory["PREPARED.json"] == file_sha(root / "PREPARED.json"), "Plan absent from frozen inventory")
    for name, digest in inventory.items():
        require(file_sha(under(root, name)) == digest, f"Frozen file changed: {name}")
    maps = {}
    for role, tree in (("current", "source"), ("baseline", "baseline_source")):
        identity = plan["source_identities"][role]
        files = {str(p.relative_to(root / tree / "src/clipp1d")): file_sha(p)
                 for p in sorted((root / tree / "src/clipp1d").rglob("*.py"))}
        require(files == identity["source_files"] and fingerprint(files) == identity["source_sha256"] ==
                plan["source_sha256"][role], "Frozen source fingerprint differs")
        require(all(inventory.get(f"{tree}/src/clipp1d/{name}") == digest for name, digest in files.items()),
                "Production source omitted from inventory")
        classification = source_classification(identity["source_sha256"], current, historical)
        maps[role] = dict(identity=identity, baseline_commit=BASELINE_COMMIT, overlays={},
                          production_revision=classification,
                          eligible_for_current_source_qualification=classification == "current")
    for name, digest in inventory.items():
        if name.startswith("baseline_source/"):
            original = name.removeprefix("baseline_source/")
            content = subprocess.check_output(["git", "show", BASELINE_COMMIT + ":" + original], cwd=REPO)
            require(sha(content) == digest, f"Baseline bytes differ from commit: {original}")
        elif name.startswith("source/"):
            original = name.removeprefix("source/")
            if original.startswith("benchmarks/"):
                destination = f"reproduction/helpers/{digest}/{PurePosixPath(name).name}"
                maps["current"].setdefault("helpers", {})[original] = destination
            elif digest != inventory.get("baseline_source/" + original):
                destination = f"reproduction/source-{plan['source_sha256']['current']}/overlay/{original}"
                maps["current"]["overlays"][original] = destination
            else:
                continue
            collection.add(under(root, name), destination, digest)
    for name in ("inventory.json", "PREPARED.json", "worker.py", "launch.py", "tracked-dirty.patch"):
        collection.add(root / name, f"attempts/{parent.name}/sealed/{name}")
    return plan, inventory, maps, dict(path=str(archive.relative_to(REPO)), sha256=file_sha(archive),
                                      bytes=archive.stat().st_size, reason="Redundant sealed source archive remains at original path")


def bound(parent, files, name, expected=None):
    require(name in files, f"Missing imported binding: {name}")
    path, item = under(parent / "imported", name), files[name]
    require(path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"], f"Imported file changed: {name}")
    require(expected is None or expected == item["sha256"], f"Nested receipt hash differs: {name}")
    return path


def remote_name(plan, name):
    path, root = PurePosixPath(name), PurePosixPath(plan["remote_root"])
    require(path.is_absolute() and path.is_relative_to(root), "Remote artifact outside bound attempt")
    return str(relative(str(path.relative_to(root))))


def array_bytes(value, shape):
    if not shape:
        require(type(value) in (float, int) and math.isfinite(value), "Invalid JSON float64 scalar")
        return struct.pack("<d", value)
    require(isinstance(value, list) and len(value) == shape[0], "Array JSON shape mismatch")
    return b"".join(array_bytes(row, shape[1:]) for row in value)


def tensor_descriptor(parent, files, descriptor):
    path = bound(parent, files, "results/" + str(relative(descriptor["path"])), descriptor["sha256"])
    value = load(path)
    shape = descriptor["shape"]
    require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), "Invalid tensor dimensions")
    require(value["shape"] == shape and value["dtype"] == descriptor["dtype"] == "float64", "Tensor descriptor differs")
    if value["encoding"] == "json_numbers":
        data = array_bytes(value["values"], shape)
    else:
        require(value["encoding"] == "float64_le_hex", "Unknown lossless tensor encoding")
        data = bytes.fromhex(value["values"])
    require(len(data) == 8 * math.prod(shape) and sha(data) == value["tensor_sha256"] == descriptor["tensor_sha256"],
            "Tensor bytes did not reconstruct exactly")


def nested_descriptors(parent, files, value):
    if isinstance(value, dict):
        if {"path", "sha256", "dtype", "shape", "tensor_sha256"} <= value.keys():
            tensor_descriptor(parent, files, value)
        else:
            for item in value.values():
                nested_descriptors(parent, files, item)
    elif isinstance(value, list):
        for item in value:
            nested_descriptors(parent, files, item)


def artifacts(parent, files, plan, task, receipt):
    if "events_sha256" in receipt:
        name = "results/" + receipt.get("events_file", task["name"] + ".events.jsonl")
        events = [parse(line) for line in bound(parent, files, name, receipt["events_sha256"]).read_bytes().splitlines()]
        if "cases" in receipt:
            require(receipt["cases"] == events, "Qualifier event journal differs from receipt")
    for name, digest in receipt.get("artifacts", {}).items():
        bound(parent, files, "results/" + str(relative(name)), digest)
    if "source_and_plan_sha256" in receipt:
        bound(parent, files, f"results/{task['name']}.artifacts/source-and-plan.json",
              receipt["source_and_plan_sha256"])
    for row in receipt.get("cases", []):
        if "artifact" in row:
            bound(parent, files, remote_name(plan, row["artifact"]), row["artifact_sha256"])
        if "detail_file" in row:
            bound(parent, files, remote_name(plan, row["detail_file"]), row["detail_sha256"])
        if "directory" in row and "run_sha256" in row:
            root = remote_name(plan, row["directory"])
            bound(parent, files, root + "/run.json", row["run_sha256"])
            for name, digest in row["table_sha256"].items():
                bound(parent, files, root + "/" + str(relative(name)), digest)
    for entry in receipt.get("captures", []):
        record = load(bound(parent, files, "results/" + str(relative(entry["record"])), entry["sha256"]))
        require(record["context"] == entry["context"] and record["context"]["policy"] == receipt["policy"] and
                record["context"]["source_sha256"] == receipt["source"]["source_sha256"], "Capture context differs")
        require(record["returned"]["qualified"] is False and
                record["original_compiled_certificate_replay"]["exact_bits_equal"] is True,
                "Capture does not preserve its unresolved compiled certificate proof")
        nested_descriptors(parent, files, record)
    nested_descriptors(parent, files, receipt.get("replays", []))
    if receipt.get("schema") == POOLED_SCHEMA and receipt["status"] == "passed":
        root = f"results/{task['name']}.artifacts/"
        for name in ("literal-pooled-state.json", "literal-directional-cuts.json"):
            document = load(bound(parent, files, root + name))
            nested_descriptors(parent, files, document)
        state = load(bound(parent, files, root + "literal-pooled-state.json"))
        require(state["initial_pooled"]["tensor_sha256"] == receipt["pooled_sha256"],
                "Pooled-state tensor hash differs from its receipt")
        bound(parent, files, root + "input.tsv", CAPTURE_INPUTS[("mixed_support", 256)])


def baseline_policy():
    # Parse only literal defaults from immutable baseline source; never import or execute numerical code.
    policy_source = subprocess.check_output(["git", "show", BASELINE_COMMIT + ":src/clipp1d/cuda/policy.py"], cwd=REPO)
    classes = [node for node in ast.parse(policy_source).body
               if isinstance(node, ast.ClassDef) and node.name == "CudaPolicy"]
    require(len(classes) == 1, "Baseline policy declaration is ambiguous")
    return {node.target.id: ast.literal_eval(node.value) for node in classes[0].body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def capture_profile(receipt):
    schema = receipt["schema"]
    if schema == "clipp1d.cuda.failed_qp_capture.v1":
        name = receipt.get("source_profile", "baseline430")
        require(name == "baseline430", "Capture v1 must retain its baseline430 binding")
    else:
        require(schema == "clipp1d.cuda.failed_qp_capture.v2", "Unknown capture schema")
        name = receipt["source_profile"]
        require(name in ("normalized276da", "bound8627"), "Unadmitted capture v2 profile")
    profile = CAPTURE_PROFILES[name]
    source = receipt["source"]
    require(source["source_sha256"] == profile["source_sha256"] and
            all(source["source_files"].get(key) == value for key, value in profile["source_files"].items()) and
            receipt["frozen_commit"] == profile["commit"], "Capture profile production binding differs")
    family = (receipt["fixture_family"], receipt["nodes"])
    require(family in profile["failures"] and receipt["expected_failures"] == profile["failures"][family],
            "Capture profile planned inventory differs")
    policy = baseline_policy()
    require(receipt["policy"] == policy, "Capture profile numerical policy differs")
    if receipt["status"] == "passed":
        require(receipt["input_sha256"] == CAPTURE_INPUTS[family], "Capture profile input differs")
        require(receipt["captured_failures"] == len(receipt["captures"]) == receipt["expected_failures"],
                "Capture profile completed inventory differs")
        require(all(entry["context"]["source_sha256"] == profile["source_sha256"] and
                    entry["context"]["input_sha256"] == receipt["input_sha256"] and
                    entry["context"]["policy"] == policy for entry in receipt["captures"]),
                "Capture profile context binding differs")
        expected = profile.get("expected_contexts", {}).get(family)
        if expected is not None:
            actual = [{key: entry["context"].get(key) for key in expected[0]} for entry in receipt["captures"]]
            require(actual == expected, "Capture profile failure identities differ")
    return name


def pooled_diagnostic(receipt):
    require(receipt["schema"] == POOLED_SCHEMA and receipt["source"]["source_sha256"] == POOLED_SOURCE and
            fingerprint(receipt["source"]["source_files"]) == POOLED_SOURCE and
            receipt["script_sha256"] == POOLED_SCRIPT and receipt["scientific_fit_qualified"] is False,
            "Pooled diagnostic source/script/scope differs")
    passed = receipt["status"] == "passed"
    if passed:
        require(receipt["cuda_available"] is True and receipt["device"].startswith("cuda:") and
                receipt["policy"] == baseline_policy() and receipt["reference_sha256"] == POOLED_REFERENCE,
                "Pooled diagnostic CUDA/policy/reference differs")
        inputs = [digest for name, digest in receipt["artifacts"].items() if name.endswith("/input.tsv")]
        require(inputs == [CAPTURE_INPUTS[("mixed_support", 256)]], "Pooled diagnostic input differs")
        path = receipt["path7"]
        require(path["objective_exact_match"] is True and
                path["original_objective"] == path["archived_objective"] == 4617151.7318359185,
                "Pooled diagnostic failed exact archived-objective reconstruction")
        original, raw = path["positive_original"], path["original_raw_audit"]
        require(original["qualified"] is False and raw["qualified"] is False and
                raw["status"] == "positive_unresolved" and
                original["diagnostics"] == raw["diagnostics"]["positive"] and
                original["diagnostics"]["status"] == "unresolved" and
                original["diagnostics"]["iterations"] == receipt["policy"]["inner_max_iterations"],
                "Pooled diagnostic original cut/audit identity differs")
        require(path["positive_rowwise"]["qualified"] is True and
                path["positive_rowwise"]["diagnostics"]["tolerance"] == original["diagnostics"]["tolerance"] and
                path["positive_rowwise"]["diagnostics"]["status"] == "qualified_lower_bound" and
                all(path[name]["qualified"] is False and
                    path[name]["diagnostics"]["status"] == "attained_descent"
                    for name in ("negative_original", "negative_rowwise")),
                "Pooled diagnostic cut experiment changed its stated outcomes")
    return dict(kind="pooled_outer_diagnostic", diagnostic_completed=passed, complete_fit_qualified=False,
                source_sha256=POOLED_SOURCE, reference_sha256=receipt.get("reference_sha256"),
                scope=receipt.get("diagnostic_scope"),
                command_binding="Exact immutable worker command and pinned sealed helper; this diagnostic receipt schema omits command")


def semantics(receipt):
    schema, passed = receipt.get("schema", ""), receipt["status"] == "passed"
    if schema == POOLED_SCHEMA:
        return pooled_diagnostic(receipt)
    if "failed_qp_capture" in schema:
        profile_name = capture_profile(receipt)
        if passed:
            require(receipt["diagnostic_capture_qualified"] is True and receipt["scientific_fit_qualified"] is False and
                    receipt["scientific_search_status"] == "incomplete" and
                    len(receipt["captures"]) == receipt["captured_failures"] == receipt["expected_failures"],
                    "Capture completion is inconsistent")
        return dict(kind="failed_qp_capture", diagnostic_completed=passed, complete_fit_qualified=False,
                    source_profile=profile_name, source_sha256=CAPTURE_PROFILES[profile_name]["source_sha256"],
                    captured_failures=receipt.get("captured_failures"), scope=receipt.get("qualification_scope"))
    if "failed_qp_replay" in schema:
        rows = receipt["replays"]
        require(all(type(row["qualified"]) is bool for row in rows), "Missing explicit original-start qualification")
        complete = len(rows) == receipt.get("captured_failed_qps") and bool(rows)
        if passed:
            require(receipt["diagnostic_status"] == "completed" and complete, "Replay diagnostic coverage incomplete")
            require(receipt["all_replays_qualified"] == all(row["qualified"] for row in rows) and
                    receipt["resolved_qps"] == sum(row["qualified"] for row in rows) and
                    receipt["unresolved_qps"] == sum(not row["qualified"] for row in rows), "Replay qualification counts differ")
        return dict(kind="saved_qp_replay", diagnostic_completed=passed, complete_fit_qualified=False,
                    all_replays_qualified=bool(passed and complete and receipt["all_replays_qualified"]),
                    resolved_qps=receipt.get("resolved_qps"), unresolved_qps=receipt.get("unresolved_qps"),
                    scope=receipt["qualification_scope"])
    return dict(kind="synthetic_qualification", status=receipt["status"],
                full_path=receipt.get("full_path"), scope=receipt.get("qualification_scope", receipt.get("scope")),
                note="Read exact qualification receipt; this archive does not independently rerun numerical admission")


def attempt(collection, name, current, historical):
    require(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name), "Invalid attempt name")
    parent = under(BASE, name)
    plan, inventory, source_maps, archive = seal(collection, parent, current, historical)
    imported = load(parent / "IMPORT_MANIFEST.json")
    files = imported["files"]
    observed = {str(p.relative_to(parent / "imported")) for p in (parent / "imported").rglob("*") if p.is_file()}
    require(observed == set(files), "Imported inventory coverage differs")
    for path in files:
        bound(parent, files, path)
    terminal = load(bound(parent, files, "receipts/terminal.json"))
    accepted = load(bound(parent, files, "receipts/accepted.json"))
    require(terminal == imported["terminal"] and type(terminal["passed"]) is bool, "Imported terminal status differs")
    require(terminal["job_id"] == accepted["job_id"] and accepted["job_name"] == plan["job_name"] and
            terminal["plan_sha256"] == accepted["plan_sha256"] == inventory["PREPARED.json"], "Terminal job/source identity differs")
    scheduler = imported["scheduler"]
    require(f"Job <{accepted['job_id']}>" in scheduler and f"Job Name <{plan['job_name']}>" in scheduler and
            "User <yding4>" in scheduler and any(f"Status <{s}>" in scheduler for s in ("DONE", "EXIT")),
            "Scheduler owner is not the exact terminal job")
    if "receipts/startup.json" in files:
        startup = load(bound(parent, files, "receipts/startup.json"))
        require(startup["job_id"] == accepted["job_id"] and startup["environment_sha256"] == plan["environment_sha256"] and
                startup["compiler_sha256"] == plan["compiler"]["sha256"] and
                startup["inventory_sha256"] == load(parent / "PREPARED.json")["inventory_sha256"], "Startup binding differs")
        require(set(startup["runtimes"]) == {row["source_role"] for row in plan["tasks"]}, "Startup source coverage differs")
        for role, runtime in startup["runtimes"].items():
            require(runtime["source"]["source_sha256"] == plan["source_sha256"][role] and
                    runtime["source"]["source_files"] == plan["source_identities"][role]["source_files"] and
                    "L40" in runtime["gpu"], "Startup source/GPU identity differs")
    else:
        require(not terminal["passed"], "Passed job has no CUDA startup proof")
    executed = {row["name"]: row for row in terminal["tasks"]}
    require(len(executed) == len(terminal["tasks"]) and set(executed) <= {row["name"] for row in plan["tasks"]},
            "Terminal task coverage differs")
    summaries = []
    for task in plan["tasks"]:
        task_name, role = task["name"], task["source_role"]
        result_name, command_name = f"results/{task_name}.json", f"receipts/command-{task_name}.json"
        row = dict(name=task_name, source_role=role, source_sha256=plan["source_sha256"][role], script=task["script"],
                   production_revision=source_maps[role]["production_revision"],
                   eligible_for_current_source_qualification=source_maps[role]["eligible_for_current_source_qualification"],
                   planned_arguments=task["args"], status="not_attempted")
        if command_name in files:
            command = load(bound(parent, files, command_name))
            args = [plan["remote_root"] + "/" + str(relative(value[7:])) if value.startswith("@ROOT@/") else
                    files[str(relative(value[9:]))]["sha256"] if value.startswith("@SHA256@/") else value
                    for value in task["args"]]
            expected = [plan["python"], "-B", plan["remote_root"] + "/source/benchmarks/" + task["script"],
                        *([task["subcommand"]] if "subcommand" in task else []), "--device", "cuda:0", "--out",
                        plan["remote_root"] + "/" + result_name, *args]
            require(command["argv"] == expected and command["source_role"] == role and
                    command["timeout_seconds"] == task["timeout_seconds"], "Executed command differs from exact plan")
            row["status"] = "interrupted_without_task_receipt"
        if result_name in files:
            execution = executed.get(task_name, {})
            path = bound(parent, files, result_name, execution.get("receipt_sha256"))
            receipt = load(path)
            require(receipt["status"] in ("passed", "failed") and command_name in files,
                    "Task command/status binding differs")
            if receipt.get("schema") == POOLED_SCHEMA:
                # This one immutable diagnostic schema did not duplicate argv.
                # The exact worker argv was already checked against the sealed
                # task above; admit only its reviewed script/source identities.
                require("command" not in receipt and task["script"] == "capture_outer_failure.py" and
                        receipt["script_sha256"] == POOLED_SCRIPT and
                        inventory["source/benchmarks/capture_outer_failure.py"] == POOLED_SCRIPT,
                        "Pooled diagnostic omitted-command identity differs")
            else:
                require(receipt["command"] == command["argv"][2:], "Task command/status binding differs")
            require(receipt["source"]["source_sha256"] == plan["source_sha256"][role] and
                    receipt["source"]["source_files"] == plan["source_identities"][role]["source_files"], "Task source differs")
            require(not execution or execution.get("status") == receipt["status"], "Task/worker status differs")
            expected_script = inventory[f"source/benchmarks/{task['script']}"]
            for key in ("qualification_script_sha256", "script_sha256", "capture_script_sha256"):
                if key in receipt:
                    require(receipt[key] == expected_script, "Executed benchmark bytes differ")
            for key in ("helpers", "capture_dependency_hashes"):
                for helper, digest in receipt.get(key, {}).items():
                    filename = "capture_failed_qp.py" if key == "capture_dependency_hashes" and helper == "driver" else (
                        task["script"] if helper == "driver" else HELPERS[helper])
                    require(inventory[f"source/benchmarks/{filename}"] == digest, "Transitive helper bytes differ")
            if "capture_helper_sha256" in receipt:
                require(receipt["capture_helper_sha256"] == inventory["source/benchmarks/capture_failed_qp.py"], "Capture helper differs")
            artifacts(parent, files, plan, task, receipt)
            row.update(status=receipt["status"], schema=receipt.get("schema"), receipt=result_name,
                       receipt_sha256=file_sha(path), elapsed_seconds=receipt.get("elapsed_seconds"),
                       error_type=receipt.get("error_type"), error=receipt.get("error"), claims=semantics(receipt))
            for key in ("capture", "baseline", "predecessor", "fixture_family", "nodes", "mode", "comparison"):
                if key in receipt:
                    row[key] = receipt[key]
            if receipt.get("schema") == POOLED_SCHEMA:
                def option(flag):
                    require(command["argv"].count(flag) == 1, "Pooled diagnostic reference argument differs")
                    return command["argv"][command["argv"].index(flag) + 1]
                row["reference"] = dict(path=option("--reference"), sha256=option("--reference-sha256"))
                require(row["reference"]["sha256"] == POOLED_REFERENCE, "Pooled diagnostic command reference differs")
        summaries.append(row)
    if terminal["passed"]:
        require(len(executed) == len(plan["tasks"]) and all(row["status"] == "passed" for row in summaries) and
                all(row["returncode"] == 0 for row in executed.values()), "Passed terminal has unfinished/failed tasks")
    for path in files:
        require(not any(part.startswith("compiler-cache-") or part == "__pycache__" for part in PurePosixPath(path).parts),
                "Compiler cache unexpectedly imported")
        collection.add(bound(parent, files, path), f"attempts/{name}/imported/{path}", files[path]["sha256"])
    for filename in OPERATIONS:
        path = parent / filename
        if path.exists():
            collection.add(path, f"attempts/{name}/{filename}")
    return dict(attempt=name, job_id=accepted["job_id"], remote_root=plan["remote_root"],
                status="passed" if terminal["passed"] else "failed", terminal_error=terminal.get("error"),
                source_reconstruction=source_maps, environment_sha256=plan["environment_sha256"],
                tasks=summaries, excluded_redundant_archive=archive,
                frozen_source_template=plan.get("frozen_source_template"))


def dependencies(attempts):
    lookup = {a["remote_root"] + "/" + row["receipt"]: row for a in attempts for row in a["tasks"] if "receipt" in row}
    attempt_lookup = {a["attempt"]: a for a in attempts}
    artifact_lookup = {}
    for a in attempts:
        parent = under(BASE, a["attempt"])
        files = load(parent / "IMPORT_MANIFEST.json")["files"]
        artifact_lookup.update({a["remote_root"] + "/" + name: (a, parent, files, name)
                                for name in files})
    for a in attempts:
        template = a.get("frozen_source_template")
        if template:
            require(template["attempt"] in attempt_lookup, "Missing frozen-source template attempt")
            original = attempt_lookup[template["attempt"]]
            prepared = load(under(BASE, template["attempt"]) / "PREPARED.json")
            require(prepared["inventory_sha256"] == template["inventory_sha256"] and
                    original["source_reconstruction"]["current"]["identity"] ==
                    a["source_reconstruction"]["current"]["identity"] and
                    template["source_sha256"] == a["source_reconstruction"]["current"]["identity"]["source_sha256"],
                    "Frozen-source template package identity differs")
        for row in a["tasks"]:
            if "reference" in row:
                item = row["reference"]
                require(row["claims"]["kind"] == "pooled_outer_diagnostic" and
                        item["sha256"] == POOLED_REFERENCE and item["path"] in artifact_lookup,
                        "Missing exact archived pooled diagnostic reference artifact")
                owner, parent, files, name = artifact_lookup[item["path"]]
                reference = load(bound(parent, files, name, POOLED_REFERENCE))
                require(name.endswith("/full-path.json") and reference["search_status"] == "incomplete" and
                        owner["source_reconstruction"]["current"]["identity"]["source_sha256"] == POOLED_SOURCE,
                        "Pooled diagnostic reference is not the frozen incomplete J path")
                if row["status"] == "passed":
                    local_parent = under(BASE, a["attempt"])
                    local_files = load(local_parent / "IMPORT_MANIFEST.json")["files"]
                    state = load(bound(local_parent, local_files,
                                       f"results/{row['name']}.artifacts/literal-pooled-state.json"))
                    require(state["pilot"]["tensor_sha256"] == reference["pilot_and_graph"]["pilot_sha256"] and
                            state["weights"]["tensor_sha256"] == reference["pilot_and_graph"]["weights_sha256"],
                            "Pooled diagnostic literal pilot/graph differs from J")
            for key in ("capture", "baseline", "predecessor"):
                item = row.get(key)
                if not item:
                    continue
                path, digest = item.get("path", item.get("receipt")), item.get("sha256")
                require(path in lookup and lookup[path]["receipt_sha256"] == digest, f"Missing exact archived {key} dependency")
                dependency = lookup[path]
                require(dependency["status"] == "passed", f"Unqualified {key} dependency")
                if key == "capture":
                    require(dependency["claims"]["kind"] == "failed_qp_capture" and
                            dependency["claims"]["diagnostic_completed"] and
                            dependency["source_sha256"] == dependency["claims"]["source_sha256"],
                            "Replay dependency is not an admitted diagnostic capture")
                    require(item["baseline_source_sha256"] == dependency["source_sha256"],
                            "Replay capture-source fingerprint differs")
                    require(item["baseline_commit"] == CAPTURE_PROFILES[dependency["claims"]["source_profile"]]["commit"],
                            "Replay capture-source commit binding differs")
                    if "source_profile" in item:
                        require(item["source_profile"] == dependency["claims"]["source_profile"],
                                "Replay capture-source profile differs")
                else:
                    require(dependency["claims"]["kind"] == "synthetic_qualification", "Diagnostic completion cannot gate a full-path successor")
                    if key == "predecessor":
                        require(dependency["source_sha256"] == row["source_sha256"],
                                "A different production revision cannot qualify a full-path successor")


def publish(collection):
    require(not DESTINATION.exists() and not DESTINATION.is_symlink(), "Destination exists; publication never overwrites")
    DESTINATION.parent.mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".cuda-bound-recovery-v3-build-", dir=DESTINATION.parent))
    try:
        inventory = collection.inventory()
        for name, (source, digest, size) in sorted(collection.files.items()):
            output = under(build, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                if isinstance(source, bytes):
                    stream.write(source)
                else:
                    require(file_sha(source) == digest, f"Evidence changed before copy: {source}")
                    with source.open("rb") as original:
                        shutil.copyfileobj(original, stream)
                stream.flush()
                os.fsync(stream.fileno())
            require(output.stat().st_size == size and file_sha(output) == digest, "Archive readback mismatch")
        with (build / "SHA256.json").open("xb") as stream:
            stream.write(encoded(inventory))
            stream.flush()
            os.fsync(stream.fileno())
        require(load(build / "SHA256.json") == inventory, "Final inventory readback differs")
        for directory in sorted([build, *(p for p in build.rglob("*") if p.is_dir())], reverse=True):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        library = ctypes.CDLL(None, use_errno=True)
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(build), -100, os.fsencode(DESTINATION), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(DESTINATION))
        descriptor = os.open(DESTINATION.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        require(all(file_sha(under(DESTINATION, name)) == digest for name, digest in inventory.items()),
                "Published final readback differs")
        return file_sha(DESTINATION / "SHA256.json")
    finally:
        if build.exists():
            shutil.rmtree(build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", action="append", required=True, help="Explicit imported terminal attempt; repeat")
    parser.add_argument("--expected-current-source-sha256", required=True)
    parser.add_argument("--historical-source-sha256", action="append", default=[],
                        help="Explicit historical production fingerprint; repeat. Retention never qualifies current source.")
    parser.add_argument("--max-total-bytes", type=int, required=True, help="Reviewed archive ceiling, including hash inventory")
    parser.add_argument("--local-validation", required=True, help="Final source/test/helper-bound local validation, relative to study")
    parser.add_argument("--suite-log", required=True, help="Its hash-bound observed pytest output or explicitly identified summary, relative to study")
    parser.add_argument("--study-artifact", action="append", default=[], help="Exact additional reviewed study-relative analysis/helper file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default; report reconciled inventory/size without writes")
    mode.add_argument("--publish", action="store_true", help="Publish exclusively after all bindings and size checks pass")
    args = parser.parse_args()
    require(re.fullmatch("[0-9a-f]{64}", args.expected_current_source_sha256), "Invalid requested current source")
    historical = set(args.historical_source_sha256)
    require(len(historical) == len(args.historical_source_sha256) and
            all(re.fullmatch("[0-9a-f]{64}", value) for value in historical) and
            not historical.intersection((BASELINE_SHA, args.expected_current_source_sha256)),
            "Invalid, duplicate, or nonhistorical source allowlist entry")
    require(args.max_total_bytes > 0 and len(args.attempt) == len(set(args.attempt)), "Invalid ceiling/duplicate attempt")
    require(not DESTINATION.exists() and not DESTINATION.is_symlink(), "Destination exists; never overwrite")
    study = load(BASE / "STUDY_PLAN.json")
    require(study["baseline_commit"] == BASELINE_COMMIT and study["baseline_source_sha256"] == BASELINE_SHA,
            "Study baseline authority differs")
    collection = Collection()
    local = validate_local(collection, args.local_validation, args.suite_log, args.expected_current_source_sha256)
    attempts = [attempt(collection, name, args.expected_current_source_sha256, historical) for name in args.attempt]
    observed_historical = {details["identity"]["source_sha256"] for item in attempts
                           for details in item["source_reconstruction"].values()
                           if details["production_revision"] == "historical"}
    require(observed_historical == historical, "Historical source allowlist contains an unused fingerprint")
    dependencies(attempts)
    # Prevent known negative evidence from disappearing during a later publication.
    for path in BASE.glob("*/IMPORT_MANIFEST.json"):
        if load(path)["terminal"]["passed"] is False or (path.parent / "MEASUREMENT_INVALIDATION.json").exists():
            require(path.parent.name in args.attempt, f"Omitted failed/invalidated attempt: {path.parent.name}")
    for name in ("STUDY_PLAN.json", "prepare.py", "worker.py", "launch.py", "launch_prepared.py", "watch.py",
                 "import_evidence.py", "publish_evidence.py", *args.study_artifact):
        collection.add(under(BASE, name), "study/" + str(relative(name)))
    summary = dict(schema="clipp1d.cuda.bound.evidence.v1", status="evidence_reconciled",
                   generated_utc=datetime.now(timezone.utc).isoformat(), baseline_commit=BASELINE_COMMIT,
                   baseline_source_sha256=BASELINE_SHA, current_source_sha256=args.expected_current_source_sha256,
                   historical_source_sha256=sorted(historical),
                   local_validation=local, attempts=attempts,
                   task_status_counts=dict(Counter(row["status"] for a in attempts for row in a["tasks"])),
                   scope="Source-bound synthetic diagnostics and qualification. Diagnostic passed is not QP qualification, complete inference, or cohort accuracy.",
                   selection="Explicit attempt allowlist; known failed/invalidated imported attempts must be included.",
                   size_policy=dict(per_file_limit_bytes=FILE_LIMIT, reviewed_total_limit_bytes=args.max_total_bytes),
                   reconstruction="Baseline commit plus deduplicated changed source/helper bytes and original sealed inventories/patches; no repeated full source trees.")
    collection.generated("FINAL_STUDY.json", summary)
    collection.generated("COPY_MAP.json", collection.mapping)
    lines = ["# Bound recovery evidence", "", "Exact copied bytes from the explicitly selected terminal attempts.",
             "`FINAL_STUDY.json` preserves individual task outcomes and scientific scopes, including failures.",
             "`SHA256.json` covers every published file except itself; `COPY_MAP.json` records exact origins.", "",
             "Capture/replay diagnostic completion does not imply original-QP or complete-fit qualification.",
             "Historical-source receipts remain historical evidence and cannot qualify the current production source.",
             "Use each receipt's explicit qualification fields. No cohort accuracy is claimed.", "",
             "| Attempt | Job | Terminal | Planned task outcomes |", "| --- | --- | --- | --- |"]
    for a in attempts:
        states = "; ".join(row["name"] + ": " + row["status"] for row in a["tasks"])
        lines.append(f"| {a['attempt']} | {a['job_id']} | {a['status']} | {states} |")
    collection.generated("README.md", ("\n".join(lines) + "\n").encode())
    total = sum(item[2] for item in collection.files.values()) + len(encoded(collection.inventory()))
    require(total <= args.max_total_bytes, f"Reviewed aggregate ceiling exceeded: {total} > {args.max_total_bytes}")
    report = dict(status="dry_run_reconciled", destination=str(DESTINATION), files=len(collection.files) + 1,
                  bytes=total, max_total_bytes=args.max_total_bytes, attempts=args.attempt,
                  largest_files=sorted([dict(path=name, bytes=row[2]) for name, row in collection.files.items()],
                                       key=lambda row: (-row["bytes"], row["path"]))[:10])
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.publish:
        digest = publish(collection)
        print(json.dumps(dict(status="evidence_archived", destination=str(DESTINATION), files=report["files"],
                              bytes=total, inventory_sha256=digest), sort_keys=True))


if __name__ == "__main__":
    main()
