"""Publish exact optimization-study evidence after explicit terminal import.

Preparation only until the parent reviews and invokes this script. No remote
commands, numerical fits, original edits, or overwrite publication are possible.
The archive reports observed coverage; failed and unattempted tasks stay visible.
"""

import argparse
import codecs
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
DESTINATION = REPO / "validation/cuda-qp-v3"
BASELINE_COMMIT = "daaf50ad5a2ae7301e9b54b9c31e87d87b24cfa6"
BASELINE_SHA = "ea7071788094e0cd3537530fcdc62e4228054fce19ff02ba8f2cf4eacce1dd26"
CURRENT_SHA = "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88"
LIMIT = 10_000_000
TEXT_SUFFIXES = {".json", ".jsonl", ".tsv", ".err", ".py", ".patch", ".toml", ".log", ".md", ".txt"}
LOCAL_OPERATIONS = (
    "PREPARED.json",
    "tasks.json",
    "IMPORT_MANIFEST.json",
    "transfer-targets.json",
    "remote-preflight.json",
    "mac-preflight.json",
    "publication.json",
    "mac-upload.json",
    "remote-upload.json",
    "launch.json",
    "FAILURE_SUMMARY.json",
    "cancel-intent.json",
    "cancel-result.json",
    "stop-intent.json",
    "stop-result.json",
    "MEASUREMENT_INVALIDATION.json",
    "DIAGNOSIS.json",
    "FDIAGNOSIS.json",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def relative(name):
    require(isinstance(name, str), "Evidence path must be a string")
    path = PurePosixPath(name)
    require(
        path.parts and not path.is_absolute() and not any(p in (".", "..") for p in path.parts),
        f"Unsafe evidence path: {name}",
    )
    return path


def under(root, name):
    path = root.joinpath(*relative(name).parts)
    cursor = path
    while cursor != root.parent:
        require(not cursor.is_symlink(), f"Symlink in evidence path: {path}")
        cursor = cursor.parent
    return path


def regular(path):
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
        f"Missing/nonregular evidence: {path}",
    )
    # Absolute paths outside BASE are used only for known repository evidence.
    require(all(not part.is_symlink() for part in path.parents), f"Symlink ancestry: {path}")
    return path


def file_sha(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse(payload):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(payload, parse_constant=invalid)


def load(path):
    return parse(regular(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def source_digest(files):
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


def git_bytes(name):
    return subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:{name}"], cwd=REPO)


class Collection:
    """Validate all selections before publication; copies retain original bytes."""

    def __init__(self, reviews=None):
        self.files, self.mapping, self.excluded = {}, [], []
        self.reviews, self.used_reviews = reviews or {}, set()

    def add(self, source, destination, *, expected=None, origin=None):
        source = regular(source)
        relative(destination)
        origin = origin or str(source.relative_to(BASE))
        size = source.stat().st_size
        digest = file_sha(source)
        require(expected is None or digest == expected, f"Source hash mismatch: {source}")
        require(source.suffix in TEXT_SUFFIXES, f"Non-text evidence not admitted: {source}")
        if size > LIMIT:
            review = self.reviews.get(origin, {})
            require(
                review.get("sha256") == digest
                and review.get("bytes") == size
                and isinstance(review.get("reason"), str)
                and bool(review["reason"].strip()),
                f"File exceeds10MB; a separate exact-hash/size review is required: {origin}",
            )
            self.used_reviews.add(origin)
        # Reject binary payloads even if misleadingly named with a text suffix.
        decoder = codecs.getincrementaldecoder("utf-8")()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                require(b"\0" not in block, f"Binary payload not admitted: {source}")
                decoder.decode(block)
            decoder.decode(b"", final=True)
        if destination in self.files:
            require(
                self.files[destination][1:] == (digest, size),
                f"Conflicting destination: {destination}",
            )
        else:
            self.files[destination] = (source, digest, size)
        self.mapping.append(dict(original=origin, published=destination, sha256=digest, bytes=size))
        return destination

    def generated(self, destination, value):
        require(destination not in self.files, f"Generated evidence collision: {destination}")
        relative(destination)
        payload = encoded(value) if not isinstance(value, bytes) else value
        require(
            len(payload) <= LIMIT, f"Generated metadata unexpectedly exceeds10MB: {destination}"
        )
        self.files[destination] = (payload, sha(payload), len(payload))


def validate_local(collection, validation_name, suite_name):
    validation_path, suite_path = under(BASE, validation_name), under(BASE, suite_name)
    data = load(validation_path)
    require(
        data["schema"] == "clipp1d.cuda.qp.local-validation.v1", "Wrong local validation schema"
    )
    source = data["source"]
    require(
        source["source_sha256"] == CURRENT_SHA == source_digest(source["source_files"]),
        "Local tests do not bind the final current source",
    )
    actual = {
        str(p.relative_to(REPO / "src/clipp1d")): file_sha(p)
        for p in sorted((REPO / "src/clipp1d").rglob("*.py"))
    }
    require(
        actual == source["source_files"], "Production source changed after final local validation"
    )
    for key, directory in (("test_files", "tests"), ("benchmark_files", "benchmarks")):
        observed = {
            str(p.relative_to(REPO)): file_sha(p) for p in sorted((REPO / directory).rglob("*.py"))
        }
        require(observed == data[key], f"{directory} changed after final local validation")
    tests = data["tests"]
    require(
        tests["passed"] > 0 and tests["failed"] == tests["skipped"] == 0,
        "Final local suite did not fully pass",
    )
    require(
        data["static_checks"] and all(v == "passed" for v in data["static_checks"].values()),
        "Final local static checks did not pass",
    )
    require(file_sha(suite_path) == tests["log_sha256"], "Final suite log hash mismatch")
    summaries = re.findall(
        rb"(?m)^(\d+) passed(?:, (\d+) warnings?)? in ([0-9.]+)s", suite_path.read_bytes()
    )
    require(
        len(summaries) == 1
        and int(summaries[0][0]) == tests["passed"]
        and int(summaries[0][1] or b"0") == tests.get("warnings", 0)
        and float(summaries[0][2]) == tests["seconds"],
        "Local test summary differs from its receipt",
    )
    collection.add(validation_path, "local/LOCAL_VALIDATION.json")
    collection.add(suite_path, "local/whole-suite-final.log")
    return data


def validate_seal(collection, parent):
    local = load(parent / "PREPARED.json")
    run_id = local["run_id"]
    require(
        re.fullmatch(r"clipp2_clipp1d_qp_20260923[a-z][a-z0-9]*", run_id),
        "Wrong study run identity",
    )
    sealed = under(parent, "sealed/" + run_id)
    plan_path, inventory_path = sealed / "PREPARED.json", sealed / "inventory.json"
    plan, inventory = load(plan_path), load(inventory_path)
    require(all(local.get(k) == value for k, value in plan.items()), "Prepared plans differ")
    require(file_sha(inventory_path) == local["inventory_sha256"], "Sealed inventory changed")
    require(
        file_sha(parent / (run_id + ".tar.gz")) == local["archive_sha256"], "Source archive changed"
    )
    require(
        plan["source_commit"] == plan["baseline_commit"] == BASELINE_COMMIT,
        "Study must reconstruct from exact daaf50a baseline",
    )
    require(
        plan["source_sha256"] == dict(current=CURRENT_SHA, baseline=BASELINE_SHA),
        "An attempt uses a different scientific source",
    )
    require(
        plan["remote_root"] == "/rsrch8/scratch/bcb/yding4/" + run_id,
        "Unexpected remote study root",
    )
    require(
        load(parent / "tasks.json") == plan["tasks"], "Local task list differs from sealed plan"
    )
    for name, digest in inventory.items():
        require(file_sha(under(sealed, name)) == digest, f"Changed sealed file: {name}")
    require(
        inventory["PREPARED.json"] == file_sha(plan_path), "Plan not covered by frozen inventory"
    )
    source_maps = {}
    for role, tree in (("current", "source"), ("baseline", "baseline_source")):
        identity = plan["source_identities"][role]
        require(
            source_digest(identity["source_files"])
            == identity["source_sha256"]
            == plan["source_sha256"][role],
            "Scientific source fingerprint arithmetic differs",
        )
        actual = {
            str(p.relative_to(sealed / tree / "src/clipp1d")): file_sha(p)
            for p in sorted((sealed / tree / "src/clipp1d").rglob("*.py"))
        }
        require(
            actual == identity["source_files"],
            "Sealed scientific source differs from declared identity",
        )
        for name, digest in actual.items():
            require(
                inventory.get(f"{tree}/src/clipp1d/{name}") == digest,
                "Production source omitted from seal",
            )
        source_maps[role] = dict(identity=identity, baseline_commit=BASELINE_COMMIT, overlays={})
    # Validate the baseline against the retrievable commit, then copy only exact
    # changed current files. The overlay map complements the original git patch.
    for name, digest in inventory.items():
        if name.startswith("baseline_source/"):
            relative_name = name.removeprefix("baseline_source/")
            require(
                sha(git_bytes(relative_name)) == digest,
                f"Baseline differs from commit: {relative_name}",
            )
        elif name.startswith("source/"):
            relative_name = name.removeprefix("source/")
            if relative_name.startswith("benchmarks/"):
                destination = f"reproduction/helpers/{digest}/{PurePosixPath(name).name}"
                collection.add(under(sealed, name), destination, expected=digest)
                source_maps["current"].setdefault("helpers", {})[relative_name] = destination
            else:
                base_digest = inventory.get("baseline_source/" + relative_name)
                if base_digest != digest:
                    destination = f"reproduction/source-{CURRENT_SHA}/overlay/{relative_name}"
                    collection.add(under(sealed, name), destination, expected=digest)
                    source_maps["current"]["overlays"][relative_name] = destination
    for name in (
        "inventory.json",
        "PREPARED.json",
        "worker.py",
        "launch.py",
        "tracked-dirty.patch",
    ):
        collection.add(sealed / name, f"attempts/{parent.name}/sealed/{name}")
    return plan, inventory, source_maps


def imported_binding(parent, manifest, name, expected=None):
    require(name in manifest, f"Bound artifact absent from imported inventory: {name}")
    path = under(parent / "imported", name)
    item = manifest[name]
    require(
        file_sha(path) == item["sha256"] and path.stat().st_size == item["bytes"],
        f"Imported bytes changed: {name}",
    )
    require(
        expected is None or item["sha256"] == expected, f"Receipt artifact hash differs: {name}"
    )
    return path


def remote_name(plan, value):
    path, remote = PurePosixPath(value), PurePosixPath(plan["remote_root"])
    require(
        path.is_absolute() and path.is_relative_to(remote),
        f"Artifact outside its remote attempt: {value}",
    )
    return str(relative(str(path.relative_to(remote))))


def validate_task_artifacts(parent, manifest, plan, task, receipt):
    name = task["name"]
    root = "results/"
    events_name = root + receipt.get("events_file", name + ".events.jsonl")
    events = []
    if "events_sha256" in receipt:
        events_path = imported_binding(parent, manifest, events_name, receipt["events_sha256"])
        events = [parse(line) for line in events_path.read_bytes().splitlines()]
    if "cases" in receipt:
        require(
            receipt["cases"] == events, "Common qualifier receipt and incremental journal differ"
        )
    if "artifacts" in receipt:
        for artifact, digest in receipt["artifacts"].items():
            imported_binding(parent, manifest, root + str(relative(artifact)), digest)
    # Common full qualifier binds artifacts in event rows instead of a top-level map.
    for row in receipt.get("cases", []):
        if "artifact" in row:
            imported_binding(
                parent, manifest, remote_name(plan, row["artifact"]), row["artifact_sha256"]
            )
        if "directory" in row and "run_sha256" in row:
            directory = remote_name(plan, row["directory"])
            imported_binding(parent, manifest, directory + "/run.json", row["run_sha256"])
            for filename, digest in row["table_sha256"].items():
                imported_binding(
                    parent, manifest, directory + "/" + str(relative(filename)), digest
                )
    if "source_and_plan_sha256" in receipt:
        imported_binding(
            parent,
            manifest,
            f"results/{name}.artifacts/source-and-plan.json",
            receipt["source_and_plan_sha256"],
        )
    if "fixtures" in receipt:
        qualified = [event for event in events if event["kind"] == "fixture_qualified"]
        require(
            len(qualified) == len(receipt["fixtures"]),
            "Attribution fixture completion coverage differs",
        )
        for case, event in zip(receipt["fixtures"], qualified):
            identity = case["fixture"]
            fixture_name = str(relative(identity["name"]))
            require(
                "/" not in fixture_name
                and event["name"] == fixture_name
                and event["input_sha256"] == identity["input_sha256"],
                "Attribution fixture/event identity differs",
            )
            inputs = load(
                imported_binding(
                    parent,
                    manifest,
                    f"results/{name}.artifacts/{fixture_name}.inputs.json",
                    case["input_file_sha256"],
                )
            )
            require(
                inputs["identity"] == identity,
                "Attribution input identity differs from numerical receipt",
            )
            measured = load(
                imported_binding(
                    parent,
                    manifest,
                    f"results/{name}.artifacts/{fixture_name}.measurements.json",
                    event["measurements_sha256"],
                )
            )
            require(measured == case, "Attribution measurement bytes differ from numerical receipt")
        initial = load(
            imported_binding(parent, manifest, f"results/{name}.artifacts/source-and-plan.json")
        )
        require(
            all(
                initial[key] == receipt[key]
                for key in ("source", "script_sha256", "command", "controls")
            ),
            "Attribution initial source/plan differs from final receipt",
        )
    for filename, info in manifest.items():
        if filename.startswith(f"results/{name}.artifacts/") and filename.endswith("/run.json"):
            public = load(imported_binding(parent, manifest, filename))
            for table, digest in public.get("table_sha256", {}).items():
                imported_binding(
                    parent, manifest, str(PurePosixPath(filename).parent / relative(table)), digest
                )


def validate_attempt(collection, name):
    require(re.fullmatch(r"[a-z][a-z0-9-]*", name), "Use an explicit direct study attempt name")
    parent = under(BASE, name)
    plan, inventory, reconstruction = validate_seal(collection, parent)
    manifest_doc = load(parent / "IMPORT_MANIFEST.json")
    manifest = manifest_doc["files"]
    require(isinstance(manifest, dict) and manifest, "Missing exact terminal import inventory")
    observed = {
        str(p.relative_to(parent / "imported"))
        for p in (parent / "imported").rglob("*")
        if p.is_file()
    }
    require(observed == set(manifest), "Imported tree has missing or unmanifested files")
    for filename in manifest:
        imported_binding(parent, manifest, filename)
    accepted = load(imported_binding(parent, manifest, "receipts/accepted.json"))
    terminal = load(imported_binding(parent, manifest, "receipts/terminal.json"))
    require(
        terminal == manifest_doc["terminal"],
        "Import manifest terminal differs from exact terminal bytes",
    )
    require(
        accepted["job_id"] == terminal["job_id"]
        and accepted["job_name"] == plan["job_name"]
        and accepted["plan_sha256"] == terminal["plan_sha256"] == inventory["PREPARED.json"],
        "Accepted job/plan/terminal identities differ",
    )
    scheduler = manifest_doc["scheduler"]
    require(
        f"Job <{accepted['job_id']}>" in scheduler
        and f"Job Name <{plan['job_name']}>" in scheduler
        and "User <yding4>" in scheduler
        and any(f"Status <{s}>" in scheduler for s in ("DONE", "EXIT")),
        "Exact scheduler owner is not terminal",
    )
    startup = None
    if "receipts/startup.json" in manifest:
        startup = load(imported_binding(parent, manifest, "receipts/startup.json"))
        require(
            startup["job_id"] == accepted["job_id"]
            and startup["inventory_sha256"]
            == file_sha(parent / "sealed" / plan["run_id"] / "inventory.json")
            and startup["environment_sha256"] == plan["environment_sha256"]
            and startup["compiler_sha256"] == plan["compiler"]["sha256"],
            "Startup environment/seal mismatch",
        )
        for role, runtime in startup["runtimes"].items():
            require(
                runtime["source"]["source_sha256"] == plan["source_sha256"][role]
                and runtime["source"]["source_files"]
                == plan["source_identities"][role]["source_files"]
                and "L40" in runtime["gpu"],
                "Startup scientific source/device mismatch",
            )
    require(
        not terminal["passed"] or startup is not None, "Passed attempt lacks CUDA startup receipt"
    )
    rows = {row["name"]: row for row in terminal["tasks"]}
    require(
        len(rows) == len(terminal["tasks"]) and set(rows) <= {t["name"] for t in plan["tasks"]},
        "Terminal tasks differ from the planned task inventory",
    )
    tasks = []
    for task in plan["tasks"]:
        task_name, role = task["name"], task["source_role"]
        require(role in ("baseline", "current"), "Invalid scientific source role")
        result_name = f"results/{task_name}.json"
        execution = rows.get(task_name, {})
        summary = dict(
            name=task_name,
            source_role=role,
            script=task["script"],
            planned_arguments=task["args"],
            status="not_attempted",
            source_sha256=plan["source_sha256"][role],
        )
        if task["script"] == "qualify_mixed_cuda.py":
            for argument, key, convert in (
                ("--fixture", "fixture_family", str),
                ("--nodes", "nodes", int),
                ("--mode", "mode", str),
            ):
                require(task["args"].count(argument) == 1, "Ambiguous mixed task declaration")
                summary[key] = convert(task["args"][task["args"].index(argument) + 1])
        command_name = f"receipts/command-{task_name}.json"
        if command_name in manifest:
            command = load(imported_binding(parent, manifest, command_name))
            require(
                command["source_role"] == role
                and command["timeout_seconds"] == task["timeout_seconds"],
                "Executed task resource/source declaration differs",
            )
            arguments = []
            for argument in task["args"]:
                if argument.startswith("@ROOT@/"):
                    arguments.append(plan["remote_root"] + "/" + str(relative(argument[7:])))
                elif argument.startswith("@SHA256@/"):
                    dependency = str(relative(argument[9:]))
                    imported_binding(parent, manifest, dependency)
                    arguments.append(manifest[dependency]["sha256"])
                else:
                    arguments.append(argument)
            expected_command = [
                plan["python"],
                "-B",
                plan["remote_root"] + "/source/benchmarks/" + task["script"],
                "--device",
                "cuda:0",
                "--out",
                plan["remote_root"] + "/" + result_name,
                *arguments,
            ]
            require(
                command["argv"] == expected_command,
                "Executed command differs from exact planned arguments",
            )
            summary.update(status="interrupted_without_task_receipt", command_receipt=command_name)
        if result_name in manifest:
            receipt_path = imported_binding(
                parent, manifest, result_name, execution.get("receipt_sha256")
            )
            receipt = load(receipt_path)
            require(receipt["status"] in ("passed", "failed"), "Nonterminal task receipt")
            require(
                command_name in manifest and receipt.get("command") == command["argv"][2:],
                "Task command identity differs from its worker receipt",
            )
            require(
                receipt.get("source", {}).get("source_sha256") == plan["source_sha256"][role],
                "Task receipt does not bind its exact scientific source",
            )
            if "source_files" in receipt["source"]:
                require(
                    receipt["source"]["source_files"]
                    == plan["source_identities"][role]["source_files"],
                    "Task source-file identity differs",
                )
            script_hash = inventory[f"source/benchmarks/{task['script']}"]
            declared = receipt.get("qualification_script_sha256", receipt.get("script_sha256"))
            if declared is not None:
                require(declared == script_hash, "Task executed a different qualification helper")
            for helper, filename in (
                ("driver", task["script"]),
                ("fixtures", "mixed_fixtures.py"),
                ("common_qualifier", "qualify_cuda.py"),
            ):
                if "helpers" in receipt:
                    require(
                        receipt["helpers"][helper] == inventory[f"source/benchmarks/{filename}"],
                        "Mixed qualification helper identity differs",
                    )
            validate_task_artifacts(parent, manifest, plan, task, receipt)
            require(
                not execution or execution.get("status") == receipt["status"],
                "Worker/task status differs",
            )
            summary.update(
                status=receipt["status"],
                receipt=result_name,
                receipt_sha256=file_sha(receipt_path),
                elapsed_seconds=receipt.get("elapsed_seconds"),
                schema=receipt.get("schema"),
                error_type=receipt.get("error_type"),
                error=receipt.get("error"),
            )
            for key in (
                "fixture_family",
                "nodes",
                "mode",
                "input_sha256",
                "predecessor",
                "baseline",
                "comparison",
                "full_path",
            ):
                if key in receipt:
                    if key in ("fixture_family", "nodes", "mode"):
                        require(
                            receipt[key] == summary[key],
                            "Mixed receipt differs from planned fixture/mode",
                        )
                    summary[key] = receipt[key]
            path_name = f"results/{task_name}.artifacts/full-path.json"
            if task["script"] == "qualify_mixed_cuda.py" and path_name in manifest:
                full_path = load(imported_binding(parent, manifest, path_name))
                path_rows = full_path["path_records"]
                starts = [
                    start
                    for row in path_rows
                    for start in row.get(
                        "starts", row.get("failure_diagnostics", {}).get("starts", [])
                    )
                ]
                summary["candidate_coverage"] = dict(
                    artifact=path_name,
                    artifact_sha256=manifest[path_name]["sha256"],
                    selected_raw_qualified=full_path["raw_qualified"],
                    selected_lambda=full_path["selected_lambda"],
                    planned_penalties=len(path_rows),
                    qualified_raw_winners=sum(
                        row.get("raw_status") == "qualified" for row in path_rows
                    ),
                    qualified_refits=sum(
                        row.get("refit_status") == "qualified" for row in path_rows
                    ),
                    complete_penalties=sum(row.get("search_complete") is True for row in path_rows),
                    recorded_starts=len(starts),
                    qualified_starts=sum(start.get("qualified") is True for start in starts),
                    unresolved_starts=sum(start.get("qualified") is not True for start in starts),
                    search_status=full_path["search_status"],
                    final_export_and_publication_qualified=receipt.get("full_path", {}).get(
                        "final_export_and_publication_qualified", False
                    ),
                    scope="Qualified selected raw/secondary candidates are separate from every-start path coverage and final independent export/publication. Failed coverage cannot inherit success from its candidate winner.",
                )
        require(
            not terminal["passed"]
            or (summary["status"] == "passed" and execution.get("returncode") == 0),
            "Passed terminal conceals an incomplete/failed planned task",
        )
        tasks.append(summary)
    invalidation_path = parent / "MEASUREMENT_INVALIDATION.json"
    invalidation = None
    if invalidation_path.exists():
        invalidation = load(invalidation_path)
        require(
            invalidation["source_sha256"] in (CURRENT_SHA, plan["source_sha256"]),
            "Measurement correction source mismatch",
        )
        require(
            invalidation.get("job_id", accepted["job_id"]) == accepted["job_id"],
            "Measurement correction refers to another job",
        )
        require(
            invalidation["affected_receipts"] and invalidation["invalidated_claims"],
            "Measurement correction lacks its exact affected scope",
        )
        for filename, digest in invalidation["affected_receipts"].items():
            imported_binding(parent, manifest, filename, digest)
    for filename, info in manifest.items():
        require(
            not any(
                part.startswith("compiler-cache") or part == "__pycache__"
                for part in relative(filename).parts
            ),
            "Cache was unexpectedly imported as evidence",
        )
        require(
            not filename.endswith((".trace.json", "trace.json", ".pt", ".so", ".bin")),
            "Profiler trace/binary requires a separate archive and is excluded from compact evidence",
        )
        collection.add(
            under(parent / "imported", filename),
            f"attempts/{name}/imported/{filename}",
            expected=info["sha256"],
        )
    for filename in LOCAL_OPERATIONS:
        path = parent / filename
        if path.exists():
            collection.add(path, f"attempts/{name}/operational/{filename}")
    # The parent transport embeds all imported payloads as JSON strings. Retain
    # its external hash and the exact extracted bytes, not a duplicate payload.
    for filename in ("terminal-evidence-import.json", "observer.log", plan["run_id"] + ".tar.gz"):
        path = parent / filename
        if path.exists():
            collection.excluded.append(
                dict(
                    original=str(path.relative_to(BASE)),
                    sha256=file_sha(path),
                    bytes=path.stat().st_size,
                    reason="Redundant transport, observer output, or reconstructible archive; original retained locally",
                )
            )
    return dict(
        attempt=name,
        run_id=plan["run_id"],
        job_id=accepted["job_id"],
        status="passed" if terminal["passed"] else "failed",
        tasks=tasks,
        claims_status="partially_invalidated" if invalidation else "as_recorded",
        measurement_invalidation=invalidation,
        remote_root=plan["remote_root"],
        plan_sha256=inventory["PREPARED.json"],
        source_reconstruction=reconstruction,
        environment_sha256=plan["environment_sha256"],
        terminal_error_type=terminal.get("error_type"),
        terminal_error=terminal.get("error"),
    )


def dependencies(attempts):
    receipts = {
        task["receipt_sha256"]: (attempt, task)
        for attempt in attempts
        for task in attempt["tasks"]
        if "receipt_sha256" in task
    }
    for attempt in attempts:
        for task in attempt["tasks"]:
            for key in ("baseline", "predecessor"):
                dependency = task.get(key)
                if dependency is None:
                    continue
                digest = dependency["sha256"]
                require(
                    digest in receipts,
                    f"Omitted {key} dependency receipt for {attempt['attempt']}/{task['name']}",
                )
                prior_attempt, prior = receipts[digest]
                require(prior["status"] == "passed", "A dependency did not pass")
                if key == "baseline":
                    require(
                        prior["source_sha256"] == BASELINE_SHA
                        and prior.get("mode") == "reference"
                        and prior.get("fixture_family") == task.get("fixture_family")
                        and prior.get("nodes") == task.get("nodes")
                        and prior.get("input_sha256") == task.get("input_sha256"),
                        "Paired baseline mismatch",
                    )
                else:
                    require(
                        prior["source_sha256"] == task["source_sha256"]
                        and prior.get("fixture_family") == task.get("fixture_family")
                        and prior.get("nodes") == {256: 64, 512: 256}.get(task.get("nodes")),
                        "Larger-fixture predecessor mismatch",
                    )
                task[key + "_published_receipt"] = (
                    f"attempts/{prior_attempt['attempt']}/imported/{prior['receipt']}"
                )


def mixed_coverage(attempts):
    """Archive coverage without turning baseline failure/current-only work into parity."""
    observed = [
        (attempt["attempt"], task)
        for attempt in attempts
        for task in attempt["tasks"]
        if task["script"] == "qualify_mixed_cuda.py"
    ]
    output = []
    for family in ("mixed_support", "below_one"):
        prior_qualified = None
        for nodes in (64, 256, 512):
            matching = [
                (attempt, task)
                for attempt, task in observed
                if task.get("fixture_family") == family and task.get("nodes") == nodes
            ]
            current = [
                (attempt, task) for attempt, task in matching if task["source_role"] == "current"
            ]
            complete = [
                (attempt, task)
                for attempt, task in current
                if task["status"] == "passed"
                and task.get("full_path", {}).get("search_status") == "complete"
                and task.get("full_path", {}).get("final_export_and_publication_qualified") is True
            ]
            if complete:
                state = "qualified"
            elif any(task["status"] == "failed" for _, task in current):
                state = "failed"
            elif any(task["status"] != "not_attempted" for _, task in current):
                state = "incomplete"
            else:
                state = "not_attempted"
            paired = [
                (attempt, task)
                for attempt, task in complete
                if task.get("mode") == "paired"
                and task.get("comparison", {}).get("all_fixed_problems_within_existing_gates")
                is True
            ]
            output.append(
                dict(
                    fixture_family=family,
                    nodes=nodes,
                    current_complete_path=state,
                    representative_baseline_parity=("qualified" if paired else "not_qualified")
                    if nodes == 64
                    else "not_requested_at_this_size",
                    predecessor_current_path_qualified=prior_qualified,
                    attempts=[
                        dict(
                            attempt=attempt,
                            task=task["name"],
                            source_role=task["source_role"],
                            mode=task.get("mode"),
                            status=task["status"],
                            error_type=task.get("error_type"),
                            error=task.get("error"),
                        )
                        for attempt, task in matching
                    ],
                )
            )
            prior_qualified = state == "qualified"
    return output


def atomic_publish(collection, destination):
    require(
        not destination.exists() and not destination.is_symlink(),
        "Existing publication must never be overwritten",
    )
    destination.parent.mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".cuda-qp-v3-build-", dir=destination.parent))
    try:
        hashes = {}
        for name, (source, digest, size) in sorted(collection.files.items()):
            output = under(build, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as target:
                if isinstance(source, bytes):
                    target.write(source)
                else:
                    require(file_sha(source) == digest, f"Evidence changed before copy: {source}")
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, target)
                target.flush()
                os.fsync(target.fileno())
            require(
                output.stat().st_size == size and file_sha(output) == digest,
                f"Copy readback failed: {name}",
            )
            hashes[name] = digest
        inventory = build / "SHA256.json"
        with inventory.open("xb") as stream:
            stream.write(encoded(hashes))
            stream.flush()
            os.fsync(stream.fileno())
        require(load(inventory) == hashes, "Final hash inventory readback failed")
        for directory in sorted(
            [build, *(p for p in build.rglob("*") if p.is_dir())], reverse=True
        ):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(build), -100, os.fsencode(destination), 1)
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        require(
            all(file_sha(under(destination, name)) == digest for name, digest in hashes.items()),
            "Published evidence readback differs",
        )
        return dict(
            destination=str(destination),
            files=len(hashes) + 1,
            bytes=sum(size for _, _, size in collection.files.values())
            + inventory_size(destination),
            inventory_sha256=file_sha(destination / "SHA256.json"),
        )
    finally:
        if build.exists():
            shutil.rmtree(build)


def inventory_size(destination):
    return (destination / "SHA256.json").stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt",
        action="append",
        required=True,
        help="Explicit terminal attempt, repeat for each",
    )
    parser.add_argument("--local-validation", default="LOCAL_VALIDATION.json")
    parser.add_argument("--suite-log", default="whole-suite-final.log")
    parser.add_argument(
        "--large-file-review",
        help="Optional relative JSON mapping original paths to reviewed sha256/bytes/reason",
    )
    parser.add_argument(
        "--study-artifact",
        action="append",
        default=[],
        help="Additional reviewed study-root report/helper/log to copy",
    )
    args = parser.parse_args()
    require(len(args.attempt) == len(set(args.attempt)), "Duplicate attempt argument")
    require(
        not DESTINATION.exists() and not DESTINATION.is_symlink(),
        "Evidence destination already exists",
    )
    reviews = load(under(BASE, args.large_file_review)) if args.large_file_review else {}
    collection = Collection(reviews)
    study = load(BASE / "STUDY_PLAN.json")
    require(
        study["baseline_commit"] == BASELINE_COMMIT
        and study["current_source_sha256"] == CURRENT_SHA,
        "Study authority differs from fixed source identities",
    )
    local = validate_local(collection, args.local_validation, args.suite_log)
    # A known failed imported attempt cannot disappear from the requested archive.
    for manifest in sorted(BASE.glob("*/IMPORT_MANIFEST.json")):
        if (
            load(manifest)["terminal"]["passed"] is False
            or (manifest.parent / "MEASUREMENT_INVALIDATION.json").exists()
        ):
            require(
                manifest.parent.name in args.attempt,
                f"Omitted failed/invalidated study attempt: {manifest.parent.name}",
            )
    attempts = [validate_attempt(collection, name) for name in args.attempt]
    dependencies(attempts)
    for name in (
        "STUDY_PLAN.json",
        "prepare.py",
        "worker.py",
        "launch.py",
        "watch.py",
        "import_evidence.py",
        "publish_evidence.py",
    ):
        collection.add(BASE / name, "study/" + name)
    for name in args.study_artifact:
        collection.add(under(BASE, name), "study/" + str(relative(name)))
    if args.large_file_review:
        collection.add(under(BASE, args.large_file_review), "study/LARGE_FILE_REVIEW.json")
    require(
        set(reviews) == collection.used_reviews,
        "Large-file review has unused or mismatched entries",
    )
    coverage = mixed_coverage(attempts)
    summary = dict(
        schema="clipp1d.cuda.qp.evidence.v1",
        status="evidence_archived",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Exact source-bound synthetic optimization evidence; individual failures remain failures. No cohort accuracy claim.",
        baseline_commit=BASELINE_COMMIT,
        baseline_source_sha256=BASELINE_SHA,
        current_source_sha256=CURRENT_SHA,
        local_validation=local,
        attempts=attempts,
        mixed_coverage=coverage,
        all_requested_current_mixed_paths_qualified=all(
            row["current_complete_path"] == "qualified" for row in coverage
        ),
        representative_mixed_baseline_parity_qualified=all(
            row["representative_baseline_parity"] == "qualified"
            for row in coverage
            if row["nodes"] == 64
        ),
        task_status_counts=dict(Counter(t["status"] for a in attempts for t in a["tasks"])),
        all_included_attempts_passed=all(a["status"] == "passed" for a in attempts),
        all_included_claims_valid=all(
            a["status"] == "passed" and a["claims_status"] == "as_recorded" for a in attempts
        ),
        receipt_semantics="Compiler admission is explicit; later lazy compilation remains charged to its executing phase. Timings with instrumentation are not uninstrumented throughput.",
        excluded_originals=collection.excluded,
        size_policy="No aggregate cap. Individual files above10MB require a separate exact-hash/size review.",
    )
    collection.generated("FINAL_STUDY.json", summary)
    collection.generated("COPY_MAP.json", collection.mapping)
    lines = [
        "# CUDA QP optimization evidence",
        "",
        "This directory preserves exact bytes from the explicitly listed terminal attempts.",
        "`FINAL_STUDY.json` reports each planned task, including failures, interruptions and unattempted work.",
        "`SHA256.json` covers every published file except itself; `COPY_MAP.json` maps originals to copies.",
        "",
        f"Baseline: `{BASELINE_COMMIT}`; source `{BASELINE_SHA}`.",
        f"Current source: `{CURRENT_SHA}`.",
        "",
        "Reproduce from the baseline commit plus each attempt's original `sealed/tracked-dirty.patch`.",
        "The source reconstruction maps additionally retain exact changed production files and helper bytes,",
        "including helpers absent from the tracked patch. Validate against the sealed inventories.",
        "The original baseline and current full source trees/archives are not duplicated here.",
        "",
        "Operational plans, scheduler receipts and all imported numerical receipts/inputs/path outputs are byte copies.",
        "Transport payload wrappers, repeated observer stdout, compiler caches, binaries and profiler traces are excluded.",
        "Original excluded-file hashes remain in FINAL_STUDY.json. No previous validation directory was changed.",
        "",
        "An archived failure is not qualification. CPU tests, same-job QP throughput, instrumented attribution,",
        "complete inference and cohort accuracy are separate scopes. See each exact receipt before using a result.",
        "",
        "| Attempt | Job | Terminal / claims | Planned task results |",
        "| --- | --- | --- | --- |",
    ]
    for attempt in attempts:
        task_states = "; ".join(t["name"] + ": " + t["status"] for t in attempt["tasks"])
        lines.append(
            f"| {attempt['attempt']} | {attempt['job_id']} | {attempt['status']} / {attempt['claims_status']} | {task_states} |"
        )
    lines.extend(
        [
            "",
            "| Mixed fixture | Nodes | Current complete path | Baseline parity |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in coverage:
        lines.append(
            f"| {row['fixture_family']} | {row['nodes']} | {row['current_complete_path']} | {row['representative_baseline_parity']} |"
        )
    collection.generated("README.md", ("\n".join(lines) + "\n").encode())
    print(json.dumps(atomic_publish(collection, DESTINATION), sort_keys=True))


if __name__ == "__main__":
    main()
