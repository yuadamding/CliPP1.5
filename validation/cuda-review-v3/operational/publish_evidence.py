"""Publish a complete, reconciled v3 attempt; never launch or import jobs.

This preparation script must not run before the accepted job's terminal import.
It uses only the standard library and rejects existing publication directories.
Original imported files are copied byte for byte; large Chrome traces stay remote.
"""
from collections import Counter
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from zoneinfo import ZoneInfo


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
# Bound once in main from explicit input paths and reconciled immutable receipts.
ATTEMPT = RUN_ID = JOB_ID = REMOTE = SEALED = IMPORTED = SOURCE_SHA = None
DESTINATION = REPO / "validation/cuda-review-v3"
BASELINE_SHA = "50c4d2a16b361bf34943a91350f664ac0a354c84d89e36c67ff03f59c37e0ee7"
POLICY_ID = "clipp1d_complete_cuda_unconstrained_v3"
MAX_BYTES = 16_000_000
FIXTURES = {"single_support", "mixed_multiplicity", "all_bounds_below_one"}
EXPECTED_COUNTS = {
    "exact_fusion_grouping": 2, "likelihood_parity": 7,
    "qp_attempt": 16, "qp_parity": 8, "qp_warm_attempt": 8, "qp_warm_parity": 4,
    "pipeline_attempt": 6, "pipeline_parity": 3, "independent_graph_comparison": 3,
    "matched_graph_attempt": 24, "matched_graph_parity": 12,
    "public_publication": 1, "resource_qp_attempt": 2, "resource_probe": 1,
    "scaling_started": 3, "scaling_path_complete": 3, "scaling_qualified": 3,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing/nonregular evidence: {path}")
    return path.read_bytes()


def parse(data):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, parse_constant=invalid_constant)


def load(path):
    return parse(regular_bytes(path))


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def relative_path(name):
    value = PurePosixPath(name)
    require(isinstance(name, str) and not value.is_absolute() and bool(value.parts)
            and all(part not in (".", "..") for part in value.parts), f"Unsafe relative path: {name}")
    return value


def under(root, name):
    relative = relative_path(name)
    path = root.joinpath(*relative.parts)
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), f"Symlink in evidence path: {path}")
        cursor = cursor.parent
    return path


def remote_relative(name):
    path = PurePosixPath(name)
    require(path.is_absolute() and path.is_relative_to(REMOTE), f"Wrong remote run: {name}")
    return str(relative_path(str(path.relative_to(REMOTE))))


def source_digest(files):
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


def receipt_from_transport(path):
    transport = load(path)
    require(transport.get("returncode") == 0, f"Failed evidence transport: {path}")
    rows = [line.split("=", 1)[1] for line in transport["stdout"].splitlines()
            if line.startswith("RECEIPT=")]
    require(len(rows) == 1, f"Ambiguous transport receipt: {path}")
    return parse(rows[0])


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_source(local, plan, inventory):
    source = local["source"]
    require(source["source_sha256"] == SOURCE_SHA and source_digest(source["source_files"]) == SOURCE_SHA,
            "Local scientific source fingerprint is not the accepted source")
    current = {str(p.relative_to(REPO)): sha(regular_bytes(p))
               for p in sorted((REPO / "src").rglob("*.py"))}
    for name in ("pyproject.toml", "benchmarks/qualify_cuda.py", "benchmarks/profile_scalar_cuda.py"):
        current[name] = sha(regular_bytes(under(REPO, name)))
    require(current == plan["source_files"], "Current source/config/qualifiers differ from the frozen plan")
    package_files = {name.removeprefix("src/clipp1d/"): value for name, value in current.items()
                     if name.startswith("src/clipp1d/")}
    require(package_files == source["source_files"], "Local validation omits or changes production source")
    for name, expected in inventory.items():
        require(sha(regular_bytes(under(SEALED, name))) == expected, f"Frozen inventory changed: {name}")
    for name, expected in plan["source_files"].items():
        require(inventory.get("source/" + name) == expected, f"Source missing from inventory: {name}")
    test_files = {str(p.relative_to(REPO)): sha(regular_bytes(p))
                  for p in sorted((REPO / "tests").rglob("*.py"))}
    require(test_files == local["test_files"], "Local tests changed after the final local validation receipt")


def qp_gate(row, policy, flag="qualified"):
    require(row.get(flag) is True, "An unqualified QP cannot enter final evidence")
    require(all(finite(row.get(key)) for key in ("gap", "gap_scale", "kkt")), "Nonfinite QP certificate")
    require(0 <= row["gap"] <= policy["inner_atol"] + policy["inner_rtol"] * row["gap_scale"]
            and 0 <= row["kkt"] <= policy["inner_kkt_tol"], "QP certificate exceeds unchanged gates")


def complete_path(rows):
    require(isinstance(rows, list) and bool(rows), "Missing path records")
    require(all(row.get("raw_status") == row.get("refit_status") == "qualified"
                and row.get("search_complete") is True for row in rows), "Unresolved planned path state")


def main():
    global ATTEMPT, RUN_ID, JOB_ID, REMOTE, SEALED, IMPORTED, SOURCE_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True,
                        help="Explicit successful attempt under this review root, e.g. gpu-qualification-i")
    parser.add_argument("--local-validation", required=True,
                        help="Final source-bound local validation JSON, relative to this review root")
    parser.add_argument("--suite-log", required=True,
                        help="Exact final test log bound by that JSON, relative to this review root")
    parser.add_argument("--failed-attempt", action="append",
                        default=["gpu-qualification-f", "gpu-qualification-g", "gpu-qualification-h"],
                        help="Additional unsuccessful attempt to preserve; F, G and stopped H are always retained")
    args = parser.parse_args()
    ATTEMPT = under(BASE, args.attempt)
    require(ATTEMPT.parent == BASE and re.fullmatch(r"gpu-qualification-[a-z]", ATTEMPT.name),
            "Select one explicit review attempt directory")
    local_validation_path = under(BASE, args.local_validation)
    suite_path = under(BASE, args.suite_log)
    require(not DESTINATION.exists() and not DESTINATION.is_symlink(),
            f"Existing evidence must never be overwritten: {DESTINATION}")
    publisher_sha = sha(regular_bytes(__file__))
    local = load(local_validation_path)
    local_plan = load(ATTEMPT / "PREPARED.json")
    RUN_ID = local_plan["run_id"]
    require(re.fullmatch(r"clipp2_clipp1d_cuda_\d{8}[a-z]", RUN_ID)
            and RUN_ID[-1] == ATTEMPT.name[-1], "Attempt/run identity mismatch")
    REMOTE = PurePosixPath("/rsrch8/scratch/bcb/yding4") / RUN_ID
    SEALED = ATTEMPT / "sealed" / RUN_ID
    IMPORTED = ATTEMPT / "imported-diagnostics"
    SOURCE_SHA = local["source"]["source_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", SOURCE_SHA), "Invalid final source fingerprint")
    JOB_ID = str(load(IMPORTED / "receipts/accepted.json")["job_id"])
    require(re.fullmatch(r"\d+", JOB_ID), "Invalid accepted scalar scheduler identity")
    plan_path = SEALED / "PREPARED.json"
    plan = load(plan_path)
    inventory_path = SEALED / "inventory.json"
    inventory = load(inventory_path)
    require(plan["run_id"] == RUN_ID and plan["remote_root"] == str(REMOTE)
            and plan["job_name"] == RUN_ID.removeprefix("clipp2_"), "Incorrect accepted run identity")
    require(all(local_plan.get(key) == value for key, value in plan.items()), "Local/remote prepared plans differ")
    require(local_plan["inventory_sha256"] == sha(regular_bytes(inventory_path)), "Inventory seal mismatch")
    require(local_plan["archive_sha256"] == sha(regular_bytes(ATTEMPT / (RUN_ID + ".tar.gz"))),
            "Frozen source archive seal mismatch")
    validate_source(local, plan, inventory)
    require(local.get("source_matches_gpu_seal") is True, "Local source/seal reconciliation missing")
    require(isinstance(local["tests"]["passed"], int) and local["tests"]["passed"] > 0
            and local["tests"]["failed"] == local["tests"]["skipped"] == 0,
            "The final local suite must contain a positive bound pass count and no failures/skips")
    suite_bytes = regular_bytes(suite_path)
    require(sha(suite_bytes) == local["tests"]["log_sha256"], "Final test log hash mismatch")
    summary = re.findall(rb"(?m)^(\d+) passed(?:, \d+ warnings?)? in ([0-9.]+)s", suite_bytes)
    require(len(summary) == 1 and int(summary[0][0]) == local["tests"]["passed"]
            and float(summary[0][1]) == local["tests"]["seconds"], "Final test summary does not match its receipt")
    require(local["static_checks"] and all(value == "passed" for value in local["static_checks"].values()),
            "Required static checks did not pass")

    transport = receipt_from_transport(ATTEMPT / "terminal-evidence-import.json")
    require(f"Job <{JOB_ID}>" in transport["scheduler"] and "Status <DONE>" in transport["scheduler"],
            "Accepted job did not finish DONE in the imported scheduler snapshot")
    remote_files = transport["files"]
    require(isinstance(remote_files, dict) and bool(remote_files), "Missing imported remote hash inventory")
    for name, entry in remote_files.items():
        require(sha(regular_bytes(under(IMPORTED, name))) == entry["sha256"], f"Import hash mismatch: {name}")

    def remote_file(name, expected=None):
        relative_path(name)
        require(name in remote_files, f"Required compact remote evidence missing: {name}")
        data = regular_bytes(under(IMPORTED, name))
        require(expected is None or sha(data) == expected, f"Bound remote artifact hash mismatch: {name}")
        return data

    def remote_json(name, expected=None):
        return parse(remote_file(name, expected))

    def artifact(name, expected):
        return remote_json(remote_relative(name), expected)

    accepted = remote_json("receipts/accepted.json")
    startup = remote_json("receipts/startup.json")
    terminal = remote_json("receipts/terminal.json")
    launch = receipt_from_transport(ATTEMPT / "launch.json")
    for receipt in (accepted, startup, terminal, launch):
        require(str(receipt["job_id"]) == JOB_ID, "Receipt belongs to another scheduler job")
    plan_sha = sha(regular_bytes(plan_path))
    require(accepted["plan_sha256"] == terminal["plan_sha256"] == launch["plan_sha256"] == plan_sha,
            "Accepted/startup/terminal plan binding mismatch")
    require(launch.get("released_acknowledged") is True and accepted["job_name"] == plan["job_name"],
            "Launch acceptance/release identity mismatch")
    require(startup["inventory_sha256"] == local_plan["inventory_sha256"]
            and startup["environment_sha256"] == plan["environment_sha256"]
            and startup["compiler_sha256"] == plan["compiler"]["sha256"], "Runtime environment/compiler/seal mismatch")
    require(terminal.get("passed") is True and terminal.get("returncode") == 0, "Accepted attempt terminal is not passed")
    qualification = remote_json("results/qualification.json", terminal["qualification_sha256"])
    require(qualification.get("status") == "passed" and qualification["schema"] == "clipp1d.cuda.qualification.v3",
            "Policy-v3 GPU qualification did not pass")
    require(qualification["source"]["source_sha256"] == SOURCE_SHA
            and qualification["source"]["source_files"] == local["source"]["source_files"], "GPU/source mismatch")
    require(str(qualification["lsf_job_id"]) == JOB_ID and qualification["path_grid"] == "default"
            and qualification["cuda_available"] is True and "L40" in qualification["gpu"],
            "Wrong job, device, or reduced qualification path")
    require(qualification["qualification_script_sha256"] == plan["source_files"]["benchmarks/qualify_cuda.py"],
            "GPU qualifier differs from sealed current qualifier")
    require(qualification["interpreter"] == plan["python"] and qualification["cc_resolved"] == plan["compiler"]["path"]
            and qualification["torch"] == startup["torch"] and qualification["cuda_runtime"] == startup["cuda"],
            "GPU runtime identity changed after startup")
    policy = qualification["policy"]
    require(policy["policy_id"] == POLICY_ID and policy["inner_max_iterations"] == 20000
            and (policy["inner_atol"], policy["inner_rtol"], policy["inner_kkt_tol"]) == (1e-10, 1e-11, 1e-7)
            and (policy["scalar_atol"], policy["scalar_rtol"], policy["stationarity_tol"], policy["fusion_tol"])
            == (1e-7, 1e-10, 2e-5, 2e-5), "Policy or scientific admission gates differ")
    events = remote_file("results/qualification.events.jsonl", qualification["events_sha256"])
    require([parse(line) for line in events.splitlines()] == qualification["cases"], "Qualifier events/receipt differ")
    remote_file("results/qualification.artifacts/source-and-plan.json", qualification["source_and_plan_sha256"])
    cases = qualification["cases"]
    counts = Counter(row["kind"] for row in cases)
    require(not any("failure" in kind or "mismatch" in kind for kind in counts), "Failed GPU stage retained in passed receipt")
    for kind, expected in EXPECTED_COUNTS.items():
        require(counts[kind] == expected, f"Expected {expected} {kind} events; got {counts[kind]}")

    def rows(kind):
        return [row for row in cases if row["kind"] == kind]

    compiler_admission, = rows("compiler_admission")
    require(compiler_admission["compiled"] is True and finite(compiler_admission["seconds"])
            and compiler_admission["seconds"] >= 0, "Initial compiler admission is missing or invalid")
    require({r["fixture"] for r in rows("pipeline_parity")} == FIXTURES, "Missing small-path fixture")
    for row in rows("exact_fusion_grouping"):
        require(row.get("qualified") is True and row["labels"] == row["expected"], "Exact fusions fragmented")
    for row in rows("qp_attempt") + rows("qp_warm_attempt"):
        qp_gate(row, policy)
    for row in rows("qp_warm_parity"):
        require(row["literal_problem_identical"] is True, "Warm/cold QPs differ in statistical problem")
    for kind in ("pipeline_attempt", "matched_graph_attempt"):
        for row in rows(kind):
            require(row["raw_qualified"] is True and row["search_status"] == "complete", "Unresolved small/fixed path")
            detail = artifact(row["detail_file"], row["detail_sha256"])
            require(detail["raw_qualified"] is True and detail["search_status"] == "complete"
                    and detail["clonal_constraint"] is False, "Invalid detailed raw qualification")
            if kind == "pipeline_attempt":
                complete_path(detail["path_records"])
    for kind in ("pipeline_parity", "matched_graph_parity"):
        for row in rows(kind):
            require(row["labels_equal"] is True and row["raw_and_refit_multiplicities_equal"] is True,
                    "Labels/multiplicity parity failed")
            require(all(finite(value) and value >= 0 for value in row["errors"].values()), "Invalid parity errors")
            if kind == "matched_graph_parity":
                require(row["identical_graph_and_lambda"] is True and row["errors"]["literal_lambda"] == 0,
                        "Fixed-graph comparison changed literal lambda")
    require(any(r["lambda_value"] > 0 and r["fusion_penalty"] > 0 for r in rows("matched_graph_parity")),
            "Fixed-problem parity omits nonfused positive-penalty work")

    public, = rows("public_publication")
    public_rel = remote_relative(public["directory"])
    public_receipt = remote_json(public_rel + "/run.json", public["run_sha256"])
    require(public_receipt["status"] == "success" and public_receipt["schema"] == "clipp1d.cuda.run.v3"
            and public_receipt["search_status"] == "complete", "Public output did not complete")
    require(public_receipt["provenance"]["source_sha256"] == SOURCE_SHA
            and public_receipt["provenance"]["clonal_constraint"] is False
            and public_receipt["raw_witness_mutation_id"] is None, "Public source/clonal contract mismatch")
    complete_path(public_receipt["search"])
    require(public_receipt["table_sha256"] == public["table_sha256"], "Public table manifests differ")
    for name, expected in public["table_sha256"].items():
        remote_file(public_rel + "/" + name, expected)
    public_input = str(PurePosixPath(public_rel).parent / "public-input.tsv")
    # The exact qualifier artifact filename is bound by the imported input hash.
    matching_inputs = [name for name, entry in remote_files.items()
                       if name.startswith("results/qualification.artifacts/") and entry["sha256"] == public["input_sha256"]]
    require(len(matching_inputs) == 1, f"Missing unique public input near {public_input}")

    resource, = rows("resource_probe")
    require(resource["status"] == "qualified" and resource["nodes"] == 256
            and resource["bounded_qp_iteration_budget"] == 20000 and len(resource["repeats"]) == 2,
            "Resource probe is not the required production-budget qualified N256 QP")
    for row in resource["repeats"]:
        qp_gate(row, policy, "qp_qualified")
    scaling = {}
    require({r["nodes"] for r in rows("scaling_qualified")} == {16, 64, 256}, "Missing increasing-size complete fits")
    for row in rows("scaling_qualified"):
        n = row["nodes"]
        detail = artifact(row["artifact"], row["artifact_sha256"])
        path = remote_json(f"results/qualification.artifacts/scaling-{n}-path.json")
        path_event, = [r for r in rows("scaling_path_complete") if r["nodes"] == n]
        remote_file(remote_relative(path_event["artifact"]), path_event["artifact_sha256"])
        require(row["qualified"] is True and detail["qualified"] is True
                and detail["search_status"] == "complete"
                and detail["final_device_qualification_and_export_complete"] is True,
                f"N={n} scaling fit is not fully qualified")
        require(detail["nodes"] == n and detail["provenance"]["source_sha256"] == SOURCE_SHA,
                "Scaling artifact/source mismatch")
        complete_path(path["path_records"])
        for phase in ("pilot_seconds", "graph_build_seconds", "path_seconds", "refit_seconds",
                      "final_device_qualification_seconds", "device_export_seconds"):
            require(finite(detail["phase_seconds"].get(phase)) and detail["phase_seconds"][phase] >= 0,
                    f"Missing or invalid scaling phase {phase}")
        # The qualifier preserves integrity timing in the numerical-stage
        # receipt; it does not duplicate it in the exported phase dictionary.
        integrity_seconds = detail["numerical_stages"].get("stage_integrity_seconds")
        require(finite(integrity_seconds) and integrity_seconds >= 0,
                "Missing or invalid scaling numerical stage stage_integrity_seconds")
        scaling[str(n)] = {key: detail[key] for key in (
            "nodes", "qualified", "search_status", "selected_lambda", "raw_objective", "score",
            "seconds", "phase_seconds", "numerical_stages", "model_integrity_counters", "graph_integrity_counters",
            "scalar_work_counters", "peak_allocated_bytes", "peak_reserved_bytes",
            "final_device_qualification_and_export_complete", "scope")}

    comparison = remote_json("results/scalar-profile-comparison.json", terminal["scalar_profile_comparison_sha256"])
    require(comparison["status"] == "passed" and set(comparison["cases"]) == {"analytical32", "mixed6"},
            "Matched scalar comparison did not pass")
    profiles, traces = {}, []
    for label, expected_source in (("baseline", BASELINE_SHA), ("current", SOURCE_SHA)):
        receipt_rel = f"results/scalar-profile-{label}.json"
        profile = remote_json(receipt_rel)
        require(profile["status"] == "passed" and profile["source"]["source_sha256"] == expected_source
                and str(profile["lsf_job_id"]) == JOB_ID
                and profile["script_sha256"] == plan["source_files"]["benchmarks/profile_scalar_cuda.py"],
                "Profiler source/job/shared-script mismatch")
        pevents = remote_file(f"results/scalar-profile-{label}.events.jsonl", profile["events_sha256"])
        require([parse(line) for line in pevents.splitlines()] == profile["cases"], "Profiler event/receipt mismatch")
        qualified = {r["name"]: r for r in profile["cases"] if r["kind"] == "profile_qualified"}
        require(set(qualified) == {"analytical32", "mixed6"}, "Profiler fixture coverage incomplete")
        fixtures = {r["name"]: r["fixture"] for r in profile["cases"] if r["kind"] == "fixture_started"}
        for name, row in qualified.items():
            require(all(row["result"]["qualified"]), "Profiler contains unqualified scalar lanes")
            counts_json = artifact(row["counts_file"], row["counts_sha256"])
            observed = counts_json["aten::_local_scalar_dense"]["count"]
            require(observed == row["scalar_read_proxy_counts"]["aten::_local_scalar_dense"], "Profiler counts differ")
            require(observed == comparison["cases"][name][label + "_scalar_reads"], "Profile comparison count mismatch")
            remote_relative(row["trace"])
            require(re.fullmatch(r"[0-9a-f]{64}", row["trace_sha256"]), "Missing remote trace hash")
            traces.append(dict(remote_path=row["trace"], sha256=row["trace_sha256"],
                               profile_receipt=receipt_rel, fixture=name,
                               scope="Recorded remote Chrome trace; not copied or re-read by this publisher"))
        profiles[label] = dict(receipt=profile, qualified=qualified, fixtures=fixtures)
    for name in ("analytical32", "mixed6"):
        row = comparison["cases"][name]
        baseline_reads, current_reads = row["baseline_scalar_reads"], row["current_scalar_reads"]
        require(type(baseline_reads) is int and type(current_reads) is int
                and baseline_reads > current_reads >= 0,
                "Matched scalar-read proxy counts did not decrease")
        require(profiles["baseline"]["fixtures"][name] == profiles["current"]["fixtures"][name],
                "Profiler inputs are not matched")
        a = profiles["baseline"]["qualified"][name]["result"]
        b = profiles["current"]["qualified"][name]["result"]
        require(len(a["loss"]) == len(b["loss"]), "Profiler result dimensions differ")
        for x, y, gx, gy in zip(a["loss"], b["loss"], a["gap"], b["gap"]):
            require(all(finite(value) for value in (x, y, gx, gy)) and gx >= 0 and gy >= 0,
                    "Nonfinite profiler scalar comparison")
            margin = gx + gy + 128 * 2.220446049250313e-16 * (1 + abs(x) + abs(y))
            require(abs(x - y) <= margin, "Matched scalar losses differ beyond their qualified gaps")

    copies = {}
    origins = {}

    def include(destination, path, origin, expected=None):
        relative_path(destination)
        require(destination not in copies, f"Duplicate publication path: {destination}")
        data = regular_bytes(path)
        require(expected is None or sha(data) == expected, f"Evidence changed after validation: {path}")
        copies[destination] = data
        origins[destination] = dict(original=origin, sha256=sha(data), bytes=len(data))

    def include_source_overlay(prefix, sealed, prepared, sealed_inventory):
        require(prepared["source_commit"] == prepared["baseline_commit"],
                "Source patch and baseline inventory use different base commits")
        patch_path = sealed / "tracked-dirty.patch"
        patch = regular_bytes(patch_path)
        patch_sha = sha(patch)
        require(patch_sha == prepared["tracked_patch_sha256"]
                == sealed_inventory["tracked-dirty.patch"], "Tracked source patch seal mismatch")
        include(prefix + "tracked-dirty.patch", patch_path, str(patch_path.relative_to(REPO)), patch_sha)
        patched_paths = {line.decode() for line in re.findall(rb"(?m)^\+\+\+ b/([^\r\n]+)$", patch)}
        base_files = prepared["baseline_source_files"]
        additions = {}
        for name, expected in sorted(prepared["source_files"].items()):
            require(sealed_inventory.get("source/" + name) == expected,
                    f"Source plan differs from its sealed inventory: {name}")
            # Preserve complete source files absent from both the baseline
            # inventory and the tracked patch, notably the new profiler. An
            # unchanged baseline file does not need a redundant full copy.
            if base_files.get(name) == expected or name in patched_paths:
                continue
            relative_path(name)
            path = under(sealed, "source/" + name)
            target = prefix + "source/" + name
            include(target, path, str(path.relative_to(REPO)), expected)
            additions[name] = dict(path=target, sha256=expected)
        reconstruction = dict(
            base_commit=prepared["source_commit"], tracked_patch=prefix + "tracked-dirty.patch",
            tracked_patch_sha256=patch_sha, additional_source_files=additions,
            expected_source_files=prepared["source_files"],
            scope="Apply the exact tracked patch to the base commit, then overlay these additional source files; verify every expected source hash. Untracked tests and source archives are not duplicated.",
        )
        copies[prefix + "source-reconstruction.json"] = json_bytes(reconstruction)
        return reconstruction

    for name, entry in sorted(remote_files.items()):
        value = relative_path(name)
        if name == "results/qualification.json":
            target = "qualification.json"
        elif name == "results/qualification.events.jsonl":
            target = "qualification.events.jsonl"
        elif name.startswith("results/qualification.artifacts/"):
            target = "artifacts/" + name.removeprefix("results/qualification.artifacts/")
        elif name.startswith("results/scalar-profile-"):
            require(not name.endswith(".trace.json"), "Large profiler traces must remain remote")
            target = "profiles/" + name.removeprefix("results/")
        elif value.parts[0] == "receipts" and value.suffix == ".json":
            target = "operational/" + str(value.relative_to("receipts"))
        elif value.parts[0] == "case-logs" and value.suffix == ".err":
            target = "logs/" + value.name
        else:
            raise RuntimeError(f"Unmapped imported evidence must be reviewed, not omitted: {name}")
        include(target, under(IMPORTED, name), str(REMOTE / name), entry["sha256"])
    for target, path in {
        "LOCAL_VALIDATION.json": local_validation_path,
        "whole-suite.log": suite_path,
        "BASELINE.json": BASE / "BASELINE.json",
        "operational/PREPARED.json": plan_path,
        "operational/LOCAL_PREPARED.json": ATTEMPT / "PREPARED.json",
        "operational/inventory.json": inventory_path,
        "operational/worker.py": SEALED / "worker.py",
        "operational/launch.py": SEALED / "launch.py",
        "operational/publish_evidence.py": Path(__file__),
    }.items():
        include(target, path, str(path.relative_to(REPO)))
    source_reconstruction = include_source_overlay("operational/", SEALED, plan, inventory)

    cpu_inventory_path = BASE / "n256-cpu-reference-scaled-audit-evidence.json"
    cpu_inventory = load(cpu_inventory_path)
    require(cpu_inventory["directory"] == "n256-cpu-reference-scaled-audit"
            and cpu_inventory["source_sha256"] == SOURCE_SHA and cpu_inventory["status"] == "complete",
            "Post-fix CPU reference is not bound to the successful source")
    cpu_root = BASE / cpu_inventory["directory"]
    cpu_names = {"driver.py", "source-and-plan.json", "terminal.json", "events.jsonl", "path.json",
                 "AUDIT_SCALING_DERIVATION.md"}
    require(set(cpu_inventory["files"]) == cpu_names, "CPU reference compact file inventory differs")
    cpu_prefix = "cpu-reference/scaled-audit/"
    for name, identity in cpu_inventory["files"].items():
        path = under(cpu_root, name)
        require(len(regular_bytes(path)) == identity["bytes"], "CPU reference artifact size changed")
        include(cpu_prefix + name, path, str(path.relative_to(REPO)), identity["sha256"])
    require(cpu_inventory["files"]["path.json"]["bytes"] <= 1_000_000,
            "CPU reference path exceeded the reviewed compact-artifact size")
    include(cpu_prefix + "EVIDENCE_SHA256.json", cpu_inventory_path, str(cpu_inventory_path.relative_to(REPO)))
    cpu_plan = parse(copies[cpu_prefix + "source-and-plan.json"])
    cpu_terminal = parse(copies[cpu_prefix + "terminal.json"])
    cpu_path = parse(copies[cpu_prefix + "path.json"])
    cpu_events = [parse(line) for line in copies[cpu_prefix + "events.jsonl"].splitlines()]
    require(cpu_plan["source"]["source_sha256"] == SOURCE_SHA
            and cpu_plan["source"]["source_files"] == local["source"]["source_files"]
            and cpu_plan["driver_sha256"] == cpu_inventory["files"]["driver.py"]["sha256"]
            and cpu_plan["fixture"]["arrays"]["alt"]["shape"] == [256],
            "CPU reference driver/source/fixture identity differs")
    require(cpu_terminal["source_unchanged"] is True
            and cpu_terminal["status"] == cpu_terminal["search_status"] == "complete"
            and cpu_terminal["elapsed_seconds"] == cpu_inventory["elapsed_seconds"],
            "CPU reference terminal is not complete under its unchanged source")
    complete_path(cpu_path)
    cpu_finished = [row for row in cpu_events if row["kind"] == "lambda_finished"]
    require(len(cpu_path) == cpu_inventory["completed_penalties"] == len(cpu_finished) == 26
            and all(row["qualified"] is True and row["complete"] is True for row in cpu_finished)
            and all(start["qualified"] is True for row in cpu_path for start in row.get("starts", [])),
            "CPU reference complete-path/starting-state evidence differs")
    for index, (path_row, event_row) in enumerate(zip(cpu_path, cpu_finished)):
        require(event_row["index"] == index and event_row["lambda_value"] == path_row["lambda_value"],
                "CPU reference path/events do not represent the same penalties")
    cpu_reference = dict(
        directory=cpu_prefix.rstrip("/"), source_sha256=SOURCE_SHA, status="complete",
        nodes=256, completed_penalties=len(cpu_path), elapsed_seconds=cpu_terminal["elapsed_seconds"],
        driver_sha256=cpu_plan["driver_sha256"], model_sha256=cpu_plan["fixture"]["model_sha256"],
        inventory_sha256=sha(regular_bytes(cpu_inventory_path)),
        high_lambda_event=cpu_finished[23],
        scope="Bounded CPU reference diagnosis only; this is separate from allocated-CUDA acceptance and provides no GPU timing or speedup claim",
    )

    failed_attempts = []
    for failed_name in sorted(set(args.failed_attempt)):
        failed_dir = under(BASE, failed_name)
        require(failed_dir.parent == BASE and re.fullmatch(r"gpu-qualification-[a-z]", failed_name)
                and failed_dir != ATTEMPT, "Invalid failed-attempt evidence selection")
        failed_local_plan = load(failed_dir / "PREPARED.json")
        failed_run = failed_local_plan["run_id"]
        failed_root = PurePosixPath("/rsrch8/scratch/bcb/yding4") / failed_run
        require(re.fullmatch(r"clipp2_clipp1d_cuda_\d{8}[a-z]", failed_run)
                and failed_run[-1] == failed_name[-1]
                and failed_local_plan["remote_root"] == str(failed_root), "Failed run/attempt mismatch")
        failed_seal = failed_dir / "sealed" / failed_run
        failed_plan_bytes = regular_bytes(failed_seal / "PREPARED.json")
        failed_plan = parse(failed_plan_bytes)
        require(all(failed_local_plan.get(k) == v for k, v in failed_plan.items()), "Failed attempt plans differ")
        failed_inventory = load(failed_seal / "inventory.json")
        require(sha(regular_bytes(failed_seal / "inventory.json")) == failed_local_plan["inventory_sha256"],
                "Failed attempt inventory hash mismatch")
        for name, expected in failed_inventory.items():
            require(sha(regular_bytes(under(failed_seal, name))) == expected, "Failed attempt seal changed")
        failed_import = failed_dir / "imported-diagnostics"
        failed_transport = receipt_from_transport(failed_dir / "terminal-evidence-import.json")
        failed_files = failed_transport["files"]
        for name, entry in failed_files.items():
            require(sha(regular_bytes(under(failed_import, name))) == entry["sha256"], "Failed attempt import mismatch")
        fa = load(failed_import / "receipts/accepted.json")
        ft = load(failed_import / "receipts/terminal.json")
        failed_job = str(fa["job_id"])
        require(str(ft["job_id"]) == failed_job and ft["passed"] is False
                and ft["plan_sha256"] == fa["plan_sha256"] == sha(failed_plan_bytes), "Invalid failed terminal identity")
        require(f"Job <{failed_job}>" in failed_transport["scheduler"]
                and "Status <EXIT>" in failed_transport["scheduler"], "Failed attempt has no bound terminal scheduler state")
        originals = {"gpu-qualification-f": "77333999", "gpu-qualification-g": "77334114",
                     "gpu-qualification-h": "77334195"}
        if failed_name in originals:
            original_job = originals[failed_name]
            require(failed_job == original_job, "Original failed attempt changed its accepted job")
        failed_q_path = failed_import / "results/qualification.json"
        failed_q = load(failed_q_path) if failed_q_path.exists() else None
        if failed_name in originals:
            require(failed_q is not None, "The original F/G/H unsuccessful receipts are required")
        failed_source = source_digest({name.removeprefix("src/clipp1d/"): value
                                       for name, value in failed_plan["source_files"].items()
                                       if name.startswith("src/clipp1d/")})
        if failed_q is not None:
            require(failed_q["status"] == "failed" and str(failed_q["lsf_job_id"]) == failed_job
                    and failed_q["source"]["source_sha256"] == failed_source,
                    "Failed qualifier receipt changed status/source/job")
            require(sha(regular_bytes(failed_import / "results/qualification.events.jsonl"))
                    == failed_q["events_sha256"], "Failed qualifier event hash mismatch")
        prefix = "failed-attempts/" + failed_name[-1] + "/"
        for name, entry in sorted(failed_files.items()):
            require(not name.endswith(".trace.json"), "Large failed-attempt traces must remain remote")
            include(prefix + name, under(failed_import, name), str(failed_root / name), entry["sha256"])
        for label in ("baseline", "current"):
            profile_name = f"results/scalar-profile-{label}.json"
            if profile_name in failed_files:
                old_profile = load(under(failed_import, profile_name))
                for row in old_profile["cases"]:
                    if row["kind"] != "profile_qualified":
                        continue
                    require(PurePosixPath(row["trace"]).is_relative_to(failed_root)
                            and re.fullmatch(r"[0-9a-f]{64}", row["trace_sha256"]),
                            "Failed-attempt trace identity is malformed")
                    traces.append(dict(remote_path=row["trace"], sha256=row["trace_sha256"],
                                       profile_receipt=prefix + profile_name, fixture=row["name"],
                                       scope="Recorded remote trace from preserved failed attempt; not copied or re-read"))
        for name in ("PREPARED.json", "inventory.json", "worker.py", "launch.py"):
            include(prefix + "sealed/" + name, failed_seal / name,
                    str((failed_seal / name).relative_to(REPO)))
        failed_reconstruction = include_source_overlay(
            prefix + "sealed/", failed_seal, failed_plan, failed_inventory)
        include(prefix + "LOCAL_PREPARED.json", failed_dir / "PREPARED.json",
                str((failed_dir / "PREPARED.json").relative_to(REPO)))
        failure_summary_path = failed_dir / "FAILURE_SUMMARY.json"
        summary = None
        if failed_name in originals:
            require(failure_summary_path.is_file(), "The reviewed F/G/H failure summaries are required")
        if failure_summary_path.exists():
            summary = load(failure_summary_path)
            require(summary["status"] == "failed" and str(summary["job_id"]) == failed_job
                    and summary["source_sha256"] == failed_source
                    and summary["qualification_sha256"] == sha(regular_bytes(failed_q_path)),
                    "Failed-attempt summary is not bound to its original receipt")
            for name, identity in summary.get("evidence", {}).items():
                require(name in failed_files and identity["sha256"] == failed_files[name]["sha256"]
                        and identity["bytes"] == len(regular_bytes(under(failed_import, name))),
                        "Failed-attempt summary artifact binding differs from its exact import")
            local_evidence = summary.get("local_evidence", {})
            for name, identity in local_evidence.items():
                parts = relative_path(name).parts
                require(len(parts) == 2 and re.fullmatch(r"n256-cpu-reference-\d{8}T\d{6}Z", parts[0])
                        and parts[1] in ("source-and-plan.json", "terminal.json", "events.jsonl")
                        and identity["published_path"] == "cpu-reference/" + parts[1],
                        "Unexpected local CPU diagnostic evidence scope")
                path = under(BASE, name)
                require(len(regular_bytes(path)) == identity["bytes"], "CPU diagnostic evidence size changed")
                include(prefix + identity["published_path"], path, str(path.relative_to(REPO)), identity["sha256"])
            if failed_name == "gpu-qualification-h":
                require(summary.get("disposition") == "explicitly_stopped_after_confirmed_diagnostic_failure"
                        and failed_q["error_type"] == "KeyboardInterrupt", "H stop was relabeled as a natural fit failure")
                intent = load(failed_import / "receipts/cancellation-intent.json")
                result = load(failed_import / "receipts/cancellation-result.json")
                require(str(intent["job_id"]) == str(result["job_id"]) == failed_job
                        and result["returncode"] == 0 and intent["plan_sha256"] == sha(failed_plan_bytes),
                        "H cancellation is not bound to the accepted attempt")
                cpu_terminal = copies[prefix + "cpu-reference/terminal.json"]
                cpu_plan = parse(copies[prefix + "cpu-reference/source-and-plan.json"])
                require(sha(cpu_terminal) == intent["diagnostic_sha256"]
                        and cpu_plan["source"]["source_sha256"] == failed_source
                        and parse(cpu_terminal)["source_unchanged"] is True,
                        "H cancellation diagnostic is not bound to its exact CPU source/trace")
            include(prefix + "FAILURE_SUMMARY.json", failure_summary_path,
                    str(failure_summary_path.relative_to(REPO)))
        failed_attempts.append(dict(
            attempt=failed_name, job_id=failed_job, run_id=failed_run, status="failed",
            source_sha256=failed_source, plan_sha256=sha(failed_plan_bytes),
            source_reconstruction=failed_reconstruction,
            failure_summary_path=None if summary is None else prefix + "FAILURE_SUMMARY.json",
            failure_summary=summary,
            disposition=None if summary is None else summary.get("disposition", "natural_failure"),
            qualification_path=None if failed_q is None else prefix + "results/qualification.json",
            terminal_path=prefix + "receipts/terminal.json",
            error_type=ft.get("error_type"), error=ft.get("error"),
            qualification_error=None if failed_q is None else failed_q.get("error"),
            resource_attempts=[] if failed_q is None else [r for r in failed_q["cases"]
                                                          if r["kind"] == "resource_qp_attempt"],
            scope="Preserved unsuccessful source-bound attempt; not acceptance evidence"))

    copies["REMOTE_TRACES.json"] = json_bytes(traces)
    copies["EVIDENCE_MAP.json"] = json_bytes(origins)
    import_manifest = {name: dict(sha256=row["sha256"], bytes=len(regular_bytes(under(IMPORTED, name))))
                       for name, row in sorted(remote_files.items())}
    copies["operational/IMPORT_MANIFEST.json"] = json_bytes(dict(
        remote_root=str(REMOTE), job_id=JOB_ID, files=import_manifest,
        transport_receipt_sha256=sha(regular_bytes(ATTEMPT / "terminal-evidence-import.json")),
        scope="Derived compact hash inventory; original imported remote bytes are copied unchanged"))
    now = datetime.now(timezone.utc)
    final = dict(
        schema="clipp1d.cuda.review.final_validation.v1", status="passed", created_utc=now.isoformat(),
        job_id=JOB_ID, run_id=RUN_ID, source=local["source"], policy=policy,
        source_reconstruction=source_reconstruction,
        tests=local["tests"], static_checks=local["static_checks"],
        execution={key: qualification[key] for key in (
            "gpu", "capability", "torch", "cuda_runtime", "interpreter", "cc_resolved",
            "lsf_job_id", "lsf_queue", "started_utc", "finished_utc", "elapsed_seconds", "scope")},
        gpu_case_counts=dict(sorted(counts.items())),
        small_paths=rows("pipeline_attempt"), adaptive_graph_parity=rows("pipeline_parity"),
        adaptive_graph_differences=rows("independent_graph_comparison"),
        fixed_problem_parity=rows("matched_graph_parity"), warm_qp_parity=rows("qp_warm_parity"),
        public_publication=public, scaling=scaling, resource_qp=resource, failed_attempts=failed_attempts,
        cpu_reference=cpu_reference,
        initial_compiler_admission=compiler_admission,
        compilation_timing_scope="Initial compiler-admission measurement only; subsequent lazy Torch specializations remain charged to their executing phases. Fit totals are not compilation-separated.",
        scalar_profile_comparison=comparison,
        profiler_scope="CPU host-dispatch scalar-read counts during actual CUDA scalar inference; synchronization proxies, not GPU synchronization duration or end-to-end speedup",
        seals=dict(production_source_sha256=SOURCE_SHA, baseline_source_sha256=BASELINE_SHA,
                   archive_sha256=local_plan["archive_sha256"], plan_sha256=plan_sha,
                   inventory_sha256=local_plan["inventory_sha256"],
                   qualification_sha256=terminal["qualification_sha256"],
                   qualification_script_sha256=qualification["qualification_script_sha256"],
                   profiler_script_sha256=plan["source_files"]["benchmarks/profile_scalar_cuda.py"],
                   publisher_sha256=publisher_sha, terminal_sha256=sha(remote_file("receipts/terminal.json")),
                   local_validation_sha256=sha(regular_bytes(local_validation_path))),
        limits=["Synthetic fixtures only; no cohort accuracy or cohort reproduction claim",
                "Dense complete-graph memory remains quadratic", "No GPU speedup claim",
                "No global optimality proof for the nonconvex likelihood",
                "CPU reference test totals are not CUDA test totals",
                "Return-only durable-publication metrics differ from self-excluding receipt elapsed time"],
    )
    copies["FINAL_VALIDATION.json"] = json_bytes(final)
    readme = f"""# CUDA policy-v3 qualification evidence

Published {now.astimezone(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M %Z')} from accepted
Seadragon scalar LSF job **{JOB_ID}**, run `{RUN_ID}`.
Production source SHA-256: `{SOURCE_SHA}`.

- [Final reconciliation](FINAL_VALIDATION.json): {local['tests']['passed']} local CPU-reference tests,
  strict allocated-CUDA case coverage, complete synthetic N16/64/256 paths,
  qualified production-budget QPs, phase timings, memory and reuse counters.
- [Actual GPU receipt](qualification.json) and [events](qualification.events.jsonl).
- [Local validation](LOCAL_VALIDATION.json) and [test log](whole-suite.log).
- [Matched scalar comparison](profiles/scalar-profile-comparison.json).
- [Operational plan](operational/PREPARED.json), [sealed inventory](operational/inventory.json),
  [worker](operational/worker.py), [accepted](operational/accepted.json),
  [startup](operational/startup.json) and [terminal](operational/terminal.json).
- [Exact original-to-published map](EVIDENCE_MAP.json) and [all-file hashes](SHA256.json).
- [Original F failure](failed-attempts/f/results/qualification.json) and
  [F terminal](failed-attempts/f/receipts/terminal.json) remain unsuccessful evidence;
  later qualification does not relabel that failed resource QP.
- [Original G failure](failed-attempts/g/FAILURE_SUMMARY.json) retains its incomplete
  N64 path despite a qualified selected raw candidate and earlier passed checks.
- [Stopped H](failed-attempts/h/FAILURE_SUMMARY.json) retains cancellation intent/result,
  completed N16/N64 fits and its source-matched [CPU-only trace](failed-attempts/h/cpu-reference/terminal.json).
  H has no completed N256 fit and does not qualify the subsequent audit-scale correction.
- The [post-fix CPU reference](cpu-reference/scaled-audit/EVIDENCE_SHA256.json) preserves its saved
  driver, source/plan, terminal, events, complete bounded path and [arithmetic derivation](cpu-reference/scaled-audit/AUDIT_SCALING_DERIVATION.md).
  Its source matches this accepted revision; its results are CPU diagnosis, not allocated-CUDA acceptance.

Original remote receipt/artifact bytes are unchanged. Remote `results/qualification.artifacts/`
maps to `artifacts/`; `results/scalar-profile-*` maps to `profiles/`;
`receipts/` maps to `operational/`; selected stderr logs map to `logs/`.
Absolute paths inside original receipts retain their original meaning.
`operational/LOCAL_PREPARED.json` is the controller's extended archive receipt;
`operational/PREPARED.json` is the exact plan hashed by the worker.
The [source reconstruction map](operational/source-reconstruction.json) binds the
baseline commit, exact sealed tracked patch and additional source files omitted
from that patch. Each preserved failed attempt includes the same reconstruction
material under its `sealed/` directory. Expected source-file hashes and archive
inventories remain available; full archives and untracked tests are not duplicated.

Large Chrome traces remain under the original remote run. [Their exact paths and
recorded hashes](REMOTE_TRACES.json) are retained; the publisher neither downloaded
nor re-read them. Compact counts files and profile receipts are included.
Scalar-read proxies decreased from {comparison['cases']['analytical32']['baseline_scalar_reads']:,}
to {comparison['cases']['analytical32']['current_scalar_reads']:,} (`analytical32`) and
{comparison['cases']['mixed6']['baseline_scalar_reads']:,} to
{comparison['cases']['mixed6']['current_scalar_reads']:,} (`mixed6`) on matched inputs.
These are CPU profiler host-dispatch operation counts
while numerical inference ran on CUDA. They are not measured GPU synchronization
durations or an end-to-end speedup.

Peak GPU measurements follow required final device qualification and export.
Receipt elapsed time excludes writing/durably publishing the receipt itself;
return-only publication metrics are retained separately in the public qualification
record. Compilation/cache histories and numerical scopes remain explicit.

The [prior policy-v2 evidence](../371003f/README.md) remains separate. Neither this
synthetic qualification nor its increasing-size cases establishes cohort accuracy,
arbitrary-size scalability, global nonconvex optimality or a GPU speedup.
"""
    copies["README.md"] = readme.encode()
    manifest = {name: dict(sha256=sha(data), bytes=len(data)) for name, data in sorted(copies.items())}
    copies["SHA256.json"] = json_bytes(manifest)
    require(sum(map(len, copies.values())) < MAX_BYTES, "Compact evidence exceeds reviewed 16 MB limit; no files omitted")

    # Reconcile immutable inputs once more before any destination mutation.
    validate_source(local, plan, inventory)
    require(sha(regular_bytes(__file__)) == publisher_sha, "Publisher changed during validation")
    for target, origin in origins.items():
        require(sha(copies[target]) == origin["sha256"], "Publication buffer changed")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cuda-review-v3-publish-", dir=DESTINATION.parent))
    try:
        for name, data in sorted(copies.items()):
            path = under(stage, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            require(sha(regular_bytes(path)) == sha(data), f"Copy readback failed: {name}")
        for folder in sorted([stage, *(p for p in stage.rglob("*") if p.is_dir())], key=lambda p: len(p.parts), reverse=True):
            descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        # Linux renameat2 gives an atomic directory publication that CANNOT replace
        # a concurrently created target, including an empty directory or symlink.
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        require(rename is not None, "Atomic no-replace rename unavailable; refusing weaker publication")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(DESTINATION), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(DESTINATION))
        descriptor = os.open(DESTINATION.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    require(set(manifest) == {str(p.relative_to(DESTINATION)) for p in DESTINATION.rglob("*")
                              if p.is_file() and str(p.relative_to(DESTINATION)) != "SHA256.json"},
            "Published file inventory differs")
    for name, expected in manifest.items():
        require(sha(regular_bytes(under(DESTINATION, name))) == expected["sha256"], f"Published hash differs: {name}")
    print(json.dumps(dict(status="published", directory=str(DESTINATION), job_id=JOB_ID,
                          files=len(copies), bytes=sum(map(len, copies.values())), source_sha256=SOURCE_SHA)))


if __name__ == "__main__":
    main()
