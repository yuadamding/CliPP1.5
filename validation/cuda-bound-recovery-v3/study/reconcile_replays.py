"""Reconcile the 13 literal captured QPs and two independently bound replays.

Read-only evidence processing; no inference, source import, or remote operations.
Writes deterministic JSON/TSV summaries. Existing identical outputs are accepted;
different existing outputs are never overwritten. Run with the study directory
as --study-root, or an archive whose attempts/ directory preserves that layout.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct


BASELINE = "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88"
CONE = "3d374f47e25067352250a2f29da35c83a99bbca68fa9aca84e64a8de828d762d"
NORMALIZED = "276da27b9a8718f3f06182108ccb0a6b7f04f3937a67f050f9721e031554aea5"
COMMIT = "430db26cf07466e88e53c6e1a8fbe2be7b7b25e9"
FAMILIES = (
    ("below_one", 64, 2, "capture-below-a", "replay-below-c2", "replay-scaled-below-e"),
    ("mixed_support", 256, 11, "capture-mixed-b", "replay-mixed-d", "replay-scaled-mixed-f"),
)
COMPONENTS = ("node_quadratic", "box_normal", "edge", "total")
PROBLEM = ("h", "target", "lower", "upper", "caps", "start", "dual")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_under(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and str(relative) == name and
            all(part not in (".", "..") for part in relative.parts), "Unsafe evidence path")
    path = root / str(relative)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            f"Missing regular evidence: {path}")
    return path


def parse(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON token: {value}")
    return json.loads(data, parse_constant=invalid)


def fingerprint(source):
    files = source["source_files"]
    return sha("".join(f"{name}\0{digest}\n" for name, digest in sorted(files.items())).encode())


class Attempt:
    def __init__(self, root, name, result_name):
        self.name = name
        self.parent = root / name
        manifest_path = safe_under(self.parent, "IMPORT_MANIFEST.json")
        self.manifest_sha256 = sha(manifest_path.read_bytes())
        self.files = parse(manifest_path.read_bytes())["files"]
        self.root = self.parent / "imported"
        self.cache = {}
        self.result_name = result_name
        self.receipt = self.load("results/" + result_name)
        self.receipt_sha256 = self.files["results/" + result_name]["sha256"]
        require(self.receipt["status"] == "passed", f"Diagnostic did not pass: {name}")
        for artifact, digest in self.receipt["artifacts"].items():
            self.read("results/" + artifact, digest)
        require(fingerprint(self.receipt["source"]) == self.receipt["source"]["source_sha256"],
                "Source file inventory/fingerprint mismatch")

    def read(self, name, expected=None):
        require(name in self.files, f"Artifact absent from import binding: {name}")
        bound = self.files[name]
        if name not in self.cache:
            data = safe_under(self.root, name).read_bytes()
            require(len(data) == bound["bytes"] and sha(data) == bound["sha256"],
                    f"Imported evidence changed: {self.name}/{name}")
            self.cache[name] = data
        require(expected is None or expected == bound["sha256"], "Nested artifact hash differs")
        return self.cache[name]

    def load(self, name, expected=None):
        return parse(self.read(name, expected))

    def tensor(self, descriptor):
        value = self.load("results/" + descriptor["path"], descriptor["sha256"])
        require(value["dtype"] == descriptor["dtype"] == "float64" and
                value["shape"] == descriptor["shape"], "Tensor metadata mismatch")
        shape = descriptor["shape"]
        require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape),
                "Invalid tensor shape")
        require(value["encoding"] == "json_numbers", "Unexpected nonfinite tensor encoding")

        def packed(values, sizes):
            if not sizes:
                require(type(values) in (int, float) and math.isfinite(values), "Invalid float64")
                return struct.pack("<d", values)
            require(isinstance(values, list) and len(values) == sizes[0], "Tensor shape mismatch")
            return b"".join(packed(row, sizes[1:]) for row in values)

        require(sha(packed(value["values"], shape)) == value["tensor_sha256"] == descriptor["tensor_sha256"],
                "Tensor bytes/hash mismatch")
        return value["values"]

    def identity(self):
        return dict(attempt=self.name, receipt="results/" + self.result_name,
                    receipt_sha256=self.receipt_sha256,
                    import_manifest_sha256=self.manifest_sha256,
                    source_sha256=self.receipt["source"]["source_sha256"],
                    lsf_job_id=self.receipt["lsf_job_id"])


def certificate(values, policy):
    result = {key: values[key] for key in ("gap", "scale", "kkt")}
    require(all(math.isfinite(x) and x >= 0 for x in result.values()), "Invalid certificate")
    result["allowed_gap"] = policy["inner_atol"] + policy["inner_rtol"] * result["scale"]
    result["gap_qualified"] = result["gap"] <= result["allowed_gap"]
    result["kkt_qualified"] = result["kkt"] <= policy["inner_kkt_tol"]
    result["qualified"] = result["gap_qualified"] and result["kkt_qualified"]
    result["gap_to_allowance"] = result["gap"] / result["allowed_gap"]
    result["kkt_to_tolerance"] = result["kkt"] / policy["inner_kkt_tol"]
    if "independently_qualified" in values:
        require(values["independently_qualified"] == result["qualified"], "Certificate gate mismatch")
    if "allowed_gap" in values:
        require(values["allowed_gap"] == result["allowed_gap"], "Certificate allowance changed")
    return result


def component_values(decomposition):
    result = {key: decomposition[key] for key in COMPONENTS}
    require(all(math.isfinite(x) and x >= 0 for x in result.values()), "Invalid gap component")
    return result


def replay_summary(attempt, item, entry, captured, record, policy, expected_qualified):
    require(item["context"] == entry["context"] == record["context"], "Replay QP context differs")
    require(item["capture_record_sha256"] == entry["sha256"], "Replay captured record differs")
    baseline = item["original_baseline"]
    for flag in ("original_capture_compiled_proof_exact", "eager_certificate_exactly_reconstructed",
                 "compiled_gate_classifications_unchanged", "terminal_admm_equals_returned"):
        require(baseline[flag] is True, f"Missing terminal reconstruction proof: {flag}")
    require(component_values(baseline["returned"]) == component_values(record["decomposition"]["returned"]),
            "Baseline component decomposition differs between replays")
    for key in ("gap", "scale", "kkt"):
        require(baseline["returned_certificate"][key] == captured.tensor(record["returned"][key]),
                "Replay saved terminal certificate changed")
    eager = certificate(item["eager_original_certificate"], policy)
    compiled = certificate(item["compiled_original_certificate"], policy)
    require(item["production_qualified"] is expected_qualified and item["qualified"] is expected_qualified
            and eager["qualified"] is expected_qualified and compiled["qualified"] is expected_qualified,
            "Unexpected original-start replay qualification")
    require(all(item[key] is True for key in ("primal_box_feasible", "dual_capacity_feasible", "dual_exact_skew")),
            "Replay feasibility failed")
    require(type(item["admm_iterations"]) is int and 0 <= item["admm_iterations"] <= policy["inner_max_iterations"],
            "Original per-QP iteration budget changed")
    require(type(item["polish_iterations"]) is int and item["polish_iterations"] >= 0, "Invalid flow work")
    for descriptor in item["result"].values():
        attempt.tensor(descriptor)
    return dict(qualified=item["qualified"], compiled_certificate=compiled, eager_certificate=eager,
                gap_components=component_values(item["decomposition"]),
                admm_iterations=item["admm_iterations"], polish_iterations=item["polish_iterations"],
                seconds=item["seconds"], timing_scope=item["timing_scope"],
                original_shifted_objective=item["original_shifted_objective"],
                result_tensor_sha256={key: value["tensor_sha256"] for key, value in sorted(item["result"].items())})


def interval(values):
    return dict(minimum=min(values), maximum=max(values))


def summarize(rows):
    result = dict(captured_qps=len(rows))
    for stage in ("baseline", "cone_only", "normalized"):
        values = [row[stage] for row in rows]
        result[stage] = dict(qualified=sum(row["qualified"] for row in values),
                            admm_iterations_sum=sum(row["admm_iterations"] for row in values),
                            admm_iterations=interval([row["admm_iterations"] for row in values]),
                            polish_iterations_sum=sum(row["polish_iterations"] for row in values),
                            compiled_gap=interval([row["compiled_certificate"]["gap"] for row in values]),
                            compiled_gap_to_allowance=interval([row["compiled_certificate"]["gap_to_allowance"] for row in values]),
                            compiled_kkt=interval([row["compiled_certificate"]["kkt"] for row in values]),
                            gap_components={key: interval([row["gap_components"][key] for row in values]) for key in COMPONENTS})
        if stage != "baseline":
            result[stage]["recorded_solve_seconds_sum"] = sum(row["seconds"] for row in values)
            result[stage]["timing_scope"] = sorted(set(row["timing_scope"] for row in values))
    return result


def reconcile(root):
    rows, inputs, policies, source_files, scripts = [], [], [], {}, []
    for family, nodes, count, captured_name, cone_name, normalized_name in FAMILIES:
        captured = Attempt(root, captured_name, f"capture-{family}-{nodes}.json")
        cone = Attempt(root, cone_name, f"replay-{family}-{nodes}.json")
        normalized = Attempt(root, normalized_name, f"replay-{family}-{nodes}.json")
        cap = captured.receipt
        require(cap["source"]["source_sha256"] == BASELINE and cap["frozen_commit"] == COMMIT,
                "Wrong capture source")
        require(cap["fixture_family"] == family and cap["nodes"] == nodes, "Wrong fixture identity")
        require(cap["diagnostic_capture_qualified"] is True and cap["scientific_fit_qualified"] is False
                and cap["scientific_search_status"] == "incomplete", "Capture success improperly promoted")
        require(len(cap["captures"]) == cap["captured_failures"] == cap["expected_failures"] == count,
                "Capture coverage mismatch")
        policy = cap["policy"]
        require(policy["inner_atol"] == 1e-10 and policy["inner_rtol"] == 1e-11
                and policy["inner_kkt_tol"] == 1e-7 and policy["inner_max_iterations"] == 20000,
                "Original QP gates or budget changed")
        policies.append(policy)
        for attempt, expected_source, expected_passes in ((cone, CONE, 0), (normalized, NORMALIZED, count)):
            replay = attempt.receipt
            require(replay["source"]["source_sha256"] == expected_source, "Unexpected replay source")
            require(replay["capture"]["sha256"] == captured.receipt_sha256 and
                    replay["capture"]["baseline_source_sha256"] == BASELINE and
                    replay["capture"]["baseline_commit"] == COMMIT, "Replay names a different capture")
            require(replay["policy"] == policy, "Replay policy differs from original")
            require(replay["captured_failed_qps"] == len(replay["replays"]) == count
                    and replay["resolved_qps"] == expected_passes
                    and replay["unresolved_qps"] == count - expected_passes
                    and replay["all_replays_qualified"] is bool(expected_passes), "Replay summary mismatch")
            require(sorted(item["capture_index"] for item in replay["replays"]) == list(range(count)),
                    "Duplicate or missing replay capture index")
            scripts.append({key: replay[key] for key in ("script_sha256", "capture_helper_sha256", "capture_dependency_hashes")})
        for attempt in (captured, cone, normalized):
            inputs.append(attempt.identity())
            source = attempt.receipt["source"]
            previous = source_files.setdefault(source["source_sha256"], source["source_files"])
            require(previous == source["source_files"], "Same source fingerprint has different files")
        cone_rows = {row["capture_record_sha256"]: row for row in cone.receipt["replays"]}
        norm_rows = {row["capture_record_sha256"]: row for row in normalized.receipt["replays"]}
        expected_keys = sorted(entry["sha256"] for entry in cap["captures"])
        require(len(set(expected_keys)) == count and sorted(cone_rows) == sorted(norm_rows) == expected_keys,
                "Sorted captured-QP inventories differ")
        for index, entry in enumerate(cap["captures"]):
            record = captured.load("results/" + entry["record"], entry["sha256"])
            context = record["context"]
            require(context == entry["context"] and context["source_sha256"] == BASELINE
                    and context["policy"] == policy and context["input_sha256"] == cap["input_sha256"],
                    "Captured record context differs")
            require(set(record["problem"]) == set(PROBLEM), "Missing literal QP or initialization")
            problem_hashes = {}
            for key, descriptor in sorted(record["problem"].items()):
                if descriptor is not None:
                    captured.tensor(descriptor)
                problem_hashes[key] = None if descriptor is None else descriptor["tensor_sha256"]
            baseline_certificate = certificate({key: captured.tensor(record["returned"][key])
                                                for key in ("gap", "scale", "kkt")}, policy)
            require(not baseline_certificate["qualified"] and record["returned"]["qualified"] is False,
                    "Captured baseline was qualified")
            baseline = dict(qualified=False, compiled_certificate=baseline_certificate,
                            eager_certificate=certificate(record["decomposition"]["returned"]["certificate"], policy),
                            gap_components=component_values(record["decomposition"]["returned"]),
                            admm_iterations=record["returned"]["iterations"],
                            polish_iterations=record["returned"]["polish_iterations"], seconds=None,
                            timing_scope="No isolated baseline QP solve timing in capture; enclosing path is instrumented diagnostic execution")
            require(baseline["admm_iterations"] == 20000, "Baseline failed QP did not exhaust original budget")
            items = {"cone_only": (cone, cone_rows[entry["sha256"]], False),
                     "normalized": (normalized, norm_rows[entry["sha256"]], True)}
            row = dict(family=family, nodes=nodes, capture_index=index,
                       capture_receipt_sha256=captured.receipt_sha256, capture_record_sha256=entry["sha256"],
                       context=context, context_sha256=sha(canonical(context)),
                       problem_and_initialization_tensor_sha256=problem_hashes, baseline=baseline)
            for stage, (attempt, item, expected) in items.items():
                require(item["capture_index"] == index, "Replay ordering differs")
                row[stage] = replay_summary(attempt, item, entry, captured, record, policy, expected)
            rows.append(row)
    require(len(rows) == 13 and all(policy == policies[0] for policy in policies), "Study coverage/policy mismatch")
    require(all(script == scripts[0] for script in scripts), "Replay helper bytes differ between compared sources")
    changed = {}
    for first, second, expected in ((BASELINE, CONE, {"cuda/kernels.py", "cuda/qp.py"}),
                                    (CONE, NORMALIZED, {"cuda/qp.py"})):
        require(set(source_files[first]) == set(source_files[second]), "Production inventory changed")
        names = sorted(name for name in source_files[first] if source_files[first][name] != source_files[second][name])
        require(set(names) == expected, "Unexpected production files changed")
        changed[first + "->" + second] = names
    rows.sort(key=lambda row: (row["family"], row["nodes"], row["context"]["path_index"],
                               row["context"]["start_index"], row["context"]["outer_iteration"],
                               row["context"]["qp_ordinal_within_start"], row["capture_record_sha256"]))
    summary = summarize(rows)
    require(summary["baseline"]["qualified"] == summary["cone_only"]["qualified"] == 0
            and summary["normalized"]["qualified"] == 13, "Required replay outcome was not established")
    return dict(schema="clipp1d.failed_qp_replay_comparison.v1", status="passed",
                scope="Exact 13 failed surrogate QPs with original starts/duals and unchanged gates; not full observed-likelihood paths, cohort accuracy, or repeated latency benchmarking",
                inputs=inputs, sources=dict(baseline=BASELINE, cone_only=CONE, normalized=NORMALIZED),
                changed_production_files=changed, policy=policies[0], common_replay_helpers=scripts[0],
                identity_checks=dict(exact_capture_receipt_and_record_hashes=True,
                                     exact_context_and_literal_problem_initialization_hashes=True,
                                     original_compiled_proof_and_eager_reconstruction=True,
                                     original_gates_and_budget_unchanged=True, sorted_keys_complete=True),
                summary=summary,
                families={f"{family}{nodes}": summarize([row for row in rows if row["family"] == family and row["nodes"] == nodes])
                          for family, nodes, *_ in FAMILIES}, rows=rows)


def table(value):
    rows = []
    for record in value["rows"]:
        row = {key: record[key] for key in ("family", "nodes", "capture_index", "capture_record_sha256", "context_sha256")}
        row.update({key: record["context"][key] for key in ("path_index", "start_index", "lambda_value", "outer_iteration", "backtrack_index", "inflation", "qp_ordinal_within_start")})
        for stage in ("baseline", "cone_only", "normalized"):
            item = record[stage]
            row.update({stage + "_" + key: item[key] for key in ("qualified", "admm_iterations", "polish_iterations", "seconds", "timing_scope")})
            row.update({stage + "_compiled_" + key: val for key, val in item["compiled_certificate"].items()})
            row.update({stage + "_eager_" + key: val for key, val in item["eager_certificate"].items()})
            row.update({stage + "_" + key: val for key, val in item["gap_components"].items()})
        rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_exact(path, content):
    if path.exists():
        require(path.read_bytes() == content, f"Refusing to overwrite different reconciliation: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    root = args.study_root.resolve()
    if not (root / FAMILIES[0][3]).is_dir() and (root / "attempts").is_dir():
        root /= "attempts"
    prefix = args.output_prefix or args.study_root / "REPLAY_COMPARISON"
    result = reconcile(root)
    result["script_sha256"] = sha(Path(__file__).read_bytes())
    write_exact(prefix.with_suffix(".json"), (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    write_exact(prefix.with_suffix(".tsv"), table(result))
    print(json.dumps(result["families"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
