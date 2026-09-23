"""Read-only source-bound final-path comparison; never run numerical fitting.

Accepts either the original study directory or its published evidence directory
with an attempts/ tree. Only imported terminal receipts are considered. Outputs
are deterministic for an unchanged evidence inventory and never overwritten.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


FINAL_SOURCE = "8627daf96b3b72be689a756445dfb60a7798a03d8bdd8d896f837eb1f11184c2"
BASELINE_SOURCE = "726eaf873550d80cd89d72d7c1253e1c70aa97135011fbad7f9d6c97c9382a88"
COMPARISON_KEYS = ("selected_lambda", "raw_ccf", "raw_objective", "refitted_ccf",
                   "cluster_centers", "cluster_labels", "score")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    require(path.is_file() and not path.is_symlink(), f"Not a regular evidence file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    digest(path)
    return json.loads(path.read_bytes())


def attempt_root(root, name):
    return root / "attempts" / name if (root / "attempts").is_dir() else root / name


def manifest_path(parent):
    return parent / "IMPORT_MANIFEST.json" if (parent / "IMPORT_MANIFEST.json").is_file() else (
        parent / "operational/IMPORT_MANIFEST.json")


class Task:
    def __init__(self, root, attempt, name, source):
        self.attempt, self.name = attempt, name
        self.parent = attempt_root(root, attempt)
        self.manifest = load(manifest_path(self.parent))
        self.bindings = {}
        self.receipt = self.read("results/" + name + ".json")
        require(self.receipt["source"]["source_sha256"] == source, "Unexpected production fingerprint")
        require(self.receipt["status"] in ("passed", "failed"), "Task is not terminal")
        require(self.receipt["cuda_available"] and self.receipt.get("device", "").startswith("cuda:"),
                "Missing allocated-CUDA identity")
        terminal = self.read("receipts/terminal.json")
        require(terminal == self.manifest["terminal"], "Imported terminal receipt differs")
        execution = [row for row in terminal["tasks"] if row["name"] == name]
        require(len(execution) == 1 and execution[0]["receipt_sha256"] ==
                self.bindings["results/" + name + ".json"], "Terminal task receipt binding differs")
        require(execution[0]["status"] == self.receipt["status"], "Worker/task outcome differs")
        self.job_id = terminal["job_id"]

    def path(self, name, expected=None):
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe artifact name")
        item = self.manifest["files"][name]
        path = self.parent / "imported" / name
        actual = digest(path)
        require(actual == item["sha256"] and path.stat().st_size == item["bytes"], "Imported artifact changed")
        require(expected is None or actual == expected, "Nested artifact hash differs")
        self.bindings[name] = actual
        return path

    def read(self, name, expected=None):
        return load(self.path(name, expected))

    def artifact(self, basename):
        name = self.name + ".artifacts/" + basename
        return self.read("results/" + name, self.receipt.get("artifacts", {}).get(name))

    def summary(self):
        return dict(attempt=self.attempt, task=self.name, job_id=self.job_id,
                    source_sha256=self.receipt["source"]["source_sha256"],
                    receipt_status=self.receipt["status"], elapsed_seconds=self.receipt["elapsed_seconds"],
                    gpu=self.receipt.get("gpu"), evidence=dict(sorted(self.bindings.items())))


def path_counts(path):
    records = path["path_records"]
    starts = [(i, j, s) for i, row in enumerate(records) for j, s in enumerate(row.get("starts", []))]
    return dict(search_status=path["search_status"], planned_penalties=len(records),
                complete_penalties=sum(bool(r.get("search_complete")) for r in records),
                qualified_winners=sum(r.get("raw_status") == "qualified" for r in records),
                qualified_refits=sum(r.get("refit_status") == "qualified" for r in records),
                starts_attempted=len(starts), starts_qualified=sum(bool(s["qualified"]) for _, _, s in starts),
                unresolved_starts=[dict(path_index=i, start_index=j, status=s["status"],
                                        audit_status=s.get("audit_status")) for i, j, s in starts if not s["qualified"]])


def maximum_error(left, right):
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return None
        errors = [maximum_error(a, b) for a, b in zip(left, right)]
        return None if None in errors else max(errors, default=0.)
    return abs(left - right)


def compare_paths(current, baseline):
    equal = {key: current[key] == baseline[key] for key in COMPARISON_KEYS}
    return dict(equal=equal, all_selected_fields_exact=all(equal.values()),
                maximum_absolute_errors={key: maximum_error(current[key], baseline[key])
                                         for key in COMPARISON_KEYS if key != "cluster_labels"},
                graph_identity={key: current["pilot_and_graph"][key] == baseline["pilot_and_graph"][key]
                                for key in ("pilot_sha256", "weights_sha256")},
                baseline_search_status=baseline["search_status"], baseline_counts=path_counts(baseline))


def timing(current, baseline, scope):
    return dict(current_seconds=current, baseline_seconds=baseline,
                current_over_baseline=None if baseline in (None, 0) else current / baseline,
                scope=scope,
                interpretation="Single observed runs, potentially different allocations/cache histories; not repeated latency or a general speedup claim")


def public_tables(task):
    tables = {}
    for filename in ("mutation_clusters.tsv", "cluster_centers.tsv", "mutation_multiplicity.tsv"):
        name = task.name + ".artifacts/public-output/" + filename
        if "results/" + name not in task.manifest["files"]:
            continue
        expected = task.receipt.get("artifacts", {}).get(name)
        with task.path("results/" + name, expected).open(newline="") as stream:
            rows = csv.reader(stream, delimiter="\t")
            fields = next(rows)
            count = sum(1 for _ in rows)
        tables[filename] = dict(columns=fields, rows=count, sha256=task.bindings["results/" + name])
    return tables


def mixed(task, root, previous):
    receipt = task.receipt
    path = task.artifact("full-path.json")
    baseline_map = {
        ("mixed_support", 64): (root, "qualification-mixed64-h", "mixed64-reference"),
        ("below_one", 64): (previous, "below64-f", "below64-current"),
        ("mixed_support", 256): (previous, "mixed256-e", "mixed256-current"),
    }
    key = (receipt["fixture_family"], receipt["nodes"])
    row = dict(family=key[0], nodes=key[1], counts=path_counts(path), selected_lambda=path["selected_lambda"],
               selected_clusters=len(path["cluster_centers"]), score=path["score"],
               final_export_and_publication_qualified=receipt["full_path"]["final_export_and_publication_qualified"],
               public_tables=public_tables(task), qualification_scope=receipt.get("qualification_scope", receipt["scope"]),
               numerical_stages=path["timings"], source_qualified=receipt["status"] == "passed")
    baseline_seconds = None
    if key in baseline_map:
        baseline = Task(*baseline_map[key], BASELINE_SOURCE)
        old = baseline.artifact("full-path.json")
        require(receipt["input_sha256"] == baseline.receipt["input_sha256"] and
                receipt["policy"] == baseline.receipt["policy"], "Baseline input/policy does not match")
        row["baseline_comparison"] = compare_paths(path, old)
        baseline_seconds = baseline.receipt["full_path"]["fit_seconds"]
        row["baseline_comparison"]["baseline_task"] = baseline.summary()
        row["baseline_comparison"]["scope"] = "Saved numerical selection; full-path and publication status remain separate"
    else:
        row["baseline_comparison"] = None
        row["baseline_absence"] = "No source-bound same-input baseline in this study; no performance or parity comparison inferred"
    row["fit_timing"] = timing(receipt["full_path"]["fit_seconds"], baseline_seconds,
                                "Instrumented complete numerical path, excluding final export/publication")
    if receipt.get("comparison"):
        row["matched_problem_qualification"] = receipt["comparison"]
    row["task"] = task.summary()
    return row


def existing(task, root):
    baseline = Task(root, "qualification-existing-g", "existing-baseline", BASELINE_SOURCE)
    require(task.receipt["policy"] == baseline.receipt["policy"] and
            task.receipt["path_grid"] == baseline.receipt["path_grid"] and
            task.receipt["qualification_script_sha256"] == baseline.receipt["qualification_script_sha256"],
            "Existing-qualifier policy/grid/fixture implementation differs")
    scaling = []
    for event in task.receipt["cases"]:
        if event["kind"] != "scaling_qualified":
            continue
        n = event["nodes"]
        current, old = task.artifact(f"scaling-{n}-path.json"), baseline.artifact(f"scaling-{n}-path.json")
        exported, old_exported = task.artifact(f"scaling-{n}-qualified.json"), baseline.artifact(f"scaling-{n}-qualified.json")
        scaling.append(dict(nodes=n, counts=path_counts(current), comparison=compare_paths(current, old),
                            final_device_qualification_and_export_complete=exported["final_device_qualification_and_export_complete"],
                            publication_scope="Numerical export only; these scaling probes do not publish tumor TSVs",
                            path_timing=timing(current["fit_seconds"], old["fit_seconds"], "Complete synthetic numerical path"),
                            export_timing=timing(exported["seconds"], old_exported["seconds"], "Path through final qualification/device export"),
                            numerical_stages=current["timings"], baseline_numerical_stages=old["timings"]))
    pipelines = []
    old_events = {(e["fixture"], e["execution"]): e for e in baseline.receipt["cases"] if e["kind"] == "pipeline_attempt"}
    for event in task.receipt["cases"]:
        if event["kind"] != "pipeline_attempt":
            continue
        key = (event["fixture"], event["execution"])
        basename = f"pipeline-{key[0]}-{key[1]}.json"
        current, old = task.artifact(basename), baseline.artifact(basename)
        pipelines.append(dict(fixture=key[0], execution=key[1], counts=path_counts(current),
                              comparison=compare_paths(current, old),
                              timing=timing(event["seconds"], old_events[key]["seconds"], "Instrumented toy numerical path")))
    public = [e for e in task.receipt["cases"] if e["kind"] == "public_publication"]
    old_public = [e for e in baseline.receipt["cases"] if e["kind"] == "public_publication"]
    require(len(public) == len(old_public) == 1, "Existing qualifier lacks public publication proof")
    tables = public_tables(task)
    public_equal = {name: data["sha256"] == old_public[0]["table_sha256"][name] for name, data in tables.items()}
    result = dict(scaling=scaling, pipelines=pipelines, public_tables=tables, public_table_hashes_equal=public_equal,
                  public_timing=timing(public[0]["seconds"], old_public[0]["seconds"], "Public fit through durable publication"),
                  overall_timing=timing(task.receipt["elapsed_seconds"], baseline.receipt["elapsed_seconds"],
                                        "Whole heterogeneous qualifier including compilation, kernels, QPs, paths and publication"))
    result["task"], result["baseline_task"] = task.summary(), baseline.summary()
    return result


def markdown(summary):
    lines = ["# Final source path comparison", "", f"Production fingerprint: `{summary['source_sha256']}`.", "",
             "Single observed fits are not repeated latency measurements. Incomplete baseline searches remain incomplete.", "",
             "| Fixture | Starts | Penalties complete | Final publication | Baseline selected estimates | Fit seconds, current / baseline |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in summary["heterogeneous_paths"]:
        c, t, b = row["counts"], row["fit_timing"], row["baseline_comparison"]
        seconds = f"{t['current_seconds']:.3f} / " + ("unavailable" if t["baseline_seconds"] is None else f"{t['baseline_seconds']:.3f}")
        lines.append(f"| {row['family']} N{row['nodes']} | {c['starts_qualified']}/{c['starts_attempted']} | "
                     f"{c['complete_penalties']}/{c['planned_penalties']} | {row['final_export_and_publication_qualified']} | "
                     f"{'no baseline' if b is None else 'exact' if b['all_selected_fields_exact'] else 'differs'} | {seconds} |")
    for run in summary["existing_qualification"]:
        lines += ["", "| Existing qualifier case | Current seconds | Baseline seconds | Ratio | Selected estimates |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for row in run["scaling"]:
            t = row["path_timing"]
            lines.append(f"| scaling N{row['nodes']} | {t['current_seconds']:.3f} | {t['baseline_seconds']:.3f} | "
                         f"{t['current_over_baseline']:.3f} | {'exact' if row['comparison']['all_selected_fields_exact'] else 'differs'} |")
        for row in run["pipelines"]:
            t = row["timing"]
            lines.append(f"| {row['fixture']} {row['execution']} | {t['current_seconds']:.3f} | {t['baseline_seconds']:.3f} | "
                         f"{t['current_over_baseline']:.3f} | {'exact' if row['comparison']['all_selected_fields_exact'] else 'differs'} |")
    if summary["not_imported"]:
        lines += ["", "Not yet imported: " + ", ".join(summary["not_imported"]) + "."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=FINAL_SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists() and not args.markdown.exists(), "Summary outputs are never overwritten")
    summary = dict(schema="clipp1d.cuda.bound.final_path_comparison.v1", source_sha256=args.expected_source_sha256,
                   scope="Read-only source-bound numerical comparisons, qualification coverage and observed timings; no cohort inference",
                   heterogeneous_paths=[], existing_qualification=[], not_imported=[],
                   script_sha256=digest(Path(__file__)))
    parents = args.root / "attempts" if (args.root / "attempts").is_dir() else args.root
    for parent in sorted(parents.glob("qualification-final-*")):
        if not manifest_path(parent).is_file():
            summary["not_imported"].append(parent.name)
            continue
        for path in sorted((parent / "imported/results").glob("*.json")):
            receipt = load(path)
            if receipt.get("source", {}).get("source_sha256") != args.expected_source_sha256:
                continue
            task = Task(args.root, parent.name, path.stem, args.expected_source_sha256)
            if receipt["schema"] == "clipp1d.cuda.mixed_qualification.v1":
                summary["heterogeneous_paths"].append(mixed(task, args.root, args.previous_root))
            elif receipt["schema"] == "clipp1d.cuda.qualification.v3":
                summary["existing_qualification"].append(existing(task, args.root))
            else:
                raise RuntimeError("Unexpected final-path receipt schema")
    for path, text in ((args.out, json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"),
                       (args.markdown, markdown(summary))):
        with path.open("x") as stream:
            stream.write(text)
    print(markdown(summary), end="")


if __name__ == "__main__":
    main()
