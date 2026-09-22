"""Verify and compare two completed scaling panels without running any fits.

Source manifests must match their recorded package trees. Model arrays and the
chain are rebuilt from input and recorded pilot values in each bound package;
no pilot, optimizer, or refit is executed. Statistical identities must match
between arms. Different nonlinear outcomes are reported, never silently equated.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_clipp2 import ari, metrics, read_rows


TABLES = ("mutation_clusters.tsv", "cluster_centers.tsv", "mutation_multiplicity.tsv")
PATH_FIELDS = ("raw_status", "refit_status", "raw_solver_status", "search_complete",
               "witness_search_complete", "search_policy", "raw_witness_index", "raw_witness_mutation_id",
               "witnesses_eligible", "witnesses_solved", "witnesses_screened", "witnesses_unresolved",
               "starts_attempted", "starts_qualified", "starts_unresolved", "search_profile_calls",
               "search_surrogate_witnesses_profiled", "search_inner_iterations", "witness_switches",
               "outer_iterations", "backtracks", "partition_sha256", "start_failure_counts",
               "start_failure_status_counts")
READBACK_CODE = r'''
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
payload=json.load(sys.stdin)
sys.path.insert(0,payload["package_source"])
import numpy as np
from clipp1d import model as model_module
from clipp1d.chain import build_chain
from clipp1d.io import read_tumor
from clipp1d.policy import Policy
if Path(model_module.__file__).resolve().parent != Path(payload["package_source"]).resolve()/"clipp1d":
    raise ValueError("Readback imported the wrong package")
policy=Policy(**payload["policy"])
data=read_tumor(payload["input"],policy)
model=model_module.compile_model(data,policy)
if set(model.mutation_ids)!=set(payload["pilots"]):
    raise ValueError("Compiled retained IDs differ from output IDs")
digest=hashlib.sha256(json.dumps(model.mutation_ids).encode())
for name in ("alt","ref","lower","upper","slope","log_prior","valid"):
    digest.update(name.encode());digest.update(getattr(model,name).tobytes())
pilot=SimpleNamespace(phi=np.array([payload["pilots"][mid] for mid in model.mutation_ids]))
chain=build_chain(pilot,model.mutation_ids,policy)
print(json.dumps(dict(model_sha256=digest.hexdigest(),chain_sha256=chain.fingerprint,
                     chain_gap_floor=chain.gap_floor,input_sha256=data.input_sha256,
                     retained_ids=model.mutation_ids,
                     input_ids=[mutation.mutation_id for mutation in data.mutations],
                     chain_ranks={mid:int(chain.inverse_order[i]) for i,mid in enumerate(model.mutation_ids)})))
'''


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require_equal(left, right, message):
    if left != right:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_source(provenance, package_source):
    files = provenance["source_files"]
    package = Path(package_source) / "clipp1d"
    actual = {path.name: file_hash(path) for path in sorted(package.glob("*.py"))}
    require_equal(actual, files, f"Source file manifest mismatch: {package}")
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(f"{name}\0{value}\n".encode())
    require_equal(digest.hexdigest(), provenance["source_sha256"], "Source aggregate hash mismatch")
    return digest.hexdigest()


def verified_case(directory):
    directory = Path(directory)
    benchmark = read_json(directory / "benchmark.json")
    setup = benchmark["setup"]
    source = setup["package_source"]
    source_sha = verify_source(setup["provenance"], source)
    input_file, truth_file = directory / "input/tumor.tsv", directory / "input/truth.tsv"
    input_hash, truth_hash = file_hash(input_file), file_hash(truth_file)
    require_equal(input_hash, setup["input_sha256"], "Input hash mismatch")
    require_equal(truth_hash, setup["truth_sha256"], "Truth hash mismatch")
    common = dict(directory=str(directory.resolve()), status=benchmark["status"],
                  wall_seconds=benchmark["wall_seconds"], source_sha256=source_sha,
                  input_sha256=input_hash, truth_sha256=truth_hash,
                  policy=setup["policy"], policy_sha256=canonical_hash(setup["policy"]),
                  benchmark_sha256=file_hash(directory / "benchmark.json"))
    if benchmark["status"] != "success":
        return dict(**common, comparable=False, benchmark=benchmark)
    run_path = directory / "fit/run.json"
    run = read_json(run_path)
    if run["status"] != "success":
        raise ValueError("Successful benchmark lacks successful publication receipt")
    provenance = run["provenance"]
    require_equal(verify_source(provenance, source), source_sha, "Fit/setup source identity mismatch")
    require_equal(provenance["policy"], setup["policy"], "Fit/setup policy mismatch")
    require_equal(provenance["policy_sha256"], common["policy_sha256"], "Policy hash mismatch")
    require_equal(provenance["input_sha256"], input_hash, "Fit/input hash mismatch")
    for name in TABLES:
        require_equal(file_hash(directory / "fit" / name), run["table_sha256"][name], f"Table hash mismatch: {name}")
    rows = read_rows(directory / "fit/mutation_clusters.tsv")
    truth = read_rows(truth_file)
    calls = read_rows(directory / "fit/mutation_multiplicity.tsv")
    if not rows or not set(rows).issubset(truth):
        raise ValueError("Retained IDs must be nonempty and included in truth")
    require_equal(set(calls), set(rows), "Multiplicity and retained ID populations differ")
    with (directory / "fit/mutation_clusters.tsv").open(newline="") as handle:
        all_rows = list(csv.DictReader(handle, delimiter="\t"))
    output_ids = [row["mutation_id"] for row in all_rows]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Duplicate mutation IDs in public output")
    payload = dict(package_source=source, input=str(input_file.resolve()), policy=setup["policy"],
                   pilots={mid: float(row["pilot_ccf"]) for mid, row in rows.items()})
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    result = subprocess.run([sys.executable, "-c", READBACK_CODE], input=json.dumps(payload),
                            capture_output=True, text=True, env=env, timeout=60, check=False)
    if result.returncode:
        raise ValueError(f"Bound-source model/chain readback failed: {result.stderr.strip()}")
    checked = json.loads(result.stdout)
    require_equal(checked["input_sha256"], input_hash, "Compiled input identity mismatch")
    require_equal(checked["model_sha256"], provenance["model_sha256"], "Model hash mismatch")
    require_equal(checked["chain_sha256"], run["chain_sha256"], "Chain hash mismatch")
    require_equal(checked["chain_sha256"], benchmark["chain"]["chain_sha256"], "Benchmark/fit chain mismatch")
    require_equal(checked["chain_gap_floor"], run["chain_gap_floor"], "Chain gap floor mismatch")
    require_equal(checked["chain_ranks"], {mid: int(row["chain_rank"]) for mid, row in rows.items()}, "Chain ranks mismatch")
    require_equal(checked["input_ids"], run["input_identifiers"]["mutation_ids"], "Receipt input IDs mismatch")
    require_equal(output_ids, checked["input_ids"], "Output/input IDs mismatch")
    return dict(**common, comparable=True, run=run, rows=rows, truth=truth, calls=calls,
                run_sha256=file_hash(run_path), model_sha256=checked["model_sha256"],
                chain_sha256=checked["chain_sha256"], retained_ids=sorted(rows))


def numeric_change(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Compared vectors must have equal shapes and finite values")
    difference = right - left
    return dict(exactly_equal=bool(np.array_equal(left, right)),
                max_abs_difference=float(np.max(np.abs(difference))) if difference.size else 0.,
                rmse=float(np.sqrt(np.mean(difference**2))) if difference.size else 0.)


def compare_case(baseline, revised):
    for name in ("input_sha256", "truth_sha256", "policy_sha256"):
        require_equal(baseline[name], revised[name], f"Paired {name} mismatch")
    identity_keys = ("directory", "source_sha256", "input_sha256", "truth_sha256", "policy_sha256", "benchmark_sha256")
    record = dict(baseline={name: baseline[name] for name in identity_keys},
                  revised={name: revised[name] for name in identity_keys},
                  baseline_status=baseline["status"], revised_status=revised["status"],
                  baseline_wall_seconds=baseline["wall_seconds"], revised_wall_seconds=revised["wall_seconds"])
    if not baseline["comparable"] or not revised["comparable"]:
        return dict(**record, comparable=False, warning="At least one case has no completed published fit")
    for name in ("model_sha256", "chain_sha256", "retained_ids"):
        require_equal(baseline[name], revised[name], f"Paired {name} mismatch")
    ids, first, second = baseline["retained_ids"], baseline["rows"], revised["rows"]
    left_run, right_run = baseline["run"], revised["run"]
    vector_changes = {name: numeric_change([first[mid][name] for mid in ids], [second[mid][name] for mid in ids])
                      for name in ("pilot_ccf", "raw_ccf", "refitted_ccf")}
    left_labels, right_labels = [first[mid]["cluster_label"] for mid in ids], [second[mid]["cluster_label"] for mid in ids]
    changes = dict(**vector_changes, identical_labels=left_labels == right_labels,
                   label_ari=ari(left_labels, right_labels),
                   selected_lambda=dict(baseline=left_run["selected_lambda"], revised=right_run["selected_lambda"]),
                   selected_score_difference=right_run["selection_score"] - left_run["selection_score"],
                   selected_raw_objective_difference=right_run["raw_objective"] - left_run["raw_objective"],
                   identical_table_hashes=left_run["table_sha256"] == right_run["table_sha256"])
    paths = []
    left_path = {row["lambda"]: row for row in left_run["search"]["path"]}
    right_path = {row["lambda"]: row for row in right_run["search"]["path"]}
    for penalty in sorted(left_path.keys() | right_path.keys()):
        left, right = left_path.get(penalty), right_path.get(penalty)
        a = {name: left.get(name) for name in PATH_FIELDS} if left is not None else None
        b = {name: right.get(name) for name in PATH_FIELDS} if right is not None else None
        differing = [name for name in PATH_FIELDS if a is None or b is None or a[name] != b[name]]
        item = dict(lambda_value=penalty, baseline=a, revised=b, differing_fields=differing)
        for field in ("raw_objective", "score"):
            av, bv = left.get(field) if left else None, right.get(field) if right else None
            item[field] = dict(baseline=av, revised=bv, difference=bv - av if av is not None and bv is not None else None)
        paths.append(item)
    warnings = []
    if any(row["differing_fields"] or row["raw_objective"]["difference"] not in (None, 0.) for row in paths):
        warnings.append("Recorded nonlinear trajectories differ; identical selected outputs would not establish path equivalence")
    if not changes["raw_ccf"]["exactly_equal"] or not changes["refitted_ccf"]["exactly_equal"] or not changes["identical_labels"]:
        warnings.append("Selected raw or refitted outputs differ; inspect magnitudes and truth metrics")
    return dict(**record, comparable=True, retained_mutations=len(ids), retained_ids_sha256=canonical_hash(ids),
                model_sha256=baseline["model_sha256"], chain_sha256=baseline["chain_sha256"],
                baseline_run_sha256=baseline["run_sha256"], revised_run_sha256=revised["run_sha256"],
                baseline_search_status=left_run["search_status"], revised_search_status=right_run["search_status"],
                selected_comparison=changes, paths=paths, warnings=warnings,
                baseline_truth_metrics=metrics(first, baseline["truth"], baseline["calls"], "refitted_ccf"),
                revised_truth_metrics=metrics(second, revised["truth"], revised["calls"], "refitted_ccf"))


def compare_panels(baseline, revised):
    baseline, revised = Path(baseline), Path(revised)
    left_manifest, right_manifest = read_json(baseline / "scaling.json"), read_json(revised / "scaling.json")
    def cases(manifest):
        return {f"{row['scenario']}-n{row['mutations']}" for row in manifest["results"]}
    names = cases(left_manifest)
    require_equal(names, cases(right_manifest), "Panel case populations differ")
    if not names:
        raise ValueError("No panel cases")
    comparisons = {name: compare_case(verified_case(baseline / name), verified_case(revised / name)) for name in sorted(names)}
    return dict(schema="clipp1d.revision_outputs_comparison.v1", benchmark_sha256=file_hash(__file__),
                metrics_implementation_sha256=file_hash(Path(__file__).with_name("compare_clipp2.py")),
                baseline_manifest_sha256=file_hash(baseline / "scaling.json"),
                revised_manifest_sha256=file_hash(revised / "scaling.json"), cases=comparisons,
                scope="Read-only bound-source input/model/chain readback and published-output comparison; no fits or global-optimality claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh comparison output")
    report = compare_panels(args.baseline, args.revised)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    with args.output.open("x") as handle:
        handle.write(encoded + "\n")
    print(json.dumps({name: dict(comparable=case["comparable"], warnings=case.get("warnings", [case.get("warning")]))
                      for name, case in report["cases"].items()}), flush=True)


if __name__ == "__main__":
    main()
