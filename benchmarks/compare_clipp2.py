"""Compare final-refit outputs on exactly matched retained mutation IDs.

This tool consumes previously validated single-region CliPP2 tables. It launches
no fitting or remote jobs. Mixed-CN multiplicity requires an explicit truth target.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    result = {}
    for field in ("tumor_id", "sample_id", "region_id"):
        if len({row[field] for row in rows if field in row}) > 1:
            raise ValueError(f"Tables must contain exactly one {field}")
    for row in rows:
        if row.get("status", "retained") != "retained":
            continue
        mid = row["mutation_id"]
        if mid in result:
            raise ValueError("Tables must contain exactly one row per mutation; multi-sample inputs are unsupported")
        result[mid] = row
    return result


def validate_clonal_designation(run, centers):
    """Keep unconstrained v5 and post-fit-designated v6 output semantics distinct."""
    if run['schema'] == 'clipp1d.run.v5':
        assert run['designated_clonal_block'] is None
        assert centers and all(row['designated_clonal'] == '0' for row in centers)
        return
    assert run['schema'] == 'clipp1d.run.v6'
    assert run['provenance']['clonal_label_rule'] == 'nearest_to_one_l2_v1'
    assert isinstance(run['designated_clonal_block'], int)
    assert centers and all(row['designated_clonal'] == str(int(row['cluster_label'] == '0')) for row in centers)
    zero, = [row for row in centers if row['cluster_label'] == '0']
    distance = abs(float(zero['refitted_ccf']) - 1)
    assert distance == min(abs(float(row['refitted_ccf']) - 1) for row in centers)
    assignment = run['clonality']
    clonal, total = int(zero['cluster_size']), sum(int(row['cluster_size']) for row in centers)
    assert assignment['clonal_cluster_label'] == 0 and assignment['clonal_ccf'] == float(zero['refitted_ccf'])
    assert assignment['distance_to_one'] == distance and assignment['total_mutations'] == total
    assert assignment['clonal_mutations'] == clonal and assignment['subclonal_mutations'] == total - clonal
    assert assignment['subclonal_mutation_fraction'] == (total - clonal) / total


def ari(truth, labels):
    if len(truth) != len(labels):
        raise ValueError("ARI partitions must contain the same number of mutations")
    # Count pairs without constructing an M x M matrix.
    def choose2(n):
        return n * (n - 1) / 2
    joint, a, b = {}, {}, {}
    for x, y in zip(truth, labels):
        joint[x, y] = joint.get((x, y), 0) + 1
        a[x], b[y] = a.get(x, 0) + 1, b.get(y, 0) + 1
    pairs = choose2(len(truth))
    if pairs == 0:
        return 1.0
    sa, sb = sum(choose2(v) for v in a.values()), sum(choose2(v) for v in b.values())
    expected = sa * sb / pairs
    denominator = .5 * (sa + sb) - expected
    return 1.0 if denominator == 0 else (sum(choose2(v) for v in joint.values()) - expected) / denominator


def f1_metrics(true, called, eligible):
    # Missing calls count as false negatives, preserving coverage in the denominator.
    pairs = [(true[i], called[i]) for i in eligible]
    classes = sorted({x for x, _ in pairs} | {y for _, y in pairs if y is not None})
    per_class = {}
    for k in classes:
        tp = sum(x == y == k for x, y in pairs)
        fp = sum(x != k and y == k for x, y in pairs)
        fn = sum(x == k and y != k for x, y in pairs)
        per_class[str(k)] = {"f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                             "support": sum(x == k for x, _ in pairs), "tp": tp, "fp": fp, "fn": fn}
    tp, fp, fn = (sum(v[k] for v in per_class.values()) for k in ("tp", "fp", "fn"))
    return {"eligible": len(pairs), "called": sum(y is not None for _, y in pairs),
            "coverage": sum(y is not None for _, y in pairs) / len(pairs) if pairs else None,
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])) if classes else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "weighted_f1": sum(v["f1"] * v["support"] for v in per_class.values()) / len(pairs) if pairs else None,
            "per_class": per_class}


def metrics(rows, truth, multiplicity, ccf_column, *, designated_label="0"):
    ids = sorted(rows)
    if not ids or not set(ids).issubset(truth):
        raise ValueError("Scored mutations must be nonempty and contained in truth")
    estimated = np.array([float(rows[i][ccf_column]) for i in ids])
    actual = np.array([float(truth[i]["true_ccf"]) for i in ids])
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(actual)):
        raise ValueError("CCFs must be finite")
    labels = [rows[i]["cluster_label"] for i in ids]
    true_labels = [truth[i]["true_cluster"] for i in ids]
    denominator = np.var(actual) + np.var(estimated) + (np.mean(actual) - np.mean(estimated))**2
    constant = bool(np.ptp(actual) == 0 or np.ptp(estimated) == 0)
    if constant:
        ccc = float(np.array_equal(actual, estimated))
    else:
        ccc = 2 * np.mean((actual - actual.mean()) * (estimated - estimated.mean())) / denominator
    eligible = []
    for mid in ids:
        if truth[mid].get("mixed_cn") == "1" and not truth[mid].get("multiplicity_truth_target"):
            raise ValueError("Mixed CN requires a declared multiplicity truth target")
        cn = [float(truth[mid][key]) for key in ("major_cn", "minor_cn")]
        if not all(np.isfinite(v) and v >= 0 and v == int(v) for v in cn):
            raise ValueError("CNA eligibility requires finite nonnegative integer CN values")
        if cn != [1., 1.]:
            eligible.append(mid)
    calls = {i: None if i not in multiplicity or multiplicity[i]["multiplicity_call"] in (".", "", "NA", "nan")
             else int(multiplicity[i]["multiplicity_call"]) for i in ids}
    return {"mutations": len(ids), "rmse": float(math.sqrt(np.mean((estimated - actual)**2))),
            "ccc": float(ccc), "ccc_constant_vector_case": constant,
            "ccc_convention": "identical constants=1; other constant-vector cases=0",
            "ari": ari(true_labels, labels), "true_k_one": len(set(true_labels)) == 1,
            "ari_convention": "degenerate identical partitions=1", "selected_k": len(set(labels)),
            "designated_clonal_fraction": (labels.count(designated_label) / len(ids)
                                           if designated_label is not None else None),
            "all_exact_one_fraction": float(np.mean(estimated == 1)),
            "cna_only_multiplicity": f1_metrics({i: int(truth[i]["true_multiplicity"]) for i in ids}, calls, eligible)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--clipp1d-results", type=Path, required=True)
    parser.add_argument("--clipp2-clusters", type=Path, required=True)
    parser.add_argument("--clipp2-multiplicity", type=Path, required=True)
    parser.add_argument("--clipp2-commit", required=True, help="Full commit from the validated baseline receipt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.clipp2_commit) != 40 or any(c not in "0123456789abcdef" for c in args.clipp2_commit):
        raise ValueError("Supply the full lowercase baseline commit")
    one_file = args.clipp1d_results / "mutation_clusters.tsv"
    mult_file = args.clipp1d_results / "mutation_multiplicity.tsv"
    one, two, truth = read_rows(one_file), read_rows(args.clipp2_clusters), read_rows(args.truth)
    if not one or set(one) != set(two) or not set(one).issubset(truth):
        raise ValueError("Retained ID populations must match exactly and be contained in truth")
    run = json.loads((args.clipp1d_results / "run.json").read_text())
    if run["status"] != "success":
        raise ValueError("CliPP1D result is not a published success")
    for path in (one_file, mult_file, args.clipp1d_results / "cluster_centers.tsv"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["table_sha256"][path.name]:
            raise ValueError("CliPP1D table hash mismatch")
    phi_columns = [k for k in next(iter(two.values())) if k.startswith("phi_")]
    if len(phi_columns) != 1:
        raise ValueError("Expected exactly one final-refit phi_<region> column")
    report = {"schema": "clipp1d.comparison.v1", "clipp2_commit": args.clipp2_commit,
              "clipp1d_source_sha256": run["provenance"]["source_sha256"],
              "selection_score": run["selection_score"],
              "clipp1d": metrics(one, truth, read_rows(mult_file), "refitted_ccf",
                                 designated_label=None if run.get("schema") == "clipp1d.run.v5" else "0"),
              "clipp2": metrics(two, truth, read_rows(args.clipp2_multiplicity), phi_columns[0]),
              "file_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (args.truth, one_file, mult_file, args.clipp2_clusters, args.clipp2_multiplicity)},
              "qualification": "matched supplied outputs; baseline validation is external"}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
