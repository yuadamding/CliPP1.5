"""Frozen development/protected panel; failures remain in the denominator.

Use --split protected only after freezing the implementation and protocol.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from clipp.api import fit
from clipp.config import FitConfig
from clipp.native import load_native
from clipp.simulation import SimulationConfig, generate

PROTOCOL = "chain_engineering_panel_v1"


def ari(a, b):
    _, i = np.unique(a, return_inverse=True)
    _, j = np.unique(b, return_inverse=True)
    table = np.zeros((i.max() + 1, j.max() + 1), dtype=np.int64)
    np.add.at(table, (i, j), 1)

    def choose(x):
        return np.sum(x * (x - 1) / 2)

    total = len(a) * (len(a) - 1) / 2
    if total == 0:
        return 1.0
    s = choose(table)
    row = choose(table.sum(axis=1))
    col = choose(table.sum(axis=0))
    expected = row * col / total
    maximum = (row + col) / 2
    return 1.0 if maximum == expected else float((s - expected) / (maximum - expected))


def scenarios():
    base = SimulationConfig(mutations=48)
    return {
        "diploid": replace(base, cn_states=((1, 1),)),
        "loh": replace(base, cn_states=((2, 0),)),
        "balanced_amplification": replace(base, cn_states=((2, 2), (4, 4))),
        "high_major_cn": replace(base, cn_states=((6, 2), (10, 1))),
        "rare_close": replace(base, centers=(1.0, 0.55, 0.5), proportions=(0.5, 0.46, 0.04)),
        "no_clonal": replace(base, centers=(0.8, 0.45, 0.2)),
        "physical_endpoints": replace(base, centers=(1.0, 0.0), proportions=(0.7, 0.3)),
        "above_cap": replace(
            base,
            mutations=72,
            centers=tuple(np.linspace(0.1, 1.0, 12)),
            proportions=tuple(np.full(12, 1 / 12)),
        ),
        "purity_error": replace(base, reported_purity=0.7),
        "nonuniform_multiplicity": replace(base, multiplicity_policy="one"),
        "overdispersion": replace(base, overdispersion=0.02),
        "ascertainment": replace(base, minimum_alt=5),
    }


def evaluate(out, inputs):
    truth = pd.read_csv(inputs / "truth.tsv", sep="\t")
    truth = truth[truth.ascertained]
    best = out / "final_result/Best_K"
    assignments = pd.read_csv(next(best.glob("mutation_assignments*")), sep="\t")
    structure = pd.read_csv(next(best.glob("subclonal_structure*")), sep="\t")
    posterior = pd.read_csv(next(best.glob("posterior_multiplicity*")), sep="\t")
    joined = truth.merge(assignments, on="mutation_id", validate="one_to_one")
    if len(joined) != len(truth):
        raise ValueError("Missing truth/input coverage")
    centers = structure.set_index("cluster_index").cancer_cell_fraction
    labels = joined.cluster_index.to_numpy()
    cna = truth[(truth.major_cn != 1) | (truth.minor_cn != 1)].merge(
        posterior[["mutation_id", "multiplicity"]],
        on="mutation_id",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    classes = sorted(set(cna.multiplicity_true) | set(cna.multiplicity_pred))
    metrics = []
    for m in classes:
        tp = ((cna.multiplicity_true == m) & (cna.multiplicity_pred == m)).sum()
        fp = ((cna.multiplicity_true != m) & (cna.multiplicity_pred == m)).sum()
        fn = ((cna.multiplicity_true == m) & (cna.multiplicity_pred != m)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics.append((precision, recall, 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0))
    multiplicity = np.mean(metrics, axis=0) if metrics else [None] * 3
    # Rare recovery: best per-cluster F1 for truth groups occupying <=10%.
    recovery = []
    for cluster, block in joined.groupby("cluster"):
        if len(block) > len(joined) * 0.1:
            continue
        recovery.append(
            max(
                2
                * int(((joined.cluster == cluster) & (joined.cluster_index == k)).sum())
                / (int((joined.cluster == cluster).sum()) + int((joined.cluster_index == k).sum()))
                for k in np.unique(labels)
            )
        )
    completion = json.loads((out / "COMPLETE.json").read_text())
    return {
        "n": len(joined),
        "true_q": joined.cluster.nunique(),
        "fitted_q": len(structure),
        "ari": ari(joined.cluster, labels),
        "ccf_mae": float(np.mean(np.abs(centers.loc[labels].to_numpy() - joined.ccf))),
        "cluster_count_error": len(structure) - joined.cluster.nunique(),
        "rare_cluster_best_f1": float(np.mean(recovery)) if recovery else None,
        "cna_multiplicity_n": len(cna),
        "multiplicity_macro_precision": multiplicity[0],
        "multiplicity_macro_recall": multiplicity[1],
        "multiplicity_macro_f1": multiplicity[2],
        "end_to_end_seconds": completion["end_to_end_seconds"],
        "peak_rss_bytes": completion["peak_process_tree_rss_bytes_sampled"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "protected"], required=True)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("replicates must be positive")
    _, native = load_native()
    args.output.mkdir(parents=True, exist_ok=False)
    configurations = scenarios()
    offset = 70001 if args.split == "development" else 970001
    receipt = {
        "protocol": PROTOCOL,
        "split": args.split,
        "replicates": args.replicates,
        "native": native,
        "seed_rule": "offset + 1000*scenario_index + replicate",
        "seed_offset": offset,
        "scenario_names": list(configurations),
        "device": args.device,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification_scope": "small correctness/robustness panel; not a cohort accuracy ranking",
    }
    (args.output / "panel.json").write_text(json.dumps(receipt, indent=2) + "\n")
    rows = []
    for index, (name, config) in enumerate(configurations.items()):
        for rep in range(args.replicates):
            case = f"{name}-{rep + 1}"
            inputs = args.output / "inputs" / case
            out = args.output / "runs" / case
            record = {"case": case, "scenario": name, "replicate": rep + 1, "status": "failed"}
            start = time.perf_counter()
            try:
                generate(inputs, replace(config, seed=offset + 1000 * index + rep))
                fit(
                    inputs / "snv.tsv",
                    inputs / "cna.tsv",
                    inputs / "purity.txt",
                    out,
                    config=FitConfig(device=args.device, sample_id=case),
                )
                record.update(evaluate(out, inputs))
                record["status"] = "complete"
            except (OSError, ValueError, RuntimeError) as error:
                record["error"] = str(error)
            record["attempt_seconds"] = time.perf_counter() - start
            rows.append(record)
            pd.DataFrame(rows).to_csv(args.output / "performance.tsv", sep="\t", index=False)
            print(json.dumps(record), flush=True)
    successful = [r for r in rows if r["status"] == "complete"]
    times = [r["end_to_end_seconds"] for r in successful]
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "attempted": len(rows),
                "complete": len(successful),
                "failed": len(rows) - len(successful),
                "runtime_median": float(np.median(times)) if times else None,
                "runtime_p95": float(np.quantile(times, 0.95)) if times else None,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
