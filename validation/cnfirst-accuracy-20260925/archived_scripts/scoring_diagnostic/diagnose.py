"""Offline fixed-membership diagnostic; never invokes the production fitter.

Uses frozen host likelihood, feasible scalar proposals, and the exact CUDA score
formula. Dense grid/local minima are NOT scalar certificates. A lower feasible
score is nevertheless evidence that the existing candidate family missed a
better partition under its own scoring objective. Truth is an oracle diagnostic
only and is never used by the greedy merge proposal.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.metrics import adjusted_rand_score

H = Path(__file__).resolve().parent
REPORT = H.parents[1] / "CNfirst4K_CliPP15_performance_20260925T181236Z"
SOURCE = H.parents[1] / "CliPP1.5/results/cnfirst-pool14-20260925-v2/a100/payload/source"
INPUT = H.parents[1] / "CliPP1.5/results/cnfirst-pool14-20260925-v1/inputs-v2"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(REPORT))
from clipp1d.io import read_tumor
from clipp1d.model import compile_model, loss, loss_at_rows
import evaluation_helpers as e

BOUND_PATH = H.parent / "BOUND_DIAGNOSTICS.json"
BOUND = {r["case_id"]: r["content"]["run.json"]
         for r in json.loads(BOUND_PATH.read_text())["records"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def penalty(sizes):
    sizes = np.asarray(sizes, float)
    n, k = sizes.sum(), len(sizes)
    mass = gammaln(k) - gammaln(n + k) + gammaln(sizes + 1).sum() + gammaln(k + 1)
    return float(k * np.log(n) - 1.4 * mass)


class ScalarDiagnostic:
    def __init__(self, model, baseline, grid_size):
        self.model = model
        self.baseline = baseline
        self.grid = np.linspace(float(model.lower.min()), float(model.upper.max()), grid_size)
        n = len(model)
        self.grid_loss = loss_at_rows(model, np.repeat(np.arange(n), grid_size),
                                     np.tile(self.grid, n)).reshape(n, grid_size)
        self.cache = {}
        self.evaluations = 0

    def group(self, members):
        key = tuple(sorted(int(i) for i in members))
        if key in self.cache:
            return self.cache[key]
        idx = np.asarray(key, int)
        sub = self.model.subset(idx)
        lo, hi = float(sub.lower.max()), float(sub.upper.min())
        assert lo <= hi

        def objective(value):
            self.evaluations += 1
            return float(loss(sub, np.full(len(idx), value)).sum())

        mask = (self.grid >= lo) & (self.grid <= hi)
        x = self.grid[mask]
        y = self.grid_loss[idx][:, mask].sum(axis=0)
        candidates = [(float(v), float(z)) for v, z in zip(x, y)]
        for point in np.unique(np.r_[lo, hi, self.baseline[idx]]):
            if lo <= point <= hi:
                candidates.append((float(point), objective(point)))
        for j in np.flatnonzero((y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:])) + 1:
            result = minimize_scalar(objective, bounds=(float(x[j-1]), float(x[j+1])),
                                     method="bounded", options={"xatol": 1e-12, "maxiter": 100})
            if result.success and lo <= result.x <= hi and math.isfinite(result.fun):
                candidates.append((float(result.x), float(result.fun)))
        phi, nll = min(candidates, key=lambda p: (p[1], p[0]))
        # Recompute exact frozen scalar likelihood at the retained feasible point.
        result = dict(members=key, phi=phi, nll=objective(phi))
        assert lo <= phi <= hi and math.isfinite(result["nll"])
        self.cache[key] = result
        return result


def summarize_partition(name, groups, truth_labels, truth_phi, published_score):
    n = len(truth_labels)
    labels, estimate = np.empty(n, int), np.empty(n)
    for k, group in enumerate(groups):
        labels[list(group["members"])] = k
        estimate[list(group["members"])] = group["phi"]
    centers = np.array([g["phi"] for g in groups])
    sizes = np.array([len(g["members"]) for g in groups])
    nll = sum(g["nll"] for g in groups)
    score = 2 * nll + penalty(sizes)
    clonal = min(range(len(groups)), key=lambda k: (abs(centers[k]-1), min(groups[k]["members"])))
    return dict(candidate=name, n=n, k=len(groups), nll=nll,
                penalty=penalty(sizes), score=score, score_delta=score-published_score,
                lower_score_than_published=score < published_score - 1e-4,
                ari=float(adjusted_rand_score(truth_labels, labels)),
                ccf_mae=float(np.abs(estimate-truth_phi).mean()),
                true_smf=float(np.mean(truth_phi < 1-1e-12)),
                estimated_smf=float(np.mean(labels != clonal)),
                centers=centers.tolist(), sizes=sizes.tolist(), labels=labels.tolist())


def diagnose(case_id, grid_size=1025):
    started = time.perf_counter()
    gpu = e.GPU[case_id]
    path = INPUT / (case_id + ".tsv")
    assert sha(path) == gpu["case"]["input_sha256"]
    model = compile_model(read_tumor(path))
    ids = list(model.mutation_ids)
    calls = pd.read_csv(io.StringIO(gpu["tables"]["mutation_clusters.tsv"]), sep="\t").set_index("mutation_id").loc[ids]
    assert len(calls) == len(model) and set(calls.status) == {"retained"}
    original_labels = calls.cluster_label.to_numpy(int)
    published_phi = calls.refitted_ccf.to_numpy(float)
    truth_frame = e.table(e.COHORT / case_id / "truth.txt")
    truth_frame.index = ["chr" + e.mid(ch, pos) for ch, pos in zip(truth_frame.chromosome_index, truth_frame.position)]
    truth = truth_frame.loc[ids]
    truth_labels, truth_phi = truth.cluster_id.to_numpy(int), truth.ccf.to_numpy(float)
    assert set(truth_frame.index) == set(ids)
    groups = [np.flatnonzero(original_labels == k) for k in np.unique(original_labels)]
    published_groups = [dict(members=tuple(g.tolist()), phi=float(published_phi[g[0]]),
                            nll=float(loss(model.subset(g), published_phi[g]).sum())) for g in groups]
    assert all(np.ptp(published_phi[g]) == 0 for g in groups)
    published_score = 2 * sum(g["nll"] for g in published_groups) + penalty([len(g) for g in groups])
    recorded_score = float(BOUND[case_id]["selection_score"])
    score_reconstruction_error = published_score - recorded_score
    assert abs(score_reconstruction_error) < 1e-6, (case_id, score_reconstruction_error)
    scalar = ScalarDiagnostic(model, published_phi, grid_size)
    refitted = [scalar.group(g) for g in groups]
    rows = [summarize_partition("published", published_groups, truth_labels, truth_phi, published_score),
            summarize_partition("published_approx_refit", refitted, truth_labels, truth_phi, published_score)]
    # Truth-free all-pairs agglomeration; accept only strict same-score decreases.
    current, history = refitted, []
    while len(current) > 1:
        old_sizes = [len(g["members"]) for g in current]
        old_score = 2 * sum(g["nll"] for g in current) + penalty(old_sizes)
        winner = None
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                merged = scalar.group(current[i]["members"] + current[j]["members"])
                new_sizes = [s for k, s in enumerate(old_sizes) if k not in (i, j)] + [old_sizes[i] + old_sizes[j]]
                change = (2 * (merged["nll"] - current[i]["nll"] - current[j]["nll"])
                          + penalty(new_sizes) - penalty(old_sizes))
                if change < -1e-4 and (winner is None or (change, i, j) < winner[:3]):
                    winner = change, i, j, merged
        if winner is None:
            break
        delta, i, j, merged = winner
        history.append(dict(k_before=len(current), score_before=old_score,
                            delta=delta, size_i=old_sizes[i], size_j=old_sizes[j]))
        current = [g for k, g in enumerate(current) if k not in (i, j)] + [merged]
        current.sort(key=lambda g: min(g["members"]))
    rows.append(summarize_partition("greedy_merge", current, truth_labels, truth_phi, published_score))
    # External partitions only after the truth-free merge search is finished.
    rows.append(summarize_partition("truth_partition_approx_refit",
        [scalar.group(np.flatnonzero(truth_labels == k)) for k in np.unique(truth_labels)],
        truth_labels, truth_phi, published_score))
    bound = e.PVM[case_id]
    pv_path = Path(bound["root"]) / "results.tsv"
    assert sha(pv_path) == bound["output_sha256"]["results.tsv"]
    pv = e.table(pv_path).set_index("mutation_id")
    pv_labels = pv.loc[[i.removeprefix("chr") for i in ids]].cluster_id.to_numpy(int)
    assert len(pv) == len(ids)
    rows.append(summarize_partition("pyclone_partition_approx_refit",
        [scalar.group(np.flatnonzero(pv_labels == k)) for k in np.unique(pv_labels)],
        truth_labels, truth_phi, published_score))
    for row in rows:
        row.update(case_id=case_id, true_k=len(np.unique(truth_labels)),
                   search_status=gpu["validated"]["search_status"])
    return dict(case_id=case_id, rows=rows, merge_history=history,
                published_selection_score=recorded_score,
                published_score_reconstruction_error=score_reconstruction_error,
                elapsed_seconds=time.perf_counter()-started, scalar_groups=len(scalar.cache),
                scalar_likelihood_evaluations=scalar.evaluations, input_sha256=sha(path),
                truth_sha256=sha(e.COHORT/case_id/"truth.txt"), pyclone_output_sha256=sha(pv_path),
                mutation_ids=ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["pilot", "all"], required=True)
    parser.add_argument("--grid-size", type=int, default=1025)
    args = parser.parse_args()
    metrics = pd.read_csv(REPORT/"per_case_metrics.tsv", sep="\t")
    native = metrics[(metrics.method == "CliPP1.5") & (metrics.population == "native")].set_index("case_id")
    ids = sorted(e.GPU)
    if args.panel == "pilot":
        ids = sorted(set(native.nlargest(4, "selected_k").index)
                     | set(native.nsmallest(3, "ari").index)
                     | set(native[native.true_k == 1].head(2).index)
                     | set(native[(native.true_k > 1) & (native.selected_k == native.true_k)].head(3).index))
    output = H / args.panel
    output.mkdir(exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(diagnose, ids, [args.grid_size] * len(ids)):
            records.append(result)
            (output/(result["case_id"]+".json")).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            print(f'{len(records)}/{len(ids)} {result["case_id"]} {result["elapsed_seconds"]:.2f}s', flush=True)
    table = pd.DataFrame([{k:v for k,v in row.items() if k not in ("centers", "sizes", "labels")}
                          for result in records for row in result["rows"]])
    table.to_csv(output/"per_case_candidates.tsv", sep="\t", index=False)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=len(ids), grid_size=args.grid_size,
                   maximum_processes=4, diagnostic="approximate scalar refits; no production solver or scalar qualification",
                   score_formula="2*NLL + K*log(N) - 1.4*(lgamma(K)-lgamma(N+K)+sum(lgamma(nk+1))+lgamma(K+1))",
                   likelihood_source=str(SOURCE/"src/clipp1d/model.py"),
                   likelihood_source_sha256=sha(SOURCE/"src/clipp1d/model.py"),
                   script_sha256=sha(__file__), export_sha256=sha(REPORT/"CLIPP15_RESULTS.json"),
                   bound_diagnostics_sha256=sha(BOUND_PATH),
                   maximum_published_score_reconstruction_error=max(abs(r["published_score_reconstruction_error"]) for r in records),
                   candidate_summary=[])
    base = table[table.candidate == "published"].set_index("case_id")
    for name, frame in table.groupby("candidate"):
        frame = frame.set_index("case_id").loc[base.index]
        summary["candidate_summary"].append(dict(candidate=name, lower_score_cases=int(frame.lower_score_than_published.sum()),
            median_score_delta=float(frame.score_delta.median()), mean_ari=float(frame.ari.mean()),
            ari_improved=int((frame.ari > base.ari + 1e-10).sum()), ari_worsened=int((frame.ari < base.ari-1e-10).sum()),
            mean_ccf_mae=float(frame.ccf_mae.mean()), mean_k=float(frame.k.mean()),
            smf_ccc=e.ccc(frame.true_smf, frame.estimated_smf)))
    (output/"SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
