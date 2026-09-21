"""Deterministic tables and compact provenance; never overwrite a prior run."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def write_json(path, value):
    # Nonfinite diagnostic values are explicit nulls, never invalid JSON numbers.
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        return v
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, default=_json_default, allow_nan=False)
        handle.write("\n")


def write_result(result, outdir):
    diagnostics = result.raw_diagnostics
    if (not diagnostics.get("clonal_feasible") or
            not (diagnostics.get("raw_branch_stationarity_qualified") or
                 diagnostics.get("separable_scalar_gap_qualified")) or
            not len(result.cluster_centers) or result.cluster_centers[0] != 1 or
            not np.all(np.isfinite(result.refitted_phi))):
        raise ValueError("Refusing to publish an unqualified fit")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = ("mutation_clusters.tsv", "cluster_centers.tsv", "mutation_multiplicity.tsv", "run.json")
    if any((outdir / n).exists() for n in names):
        raise FileExistsError("Output files already exist; use a fresh output directory")
    ids = result.input_identifiers
    retained_ids = [m for m, keep in zip(ids["mutation_ids"], result.retained_mask) if keep]
    lookup = {mid: i for i, mid in enumerate(retained_ids)}
    ranks = result.frozen_chain.inverse_order
    with (outdir / names[0]).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["tumor_id", "sample_id", "mutation_id", "status", "chain_rank", "pilot_ccf",
                         "raw_ccf", "refitted_ccf", "cluster_label"])
        for mid, reason in zip(ids["mutation_ids"], result.exclusion_reasons):
            row = [ids["tumor_id"], ids["sample_id"], mid]
            if reason:
                writer.writerow(row + [reason] + ["."] * 5)
            else:
                i = lookup[mid]
                writer.writerow(row + ["retained", ranks[i], result.pilot.phi[i], result.raw_phi[i],
                                       result.refitted_phi[i], result.cluster_labels[i]])
    with (outdir / names[1]).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["tumor_id", "sample_id", "cluster_label", "cluster_size", "refitted_ccf", "designated_clonal"])
        sizes = np.bincount(result.cluster_labels, minlength=len(result.cluster_centers))
        for label, center in enumerate(result.cluster_centers):
            writer.writerow([ids["tumor_id"], ids["sample_id"], label,
                             int(sizes[label]), center, int(label == 0)])
    with (outdir / names[2]).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["tumor_id", "sample_id", "mutation_id", "refitted_ccf", "multiplicity_call"])
        for i, mid in enumerate(retained_ids):
            writer.writerow([ids["tumor_id"], ids["sample_id"], mid, result.refitted_phi[i], result.multiplicity_calls[i]])
    tables = {n: hashlib.sha256((outdir / n).read_bytes()).hexdigest() for n in names[:3]}
    write_json(outdir / "run.json", {"schema": "clipp1d.run.v3", "status": "success",
               "search_status": result.search_status,
               "input_identifiers": ids, "provenance": result.provenance,
               "selected_lambda": result.selected_lambda, "selection_score": result.selection_score,
               "raw_objective": result.raw_objective, "raw_witness_index": result.raw_witness_index,
               "raw_witness_mutation_id": result.raw_witness_mutation_id,
               "score_components": result.score_components, "raw_diagnostics": result.raw_diagnostics,
               "search": result.search_diagnostics, "chain_sha256": result.frozen_chain.fingerprint,
               "chain_gap_floor": result.frozen_chain.gap_floor,
               "partition_cuts": result.partition,
               "pilot_qualification": {"all_qualified": True,
                   "maximum_scalar_gap": float(np.max(result.pilot.gaps)),
                   "sum_scalar_gaps": float(np.sum(result.pilot.gaps)),
                   "likelihood_evaluations": sum(r.evaluations for r in result.pilot.scalar_results),
                   "bound_evaluations": sum(r.bound_evaluations for r in result.pilot.scalar_results),
                   "exact_scalar_fits": sum(r.method == "same_slope_single_candidate_exact" for r in result.pilot.scalar_results),
                   "alternative_well_mutations": int(np.count_nonzero(result.pilot.alternative_phi != result.pilot.phi))},
               "exclusions": {mid: r for mid, r in zip(ids["mutation_ids"], result.exclusion_reasons) if r},
               "designated_clonal_block": result.designated_clonal_block,
               "table_sha256": tables})
