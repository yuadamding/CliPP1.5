"""Deterministic tables and compact provenance; never overwrite a prior run."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .chain import extract_blocks
from .labeling import CLONAL_LABEL_RULE, clonality_summary, public_cluster_order
from .selection import partition_score


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _json_text(value):
    # Nonfinite diagnostic values are explicit nulls, never invalid JSON numbers.
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, np.ndarray) and v.ndim == 0:
            return clean(v.item())
        if isinstance(v, (tuple, list, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        return v
    return json.dumps(clean(value), indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n"


def write_json(path, value):
    """Publish a complete receipt without overwriting an existing path.

    The destination filesystem must support same-directory hard links (exFAT
    does not). Unsupported publication raises OSError; there is no unsafe
    fallback that exposes a partial marker or replaces existing evidence.
    """
    # Serialize before creating anything. Link a complete same-filesystem file
    # atomically: readers cannot observe a truncated success marker, and an
    # existing receipt (including one created concurrently) is never replaced.
    payload = _json_text(value)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".clipp1d-json-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require(condition, message):
    if not condition:
        raise ValueError("Refusing to publish " + message)


def _finite_scalar(value, *, nonnegative=False):
    return (isinstance(value, (int, float, np.integer, np.floating)) and
            not isinstance(value, (bool, np.bool_)) and np.isfinite(value) and
            (not nonnegative or value >= 0))


def _partition(cuts, n):
    valid = (isinstance(cuts, (tuple, list)) and len(cuts) >= 2 and
             all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in cuts) and
             cuts[0] == 0 and cuts[-1] == n and all(a < b for a, b in zip(cuts[:-1], cuts[1:])))
    _require(valid, "invalid chain partition cuts")
    return hashlib.sha256(np.asarray(cuts, dtype=np.int64).tobytes()).hexdigest()


def _validate_result(result):
    """Cross-check the public arrays and their independent raw reference.

    This validates identities and recorded qualification; it does not rerun the
    solver or confer a raw certificate on a direct partition.
    """
    diagnostics = result.raw_diagnostics
    _require(diagnostics.get("box_feasible") and diagnostics.get("clonal_constraint") is False and
             result.provenance.get("clonal_constraint") is False and
             (diagnostics.get("raw_branch_stationarity_qualified") or
              diagnostics.get("separable_scalar_gap_qualified")) and
             result.provenance.get("clonal_label_rule") == CLONAL_LABEL_RULE and
             result.raw_witness_index is None and
             result.raw_witness_mutation_id is None and _finite_scalar(result.raw_objective),
             "an unqualified unconstrained fit")
    ids = result.input_identifiers["mutation_ids"]
    mask = np.asarray(result.retained_mask)
    _require(len(set(ids)) == len(ids) and mask.shape == (len(ids),) and mask.dtype.kind == "b" and
             len(result.exclusion_reasons) == len(ids) and
             all(reason is None or isinstance(reason, str) and bool(reason) for reason in result.exclusion_reasons) and
             np.array_equal(mask, [r is None for r in result.exclusion_reasons]),
             "inconsistent retained mutation identities")
    retained = tuple(mid for mid, keep in zip(ids, mask) if keep)
    n = len(retained)
    centers = np.asarray(result.cluster_centers)
    k = centers.size
    _require(n > 0 and centers.shape == (k,) and k > 0 and
             np.all(np.isfinite(centers)) and np.all((centers >= 0) & (centers <= 1)) and
             np.all(np.diff(centers) <= 0), "invalid public cluster centers")
    for name in ("raw_phi", "refitted_phi", "cluster_labels", "multiplicity_calls"):
        values = np.asarray(getattr(result, name))
        _require(values.shape == (n,) and np.all(np.isfinite(values)), "invalid " + name)
    labels = result.cluster_labels
    calls = result.multiplicity_calls
    _require(labels.dtype.kind in "iu" and np.array_equal(np.unique(labels), np.arange(k)) and
             np.array_equal(result.refitted_phi, centers[labels]) and
             np.all((result.raw_phi >= 0) & (result.raw_phi <= 1)) and
             calls.dtype.kind in "iu" and np.all((calls >= 1) & (calls <= 4)),
             "inconsistent cluster assignments or multiplicities")
    chain = result.frozen_chain
    _require(chain.order.shape == chain.inverse_order.shape == (n,) and
             chain.order.dtype.kind in "iu" and chain.inverse_order.dtype.kind in "iu" and
             np.array_equal(np.sort(chain.order), np.arange(n)) and
             np.array_equal(chain.inverse_order[chain.order], np.arange(n)), "invalid frozen chain indices")
    fingerprint = _partition(result.partition, n)
    _require(len(result.partition) == k + 1, "a partition inconsistent with its cluster count")
    block_labels = []
    ordered_labels = labels[chain.order]
    for start, stop in zip(result.partition[:-1], result.partition[1:]):
        block_labels.append(int(ordered_labels[start]))
        _require(np.all(ordered_labels[start:stop] == ordered_labels[start]),
                 "cluster assignments inconsistent with the selected partition")
    expected_order = public_cluster_order(centers[block_labels], result.partition)
    _require([block_labels[i] for i in expected_order] == list(range(k)),
             "cluster labels inconsistent with nearest-to-one designation and chain tie order")
    _require(isinstance(result.designated_clonal_block, (int, np.integer)) and
             not isinstance(result.designated_clonal_block, (bool, np.bool_)) and
             result.designated_clonal_block == expected_order[0],
             "a clonal designation inconsistent with public cluster zero")
    pilot = result.pilot
    _require(pilot.mutation_ids == retained and pilot.phi.shape == pilot.gaps.shape == (n,) and
             np.all(np.isfinite(pilot.phi)) and np.all(np.isfinite(pilot.gaps)) and np.all(pilot.gaps >= 0) and
             len(pilot.scalar_results) == n and all(r.qualified for r in pilot.scalar_results),
             "an unqualified or mismatched pilot")
    selection = result.candidate_provenance
    family = selection.get("candidate_family")
    _require(family in ("direct_chain_partition", "fusion_path") and
             selection.get("origin") in ("production_path", "boundary_refinement",
                                         "adjacent_ward_pilot", "adjacent_ward_selected_raw") and
             selection.get("chain_sha256") == chain.fingerprint and
             selection.get("model_sha256") == result.provenance.get("model_sha256") and
             selection.get("refit_qualified") is True and
             _finite_scalar(selection.get("refit_gap"), nonnegative=True) and
             selection.get("partition_sha256") == fingerprint and
             selection.get("global_optimality_proven") is False and
             _finite_scalar(result.selection_score), "inconsistent candidate provenance")
    components = result.score_components
    _require(isinstance(components, dict) and _finite_scalar(components.get("twice_refit_loss")),
             "invalid selected score components")
    score, expected_components = partition_score(components["twice_refit_loss"] / 2, np.diff(result.partition))
    _require(score == result.selection_score and _json_text(components) == _json_text(expected_components),
             "selected score components inconsistent with the partition")
    reference = selection.get("raw_reference")
    _require(isinstance(reference, dict), "a missing independent raw reference")
    reference_hash = _partition(reference.get("partition_cuts"), n)
    tolerance = result.provenance.get("policy", {}).get("fusion_tol")
    _require(_finite_scalar(tolerance, nonnegative=True) and
             tuple(reference["partition_cuts"]) == extract_blocks(result.raw_phi[chain.order], tolerance),
             "a raw-reference partition inconsistent with the raw state")
    reference_centers = np.asarray(reference.get("refit_centers"))
    _require(reference.get("chain_sha256") == chain.fingerprint and
             reference.get("model_sha256") == result.provenance.get("model_sha256") and
             reference.get("partition_sha256") == reference_hash and
             reference.get("raw_phi_sha256") == hashlib.sha256(result.raw_phi[chain.order].tobytes()).hexdigest() and
             _finite_scalar(reference.get("lambda"), nonnegative=True) and
             reference["lambda"] == result.raw_reference_lambda and
             reference.get("clonal_label_rule") == CLONAL_LABEL_RULE and
             _finite_scalar(reference.get("refit_score")) and
             _finite_scalar(reference.get("refit_gap"), nonnegative=True) and
             reference_centers.shape == (len(reference["partition_cuts"]) - 1,) and
             np.all(np.isfinite(reference_centers)) and
             np.all((reference_centers >= 0) & (reference_centers <= 1)),
             "a mismatched independent raw reference")
    _require(reference.get("designated_clonal_block") ==
             public_cluster_order(reference_centers, reference["partition_cuts"])[0],
             "an incorrect raw-reference clonal designation")
    seed = selection.get("seed_origin")
    _require(seed in ("production_path", "adjacent_ward_pilot", "adjacent_ward_selected_raw"),
             "unknown candidate seed provenance")
    parent = selection.get("raw_parent")
    _require((parent is None if seed == "adjacent_ward_pilot" else _json_text(parent) == _json_text(reference)),
             "inconsistent raw-parent lineage")
    if family == "direct_chain_partition":
        _require(result.selected_lambda is None and selection.get("selected_raw_certificate") is None and
                 selection.get("selected_partition_certified") is False and reference_hash != fingerprint,
                 "a direct partition carrying an inherited raw certificate")
    else:
        _require(result.selected_lambda == result.raw_reference_lambda and
                 selection.get("selected_partition_certified") is True and reference_hash == fingerprint and
                 _json_text(selection.get("selected_raw_certificate")) == _json_text(diagnostics) and
                 np.array_equal(reference_centers, centers[block_labels]) and
                 result.selection_score == reference["refit_score"],
                 "a fusion partition without its exact raw certificate and refit")
    _require(result.search_status in ("complete", "incomplete") and
             result.search_status == result.search_diagnostics.get("search_status") and
             result.search_diagnostics.get("selected_refit_gap") == selection["refit_gap"],
             "inconsistent selected search status")


def _result_receipt(result):
    ids = result.input_identifiers
    return {"schema": "clipp1d.run.v6", "status": "success",
               "search_status": result.search_status,
               "input_identifiers": ids, "provenance": result.provenance,
               "selected_lambda": result.selected_lambda, "selection_score": result.selection_score,
               "candidate_provenance": result.candidate_provenance,
               "raw_reference_lambda": result.raw_reference_lambda,
               "raw_fields_scope": "independent qualified fusion reference, not a direct-partition certificate",
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
               "clonality": clonality_summary(result.cluster_centers, result.cluster_labels),
               "table_sha256": {}}


def write_result(result, outdir):
    _validate_result(result)
    # Validate every receipt field before any table is created. Keep the
    # normalized JSON snapshot so only known-serializable hashes are added
    # after publication of the tables.
    receipt = json.loads(_json_text(_result_receipt(result)))
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
                         "raw_reference_ccf", "refitted_ccf", "cluster_label"])
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
    receipt["table_sha256"] = {n: hashlib.sha256((outdir / n).read_bytes()).hexdigest() for n in names[:3]}
    write_json(outdir / "run.json", receipt)
