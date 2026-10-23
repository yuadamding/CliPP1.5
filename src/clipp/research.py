"""Opt-in CPU joint-fitting experiment from a verified, saved production bank.

Usage: python -m clipp.research refine RUN --output NEW_DIRECTORY
       python -m clipp.research verify NEW_DIRECTORY

This entry point never changes production defaults or overwrites an existing
run. Its posterior MAP labels are experimental mixture summaries, distinct
from the immutable chain proposals retained in the output.
"""

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logsumexp

from ._io import partition_labels, read_table, write_json
from .environment import fingerprint
from .joint_fitting import EXPERIMENTAL_VERSION, fit_joint, joint_diagnostics, reduced_candidates
from .model import MultiplicityModel
from .native import sha256
from .output import write_posterior_diagnostics
from .verify import _close, _kernel, _require, verify_run


SCHEMA = 1
_TOLERANCES = {"likelihood": 1e-9, "center": 1e-7, "weight": 1e-8}
_SOURCE_FILES = {
    "manifest.json": "manifest.json", "COMPLETE.json": "COMPLETE.json",
    "retained.tsv": "preprocess_result/retained.tsv",
    "chain_order.txt": "preliminary_result/chain_order.txt",
    "candidates.jsonl.gz": "evidence/candidates.jsonl.gz",
}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        _require(header.get("candidate_evidence_schema") == 1, "experimental source bank schema")
        count = 0
        for line in stream:
            count += 1
            yield json.loads(line)
        _require(count == header["count"], "experimental source bank count")


def starting_bank(path):
    """At most three distinct center sets per actual q, using bounded memory.

Each of three streaming passes selects one role. The complete source bank is
preserved, including candidates deduplicated here, so their provenance is not
lost. Native and conditional roles are selected from remaining distinct sets.
"""
    selected, used = {}, {}
    for role in ("best_bic", "native", "best_conditional"):
        best = {}
        for row in _records(path):
            if row.get("status") != "scored" or not row.get("publication_eligible"):
                continue
            if role == "native" and row.get("candidate_kind") != "native":
                continue
            parameters = json.loads(row["partition_parameters"])
            q = len(parameters["centers"])
            identity = tuple(sorted(parameters["centers"]))
            if identity in used.get(q, set()):
                continue
            key = ((-row["conditional_log_likelihood"] if role == "best_conditional" else row["bic"]),
                   row["bic"], row["replicate"], row["candidate_id"])
            if q not in best or key < best[q][0]:
                best[q] = (key, row, identity)
        for q, (_, row, identity) in sorted(best.items()):
            used.setdefault(q, set()).add(identity)
            selected.setdefault(q, []).append({"role": role, "actual_components": q, "source_candidate": row})
    bank = [record for q in sorted(selected) for record in selected[q]]
    for index, record in enumerate(bank):
        record["start_id"] = index
    _require(bool(bank), "no eligible experimental starting states")
    return bank


def _groups(result, input_groups):
    return [sorted(index for component in group for index in input_groups[component])
            for group in result["component_groups"]]


def _mapping(groups, count):
    mapping = np.full(count, -1, dtype=int)
    for component, group in enumerate(groups):
        mapping[group] = component
    return mapping


def _model(data, purity):
    return MultiplicityModel(*(data[name].to_numpy() for name in (
        "alt_count", "depth", "major_cn", "total_cn"
    )), purity)


def _snapshot(source, target):
    target.mkdir()
    for destination, relative in _SOURCE_FILES.items():
        path = source / relative
        digest = sha256(path)
        shutil.copyfile(path, target / destination)
        _require(sha256(path) == digest == sha256(target / destination), "source changed while snapshotting")


def refine_run(directory, output, *, max_iterations=500, reductions=True):
    """Refine a verified schema-5 bank on CPU and publish a distinct experiment.

The source's manifest, completion, canonical observations, order and complete
bank are copied. Other native/input artifacts remain in the original verified
run; verification of this experiment replays its mixture arithmetic, not those
original native solver certificates. An all-failed bank has no COMPLETE marker
and publishes no posterior winner or substituted legacy result.
"""
    source, output = Path(directory).resolve(strict=True), Path(output).absolute()
    if type(max_iterations) is not int or max_iterations < 1 or type(reductions) is not bool:
        raise ValueError("Expected a positive iteration count and Boolean reduction flag")
    if output.resolve().is_relative_to(source):
        raise ValueError("Research output cannot alter the verified source run")
    if not (source / "manifest.json").is_file():
        raise ValueError("Joint refinement requires historical schema-5 evidence; two-table outputs have no candidate bank")
    source_hash = sha256(source / "manifest.json")
    source_verification = verify_run(source)
    source_manifest = json.loads((source / "manifest.json").read_text())
    _require(source_manifest.get("output_schema_version") == 5, "joint experiment requires schema-5 evidence")
    _require(source_manifest["candidate_evidence"]["path"] == _SOURCE_FILES["candidates.jsonl.gz"],
             "source candidate-bank path")
    if output.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Research output must be outside the installed package")
    output.mkdir(parents=True, exist_ok=False)
    _snapshot(source, output / "source")
    _require(sha256(output / "source/manifest.json") == source_hash, "source manifest changed during verification")
    data = read_table(output / "source/retained.tsv")
    order = np.loadtxt(output / "source/chain_order.txt", dtype=int, ndmin=1)
    model = _model(data, source_manifest["purity"])
    starts = starting_bank(output / "source/candidates.jsonl.gz")
    write_json(output / "starts.json", starts)
    attempts = []

    def save(result, start, input_groups, *, parent=None):
        record = {
            "attempt_id": len(attempts), "start_id": start["start_id"], "parent_attempt_id": parent,
            "proposal_component_groups": _groups(result, input_groups), "fit": _plain(result),
        }
        attempts.append(record)
        # Persist each completed local attempt before starting further work.
        write_json(output / "attempts.json", attempts)
        return record

    for start in starts:
        parameters = json.loads(start["source_candidate"]["partition_parameters"])
        result = fit_joint(model, parameters["centers"], parameters["weights"], max_iterations=max_iterations)
        parent = save(result, start, [[i] for i in range(len(parameters["centers"]))])
        if result["converged"] and reductions:
            for lower in reduced_candidates(model, result, max_iterations=max_iterations):
                first, last = lower["reduction_proposal"]["merged_parent_components"]
                groups = parent["proposal_component_groups"]
                merged = groups[:first] + [groups[first] + groups[last]] + groups[last + 1:]
                save(lower, start, merged, parent=parent["attempt_id"])
    qualified = [record for record in attempts if record["fit"]["converged"]]
    winner = min(qualified, key=lambda record: (
        record["fit"]["bic"], record["fit"]["effective_component_count"], record["attempt_id"]
    )) if qualified else None
    if winner is not None:
        parameters = json.loads(starts[winner["start_id"]]["source_candidate"]["partition_parameters"])
        labels = partition_labels(order, parameters["cuts"], parameters["block_labels"])
        mapping = _mapping(winner["proposal_component_groups"], len(parameters["centers"]))
        write_posterior_diagnostics(
            model, data[["chromosome_index", "position", "mutation_id", "original_row"]],
            {"centers": np.asarray(winner["fit"]["centers"]),
             "cluster_weights": np.asarray(winner["fit"]["cluster_weights"]), "labels": labels},
            output / "posterior", marginal_multiplicity=True, chain_to_component=mapping,
        )
    manifest = {
        "experimental_schema_version": SCHEMA, "experimental_version": EXPERIMENTAL_VERSION,
        "status": "qualified_joint_fit" if winner is not None else "no_qualified_joint_fit",
        "backend": "cpu", "precision": "float64", "purity": model.purity, "num_mutations": len(model),
        "configuration": {"max_iterations": max_iterations, "reductions": reductions, "tolerances": _TOLERANCES},
        "winner_attempt_id": None if winner is None else winner["attempt_id"],
        "qualified_attempts": len(qualified), "total_attempts": len(attempts),
        "source": {
            "run": str(source), "manifest_sha256": source_hash, "verification": source_verification,
            "inputs": source_manifest["inputs"],
            "canonical_input_sha256": source_manifest["canonical_input_sha256"],
            "chain_order_sha256": source_manifest["chain_order_sha256"],
            "candidate_evidence_sha256": sha256(output / "source/candidates.jsonl.gz"),
        },
        "environment": fingerprint(),
        "code_sha256": {
            str(path.relative_to(Path(__file__).parent)): sha256(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "label_definition": "posterior_map_cluster_index is final; proposal_chain_cluster_index is provenance",
        "scientific_scope": "local experimental fit, not global optimality or benchmark qualification",
        "artifacts": {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                      if path.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    verified = verify_experiment(output, require_complete=False)
    if winner is not None:
        write_json(output / "COMPLETE.json", {
            "status": "complete", "manifest_sha256": sha256(output / "manifest.json"),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
    return {"output": str(output), **verified}


def _independent_likelihood(data, purity, centers, weights):
    columns = np.column_stack([_kernel(data, cp, purity, posterior=False)[0] for cp in centers])
    log_weights = np.full(len(weights), -np.inf)
    np.log(weights, out=log_weights, where=np.asarray(weights) > 0)
    values = logsumexp(columns + log_weights, axis=1)
    _require(np.isfinite(values).all(), "experimental finite likelihood")
    return float(values.sum())


def verify_experiment(directory, *, require_complete=True):
    """Replay source identities, complete attempt selection and posterior arithmetic.

No optimizer is called. Likelihoods independently enumerate the full binomial
multiplicity support; pure gradient calculations qualify local stationarity.
The copied source receipt binds the earlier production verification rather
than claiming to rerun missing native artifacts from the original directory.
"""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    _require(manifest.get("experimental_schema_version") == SCHEMA
             and manifest.get("experimental_version") == EXPERIMENTAL_VERSION
             and manifest.get("backend") == "cpu" and manifest.get("precision") == "float64",
             "experimental contract")
    config = manifest["configuration"]
    _require(type(config["max_iterations"]) is int and config["max_iterations"] > 0
             and type(config["reductions"]) is bool and config["tolerances"] == _TOLERANCES,
             "experimental numerical contract")
    if require_complete:
        complete = json.loads((root / "COMPLETE.json").read_text())
        _require(complete["status"] == "complete"
                 and complete["manifest_sha256"] == sha256(root / "manifest.json"), "experimental completion")
    files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    _require(not any(path.is_symlink() for path in root.rglob("*")), "experimental symlinks")
    _require(files - {"manifest.json", "COMPLETE.json"} == set(manifest["artifacts"]), "experimental inventory")
    for name, digest in manifest["artifacts"].items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts, "experimental artifact path")
        _require(sha256(root / relative) == digest, "experimental checksum " + name)
    source = json.loads((root / "source/manifest.json").read_text())
    source_complete = json.loads((root / "source/COMPLETE.json").read_text())
    source_hash = sha256(root / "source/manifest.json")
    _require(source_complete["status"] == "complete" and source_complete["manifest_sha256"] == source_hash
             and manifest["source"]["manifest_sha256"] == source_hash
             and source["output_schema_version"] == 5, "experimental source completion/identity")
    _require(manifest["source"]["inputs"] == source["inputs"], "experimental input identities")
    for name, relative in _SOURCE_FILES.items():
        if name not in {"manifest.json", "COMPLETE.json"}:
            _require(sha256(root / "source" / name) == source["artifacts"][relative], "source snapshot " + name)
    data = read_table(root / "source/retained.tsv")
    purity = manifest["purity"]
    _require(purity == source["purity"] and len(data) == manifest["num_mutations"] == source["num_retained"],
             "experimental population")
    model = _model(data, purity)
    order = np.loadtxt(root / "source/chain_order.txt", dtype=int, ndmin=1)
    _require(np.array_equal(np.sort(order), np.arange(len(data))), "experimental chain order")
    starts = json.loads((root / "starts.json").read_text())
    _require(starts == starting_bank(root / "source/candidates.jsonl.gz"), "experimental bounded starting bank")
    attempts = json.loads((root / "attempts.json").read_text())
    _require(len(attempts) == manifest["total_attempts"] and bool(attempts), "experimental attempt count")
    base_ids, qualified, reduction_visits = [], [], {}
    for index, attempt in enumerate(attempts):
        _require(attempt["attempt_id"] == index and 0 <= attempt["start_id"] < len(starts), "attempt identity")
        fit = attempt["fit"]
        _require(fit["experimental_version"] == EXPERIMENTAL_VERSION
                 and fit["tolerances"] == _TOLERANCES
                 and fit["status"] in {"iteration_limit", "locally_converged", "optimization_failed",
                                       "nonmonotone_step", "reduction_failed", "final_verification_failed"},
                 "attempt numerical contract")
        start = starts[attempt["start_id"]]
        parameters = json.loads(start["source_candidate"]["partition_parameters"])
        parent_id = attempt["parent_attempt_id"]
        if parent_id is None:
            base_ids.append(attempt["start_id"])
            input_groups = [[i] for i in range(len(parameters["centers"]))]
            initial_centers, initial_weights = np.asarray(parameters["centers"]), np.asarray(parameters["weights"])
        else:
            _require(manifest["configuration"]["reductions"] and 0 <= parent_id < index, "reduction parent")
            parent = attempts[parent_id]
            _require(parent["start_id"] == attempt["start_id"] and parent["fit"]["converged"]
                     and parent["parent_attempt_id"] is None, "qualified same-start parent")
            left, right = fit["reduction_proposal"]["merged_parent_components"]
            pcp, pw = np.asarray(parent["fit"]["centers"]), np.asarray(parent["fit"]["cluster_weights"])
            _require(right == left + 1 and 0 <= left < right < len(pcp), "adjacent reduction indices")
            reduction_visits.setdefault(parent_id, []).append((left, right))
            _require(fit["reduction_proposal"]["parent_bic"] == parent["fit"]["bic"], "reduction score provenance")
            mass = pw[left] + pw[right]
            initial_centers = np.concatenate((pcp[:left], [(pcp[left] * pw[left] + pcp[right] * pw[right]) / mass], pcp[right + 1:]))
            initial_weights = np.concatenate((pw[:left], [mass], pw[right + 1:]))
            pg = parent["proposal_component_groups"]
            input_groups = pg[:left] + [pg[left] + pg[right]] + pg[right + 1:]
        _require(np.array_equal(fit["initial_centers"], initial_centers), "starting center provenance")
        _require(np.array_equal(fit["initial_weights"], initial_weights / initial_weights.sum()), "starting weight provenance")
        _require(attempt["proposal_component_groups"] == _groups(fit, input_groups), "proposal component mapping")
        centers, weights = np.asarray(fit["centers"]), np.asarray(fit["cluster_weights"])
        _require(centers.ndim == 1 and len(centers) > 0 and weights.shape == centers.shape
                 and np.isfinite(centers).all() and np.isfinite(weights).all()
                 and np.all((centers >= 0) & (centers <= purity)) and np.all(weights > 0)
                 and np.isclose(weights.sum(), 1, atol=5e-12, rtol=0)
                 and np.all(np.diff(centers) < 0), "effective mixture parameters")
        before_cp, before_w = np.asarray(fit["unreduced_centers"]), np.asarray(fit["unreduced_weights"])
        _require(before_cp.shape == before_w.shape == initial_centers.shape
                 and np.isfinite(before_cp).all() and np.isfinite(before_w).all()
                 and np.all((before_cp >= 0) & (before_cp <= purity)) and np.all(before_w >= 0)
                 and np.isclose(before_w.sum(), 1, atol=5e-12, rtol=0), "unreduced parameters")
        groups = [[int(i) for i in np.flatnonzero((before_cp == cp) & (before_w > 0))] for cp in centers]
        _require(groups == fit["component_groups"] and all(groups)
                 and np.array_equal(weights, [before_w[group].sum() for group in groups])
                 and fit["discarded_zero_components"] == np.flatnonzero(before_w == 0).tolist()
                 and sorted(i for group in groups for i in group) == np.flatnonzero(before_w > 0).tolist(),
                 "exact reduction provenance")
        likelihood = _independent_likelihood(data, purity, centers, weights)
        _close(fit["log_likelihood"], likelihood, "joint likelihood")
        initial_ll = _independent_likelihood(data, purity, initial_centers, initial_weights / initial_weights.sum())
        _close(fit["initial_log_likelihood"], initial_ll, "initial joint likelihood")
        _require(fit["effective_component_count"] == len(centers)
                 and fit["initial_component_count"] == len(initial_centers), "effective component counts")
        _close(fit["bic"], -2 * likelihood + (2 * len(centers) - 1) * np.log(len(data)), "joint BIC")
        history = np.asarray([row["log_likelihood"] for row in fit["history"]])
        _require(len(history) > 0 and np.isfinite(history).all()
                 and np.all(np.diff(history) >= -64 * np.finfo(float).eps * np.maximum(1, np.abs(history[:-1]))),
                 "monotone accepted history")
        for i, row in enumerate(fit["history"]):
            if row["phase"] == "joint":
                _require(i >= 2 and row["iteration"] == i - 1, "joint history sequence")
                _close(row["relative_improvement"], (history[i] - history[i - 1]) / max(1.0, abs(history[i - 1])),
                       "relative improvement", atol=1e-15, rtol=1e-12)
            else:
                _require(row["iteration"] == 0 and row["phase"] == ("initial" if i == 0 else "initial_weights")
                         and i <= 1, "initial history sequence")
        _require(type(fit["iterations"]) is int and fit["history"][-1]["iteration"] <= fit["iterations"]
                 <= config["max_iterations"], "joint iteration count")
        _close(history[0], initial_ll, "initial history")
        _close(history[-1], likelihood, "final history")
        diagnostics = joint_diagnostics(model, centers, weights)
        for name in ("center_gradient", "center_stationarity", "center_stationarity_gap",
                     "weight_optimality_gap", "weight_active_score_gap", "posterior_expected_num_SNV",
                     "weight_expected_count_max_error"):
            _close(fit[name], diagnostics[name], "joint diagnostic " + name)
        _require(type(fit["converged"]) is bool and fit["converged"] == (fit["status"] == "locally_converged"),
                 "joint qualification status")
        if fit["converged"]:
            tolerances = fit["tolerances"]
            _require(likelihood >= initial_ll - 1e-8
                     and diagnostics["center_stationarity_gap"] <= tolerances["center"]
                     and diagnostics["weight_optimality_gap"] <= len(data) * tolerances["weight"]
                     and diagnostics["weight_active_score_gap"] <= tolerances["weight"]
                     and fit["history"][-1]["relative_improvement"] <= tolerances["likelihood"], "local qualification")
            qualified.append(attempt)
    _require(base_ids == list(range(len(starts))), "all declared starts attempted")
    for attempt in attempts:
        if attempt["parent_attempt_id"] is None:
            count = attempt["fit"]["effective_component_count"]
            expected = [(k, k + 1) for k in range(count - 1)] if (
                config["reductions"] and attempt["fit"]["converged"]
            ) else []
            _require(reduction_visits.get(attempt["attempt_id"], []) == expected,
                     "complete declared component-reduction bank")
    _require(len(qualified) == manifest["qualified_attempts"], "qualified attempt count")
    winner = min(qualified, key=lambda row: (row["fit"]["bic"], row["fit"]["effective_component_count"], row["attempt_id"])) if qualified else None
    _require(manifest["winner_attempt_id"] == (None if winner is None else winner["attempt_id"]), "joint winner selection")
    _require(manifest["status"] == ("qualified_joint_fit" if winner else "no_qualified_joint_fit"), "experiment status")
    if winner is not None:
        from ._verify_posterior import verify_posterior
        parameters = json.loads(starts[winner["start_id"]]["source_candidate"]["partition_parameters"])
        labels = partition_labels(order, parameters["cuts"], parameters["block_labels"])
        mapping = _mapping(winner["proposal_component_groups"], len(parameters["centers"]))
        verify_posterior(data, purity, labels, np.asarray(winner["fit"]["centers"]),
                         np.asarray(winner["fit"]["cluster_weights"]), root / "posterior",
                         chain_to_component=mapping, marginal_multiplicity=True)
    else:
        _require(not (root / "posterior").exists() and not (root / "COMPLETE.json").exists(), "failed bank has no published winner")
    return {"status": manifest["status"], "num_mutations": len(data), "total_attempts": len(attempts),
            "qualified_attempts": len(qualified), "winner_attempt_id": manifest["winner_attempt_id"],
            "bic": None if winner is None else winner["fit"]["bic"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    refine = sub.add_parser("refine", help="CPU experiment from a verified production bank")
    refine.add_argument("run")
    refine.add_argument("--output", required=True)
    refine.add_argument("--max-iterations", type=int, default=500)
    refine.add_argument("--no-reductions", action="store_true")
    verify = sub.add_parser("verify", help="independent replay without fitting")
    verify.add_argument("run")
    args = parser.parse_args(argv)
    result = refine_run(args.run, args.output, max_iterations=args.max_iterations, reductions=not args.no_reductions) if args.command == "refine" else verify_experiment(args.run)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "qualified_joint_fit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
