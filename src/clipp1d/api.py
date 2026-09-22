"""Single high-level orchestration entry point."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy

from . import __version__
from .chain import build_chain
from .io import read_tumor
from .model import compile_model, posterior_multiplicity
from .policy import Policy
from .partitions import propose_partitions
from .report import write_json, write_result
from .scalar import compute_pilot
from .selection import select_fit
from .types import FitResult

UPSTREAM_COMMIT = "77525a6875e698f6834cb73e6d2a1c27af2af8bf"


def _floating_format(dtype):
    """Describe the runtime format without equating storage and precision bits."""
    info = np.finfo(dtype)
    return {"dtype": np.dtype(dtype).name,
            "storage_bits": int(np.dtype(dtype).itemsize * 8),
            "nmant": int(info.nmant), "significand_bits": int(info.nmant + 1),
            "exponent_bits": int(info.iexp), "minexp": int(info.minexp),
            "maxexp": int(info.maxexp), "machep": int(info.machep),
            "eps": str(info.eps), "epsneg": str(info.epsneg),
            "smallest_normal": str(info.smallest_normal), "max_finite": str(info.max)}


def source_provenance():
    package = Path(__file__).resolve().parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(f"{name}\0{value}\n".encode())
    commit = None
    root = package.parent.parent
    if (root / ".git").is_dir():
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if proc.returncode == 0:
            commit = proc.stdout.strip()
    return {"package_version": __version__, "upstream_commit": UPSTREAM_COMMIT,
            "source_commit": commit, "source_sha256": digest.hexdigest(), "source_files": hashes,
            "python": platform.python_version(), "interpreter": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__, "backend": "cpu", "dtype": "float64",
            "platform": platform.platform(), "machine": platform.machine(),
            "floating_point_formats": {"float64": _floating_format(np.float64),
                                       "longdouble": _floating_format(np.longdouble)},
            "precision_scope": ("Runtime format metadata for this process; input/output use float64 and "
                                "selected internal message/dual arithmetic uses NumPy longdouble. "
                                "Storage bits do not establish significand precision. "
                                "This receipt does not qualify other platforms.")}


def fit(input_file, outdir=None, *, max_major_cn=4, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    destination = Path(outdir) if outdir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError("Output directory must be empty; existing evidence is never overwritten")
    started = perf_counter()
    provenance = {**source_provenance(),
                  "started_utc": datetime.now(timezone.utc).isoformat()}
    data = None
    try:
        policy = Policy(max_major_cn=max_major_cn)
        provenance["policy"] = asdict(policy)
        provenance["policy_sha256"] = hashlib.sha256(json.dumps(asdict(policy), sort_keys=True).encode()).hexdigest()
        data = read_tumor(input_file, policy)
        provenance["input_sha256"] = data.input_sha256
        model = compile_model(data, policy)
        model_digest = hashlib.sha256(json.dumps(model.mutation_ids).encode())
        for name in ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid"):
            model_digest.update(name.encode())
            model_digest.update(getattr(model, name).tobytes())
        provenance["model_sha256"] = model_digest.hexdigest()
        pilot_start = perf_counter()
        pilot = compute_pilot(model, policy)
        pilot_seconds = perf_counter() - pilot_start
        chain = build_chain(pilot, model.mutation_ids, policy)
        lam, raw, refit, search = select_fit(model, chain, pilot, policy)
        raw_lambda, raw_refit = lam, refit
        refit, proposals = propose_partitions(model.subset(chain.order), pilot.phi[chain.order],
                                             raw.x, raw_refit, policy)
        direct = refit.cuts != raw_refit.cuts
        lam = None if direct else raw_lambda
        def partition_hash(cuts):
            return hashlib.sha256(np.asarray(cuts, dtype=np.int64).tobytes()).hexdigest()
        reference = {"lambda": raw_lambda, "partition_sha256": partition_hash(raw_refit.cuts),
                     "partition_cuts": raw_refit.cuts, "refit_score": raw_refit.score,
                     "refit_centers": raw_refit.centers, "refit_gap": raw_refit.gap,
                     "designated_clonal_block": raw_refit.designated_clonal_block,
                     "raw_phi_sha256": hashlib.sha256(raw.x.tobytes()).hexdigest(),
                     "chain_sha256": chain.fingerprint,
                     "model_sha256": provenance["model_sha256"]}
        lineage = proposals["selected_seed_origin"]
        selection = {"candidate_family": "direct_chain_partition" if direct else "fusion_path",
                     "origin": proposals["selected_origin"],
                     "raw_parent": reference if "selected_raw" in lineage else None,
                     "refit_qualified": True, "refit_gap": refit.gap,
                     "partition_sha256": partition_hash(refit.cuts),
                     "chain_sha256": chain.fingerprint,
                     "model_sha256": provenance["model_sha256"],
                     "selected_raw_certificate": None if direct else dict(raw.diagnostics),
                     "selected_partition_certified": not direct,
                     "certification_scope": ("qualified likelihood refit only" if direct else
                                             "raw branch stationarity; final centers separately refitted"),
                     "raw_reference": reference,
                     "global_optimality_proven": False}
        search["direct_proposals"] = proposals
        search["search_status"] = ("complete" if search["path_search_complete"] and
                                   proposals["search_complete"] else "incomplete")
        search["selected_at_upper_boundary"] = not direct and search["selected_at_upper_boundary"]
        # Raw arrays and diagnostics always describe the independent fusion reference.
        phi_chain = np.empty(len(model))
        labels_chain = np.empty(len(model), dtype=int)
        public_order = [refit.designated_clonal_block] + sorted(
            (i for i in range(len(refit.centers)) if i != refit.designated_clonal_block),
            key=lambda i: (-refit.centers[i], refit.cuts[i]))
        block_label = {block: label for label, block in enumerate(public_order)}
        for block, (a, b) in enumerate(zip(refit.cuts[:-1], refit.cuts[1:])):
            phi_chain[a:b] = refit.centers[block]
            labels_chain[a:b] = block_label[block]
        refitted = phi_chain[chain.inverse_order]
        calls = np.argmax(posterior_multiplicity(model, refitted), axis=1) + 1
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=perf_counter() - started, pilot_seconds=pilot_seconds)
        search["selected_refit_gap"] = refit.gap
        result = FitResult(
            {"tumor_id": data.tumor_id, "sample_id": data.sample_id,
             "mutation_ids": [m.mutation_id for m in data.mutations]},
            np.array([m.exclusion is None for m in data.mutations]),
            tuple(m.exclusion for m in data.mutations), pilot, chain, lam,
            raw.x[chain.inverse_order].copy(), dict(raw.diagnostics), refit.cuts,
            refitted, refit.centers[public_order], labels_chain[chain.inverse_order],
            refit.designated_clonal_block, calls, refit.score, refit.score_components,
            provenance, search, raw_objective=raw.objective, raw_witness_index=raw.witness,
            raw_witness_mutation_id=model.mutation_ids[chain.order[raw.witness]],
            search_status=search["search_status"], candidate_provenance=selection,
            raw_reference_lambda=raw_lambda)
        if destination is not None:
            write_result(result, destination)
        return result
    except Exception as exc:
        if destination is not None and not (destination / "run.json").exists():
            write_json(destination / "run.json", {"schema": "clipp1d.run.v4", "status": "failure",
                       "search_status": "not_completed",
                       "error_type": type(exc).__name__, "message": str(exc), "provenance": provenance,
                       "diagnostics": getattr(exc, "diagnostics", {}),
                       "exclusions": {m.mutation_id: m.exclusion for m in data.mutations if m.exclusion}
                       if data is not None else {}})
        raise
