"""Deterministic penalty path, contiguous refits, and the compatible score."""

import hashlib
import logging
import math
from time import perf_counter

import numpy as np

from .chain import extract_blocks
from .clonal import fit_fixed_lambda
from .model import evaluate
from .policy import Policy
from .scalar import minimize_block
from .types import ClonalConstraintInfeasibleError, NumericalQualificationError, PartitionRefit

logger = logging.getLogger(__name__)


def partition_score(loss, sizes):
    sizes = np.asarray(sizes)
    if sizes.ndim != 1 or not sizes.size or np.any(sizes <= 0) or np.any(sizes != np.rint(sizes)):
        raise ValueError("Expected occupied integer block sizes")
    n, k = int(np.sum(sizes)), sizes.size
    log_mass = math.fsum([math.lgamma(k), -math.lgamma(n + k),
                          *(math.lgamma(int(s) + 1) for s in sizes), math.lgamma(k + 1)])
    components = {"twice_refit_loss": float(2 * loss), "center_penalty": float(k * np.log(n)),
                  "partition_log_mass": log_mass, "partition_penalty": -1.4 * log_mass}
    return math.fsum((components["twice_refit_loss"], components["center_penalty"],
                      components["partition_penalty"])), components


def refit_partition(model, cuts, policy=Policy()):
    """The model is in chain order. Profile the identity of the clonal block."""
    cuts = tuple(cuts)
    if (not cuts or cuts[0] != 0 or cuts[-1] != len(model) or
            any(not isinstance(v, (int, np.integer)) for v in cuts) or
            any(a >= b for a, b in zip(cuts[:-1], cuts[1:]))):
        raise ValueError("Cuts must partition the whole chain into nonempty intervals")
    centers, losses, gaps, at_one = [], [], [], []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = model.subset(np.arange(start, stop))
        scalar = minimize_block(block, policy)
        if not scalar.qualified:
            raise NumericalQualificationError("Block refit did not qualify", block=[start, stop],
                                               scalar_gap=scalar.optimality_gap)
        centers.append(scalar.argmin)
        losses.append(scalar.attained_loss)
        gaps.append(scalar.optimality_gap)
        at_one.append(float(np.sum(evaluate(block, np.ones(len(block))).loss))
                      if np.all(block.upper == 1) else np.inf)
    costs = np.asarray(at_one) - losses
    designated = int(np.argmin(costs))
    if not np.isfinite(costs[designated]):
        raise ClonalConstraintInfeasibleError("Partition has no block eligible for CCF one")
    centers = np.array(centers)
    centers[designated] = 1.0
    loss = math.fsum(at_one[i] if i == designated else losses[i] for i in range(len(losses)))
    # All scalar lower bounds also bound the profiled unknown-clonal-block refit.
    scalar_lower = np.asarray(losses) - gaps
    lower_profile = math.fsum(scalar_lower) + float(np.min(np.asarray(at_one) - scalar_lower))
    gap = max(0, loss - lower_profile)
    score, components = partition_score(loss, np.diff(cuts))
    return PartitionRefit(cuts, centers, designated, loss, gap, score, components)


def penalty_reference(model, chain, pilot):
    if len(model) == 1:
        return 0.0
    h, p = np.maximum(pilot.curvature[chain.order], 1.0), pilot.phi[chain.order]
    lower, upper = model.lower[chain.order], model.upper[chain.order]
    beta = float(np.clip(np.average(p, weights=h), np.max(lower), np.min(upper)))
    a = h * (beta - p)
    balance = -float(np.sum(a))
    if balance > 0 and beta == np.min(upper):
        active = upper == beta
    elif balance < 0 and beta == np.max(lower):
        active = lower == beta
    else:
        active = np.ones(len(model), dtype=bool)  # floating point sum correction only
    a[active] += balance / np.count_nonzero(active)
    reference = float(np.max(np.abs(np.cumsum(a)[:-1]) / chain.weights))
    return reference if reference > 1e-12 else 1e-3


def select_fit(model, chain, pilot, policy=Policy()):
    ordered = model.subset(chain.order)
    if not np.any(ordered.upper == 1):
        raise ClonalConstraintInfeasibleError("No retained mutation is clonal-eligible")
    reference = penalty_reference(model, chain, pilot)
    path = [0.0] if len(model) == 1 else [0.0] + [reference * 2.0**k for k in
                                               range(policy.path_min_exponent, policy.path_max_exponent + 1)]
    records, best, warm = [], None, None
    # Bounded by the fixed path length; store hashes, never one full state per path/witness.
    refit_cache = {}
    extensions = 0
    index = 0
    while index < len(path):
        lam = path[index]
        started = perf_counter()
        record = {"lambda": lam}
        try:
            raw = fit_fixed_lambda(model, chain, pilot, lam, warm, policy)
            record["solver_seconds"] = perf_counter() - started
            warm = raw.x.copy()
            cuts = extract_blocks(raw.x, policy.fusion_tol)
            signature = hashlib.sha256(np.array(cuts, dtype=np.int64).tobytes()).hexdigest()
            refit_started = perf_counter()
            if signature not in refit_cache:
                # A fixed-size cache of scalar summaries is unnecessary: cache only
                # the immediately preceding refit, keeping even long paths O(M).
                refit_cache.clear()
                refit_cache[signature] = refit_partition(ordered, cuts, policy)
            refit = refit_cache[signature]
            record.update(raw.diagnostics)
            record.update(status="qualified", score=refit.score, blocks=len(cuts) - 1,
                          partition_sha256=signature, refit_gap=refit.gap,
                          refit_seconds=perf_counter() - refit_started)
            candidate = (refit.score, len(cuts) - 1, lam, cuts)
            if best is None or candidate < best[0]:
                best = (candidate, lam, raw, refit)
        except NumericalQualificationError as exc:
            record.update(status="unresolved", message=str(exc), **exc.diagnostics)
        record["elapsed_seconds"] = perf_counter() - started
        records.append(record)
        logger.info("lambda=%g status=%s blocks=%s", lam, record["status"], record.get("blocks", "?"))
        index += 1
        if (index == len(path) and len(model) > 1 and best is not None and
                best[1] == path[-1] and extensions < policy.path_extensions):
            path.append(path[-1] * 2)
            extensions += 1
    if best is None:
        raise NumericalQualificationError("No qualified chain partition was found", path=records)
    _, lam, raw, refit = best
    return lam, raw, refit, {"lambda_reference": reference, "path": records,
                            "path_candidates": len(records), "extensions": extensions,
                            "extension_limit": policy.path_extensions,
                            "selected_at_upper_boundary": len(model) > 1 and lam == path[-1],
                            "path_search_complete": all(r["status"] == "qualified" and
                                                        r.get("witness_search_complete", False) for r in records)}
