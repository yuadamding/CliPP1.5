"""Bounded scalar search with interval lower bounds and explicit qualification.

Candidate-wise binomial maxima bound the mixture. On intervals without a
clipping kink, f'' = E(curvature) - Var(score) >= -range(score)**2/4 supplies
a second, tighter Taylor lower bound. Both are conservative in float64 with
an arithmetic margin; these are numerical bounds, not interval-arithmetic proofs.
"""

import heapq
import hashlib
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .model import clipping_breakpoints, evaluate
from .policy import Policy
from .types import NumericalQualificationError, PilotResult, ScalarResult


def interval_lower_bound(model, left, right):
    p_left = np.clip(model.slope * left, model.eps, 1 - model.eps)
    p_right = np.clip(model.slope * right, model.eps, 1 - model.eps)
    empirical = (model.alt / (model.alt + model.ref))[:, None]
    p_best = np.clip(empirical, p_left, p_right)
    logits = (model.alt[:, None] * np.log(p_best) +
              model.ref[:, None] * np.log1p(-p_best) + model.log_prior)
    independent = float(-np.sum(logsumexp(logits, axis=1)))
    mid = left + (right - left) / 2
    terms = evaluate(model, np.full(len(model), mid), derivatives=True)
    # Only use the smooth bound if no breakpoint lies strictly inside the interval.
    slopes = np.where(model.valid, model.slope, np.nan)
    low_kinks, high_kinks = model.eps / slopes, (1 - model.eps) / slopes
    crossing = np.any(((low_kinks > left) & (low_kinks < right)) |
                      ((high_kinks > left) & (high_kinks < right)))
    bound = independent
    if not crossing:
        mass = model.slope * mid
        moving = (mass > model.eps) & (mass < 1 - model.eps)
        s = np.where(moving, model.slope, 0)
        score_left = s * (model.alt[:, None] / p_left - model.ref[:, None] / (1 - p_left))
        score_right = s * (model.alt[:, None] / p_right - model.ref[:, None] / (1 - p_right))
        spread = (np.max(np.where(model.valid, score_left, -np.inf), axis=1) -
                  np.min(np.where(model.valid, score_right, np.inf), axis=1))
        # A singleton candidate has identically zero score variance.
        spread = np.where(model.valid.sum(axis=1) == 1, 0, spread)
        negative_curvature = float(np.sum(spread**2 / 4))
        radius = (right - left) / 2
        smooth = (float(np.sum(terms.loss)) - abs(float(np.sum(terms.gradient))) * radius -
                  0.5 * negative_curvature * radius**2)
        bound = max(bound, smooth)
    margin = 128 * np.finfo(float).eps * (1 + abs(bound) + float(np.sum(terms.loss)))
    return bound - margin


def scalar_key(model, index):
    """Bind a cached scalar result to its exact likelihood and original domain."""
    digest = hashlib.sha256(float(model.eps).hex().encode())
    for name in ("alt", "ref", "lower", "upper", "slope", "log_prior", "valid"):
        digest.update(np.asarray(getattr(model, name)[index]).tobytes())
    return digest.hexdigest()


def minimize_block(model, policy=Policy()):
    lower, upper = float(np.max(model.lower)), float(np.min(model.upper))
    if lower > upper:
        raise ValueError("Block has no common feasible CCF")
    if np.all(model.valid.sum(axis=1) == 1) and np.all(model.slope[:, 0] == model.slope[0, 0]):
        slope = float(model.slope[0, 0])
        alt, ref = math.fsum(model.alt), math.fsum(model.ref)
        p_lower = float(np.clip(slope * lower, model.eps, 1 - model.eps))
        p_upper = float(np.clip(slope * upper, model.eps, 1 - model.eps))
        empirical = alt / (alt + ref) if alt + ref else p_lower
        if p_lower == p_upper or empirical <= p_lower:
            point = lower
        else:
            point = float(np.clip(min(empirical, 1 - model.eps) / slope, lower, upper))
        value = float(np.sum(evaluate(model, np.full(len(model), point)).loss))
        margin = 128 * np.finfo(float).eps * (1 + abs(value))
        lower_bound = value - margin
        gap = value - lower_bound
        return ScalarResult(point, value, lower_bound, gap,
                            gap <= policy.scalar_atol + policy.scalar_rtol * abs(value),
                            evaluations=1, method="same_slope_single_candidate_exact")
    return _minimize_general(model, policy)


def _minimize_general(model, policy=Policy()):
    lower, upper = float(np.max(model.lower)), float(np.min(model.upper))
    if lower > upper:
        raise ValueError("Block has no common feasible CCF")

    evaluations = 0
    bound_evaluations = 0

    def loss(t):
        nonlocal evaluations
        evaluations += 1
        return float(np.sum(evaluate(model, np.full(len(model), t)).loss))

    def bound(a, b):
        nonlocal bound_evaluations
        bound_evaluations += 1
        return interval_lower_bound(model, a, b)

    if lower == upper:
        value = loss(lower)
        return ScalarResult(lower, value, value, 0, True, evaluations=evaluations)
    kinks = clipping_breakpoints(model)
    kinks = kinks[(kinks >= lower) & (kinks <= upper)]
    modes = ((model.alt / (model.alt + model.ref))[:, None] /
             np.where(model.valid, model.slope, np.nan))
    modes = np.unique(np.clip(modes[np.isfinite(modes)], lower, upper))
    if modes.size > 65:
        modes = np.quantile(modes, np.linspace(0, 1, 65))
    # A bounded initial grid: many distinct slopes must not force one complete
    # block evaluation per clipping point before adaptive refinement starts.
    extreme_kinks = kinks[[0, -1]] if kinks.size else []
    seeds = np.unique(np.r_[np.linspace(lower, upper, 33), modes, extreme_kinks])
    values = np.array([loss(t) for t in seeds])
    wells = [(float(v), float(t)) for v, t in zip(values, seeds)]
    for i in range(1, len(seeds) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            result = minimize_scalar(loss, bounds=(seeds[i - 1], seeds[i + 1]), method="bounded",
                                     options={"xatol": 1e-14, "maxiter": 150})
            wells.append((float(result.fun), float(result.x)))
    best_loss, best_x = min(wells)
    # Deterministic smallest coordinate for arithmetic ties, not broad tolerance ties.
    tie = 16 * np.finfo(float).eps * (1 + abs(best_loss))
    best_x = min(x for v, x in wells if v <= best_loss + tie)
    best_loss = loss(best_x)
    heap = [(bound(a, b), float(a), float(b))
            for a, b in zip(seeds[:-1], seeds[1:])]
    heapq.heapify(heap)
    subdivisions = 0
    while heap and best_loss - heap[0][0] > policy.scalar_atol + policy.scalar_rtol * abs(best_loss):
        if subdivisions >= policy.scalar_max_intervals:
            break
        lower_value, a, b = heapq.heappop(heap)
        mid = a + (b - a) / 2
        first, last = np.searchsorted(kinks, a, side="right"), np.searchsorted(kinks, b, side="left")
        if first < last:
            index = int(np.clip(np.searchsorted(kinks, mid), first, last - 1))
            if index > first and abs(kinks[index - 1] - mid) < abs(kinks[index] - mid):
                index -= 1
            mid = float(kinks[index])
        if mid == a or mid == b:
            heapq.heappush(heap, (lower_value, a, b))
            break
        value = loss(mid)
        if value < best_loss:
            best_x, best_loss = mid, value
        for lo, hi in ((a, mid), (mid, b)):
            heapq.heappush(heap, (bound(lo, hi), lo, hi))
        subdivisions += 1
    lb = min(best_loss, heap[0][0]) if heap else best_loss
    gap = max(0.0, best_loss - lb)
    alternatives = []
    # Retain local wells, not arbitrary points from a grid.
    for value, x in sorted(wells):
        if abs(x - best_x) > 1e-3 and value <= best_loss + 2 and all(abs(x - a) > 1e-3 for a in alternatives):
            delta = min(1e-5, (upper - lower) * 1e-4)
            if value <= loss(max(lower, x - delta)) and value <= loss(min(upper, x + delta)):
                alternatives.append(x)
        if len(alternatives) == 2:
            break
    return ScalarResult(best_x, best_loss, lb, gap,
                        gap <= policy.scalar_atol + policy.scalar_rtol * abs(best_loss),
                        tuple(alternatives), subdivisions, evaluations, bound_evaluations)


def compute_pilot(model, policy=Policy()):
    results = [minimize_block(model.subset([i]), policy) for i in range(len(model))]
    unresolved = [model.mutation_ids[i] for i, r in enumerate(results) if not r.qualified]
    if unresolved:
        raise NumericalQualificationError("Scalar pilot search did not qualify", unresolved=unresolved,
                                           gaps=[r.optimality_gap for r in results])
    phi = np.array([r.argmin for r in results])
    return PilotResult(phi, np.array([r.lower_bound for r in results]),
                       np.array([r.attained_loss for r in results]),
                       evaluate(model, phi, derivatives=True).curvature,
                       np.array([r.alternatives[0] if r.alternatives else r.argmin for r in results]),
                       np.array([r.optimality_gap for r in results]), tuple(results),
                       tuple(scalar_key(model, i) for i in range(len(model))), model.mutation_ids)
