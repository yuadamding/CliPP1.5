"""Frozen adaptive adjacency and implicit first differences. No graph matrices."""

import hashlib

import numpy as np

from .policy import Policy
from .types import FrozenChain


def difference(x):
    return np.diff(x)


def adjoint(q):
    q = np.asarray(q)
    result = np.zeros(q.size + 1, dtype=np.float64)
    result[:-1] -= q
    result[1:] += q
    return result


def build_chain(pilot, mutation_ids, policy=Policy()):
    order = np.lexsort((np.array(mutation_ids), pilot.phi))
    inverse = np.argsort(order)
    gaps = np.diff(pilot.phi[order])
    positive = gaps[gaps > 0]
    floor = max(1e-8, 0.1 * float(np.median(positive))) if positive.size else 1e-8
    weights = 1 / np.maximum(gaps, floor)
    if weights.size:
        weights /= np.mean(weights)
    digest = hashlib.sha256(policy.weight_rule.encode())
    for i in order:
        value = mutation_ids[i].encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    digest.update(pilot.phi[order].tobytes())
    digest.update(weights.tobytes())
    return FrozenChain(order, inverse, weights, floor, digest.hexdigest())


def extract_blocks(x, tolerance):
    """Contiguous ranges; no transitive tolerance chaining or nonadjacent merge."""
    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or tolerance < 0:
        raise ValueError("Expected a nonempty finite vector and nonnegative tolerance")
    cuts, low, high = [0], x[0], x[0]
    for i in range(1, len(x)):
        new_low, new_high = min(low, x[i]), max(high, x[i])
        if (new_high - new_low > tolerance or abs(x[i] - x[i - 1]) > tolerance or
                (x[i] == 1) != (x[i - 1] == 1)):
            cuts.append(i)
            low = high = x[i]
        else:
            low, high = new_low, new_high
    return tuple(cuts + [len(x)])
