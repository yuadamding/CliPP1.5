"""Fixed-order adjacent Ward proposals; no likelihood qualification.

``values`` are coordinates in an already frozen chain, never re-sorted here.
Positive weights describe a local quadratic proposal geometry (for example,
CliPP1.5's expected-curvature surrogate). They need not be the true observed
Hessian, which can be negative for a multiplicity mixture. Every emitted
partition still requires independent likelihood refits and the original score.

Global weight scaling does not change exact-arithmetic merge decisions. We
normalize in longdouble, accumulate masses in that format, and update means
relative to the heavier child, avoiding subtraction of large raw second
moments. Longdouble is an arithmetic guard, not a certificate. Unsupported
arithmetic raises rather than silently changing the merge geometry.
"""

import heapq
import operator

import numpy as np


def adjacent_ward_cuts(values, weights, max_blocks=12):
    """Return ``(candidate_cuts, work)`` for K=min(max_blocks, n), ..., 1.

    Only adjacent intervals merge, including when values are nonmonotone.
    Arithmetic ties prefer the smallest left boundary and then right boundary.
    A lazy heap stores at most O(n) records over the entire n-1 merge run;
    each merge inserts at most two new records. Node state is O(n), with no
    pairwise-distance or interval-score matrix. Materialized candidate cuts
    contain O(min(max_blocks, n)**2) indices (constant for fixed max_blocks).
    """
    if isinstance(max_blocks, (bool, np.bool_)):
        raise ValueError("max_blocks must be a positive integer")
    try:
        max_blocks = operator.index(max_blocks)
    except TypeError as exc:
        raise ValueError("max_blocks must be a positive integer") from exc
    if max_blocks < 1:
        raise ValueError("max_blocks must be a positive integer")
    x = np.asarray(values, dtype=np.longdouble)
    h = np.asarray(weights, dtype=np.longdouble)
    if x.ndim != 1 or h.shape != x.shape or x.size == 0:
        raise ValueError("values and weights must be nonempty matching vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("values must be finite and weights finite and positive")
    n = len(x)
    weight_scale = np.max(h)
    h = h / weight_scale
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        raise FloatingPointError("Weight normalization is not representable")
    capacity = 2 * n - 1
    mass = np.zeros(capacity, dtype=np.longdouble)
    mean = np.zeros(capacity, dtype=np.longdouble)
    mass[:n], mean[:n] = h, x
    start = np.full(capacity, -1, dtype=np.intp)
    stop = np.full(capacity, -1, dtype=np.intp)
    previous = np.full(capacity, -1, dtype=np.intp)
    following = np.full(capacity, -1, dtype=np.intp)
    start[:n], stop[:n] = np.arange(n), np.arange(1, n + 1)
    previous[:n], following[:n] = np.arange(-1, n - 1), np.arange(1, n + 1)
    following[n - 1] = -1
    alive = np.zeros(capacity, dtype=bool)
    alive[:n] = True
    heap = []
    work = {"heap_pushes": 0, "heap_pops": 0, "stale_pops": 0,
            "max_heap_entries": 0, "max_live_blocks": n, "merges": 0,
            "node_capacity": capacity, "candidate_count": 0,
            "weight_scale": str(weight_scale),
            "arithmetic": np.dtype(np.longdouble).name,
            "proposal_only": True}

    def push(left, right):
        delta = mean[right] - mean[left]
        small, large = sorted((mass[left], mass[right]))
        cost = small * (large / (small + large)) * delta * delta
        if not np.isfinite(cost) or cost < 0:
            raise FloatingPointError("Adjacent Ward cost is not representable")
        heapq.heappush(heap, (cost, int(start[left]), int(stop[right]), left, right))
        work["heap_pushes"] += 1
        work["max_heap_entries"] = max(work["max_heap_entries"], len(heap))

    for left in range(n - 1):
        push(left, left + 1)
    head, live, next_id = 0, n, n
    candidates = []

    def emit():
        if live <= max_blocks:
            cuts, node = [0], head
            while node != -1:
                cuts.append(int(stop[node]))
                node = int(following[node])
            candidates.append(tuple(cuts))

    emit()
    while live > 1:
        while heap:
            _, _, _, left, right = heapq.heappop(heap)
            work["heap_pops"] += 1
            if alive[left] and alive[right] and following[left] == right:
                break
            work["stale_pops"] += 1
        else:
            raise RuntimeError("No live adjacent merge remains")
        node, next_id = next_id, next_id + 1
        before, after = int(previous[left]), int(following[right])
        total = mass[left] + mass[right]
        delta = mean[right] - mean[left]
        if mass[left] >= mass[right]:
            center = mean[left] + (mass[right] / total) * delta
        else:
            center = mean[right] - (mass[left] / total) * delta
        if not np.isfinite(total) or not np.isfinite(center):
            raise FloatingPointError("Merged Ward moment is not representable")
        mass[node], mean[node] = total, center
        start[node], stop[node] = start[left], stop[right]
        previous[node], following[node] = before, after
        alive[left] = alive[right] = False
        alive[node] = True
        if before == -1:
            head = node
        else:
            following[before] = node
        if after != -1:
            previous[after] = node
        live -= 1
        work["merges"] += 1
        if before != -1:
            push(before, node)
        if after != -1:
            push(node, after)
        emit()
    work["candidate_count"] = len(candidates)
    return tuple(candidates), work
