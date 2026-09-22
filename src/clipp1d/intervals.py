"""Exact-order signed interval scan with batched independent prefix sums.

Each row is one feasible exact-fused run. Power-of-two length buckets keep
padding below twice the input size: prefix arithmetic and temporary storage are
O(M). Bucket discovery/grouping gives a conservative O(M log M) total-work bound.
No prefix subtraction across distinct runs changes the scalar scan's rounding.
"""

import numpy as np


def _scalar_interval_descent(x, left, right, lower, upper, caps, tolerance=0.0):
    """Original scalar ordering, retained for unsupported/nonfinite arithmetic."""
    best, best_merit = None, 0.0
    n = len(x)
    for sign in (-1, 1):
        prefix = absolute = 0.0
        start_key = np.inf
        active = False
        for i in range(n):
            feasible = x[i] > lower[i] if sign < 0 else x[i] < upper[i]
            if not feasible:
                active = False
                continue
            if not active or (i and x[i] != x[i - 1]):
                prefix = absolute = 0.0
                start_key = np.inf
            active = True
            left_cost = 0.0
            if i:
                jump = x[i] - x[i - 1]
                left_cost = caps[i - 1] * (1 if jump == 0 else sign * np.sign(jump))
            key = left_cost + tolerance * abs(left_cost) - prefix - tolerance * absolute
            if key < start_key:
                start_key = key
                start = i
                start_prefix, start_absolute, start_cost = prefix, absolute, left_cost
            value = -left[i] if sign < 0 else right[i]
            prefix += value
            absolute += abs(value)
            right_cost = 0.0
            if i + 1 < n:
                jump = x[i + 1] - x[i]
                right_cost = caps[i] * (1 if jump == 0 else -sign * np.sign(jump))
            merit = prefix + tolerance * absolute + right_cost + tolerance * abs(right_cost) + start_key + tolerance
            if merit < best_merit:
                derivative = prefix - start_prefix + start_cost + right_cost
                scale = 1 + absolute - start_absolute + abs(start_cost) + abs(right_cost)
                best = (start, i + 1, sign, float(derivative), float(scale))
                best_merit = merit
    return best


def interval_descent(x, left, right, lower, upper, caps, tolerance=0.0):
    """Return exactly the scalar scan's signed interval and rounded diagnostics.

    Sequential row-wise cumsums start from explicit zero, reproducing both
    prefix accumulators. Strict minima retain the first equal start; equal final
    merits retain negative sign, then the earliest stop. Arithmetic ordering and
    acceptance tolerance are unchanged. Float64 finite arrays use batching;
    other arithmetic uses the original scalar implementation.
    """
    arrays = (x, left, right, lower, upper, caps)
    n = len(x)
    # Below this size the original scan is faster than setting up array batches.
    if n < 64:
        return _scalar_interval_descent(*arrays, tolerance)
    if (any(not isinstance(a, np.ndarray) or a.dtype != np.dtype(float) or a.ndim != 1 or
            not np.all(np.isfinite(a)) for a in arrays) or not np.isfinite(tolerance)):
        return _scalar_interval_descent(*arrays, tolerance)
    best, best_merit, best_rank = None, 0.0, None
    jumps = np.diff(x)
    equal = jumps == 0
    jump_sign = np.sign(jumps)
    for sign in (-1, 1):
        feasible = x > lower if sign < 0 else x < upper
        connected = feasible[1:] & feasible[:-1] & equal
        starts = np.flatnonzero(feasible & ~np.r_[False, connected])
        stops = np.flatnonzero(feasible & ~np.r_[connected, False]) + 1
        lengths = stops - starts
        if not len(starts):
            continue
        left_cost, right_cost = np.zeros(n), np.zeros(n)
        left_cost[1:] = caps * np.where(equal, 1, sign * jump_sign)
        right_cost[:-1] = caps * np.where(equal, 1, -sign * jump_sign)
        # Integer powers avoid a log2 rounding dependency at bucket boundaries.
        boundaries = 1 << np.arange((int(np.max(lengths)) - 1).bit_length() + 1)
        buckets = np.searchsorted(boundaries, lengths)
        for bucket in np.unique(buckets):
            width = int(boundaries[bucket])
            chosen_runs = buckets == bucket
            run_starts, run_lengths = starts[chosen_runs], lengths[chosen_runs]
            offsets = np.arange(width)
            valid = offsets[None, :] < run_lengths[:, None]
            indices = np.minimum(run_starts[:, None] + offsets, n - 1)
            values = -left[indices] if sign < 0 else right[indices]
            values = np.where(valid, values, 0.)
            zero = np.zeros((len(run_starts), 1))
            # NumPy cumsum is sequential; including zero preserves the initial
            # scalar addition (including signed-zero behavior).
            prefix = np.cumsum(np.concatenate((zero, values), axis=1), axis=1)
            absolute = np.cumsum(np.concatenate((zero, np.abs(values)), axis=1), axis=1)
            before, after = prefix[:, :-1], prefix[:, 1:]
            abs_before, abs_after = absolute[:, :-1], absolute[:, 1:]
            lc, rc = left_cost[indices], right_cost[indices]
            keys = lc + tolerance * np.abs(lc) - before - tolerance * abs_before
            keys = np.where(valid, keys, np.inf)
            min_keys = np.minimum.accumulate(keys, axis=1)
            previous_keys = np.concatenate((np.full_like(zero, np.inf), min_keys[:, :-1]), axis=1)
            new_start = keys < previous_keys
            start_columns = np.maximum.accumulate(np.where(new_start, offsets, 0), axis=1)
            merits = after + tolerance * abs_after + rc + tolerance * np.abs(rc) + min_keys + tolerance
            if not (np.all(np.isfinite(prefix)) and np.all(np.isfinite(absolute)) and
                    np.all(np.isfinite(merits[valid]))):
                return _scalar_interval_descent(*arrays, tolerance)
            merits = np.where(valid, merits, np.inf)
            row, column = np.unravel_index(np.argmin(merits), merits.shape)
            merit = merits[row, column]
            stop = int(indices[row, column]) + 1
            rank = (0 if sign < 0 else 1, stop)
            if merit < best_merit or (merit == best_merit and merit < 0 and rank < best_rank):
                a = start_columns[row, column]
                start = int(run_starts[row] + a)
                derivative = after[row, column] - before[row, a] + lc[row, a] + rc[row, column]
                scale = 1 + abs_after[row, column] - abs_before[row, a] + abs(lc[row, a]) + abs(rc[row, column])
                best = (start, stop, sign, float(derivative), float(scale))
                best_merit, best_rank = merit, rank
    return best
