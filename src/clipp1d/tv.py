"""Direct bounded weighted chain TV by convex functional dynamic programming.

The derivative message is piecewise affine, with jumps at active box endpoints.
Two heaps remove its left/right tails at each TV convolution. Each knot enters
once and leaves once, giving O(n log n) time and O(n) storage, including boxes.
No segment-cost matrix, unbounded-TV/post-clipping approximation, or iterations.
"""

import heapq
import math

import numpy as np


class _DerivativeMessage:
    def __init__(self):
        self.left = []
        self.right = []
        self.knots = {}
        self.serial = 0
        # Returned CCFs remain float64; guard digits prevent
        # large, subsequently clipped unary coefficients from erasing small
        # surviving message slopes or prefix-value differences.
        self.left_m = self.right_m = np.longdouble(0)
        self.left_b = self.right_b = np.longdouble(0)
        self.inserted = self.removed = self.peak_knots = 0

    def add_unary(self, h, target):
        h, target = np.longdouble(h), np.longdouble(target)
        self.left_m += h
        self.right_m += h
        self.left_b -= h * target
        self.right_b -= h * target

    def add_knot(self, point, dm, db):
        key = self.serial
        self.serial += 1
        self.knots[key] = (point, dm, db)
        heapq.heappush(self.left, (point, key))
        heapq.heappush(self.right, (-point, key))
        self.inserted += 1
        self.peak_knots = max(self.peak_knots, len(self.knots))

    def peek(self, right=False):
        heap = self.right if right else self.left
        while heap and heap[0][1] not in self.knots:
            heapq.heappop(heap)
        return self.knots[heap[0][1]] if heap else None

    def remove(self, right=False):
        heap = self.right if right else self.left
        _, key = heapq.heappop(heap)
        self.removed += 1
        return self.knots.pop(key)

    def threshold_left(self, level, lower, upper):
        m, b = self.left_m, self.left_b
        while True:
            if not np.isfinite(m + b) or m <= 0:
                raise ArithmeticError("Nonpositive/nonfinite message slope")
            root = min(upper, max(lower, (level - b) / m))
            knot = self.peek()
            if knot is None or root < knot[0]:
                self.left_m, self.left_b = m, b
                return root, m, b
            point, dm, db = self.remove()
            m, b = m + dm, b + db
            if point >= lower and m * point + b >= level:
                self.left_m, self.left_b = m, b
                return point, m, b

    def threshold_right(self, level, lower, upper, integrate=False, integral_state=None):
        m, b = self.right_m, self.right_b
        integral_m, integral_b = (m, b) if integral_state is None else integral_state
        position, area = np.longdouble(1), np.longdouble(0)
        while True:
            if not np.isfinite(m + b) or m <= 0:
                raise ArithmeticError("Nonpositive/nonfinite message slope")
            root = min(upper, max(lower, (level - b) / m))
            knot = self.peek(right=True)
            if knot is None or root > knot[0]:
                if integrate:
                    area += (position - root) * (.5 * integral_m * (position + root) + integral_b)
                self.right_m, self.right_b = m, b
                return root, m, b, area
            point, dm, db = self.remove(right=True)
            if integrate:
                area += (position - point) * (.5 * integral_m * (position + point) + integral_b)
                position = point
            m, b = m - dm, b - db
            integral_m, integral_b = integral_m - dm, integral_b - db
            if point <= upper and m * point + b <= level:
                self.right_m, self.right_b = m, b
                return point, m, b, area

    def convolve(self, cap, lower, upper, integrate=False, integral_state=None):
        cap = np.longdouble(cap)
        if lower == upper:
            # A zero-width unary disconnects the messages. Values are needed only
            # by common-surrogate profiling, whose input has no frozen interior.
            if integrate:
                _, _, _, area = self.threshold_right(np.inf, lower, upper, True, integral_state)
            else:
                area = 0.
            a = b = lower
        else:
            a, am, ab = self.threshold_left(-cap, lower, upper)
            # Left-tail knots are already removed. Monotonicity guarantees the
            # upper threshold is >= a, including two levels inside one box jump.
            b, bm, bb, area = self.threshold_right(cap, max(lower, a), upper, integrate, integral_state)
        if a > b:
            raise ArithmeticError("Inconsistent message clipping thresholds")
        if a == b:
            self.removed += len(self.knots)
            self.knots.clear()
            self.left.clear()
            self.right.clear()
            self.add_knot(a, 0., 2 * cap)
        else:
            self.add_knot(a, am, ab + cap)
            self.add_knot(b, -bm, cap - bb)
        self.left_m = self.right_m = np.longdouble(0)
        self.left_b, self.right_b = -cap, cap
        return a, b, area


def forward_messages(h, target, lower, upper, caps, *, values_at_one=False):
    """Return reconstruction thresholds and optionally shifted prefix values at 1.

    Values subtract each unary's original boxed minimum, a witness-independent
    constant. This protects comparisons from irrelevant large quadratic offsets.
    Values at one require every upper bound <= 1; roots support arbitrary boxes.
    """
    n = len(h)
    if values_at_one and np.any(upper > 1):
        raise ValueError("Witness profiling at one requires upper bounds <= 1")
    message = _DerivativeMessage()
    low, high = np.empty(n - 1), np.empty(n - 1)
    prefix = np.full(n, np.inf, dtype=np.longdouble) if values_at_one else None
    at_one = np.longdouble(0)
    for i in range(n):
        if lower[i] == upper[i] and not values_at_one:
            # Avoid even forming a huge, irrelevant frozen unary coefficient.
            if i + 1 < n:
                low[i], high[i], _ = message.convolve(caps[i], lower[i], upper[i])
            else:
                last = lower[i]
            continue
        previous_right = message.right_m, message.right_b
        message.add_unary(h[i], target[i])
        if values_at_one:
            reference = np.longdouble(min(upper[i], max(lower[i], target[i])))
            displacement = 1 - reference
            if lower[i] <= 1 <= upper[i]:
                prefix[i] = at_one + .5 * h[i] * displacement**2 + h[i] * (reference - target[i]) * displacement
        if i + 1 < n:
            a, b, area = message.convolve(caps[i], lower[i], upper[i], values_at_one, previous_right)
            low[i], high[i] = a, b
            if values_at_one:
                # Integrate only the *previous* clipped message and evaluate the
                # new unary at b directly. Subtracting the full area from its
                # value at one would cancel a huge unary offset every iteration.
                displacement = b - reference
                at_one += (.5 * h[i] * displacement**2 + h[i] * (reference - target[i]) * displacement +
                           caps[i] * (1 - b) - area)
        else:
            last = message.threshold_left(0., lower[i], upper[i])[0]
    return low, high, last, prefix, dict(knots_inserted=message.inserted,
                                        knots_removed=message.removed, peak_live_knots=message.peak_knots)


def direct_primal(h, target, lower, upper, caps):
    low, high, last, _, stats = forward_messages(h, target, lower, upper, caps)
    x = np.empty(len(h))
    x[-1] = last
    for i in range(len(h) - 2, -1, -1):
        x[i] = min(high[i], max(low[i], x[i + 1]))
    return x, stats


def reconstruct_at_witness(forward_low, forward_high, reverse_low, reverse_high, witness, value=1.):
    """Recover one fixed-witness primal from two already-built message passes.

    Forward thresholds recover the prefix moving left from the fixed coordinate;
    reversed-chain thresholds recover the suffix moving right. No unary targets,
    curvatures, boxes, or heap messages are rebuilt. The caller must use thresholds
    from the same quadratic and fix a value feasible in the original witness box.
    """
    arrays = tuple(np.asarray(v) for v in (forward_low, forward_high, reverse_low, reverse_high))
    n = arrays[0].size + 1
    if (any(v.shape != (n - 1,) or not np.all(np.isfinite(v)) for v in arrays) or
            np.any(arrays[0] > arrays[1]) or np.any(arrays[2] > arrays[3]) or
            isinstance(witness, (bool, np.bool_)) or not isinstance(witness, (int, np.integer)) or
            not 0 <= witness < n or not np.isfinite(value)):
        raise ValueError("Invalid fixed-witness reconstruction thresholds")
    forward_low, forward_high, reverse_low, reverse_high = arrays
    x = np.empty(n)
    x[witness] = value
    for i in range(witness - 1, -1, -1):
        x[i] = min(forward_high[i], max(forward_low[i], x[i + 1]))
    for i in range(witness + 1, n):
        j = n - 1 - i
        x[i] = min(reverse_high[j], max(reverse_low[j], x[i - 1]))
    return x


def split_primal(h, target, lower, upper, caps):
    """Split at frozen nodes when incident absolute penalties are affine on boxes."""
    fixed = np.flatnonzero(lower == upper)
    if not fixed.size:
        return direct_primal(h, target, lower, upper, caps)
    for k in fixed:
        for j in (k - 1, k + 1):
            if 0 <= j < len(h) and lower[j] != upper[j] and lower[j] < lower[k] < upper[j]:
                return direct_primal(h, target, lower, upper, caps)
    x = lower.copy()
    stats = dict(knots_inserted=0, knots_removed=0, peak_live_knots=0, free_segments=0)
    for a, b in zip(np.r_[0, fixed + 1], np.r_[fixed, len(h)]):
        if a == b:
            continue
        shifted = target[a:b].copy()
        if a:
            shifted[0] += (1 if lower[a - 1] >= upper[a] else -1) * caps[a - 1] / h[a]
        if b < len(h):
            shifted[-1] += (1 if lower[b] >= upper[b - 1] else -1) * caps[b - 1] / h[b - 1]
        x[a:b], detail = direct_primal(h[a:b], shifted, lower[a:b], upper[a:b], caps[a:b - 1])
        stats["free_segments"] += 1
        for key in ("knots_inserted", "knots_removed"):
            stats[key] += detail[key]
        stats["peak_live_knots"] = max(stats["peak_live_knots"], detail["peak_live_knots"])
    return x, stats


def reconstruct_dual(x, h, target, lower, upper, caps):
    """Linear reachable-interval pass and backward recovery of box normals/duals."""
    n = len(x)
    # A represented block mean need not satisfy its summed normal equation in
    # float64, especially when its curvature spans many orders of magnitude.
    # Recover the dual at the *unrounded* block mean: distribute that unavoidable
    # primal rounding error by curvature instead of concentrating it at whichever
    # small-curvature node happens to end the interval pass. The independent
    # float64 gap/KKT audit still checks the returned, represented primal/dual.
    extended = np.longdouble
    gradients = np.zeros(n, dtype=extended)
    free = lower < upper
    gradients[free] = h[free].astype(extended) * (x[free].astype(extended) - target[free])
    cuts = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1, n]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if np.any(x[a:b] == lower[a:b]) or np.any(x[a:b] == upper[a:b]):
            continue
        left = extended(caps[a - 1]) * np.sign(x[a] - x[a - 1]) if a else extended(0)
        right = extended(caps[b - 1]) * np.sign(x[b] - x[b - 1]) if b < n else extended(0)
        curvature = h[a:b].astype(extended)
        adjustment = (np.sum(gradients[a:b]) - (right - left)) / np.sum(curvature)
        gradients[a:b] -= curvature * adjustment
    reachable_lo, reachable_hi = np.empty(n, dtype=extended), np.empty(n, dtype=extended)
    lo = hi = extended(0)
    accumulated_rounding = extended(0)
    for i in range(n):
        edge_lo = edge_hi = 0.
        if i + 1 < n:
            jump = x[i + 1] - x[i]
            edge_lo, edge_hi = (-caps[i], caps[i]) if jump == 0 else (caps[i] * np.sign(jump),) * 2
        unary_scale = h[i] * (abs(x[i]) + abs(target[i])) if free[i] else 0.
        accumulated_rounding += 128 * np.finfo(extended).eps * (
            1 + abs(gradients[i]) + abs(lo) + abs(hi))
        rounding = accumulated_rounding + 128 * np.finfo(float).eps * (
            1 + unary_scale + abs(lo) + abs(hi) + abs(edge_lo) + abs(edge_hi))
        if lower[i] == upper[i]:
            lo, hi = edge_lo, edge_hi
        else:
            lo, hi = lo + gradients[i], hi + gradients[i]
            if x[i] == lower[i]:
                lo = -np.inf
            if x[i] == upper[i]:
                hi = np.inf
            lo, hi = max(edge_lo, lo), min(edge_hi, hi)
        if lo > hi:
            if lo - hi > rounding:
                raise ArithmeticError("No feasible chain dual interval")
            # Repair only an empty intersection caused by roundoff. Unconditional
            # widening by an edge-scale allowance corrupts the recovered normal
            # when huge caps coexist with small unary gradients.
            lo = hi = min(edge_hi, max(edge_lo, .5 * lo + .5 * hi))
        reachable_lo[i], reachable_hi[i] = lo, hi
        if edge_lo == edge_hi or lower[i] == upper[i]:
            accumulated_rounding = extended(0)
    dual = np.empty(n - 1)
    next_q = 0.
    for i in range(n - 1, 0, -1):
        lo, hi = reachable_lo[i - 1], reachable_hi[i - 1]
        desired = next_q - gradients[i]
        if lower[i] == upper[i]:
            desired = 0.
        elif x[i] == lower[i]:
            desired = max(desired, lo)
        elif x[i] == upper[i]:
            desired = min(desired, hi)
        dual[i - 1] = min(hi, max(lo, desired))
        next_q = dual[i - 1]
    return dual


def polish_blocks(x, h, target, lower, upper, caps):
    """One exact-block weighted-mean pass to reduce message arithmetic roundoff."""
    result = x.copy()
    cuts = np.r_[0, np.flatnonzero(np.diff(x) != 0) + 1, len(x)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        lo, hi = np.max(lower[a:b]), np.min(upper[a:b])
        if lo == hi:
            result[a:b] = lo
            continue
        left = caps[a - 1] * np.sign(x[a] - x[a - 1]) if a else 0.
        right = caps[b - 1] * np.sign(x[b] - x[b - 1]) if b < len(x) else 0.
        reference = x[a]
        shift = math.fsum([*(h[a:b] * (target[a:b] - reference)), -left, right]) / math.fsum(h[a:b])
        result[a:b] = min(hi, max(lo, reference + shift))
    return result
