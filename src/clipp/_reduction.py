"""Sparse suffix reduction compatible with the former padded likelihood.

SciPy 1.18's real logsumexp separates all maxima before summing shifted
exponentials. NumPy's contiguous float64 sum uses an eight-lane pairwise
tree, with a 128-element leaf size. Removing a zero suffix changes that
tree and can change scalar optimizer decisions. Preserve the logical tree
while skipping its zero-only branches; no padded states are exponentiated.

This numerical compatibility is qualified against NumPy 2.2.6 / SciPy 1.18.
Dependency upgrades must run the external reduction/reference tests: it is
not a claim that NumPy's private reduction order is a stable public API.
"""

import numpy as np
from scipy.special import logsumexp


def _positive_prefix_sum(values, logical_width):
    """Sum a nonnegative prefix as if followed by zeros to logical_width.

    This mirrors the contiguous NumPy pairwise-sum association, not a new
    summation rule. Skipped additions have an exactly zero right operand.
    The reference likelihood always had a contiguous multiplicity axis.
    """
    width = values.shape[-1]
    shape = values.shape[:-1]
    if not width:
        return np.zeros(shape, dtype=np.float64)
    if logical_width < 8:
        result = np.zeros(shape, dtype=np.float64)
        for index in range(width):
            result = result + values[..., index]
        return result
    if logical_width <= 128:
        # Views plus scalar zeros avoid an eight-column allocation for the
        # common one/few-state groups. Do not mutate the input views.
        lanes = [values[..., index] if index < width else 0.0 for index in range(8)]
        full_stop = logical_width - logical_width % 8
        for index in range(8, min(full_stop, width), 8):
            count = min(8, width - index)
            for lane in range(count):
                lanes[lane] = lanes[lane] + values[..., index + lane]
        result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + (
            (lanes[4] + lanes[5]) + (lanes[6] + lanes[7])
        )
        for index in range(full_stop, width):
            result = result + values[..., index]
        return result
    split = logical_width // 2
    split -= split % 8
    if width <= split:
        return _positive_prefix_sum(values, split)
    return _positive_prefix_sum(values[..., :split], split) + _positive_prefix_sum(
        values[..., split:], logical_width - split
    )


def grouped_logsumexp(kernel, logical_width):
    """Reduce valid float64 states with an implicit negative-infinity suffix.

    ``logical_width`` is the maximum major CN of the current model/block,
    including for a prepared subset. The output matches the old contiguous
    padded SciPy 1.18 reduction rather than shortening its addition tree.
    Arithmetic and state storage scale with the valid support.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if (
        kernel.ndim < 1
        or kernel.shape[-1] < 1
        or not isinstance(logical_width, (int, np.integer))
        or logical_width < kernel.shape[-1]
    ):
        raise ValueError("Require a nonempty multiplicity axis within the logical padded width")
    if kernel.shape[-1] == logical_width:
        return logsumexp(kernel, axis=-1)
    if kernel.shape[-1] == 1:
        # One possible state has logsumexp equal to its kernel even with
        # any number of omitted -inf states, including every endpoint.
        # The reference adds log1p(0) + log(1) before the maximum, so its
        # zero is positive even if a supplied kernel contains negative zero.
        return np.add(kernel[..., 0], 0.0)
    maximum = kernel.max(axis=-1, keepdims=True)
    maxima = kernel == maximum
    count = maxima.sum(axis=-1, dtype=np.float64)
    other = np.where(maxima, -np.inf, kernel)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shifted_sum = _positive_prefix_sum(np.exp(other - maximum), logical_width)
        result = np.log1p(shifted_sum / count) + np.log(count) + maximum[..., 0]
        # Match SciPy's direct-exponential fallback for all -inf, +inf and
        # NaN rows. Ordinary finite rows never need this second evaluation.
        nonfinite = ~np.isfinite(result)
        if np.any(nonfinite):
            direct = np.log(_positive_prefix_sum(np.exp(kernel[nonfinite]), logical_width))
            result = np.array(result, copy=True)
            result[nonfinite] = direct
    return result
