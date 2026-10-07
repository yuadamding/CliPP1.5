"""Uniform full-support binomial multiplicity model; CP is the fitting scale."""

import math
import numpy as np
from scipy.special import gammaln, xlog1py, xlogy

from ._reduction import grouped_logsumexp


# Prepared data and returned arrays are separate from this transient workspace.
# Include the original-row marginals and conservative reduction temporaries.
LIKELIHOOD_WORKSPACE_BYTES = 32 * 1024 * 1024
_STATE_BYTES = 16 * 8


def _immutable(values):
    values = np.array(values, copy=True)
    values.flags.writeable = False
    return values


class MultiplicityModel:
    """Full-support mixture evaluated in groups of equal major copy number.

    Evaluation groups retain each row's counts and denominator. All groups in
    a block share the fitted CP; no invalid multiplicities are evaluated.
    """

    def __init__(self, r, n, major, total, purity):
        arrays = [np.atleast_1d(np.asarray(x, dtype=float)) for x in (r, n, major, total)]
        if (
            any(x.ndim != 1 for x in arrays)
            or not arrays[0].size
            or any(x.shape != arrays[0].shape for x in arrays)
            or any(not np.all(np.isfinite(x)) or np.any(x != np.rint(x)) for x in arrays)
        ):
            raise ValueError("Counts and copy numbers must be matching finite integer vectors")
        self.r, self.n, self.major, self.total = arrays
        self.purity = float(purity)
        if (
            not math.isfinite(self.purity)
            or not 0 < self.purity <= 1
            or np.any(self.r < 0)
            or np.any(self.n <= 0)
            or np.any(self.r > self.n)
            or np.any(self.major < 1)
            or np.any(self.total < self.major)
        ):
            raise ValueError("Invalid counts, major/total copy numbers, or purity")
        self.r, self.n, self.total = map(_immutable, (self.r, self.n, self.total))
        self.major = _immutable(self.major.astype(int))
        self.reference = _immutable(self.n - self.r)
        self.denominator = _immutable(2 * (1 - self.purity) + self.purity * self.total)
        self.log_constant = _immutable(
            gammaln(self.n + 1) - gammaln(self.r + 1) - gammaln(self.n - self.r + 1) - np.log(self.major)
        )
        self.m = _immutable(np.arange(1, int(self.major.max()) + 1))
        self.valid_states = int(self.major.sum())
        self.workspace_bytes = LIKELIHOOD_WORKSPACE_BYTES
        self._groups = tuple(
            (_immutable(rows), _immutable(self.m[None, :cn] / self.denominator[rows, None]))
            for cn in np.unique(self.major)
            for rows in [np.flatnonzero(self.major == cn)]
        )

    def __len__(self):
        return len(self.r)

    @property
    def valid(self):
        """Legacy padded view, constructed only on explicit request."""
        return self.m[None, :] <= self.major[:, None]

    @property
    def scale(self):
        """Legacy padded view; fitting uses prepared valid-state groups."""
        return self.m[None, :] / self.denominator[:, None]

    def subset(self, indices):
        # Trusted preparation from immutable validated rows, including repeated
        # or reordered indices. No repeated validation/gammaln/division is needed.
        rows = np.atleast_1d(np.arange(len(self))[indices])
        if rows.ndim != 1 or not rows.size:
            raise ValueError("Model subset must be a nonempty vector of rows")
        result = object.__new__(type(self))
        for name in ("r", "n", "major", "total", "reference", "denominator", "log_constant"):
            setattr(result, name, _immutable(getattr(self, name)[rows]))
        result.purity = self.purity
        result.m = self.m[: int(result.major.max())]
        result.valid_states = int(result.major.sum())
        result.workspace_bytes = self.workspace_bytes
        groups = []
        for parent_rows, scale in self._groups:
            selected = np.flatnonzero(result.major == scale.shape[1])
            if selected.size:
                positions = np.searchsorted(parent_rows, rows[selected])
                groups.append((_immutable(selected), _immutable(scale[positions])))
        result._groups = tuple(groups)
        return result

    def component_modes(self):
        """All valid component modes, with the original division arithmetic."""
        return np.concatenate(
            [((self.r[rows] / self.n[rows])[:, None] / scale).ravel() for rows, scale in self._groups]
        )

    def _cp(self, cp):
        cp = np.broadcast_to(np.asarray(cp, dtype=float), (len(self),))
        if np.any(~np.isfinite(cp)) or np.any(cp < 0) or np.any(cp > self.purity):
            raise ValueError("Cellular prevalence must be finite and in [0, purity]")
        return cp

    def _batch_size(self, points):
        # Reserve two marginal matrices and sixteen float64 state temporaries.
        # A single row retains its complete support in each multiplicity sum.
        minimum = 16 * len(self) + _STATE_BYTES * len(self.m)
        if self.workspace_bytes < minimum:
            raise ValueError(f"Likelihood workspace requires at least {minimum} bytes for this model")
        return max(1, min(32, points, self.workspace_bytes // minimum))

    def _chunks(self, batch):
        available = self.workspace_bytes - 16 * batch * len(self)
        for rows, scale in self._groups:
            size = max(1, available // (_STATE_BYTES * batch * scale.shape[1]))
            for first in range(0, len(rows), size):
                yield rows[first : first + size], scale[first : first + size]

    def _kernel(self, cp, rows, scale):
        p = np.minimum(1.0, cp * scale)
        return (
            xlogy(self.r[rows, None], p)
            + xlog1py(self.reference[rows, None], -p)
            + self.log_constant[rows, None]
        )

    def _log_kernel(self, cp):
        """Compatibility matrix; padded storage is allocated only on request."""
        cp = self._cp(cp)
        self._batch_size(1)
        result = np.full((len(self), len(self.m)), -np.inf)
        for rows, scale in self._chunks(1):
            result[rows, : scale.shape[1]] = self._kernel(cp[rows, None], rows, scale)
        return result

    def log_likelihood(self, cp):
        """Per-mutation marginal log likelihood, including binomial constants."""
        cp = self._cp(cp)
        self._batch_size(1)
        result = np.empty(len(self))
        for rows, scale in self._chunks(1):
            result[rows] = grouped_logsumexp(self._kernel(cp[rows, None], rows, scale), len(self.m))
        return result

    def posterior(self, cp):
        """Conditional probabilities for m=1,...,max(major), with padded zeros."""
        cp = self._cp(cp)
        self._batch_size(1)
        result = np.zeros((len(self), len(self.m)))
        for rows, scale in self._chunks(1):
            kernel = self._kernel(cp[rows, None], rows, scale)
            normalizer = grouped_logsumexp(kernel, len(self.m))
            if not np.all(np.isfinite(normalizer)):
                raise ValueError("Multiplicity posterior is undefined at zero likelihood")
            result[rows, : scale.shape[1]] = np.exp(kernel - normalizer[:, None])
        return result

    def grid_log_likelihood(self, grid, *, progress=None):
        """Evaluate every proposal/state within the transient workspace budget."""
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or np.any(~np.isfinite(grid)) or np.any(grid < 0) or np.any(grid > self.purity):
            raise ValueError("Center grid must be a finite vector in [0, purity]")
        result = np.empty(len(grid))
        batch = self._batch_size(len(grid))
        for start in range(0, len(grid), batch):
            cp = grid[start : start + batch]
            marginal = np.empty((len(cp), len(self)))
            for rows, scale in self._chunks(len(cp)):
                kernel = self._kernel(cp[:, None, None], rows, scale)
                marginal[:, rows] = grouped_logsumexp(kernel, len(self.m))
            # Restore original row order before summation, including subsets.
            result[start : start + len(cp)] = marginal.sum(axis=1)
            if progress is not None:
                progress.add(valid_state_evaluations=len(cp) * self.valid_states)
                progress.update(
                    "grid_chunk", grid_points_total=len(grid), grid_points_completed=start + len(cp)
                )
        return result
