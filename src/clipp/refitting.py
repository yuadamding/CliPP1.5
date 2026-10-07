"""Conditional scalar multimode numerical refit (not a global certificate)."""

from contextlib import nullcontext

import numpy as np
from scipy.optimize import minimize_scalar


def refit_center(model, *, progress=None):
    """Refine bracketed modes and edge intervals, retaining physical endpoints."""
    if progress is None:
        return _refit_center(model, None)
    with progress.context(
        phase="conditional_refit",
        interval_mutations=len(model),
        valid_states=model.valid_states,
        grid_points_total=0,
        grid_points_completed=0,
        scalar_intervals_total=0,
        scalar_intervals_completed=0,
    ):
        progress.add(interval_refits_started=1)
        progress.update("refit_started")
        result = _refit_center(model, progress)
        progress.add(interval_refits_completed=1)
        progress.update("refit_completed")
        return result


def _refit_center(model, progress):
    with progress.timer("grid_preparation") if progress is not None else nullcontext():
        grid = np.unique(
            np.concatenate(
                (np.linspace(0, model.purity, 513), np.clip(model.component_modes(), 0, model.purity))
            )
        )
    with progress.timer("grid_evaluation") if progress is not None else nullcontext():
        values = (
            model.grid_log_likelihood(grid)
            if progress is None
            else model.grid_log_likelihood(grid, progress=progress)
        )
    if progress is not None:
        progress.fields.update(grid_points_total=len(grid), grid_points_completed=len(grid))
    best_index = int(np.argmax(values))
    best = (float(values[best_index]), float(grid[best_index]))
    maxima = (
        np.flatnonzero(
            (values[1:-1] >= values[:-2])
            & (values[1:-1] >= values[2:])
            & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))
        )
        + 1
    )
    # A maximum just inside a finite boundary can beat that endpoint while
    # every interior grid point is worse. It then has no grid-local maximum
    # to bracket. Search both edge intervals even when neither is bracketed;
    # the explicit grid candidates above still preserve exact endpoints.
    intervals = {(float(grid[0]), float(grid[1])), (float(grid[-2]), float(grid[-1]))}
    intervals.update((float(grid[index - 1]), float(grid[index + 1])) for index in maxima)

    def objective(cp):
        value = -float(model.log_likelihood(cp).sum())
        if progress is not None:
            progress.add(valid_state_evaluations=model.valid_states)
        return value

    if progress is not None:
        progress.fields.update(scalar_intervals_total=len(intervals))
    with progress.timer("scalar_optimization") if progress is not None else nullcontext():
        for completed, (lower, upper) in enumerate(sorted(intervals), 1):
            fit = minimize_scalar(
                objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-12, "maxiter": 500},
            )
            if not fit.success or not np.isfinite(fit.fun):
                raise RuntimeError("Marginal-likelihood center refit failed: %s" % fit.message)
            candidate = (-float(fit.fun), float(fit.x))
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
            if progress is not None:
                progress.fields.update(scalar_intervals_completed=completed)
                progress.update("scalar_interval", scalar_lower=lower, scalar_upper=upper)
    if not np.isfinite(best[0]):
        raise RuntimeError("No finite marginal likelihood in cluster refit")
    return best[1], best[0]
