"""Deterministic heterogeneous single-region inputs; no simulated truth claim.

These extend the integer-support mixtures and original-box mechanisms of the
small mixed_multiplicity/all_bounds_below_one qualification fixtures. Every
row is a distinct observation. Sizes are prefixes of one frozen 512-row recipe.
"""

import csv
import hashlib
import io
from pathlib import Path

import numpy as np

from clipp1d.io import SCHEMA_COLUMNS, read_tumor
from clipp1d.model import compile_model, loss_at_rows


RECIPE = "heterogeneous_integer_mixture_v1"
FAMILIES = ("mixed_support", "below_one")
SIZES = (64, 256, 512)


def fixture_text(family, nodes):
    """Canonical TSV bytes with physical mixed-CN denominators and fixed purity."""
    if family not in FAMILIES or type(nodes) is not int or nodes not in SIZES:
        raise ValueError("Choose mixed_support/below_one and exactly 64, 256 or 512 nodes")
    rng = np.random.Generator(np.random.PCG64(91073 if family == "mixed_support" else 281357))
    purity = 0.71 if family == "mixed_support" else 0.83
    depths = 100 + rng.permutation(1900)[:512]
    support = rng.integers(2, 5, size=512)
    fractions = rng.uniform(0.04, 0.98 if family == "mixed_support" else 0.25, 512)
    normal = rng.choice([1, 2, 3], size=512) if family == "mixed_support" else np.full(512, 2)
    relative_modes = rng.uniform(0.58, 0.94, 512)
    kinds = rng.permutation(512)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(SCHEMA_COLUMNS)
    for i in range(nodes):
        k, fraction = int(support[i]), float(fractions[i])
        copies = 1 + fraction * (k - 1)
        slope = purity / ((1 - purity) * int(normal[i]) + purity * copies)
        upper = min(1.0, (1 - 1e-6) / (k * slope))
        probability = slope * upper * relative_modes[i]
        # A fixed scattered minority challenges original upper boxes and clipped
        # endpoints; the remaining rows retain several competing interior modes.
        if kinds[i] % 19 == 0:
            probability = slope * upper * rng.uniform(1.08, 1.35)
        alt = int(np.rint(depths[i] * min(probability, 0.98)))
        if kinds[i] % 37 == 0:
            alt = 0
        elif kinds[i] % 41 == 0:
            alt = int(depths[i]) - 1
        common = [
            f"m{i:06d}",
            "sample1",
            alt,
            int(depths[i]) - alt,
            1,
            purity,
            int(normal[i]),
            f"segment{i:06d}",
        ]
        writer.writerow(common + ["low", repr(1 - fraction), 1, 0])
        writer.writerow(common + ["high", repr(fraction), k, 0])
    return stream.getvalue().encode("utf-8")


def write_fixture(path, family, nodes):
    path = Path(path)
    payload = fixture_text(family, nodes)
    with path.open("xb") as stream:
        stream.write(payload)
    data = read_tumor(path)
    model = compile_model(data)
    if len(model) != nodes:
        raise AssertionError("Qualification fixture unexpectedly filtered mutations")
    return data, model


def difficulty_summary(model, data, *, grid_points=2049):
    """Descriptive bounded CPU grid, never a pilot fit or global certificate."""
    if type(grid_points) is not int or grid_points < 257:
        raise ValueError("Difficulty grid requires at least 257 deterministic points")
    support = model.valid.sum(axis=1)
    depth = model.alt + model.ref
    wells, competitive = [], []
    grid = np.linspace(0.0, 1.0, grid_points)
    for i in range(len(model)):
        points = model.lower[i] + grid * (model.upper[i] - model.lower[i])
        values = loss_at_rows(model, np.full(grid_points, i, dtype=np.intp), points)
        minima = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])) + 1
        wells.append(int(minima.size))
        competitive.append(int(np.count_nonzero(values[minima] <= values.min() + 2.0)))
    slopes = model.slope[model.valid]
    low_clips = np.divide(
        model.eps, model.slope, out=np.full_like(model.slope, np.inf), where=model.valid
    )
    high_clips = np.divide(
        1 - model.eps, model.slope, out=np.full_like(model.slope, np.inf), where=model.valid
    )
    inside_low = (
        model.valid & (low_clips >= model.lower[:, None]) & (low_clips <= model.upper[:, None])
    )
    inside_high = (
        model.valid & (high_clips >= model.lower[:, None]) & (high_clips <= model.upper[:, None])
    )
    row_hashes = set()
    for i in range(len(model)):
        row_hashes.add(
            hashlib.sha256(
                b"".join(
                    np.asarray(getattr(model, name)[i]).tobytes()
                    for name in ("alt", "ref", "slope", "log_prior", "lower", "upper", "valid")
                )
            ).hexdigest()
        )
    return dict(
        recipe=RECIPE,
        nodes=len(model),
        distinct_likelihood_rows=len(row_hashes),
        distinct_depths=int(np.unique(depth).size),
        depth_min=int(depth.min()),
        depth_max=int(depth.max()),
        support_counts={str(k): int(np.count_nonzero(support == k)) for k in (2, 3, 4)},
        purity=data.purity,
        normal_cn_values=sorted({m.normal_cn for m in data.mutations}),
        slope_min=float(slopes.min()),
        slope_max=float(slopes.max()),
        first_slope_distinct=int(np.unique(model.slope[:, 0]).size),
        upper_bound_min=float(model.upper.min()),
        upper_bound_max=float(model.upper.max()),
        original_upper_below_one=int(np.count_nonzero(model.upper < 1)),
        lower_clipping_rows=int(np.count_nonzero(inside_low.any(axis=1))),
        upper_clipping_rows=int(np.count_nonzero(inside_high.any(axis=1))),
        zero_alt_rows=int(np.count_nonzero(model.alt == 0)),
        near_all_alt_rows=int(np.count_nonzero(model.ref <= 1)),
        grid_interior_wells=wells,
        grid_competitive_wells=competitive,
        rows_with_multiple_interior_wells=int(np.count_nonzero(np.asarray(wells) >= 2)),
        rows_with_multiple_competitive_wells=int(np.count_nonzero(np.asarray(competitive) >= 2)),
        grid_points=grid_points,
        competitive_loss_window=2.0,
        scope="Fixed synthetic likelihood stress inputs; grid wells are descriptive, not qualified minima or tumor truth",
    )


def validate_difficulty(summary, family):
    n = summary["nodes"]
    if (
        summary["distinct_likelihood_rows"] != n
        or summary["distinct_depths"] != n
        or summary["first_slope_distinct"] != n
        or not all(summary["support_counts"][str(k)] > 0 for k in (2, 3, 4))
        or summary["rows_with_multiple_competitive_wells"] < n // 2
        or summary["lower_clipping_rows"] == 0
        or summary["upper_clipping_rows"] == 0
    ):
        raise AssertionError("Fixture lost its required heterogeneous/multimodal/clipping coverage")
    if family == "below_one" and summary["original_upper_below_one"] != n:
        raise AssertionError("Below-one fixture expanded an original feasible box")
