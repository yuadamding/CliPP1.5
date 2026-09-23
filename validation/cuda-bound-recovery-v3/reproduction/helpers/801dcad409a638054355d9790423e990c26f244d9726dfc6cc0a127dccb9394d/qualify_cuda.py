"""Actual allocated-CUDA qualification, with durable partial/failure evidence.

This command does not submit jobs or install anything. Host NumPy likelihoods are
independent test oracles; all fitted paths and QPs run on the requested CUDA GPU.
An eager GPU run, compiled GPU run, resource measurement and cohort accuracy are
separate evidence. There is no CPU fitting fallback or skipped-success outcome.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
from time import perf_counter
from types import SimpleNamespace
import traceback

import numpy as np
import torch

from clipp1d.api import fit as public_fit, source_provenance
from clipp1d.cuda_api import require_cuda
from clipp1d.cuda.graph import build_graph
from clipp1d.cuda.kernels import Kernels, differences, adjoint
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda.qp import solve_qp
from clipp1d.cuda.solver import fit_lambda
from clipp1d.cuda.partition import refit, grouping
from clipp1d.cuda.selection import fit_tensor_model
from clipp1d.io import SCHEMA_COLUMNS
from clipp1d.model import evaluate, one_sided_derivatives
from clipp1d.types import CountModel


SCHEMA = "clipp1d.cuda.qualification.v3"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(device, function):
    torch.cuda.synchronize(device)
    begin = perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    return result, perf_counter() - begin


def host_fixture(name):
    """Small canonical independent likelihood fixtures, not cohort samples."""
    if name == "single_support":
        alt = np.array([20.0, 21.0, 48.0])
        counts, scale = np.ones(3, dtype=int), np.full(3, 0.5)
    elif name == "mixed_multiplicity":
        alt = np.array([16.0, 28.0, 37.0, 19.0, 40.0, 54.0])
        counts, scale = np.array([1, 2, 3, 4, 2, 3]), np.array([0.4, 0.25, 0.15, 0.12, 0.35, 0.42])
    elif name == "all_bounds_below_one":
        alt = np.array([17.0, 24.0, 36.0, 43.0])
        counts, scale = np.full(4, 3), np.array([0.38, 0.45, 0.52, 0.59])
    else:
        raise ValueError(name)
    n, eps = alt.size, 1e-6
    support = np.arange(1, int(counts.max()) + 1)
    valid = support <= counts[:, None]
    slopes = scale[:, None] * np.where(valid, support, 0)
    prior = np.where(valid, -np.log(counts[:, None]), -np.inf)
    upper = np.minimum(1.0, (1 - eps) / (scale * counts))
    return CountModel(
        tuple(f"m{i:04}" for i in range(n)),
        alt,
        100 - alt,
        np.full(n, eps),
        upper,
        slopes,
        prior,
        valid,
        eps,
    )


def host_identity(model):
    digest = hashlib.sha256(json.dumps(model.mutation_ids).encode())
    arrays = {}
    for name in ("alt", "ref", "slope", "log_prior", "lower", "upper", "valid"):
        value = np.asarray(getattr(model, name))
        arrays[name] = dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
        digest.update(name.encode())
        digest.update(value.tobytes())
    return dict(
        model_sha256=digest.hexdigest(),
        mutation_ids=model.mutation_ids,
        arrays=arrays,
        lower=model.lower.tolist(),
        upper=model.upper.tolist(),
        eps=model.eps,
    )


def scaling_fixture(n):
    """Deterministic separated, nonidentical single-support count observations."""
    if n < 1:
        raise ValueError("Scaling fixtures require positive node counts")
    i = np.arange(n)
    alt = np.asarray((100, 250, 400))[i % 3] + (i // 3) % 41
    return CountModel(tuple(f"m{j:06}" for j in i), alt.astype(np.float64),
                      (1000 - alt).astype(np.float64), np.full(n, 1e-6), np.ones(n),
                      np.full((n, 1), 0.5), np.zeros((n, 1)), np.ones((n, 1), dtype=bool), 1e-6)


def upload(host, device, kernels):
    arrays = {
        key: torch.tensor(getattr(host, key).copy(), dtype=torch.float64, device=device)
        for key in ("alt", "ref", "slope", "log_prior", "lower", "upper")
    }
    return TensorModel(host.mutation_ids, eps=host.eps, kernels=kernels, **arrays)


def assert_device(*values, device):
    if any(v.device != device or v.dtype != torch.float64 for v in values):
        raise AssertionError("Numerical fitted state left the selected float64 CUDA device")


def closeness(actual, expected, *, atol, rtol):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise AssertionError("Nonfinite or mismatched comparison arrays")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return error


def kernel_cases(device, eager, compiled, record):
    """Compiled and eager CUDA each compared directly with independent NumPy."""
    for values, expected in (([0.3, 0.3, 0.300015, 0.30003], [0, 0, 1, 2]),
                             ([0.3, 0.3, 0.300015, 0.30003, 0.30003], [0, 0, 1, 2, 2])):
        point = torch.tensor(values, dtype=torch.float64, device=device)
        _, _, labels, _ = grouping(point, CudaPolicy().fusion_tol)
        if not torch.equal(labels, torch.tensor(expected, dtype=labels.dtype, device=device)):
            raise AssertionError("Exact fused groups fragmented in an overwide tolerance run")
        record("exact_fusion_grouping", dict(raw_ccf=values, labels=labels.cpu().tolist(),
                                             expected=expected, qualified=True))
    host = host_fixture("mixed_multiplicity")
    record("fixture", dict(name="kernel_mixed_multiplicity", **host_identity(host)))
    n = len(host)
    first = host.slope[:, 0]
    maximum = np.max(host.slope, axis=1)
    probes = {
        "interior": np.linspace(0.12, 0.84, n),
        "original_lower": host.lower,
        "original_upper": host.upper,
        "exact_lower_clipping": host.eps / first,
        "exact_upper_clipping": (1 - host.eps) / maximum,
        "below_lower_clipping": np.nextafter(host.eps / first, -np.inf),
        "above_upper_clipping": np.nextafter((1 - host.eps) / maximum, np.inf),
    }
    for name, phi in probes.items():
        expected = evaluate(host, phi, derivatives=True)
        left, right = one_sided_derivatives(host, phi)
        references = (
            expected.loss,
            expected.gradient,
            expected.curvature,
            expected.posterior,
            left,
            right,
        )
        details = {}
        for mode, kernels in (("eager", eager), ("compiled", compiled)):
            model = upload(host, device, kernels)
            point = torch.tensor(phi.copy(), dtype=torch.float64, device=device)
            outputs, elapsed = timed(device, lambda: model.terms(point))
            assert_device(*outputs, device=device)
            errors = {}
            for label, actual, reference in zip(
                ("loss", "gradient", "curvature", "posterior", "left", "right"), outputs, references
            ):
                represented = actual.detach().cpu().numpy()
                try:
                    errors[label] = closeness(represented, reference, atol=2e-9, rtol=5e-11)
                except AssertionError as error:
                    record(
                        "likelihood_mismatch",
                        dict(
                            probe=name,
                            execution=mode,
                            field=label,
                            actual=represented.tolist(),
                            expected=np.asarray(reference).tolist(),
                            phi=phi.tolist(),
                            phi_hex=[float(value).hex() for value in phi],
                            slope_hex=[[float(value).hex() for value in row] for row in host.slope],
                            atol=2e-9,
                            rtol=5e-11,
                        ),
                    )
                    raise AssertionError(
                        f"{mode} likelihood probe={name} field={label}: {error}"
                    ) from error
            details[mode] = dict(
                maximum_absolute_errors=errors,
                seconds=elapsed,
                timing_scope="first shape call may include compilation",
            )
            smaller, reduced_seconds = timed(
                device, lambda: (model.loss(point), *model.loss_gradient(point))
            )
            assert_device(*smaller, device=device)
            reduced_errors = {}
            for label, value, full in zip(
                ("loss_only", "loss_gradient_loss", "loss_gradient_gradient"),
                smaller, (outputs[0], outputs[0], outputs[1]),
            ):
                reduced_errors[label] = closeness(
                    value.cpu().numpy(), full.cpu().numpy(), atol=2e-9, rtol=5e-11
                )
            details[mode].update(reduced_output_errors=reduced_errors,
                                 reduced_output_seconds=reduced_seconds)
        record("likelihood_parity", dict(probe=name, **details))

    for n in (1, 2, 7, 13):
        h = torch.linspace(0.5, 5.0, n, dtype=torch.float64, device=device)
        target = torch.linspace(0.12, 0.91, n, dtype=torch.float64, device=device)
        lower, upper = torch.zeros_like(h), torch.ones_like(h)
        graph = build_graph(target, tuple(f"m{i:04}" for i in range(n)))
        probe = differences(target)
        lap_error = float((adjoint(probe) - (n * target - target.sum())).abs().max())
        if lap_error > 2e-12:
            raise AssertionError("Complete-graph Laplacian identity failed")
        for fixed in (False, True):
            lo, hi = lower.clone(), upper.clone()
            if fixed:
                lo[0] = hi[0] = 0.73
            caps = graph.weights * 0.08
            outputs = {}
            for mode, kernels in (("eager", eager), ("compiled", compiled)):
                result, elapsed = timed(device, lambda: solve_qp(h, target, lo, hi, caps, kernels))
                assert_device(result.x, result.dual, device=device)
                detail = dict(
                    qualified=result.qualified,
                    gap=float(result.gap),
                    gap_scale=float(result.scale),
                    kkt=float(result.kkt),
                    iterations=result.iterations,
                    seconds=elapsed,
                    timing_scope="first shape call may include compilation",
                )
                record("qp_attempt", dict(n=n, fixed_coordinate=fixed, execution=mode, **detail))
                if not result.qualified:
                    raise AssertionError(
                        f"Unqualified {mode} complete-graph QP, n={n}, fixed={fixed}"
                    )
                outputs[mode] = result
            error = closeness(
                outputs["compiled"].x.cpu().numpy(),
                outputs["eager"].x.cpu().numpy(),
                atol=1e-8,
                rtol=1e-9,
            )
            if fixed and any(float(result.x[0]) != 0.73 for result in outputs.values()):
                raise AssertionError("Original fixed coordinate was changed")
            record(
                "qp_parity",
                dict(
                    n=n,
                    fixed_coordinate=fixed,
                    raw_max_absolute_error=error,
                    complete_graph_laplacian_error=lap_error,
                ),
            )


def qp_certificate(result, h, target, lower, upper, caps, kernels):
    """Recompute the full original-problem certificate independently of result flags."""
    stats = kernels.gap_kkt(result.x, result.dual, h, target, lower, upper, caps)
    p = CudaPolicy()
    detail = dict(qualified=result.qualified, gap=float(stats[0]), gap_scale=float(stats[1]),
                  kkt=float(stats[2]), iterations=result.iterations)
    if not (result.qualified and bool(torch.isfinite(stats).all())
            and detail["gap"] <= p.inner_atol + p.inner_rtol * detail["gap_scale"]
            and detail["kkt"] <= p.inner_kkt_tol):
        raise AssertionError(f"QP failed recomputed production gap/KKT gates: {detail}")
    return detail


def warm_qp_cases(device, eager, compiled, record):
    n = 13
    h = torch.linspace(1.0, 9.0, n, device=device, dtype=torch.float64)
    target = torch.linspace(0.1, 0.9, n, device=device, dtype=torch.float64)
    lower, upper = h * 0, h * 0 + 1
    lower[0] = upper[0] = 0.3
    caps = build_graph(target).weights * 0.15
    for mode, kernels in (("eager", eager), ("compiled", compiled)):
        original = solve_qp(h, target, lower, upper, caps, kernels)
        qp_certificate(original, h, target, lower, upper, caps, kernels)
        for changed in (False, True):
            hh, tt, cc = (h * 7, target.flip(0) + 0.1, caps * 0.35) if changed else (h, target, caps)
            outputs = {}
            for warm in (False, True):
                fitted, elapsed = timed(device, lambda: solve_qp(
                    hh, tt, lower, upper, cc, kernels, start=original.x,
                    dual=original.dual if warm else None))
                detail = qp_certificate(fitted, hh, tt, lower, upper, cc, kernels)
                record("qp_warm_attempt", dict(execution=mode, changed_problem=changed,
                       warm=warm, seconds=elapsed, **detail))
                outputs[warm] = fitted
            difference = closeness(outputs[True].x.cpu().numpy(), outputs[False].x.cpu().numpy(),
                                   atol=2e-6, rtol=1e-8)
            record("qp_warm_parity", dict(execution=mode, changed_problem=changed,
                   raw_max_absolute_error=difference, literal_problem_identical=True,
                   cold_iterations=outputs[False].iterations, warm_iterations=outputs[True].iterations))


def canonical_labels(fit):
    centers = fit.refit.centers
    order = torch.argsort(centers, descending=True, stable=True)
    order = torch.cat((order[order == fit.refit.clonal], order[order != fit.refit.clonal]))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return inverse[fit.refit.labels], centers[order]


def tensor_sha(value):
    array = value.detach().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def graph_metadata(fit):
    """Verify each actual graph from its own frozen pilot; never equate two pilots."""
    fit.graph.validate()
    recreated = build_graph(fit.pilot.phi, fit.model.mutation_ids)
    for name in ("weights", "pilot", "gap_floor", "normalization"):
        if not torch.equal(getattr(recreated, name), getattr(fit.graph, name)):
            raise AssertionError(f"Frozen graph recipe mismatch: {name}")
    # Independent NumPy recipe oracle uses upper-triangle pairs, whereas the
    # production graph stores both orientations and normalizes their dense sum.
    pilot_array = fit.pilot.phi.detach().cpu().numpy()
    adjacent = np.diff(np.sort(pilot_array))
    positive = adjacent[adjacent > 0]
    expected_floor = max(1e-8, 0.1 * float(np.median(positive))) if positive.size else 1e-8
    distances = np.abs(pilot_array[:, None] - pilot_array[None, :])
    raw = 1 / np.maximum(distances, expected_floor)
    np.fill_diagonal(raw, 0.0)
    pairs = np.triu_indices(pilot_array.size, 1)
    expected_normalization = float(raw[pairs].mean()) if pairs[0].size else 1.0
    expected_weights = raw / expected_normalization
    rounding = 128 * np.finfo(np.float64).eps
    oracle_error = closeness(
        fit.graph.weights.cpu().numpy(), expected_weights, atol=rounding, rtol=rounding
    )
    closeness(float(fit.graph.gap_floor), expected_floor, atol=0.0, rtol=rounding)
    closeness(float(fit.graph.normalization), expected_normalization, atol=0.0, rtol=rounding)
    policy = CudaPolicy()
    if not bool(
        fit.pilot.qualified.all()
        & (fit.pilot.gap >= 0).all()
        & (fit.pilot.gap <= policy.scalar_atol + policy.scalar_rtol * fit.pilot.loss.abs()).all()
    ):
        raise AssertionError("Graph pilots do not meet their unchanged scalar gap gate")
    return dict(
        pilot_phi=fit.pilot.phi.cpu().tolist(),
        pilot_loss=fit.pilot.loss.cpu().tolist(),
        pilot_lower_bounds=fit.pilot.lower_bound.cpu().tolist(),
        pilot_gaps=fit.pilot.gap.cpu().tolist(),
        pilot_sha256=tensor_sha(fit.pilot.phi),
        pilot_subdivisions=fit.pilot.subdivisions,
        weight_rule=fit.graph.weight_rule,
        gap_floor=float(fit.graph.gap_floor),
        normalization=float(fit.graph.normalization),
        weights=fit.graph.weights.cpu().tolist(),
        weights_sha256=tensor_sha(fit.graph.weights),
        selected_caps_sha256=tensor_sha(fit.graph.weights * fit.lambda_value),
        recipe_recomputed_exactly=True,
        independent_numpy_recipe_max_weight_error=oracle_error,
        independent_numpy_recipe_rtol=rounding,
        independent_numpy_recipe_atol=rounding,
    )


def path_position(fit, path_grid):
    """Discrete path identity; independently derived reference scales may differ."""
    selected = float(fit.lambda_value)
    locations = [i for i, row in enumerate(fit.records) if row["lambda_value"] == selected]
    if len(locations) != 1:
        raise AssertionError("Selected lambda must have one exact recorded path position")
    index = locations[0]
    reference = float(fit.timings["lambda_reference"])
    recipe = {}
    if path_grid == "default":
        policy = fit.policy
        device_reference = fit.lambda_value.new_tensor(reference)
        coordinates = [None]
        expected = [device_reference.new_tensor(0.0)]
        host_expected = [0.0]
        if fit.model.n > 1:
            for exponent in range(policy.path_min_exponent, policy.path_max_exponent + 1):
                coordinates.append(exponent)
                # Match the actual device operation, not host math.ldexp. CUDA
                # torch.ldexp can differ from it by one ULP even for normal values.
                expected.append(
                    torch.ldexp(device_reference, torch.tensor(exponent, device=device_reference.device))
                )
                host_expected.append(math.ldexp(reference, exponent))
            extensions = fit.timings["extensions"]
            if not 0 <= extensions <= policy.path_extensions:
                raise AssertionError("Recorded path extensions exceed its actual policy")
            for _ in range(extensions):
                coordinates.append(coordinates[-1] + 1)
                # Production extensions double the preceding device value.
                expected.append(expected[-1] * 2.0)
                host_expected.append(host_expected[-1] * 2.0)
        expected = [float(value) for value in expected]
        if len(fit.records) != len(expected):
            raise AssertionError("Recorded default path length differs from its policy recipe")
        mismatches = [
            dict(index=i, coordinate=coordinates[i], actual=float(row["lambda_value"]).hex(),
                 expected=value.hex())
            for i, (row, value) in enumerate(zip(fit.records, expected))
            if row["lambda_value"] != value
        ]
        if mismatches:
            raise AssertionError(
                f"Recorded default path differs from its own exact device recipe: {mismatches}"
            )
        recipe = dict(
            recipe_recomputed_exactly=True,
            recipe="device torch.ldexp for base grid; repeated device multiplication by two for extensions",
            policy_min_exponent=policy.path_min_exponent,
            policy_max_exponent=policy.path_max_exponent,
            host_ldexp_differences_ulps=[
                (value - host) / math.ulp(host) for value, host in zip(expected, host_expected)
            ],
        )
    else:
        coordinates = [row["lambda_value"] for row in fit.records]
    return dict(
        selected_index=index,
        selected_coordinate=coordinates[index],
        coordinates=coordinates,
        lambda_reference=reference,
        actual_selected_lambda=selected,
        **recipe,
    )


def compare_fit_outputs(actual, reference):
    """The original raw/refit/score/objective parity gates, without a lambda assumption."""
    if actual.search_status != reference.search_status or actual.search_status != "complete":
        raise AssertionError("Compiled/eager planned search does not completely qualify")
    errors = {
        "raw_ccf": closeness(
            actual.raw.x.cpu().numpy(), reference.raw.x.cpu().numpy(), atol=2e-5, rtol=1e-8
        ),
        "refitted_ccf": closeness(
            actual.refit.phi.cpu().numpy(), reference.refit.phi.cpu().numpy(), atol=2e-5, rtol=1e-8
        ),
        "score": closeness(
            float(actual.refit.score), float(reference.refit.score), atol=1e-7, rtol=1e-10
        ),
        "raw_objective": closeness(
            float(actual.raw.objective), float(reference.raw.objective), atol=1e-7, rtol=1e-10
        ),
    }
    actual_labels, actual_centers = canonical_labels(actual)
    reference_labels, reference_centers = canonical_labels(reference)
    if not torch.equal(actual_labels, reference_labels):
        raise AssertionError("Compiled/eager selected cluster-label mismatch")
    errors["cluster_centers"] = closeness(
        actual_centers.cpu().numpy(), reference_centers.cpu().numpy(), atol=2e-5, rtol=1e-8
    )
    for field in ("raw", "refit"):
        ax = actual.raw.x if field == "raw" else actual.refit.phi
        rx = reference.raw.x if field == "raw" else reference.refit.phi
        if not torch.equal(
            actual.model.terms(ax)[3].argmax(-1), reference.model.terms(rx)[3].argmax(-1)
        ):
            raise AssertionError(f"Compiled/eager {field}-conditional multiplicity mismatch")
    return errors


def independent_graph_differences(actual, reference):
    aw, rw = actual.graph.weights, reference.graph.weights
    ac, rc = aw * actual.lambda_value, rw * reference.lambda_value
    lo, hi = reference.model.lower, reference.model.upper
    diameter = torch.maximum((hi[:, None] - lo[None, :]).abs(), (hi[None, :] - lo[:, None]).abs())
    envelope = 0.5 * ((ac - rc).abs() * diameter).sum()
    perturbations = {
        name: float(0.5 * ((ac - rc) * differences(value).abs()).sum())
        for name, value in (
            ("eager_selected_state", reference.raw.x),
            ("compiled_selected_state", actual.raw.x),
        )
    }
    return dict(
        pilot_phi_max_difference=float((actual.pilot.phi - reference.pilot.phi).abs().max()),
        pilot_loss_max_difference=float((actual.pilot.loss - reference.pilot.loss).abs().max()),
        pilot_gap_max_difference=float((actual.pilot.gap - reference.pilot.gap).abs().max()),
        gap_floor_difference=float(actual.graph.gap_floor - reference.graph.gap_floor),
        normalization_difference=float(actual.graph.normalization - reference.graph.normalization),
        weights_max_difference=float((aw - rw).abs().max()),
        weights_max_relative_difference=float(
            torch.where(
                rw > 0, (aw - rw).abs() / rw.clamp_min(torch.finfo(rw.dtype).tiny), 0.0
            ).max()
        ),
        selected_caps_max_difference=float((ac - rc).abs().max()),
        selected_lambda_difference=float(actual.lambda_value - reference.lambda_value),
        fusion_objective_perturbation_envelope=float(envelope),
        exact_fusion_objective_perturbations=perturbations,
        graphs_identical=bool(torch.equal(aw, rw)),
        note="Scalar loss-gap qualification alone does not bound pilot locations or adaptive-weight differences",
    )


def fit_summary(fit):
    labels, centers = canonical_labels(fit)
    return dict(
        raw_qualified=fit.raw.qualified,
        search_status=fit.search_status,
        selected_lambda=float(fit.lambda_value),
        score=float(fit.refit.score),
        raw_objective=float(fit.raw.objective),
        raw_ccf=fit.raw.x.cpu().tolist(),
        refitted_ccf=fit.refit.phi.cpu().tolist(),
        cluster_labels=labels.cpu().tolist(),
        cluster_centers=centers.cpu().tolist(),
        selected_raw_diagnostics=fit.raw.diagnostics,
        path_records=fit.records,
        timings=fit.timings,
        clonal_constraint=False,
        graph_edges=fit.graph.edges,
        pilot_and_graph=graph_metadata(fit),
    )


def matched_graph_cases(name, reference, actual, device, artifacts, record):
    """Same literal pilots, weights, bounds and lambdas; both backends refit independently."""
    for key in ("alt", "ref", "slope", "log_prior", "lower", "upper"):
        if not torch.equal(getattr(reference.model, key), getattr(actual.model, key)):
            raise AssertionError("Matched replay likelihood inputs differ")
    positive = [row["lambda_value"] for row in reference.records if row["lambda_value"] > 0]
    values = sorted(
        set(
            [
                0.0,
                positive[0],
                math.ldexp(reference.timings["lambda_reference"], -4),
                float(reference.lambda_value),
            ]
        )
    )
    shared_graph, shared_pilot = reference.graph, reference.pilot
    graph_digest, pilot_digest = tensor_sha(shared_graph.weights), tensor_sha(shared_pilot.phi)
    nonzero_nonfused = False
    for index, value in enumerate(values):
        lam = reference.lambda_value.new_tensor(value)
        outputs = {}
        for mode, model in (("eager", reference.model), ("compiled", actual.model)):
            record(
                "matched_graph_started",
                dict(
                    fixture=name,
                    execution=mode,
                    lambda_value=value,
                    weights_sha256=graph_digest,
                    pilot_sha256=pilot_digest,
                ),
            )

            def execute():
                raw = fit_lambda(model, shared_graph, shared_pilot, lam)
                secondary = refit(model, raw.x)
                complete = raw.qualified and raw.diagnostics.get("search_complete", False)
                return SimpleNamespace(
                    model=model,
                    graph=shared_graph,
                    pilot=shared_pilot,
                    raw=raw,
                    refit=secondary,
                    lambda_value=lam,
                    search_status="complete" if complete else "incomplete",
                    records=[],
                    timings={
                        "scope": "fixed shared pilot, graph and literal lambda; no prior continuation"
                    },
                )

            fitted, elapsed = timed(device, execute)
            detail_file = artifacts / f"matched-{name}-{index}-{mode}.json"
            summary = fit_summary(fitted)
            summary.update(
                seconds=elapsed,
                literal_lambda=value,
                compilation_banks=model.kernels.compilation_diagnostics(),
            )
            write_json(detail_file, summary)
            record(
                "matched_graph_attempt",
                dict(
                    fixture=name,
                    execution=mode,
                    lambda_value=value,
                    seconds=elapsed,
                    raw_qualified=fitted.raw.qualified,
                    search_status=fitted.search_status,
                    detail_file=str(detail_file),
                    detail_sha256=sha(detail_file),
                ),
            )
            outputs[mode] = fitted
        errors = compare_fit_outputs(outputs["compiled"], outputs["eager"])
        # A fixed statistical problem has exactly one literal lambda, not a widened scale tolerance.
        errors["literal_lambda"] = closeness(
            float(outputs["compiled"].lambda_value),
            float(outputs["eager"].lambda_value),
            atol=0.0,
            rtol=0.0,
        )
        penalty = float(
            0.5 * (shared_graph.weights * lam * differences(outputs["eager"].raw.x).abs()).sum()
        )
        nonzero_nonfused |= value > 0 and penalty > 0
        shared_graph.validate()
        if graph_digest != tensor_sha(shared_graph.weights) or pilot_digest != tensor_sha(
            shared_pilot.phi
        ):
            raise AssertionError("Matched replay altered the shared graph or pilots")
        record(
            "matched_graph_parity",
            dict(
                fixture=name,
                lambda_value=value,
                errors=errors,
                fusion_penalty=penalty,
                weights_sha256=graph_digest,
                pilot_sha256=pilot_digest,
                identical_graph_and_lambda=True,
                labels_equal=True,
                raw_and_refit_multiplicities_equal=True,
            ),
        )
    if not nonzero_nonfused:
        raise AssertionError(
            "Matched parity did not exercise a nonzero penalty on a nonfused state"
        )


def pipeline_cases(device, eager, compiled, path_grid, artifacts, record):
    paths = None if path_grid == "default" else [0.0, 0.1, 0.5]
    for name in ("single_support", "mixed_multiplicity", "all_bounds_below_one"):
        host = host_fixture(name)
        record("fixture", dict(name=name, **host_identity(host)))
        fits = {}
        for mode, kernels in (("eager", eager), ("compiled", compiled)):
            model = upload(host, device, kernels)
            record("pipeline_started", dict(fixture=name, execution=mode, path_grid=path_grid))
            torch.cuda.reset_peak_memory_stats(device)
            fit, elapsed = timed(device, lambda: fit_tensor_model(model, lambda_values=paths))
            summary = fit_summary(fit)
            summary.update(
                seconds=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                path_grid=path_grid,
                compilation_banks=kernels.compilation_diagnostics(),
                timing_scope="end-to-end numerical path; compiled run includes uncached compilation",
            )
            detail_file = artifacts / f"pipeline-{name}-{mode}.json"
            write_json(detail_file, summary)
            record(
                "pipeline_attempt",
                dict(
                    fixture=name,
                    execution=mode,
                    detail_file=str(detail_file),
                    detail_sha256=sha(detail_file),
                    search_status=fit.search_status,
                    raw_qualified=fit.raw.qualified,
                    seconds=elapsed,
                    planned_lambdas=len(fit.records),
                ),
            )
            assert_device(fit.raw.x, fit.raw.dual, fit.refit.phi, fit.refit.centers, device=device)
            if (
                not fit.raw.qualified
                or fit.raw.witness is not None
                or fit.raw.diagnostics.get("clonal_constraint") is not False
            ):
                raise AssertionError(
                    "Pipeline did not return a qualified unconstrained raw solution"
                )
            fits[mode] = fit
        ref, actual = fits["eager"], fits["compiled"]
        positions = {
            "eager": path_position(ref, path_grid),
            "compiled": path_position(actual, path_grid),
        }
        differences_record = independent_graph_differences(actual, ref)
        record(
            "independent_graph_comparison",
            dict(fixture=name, path_positions=positions, **differences_record),
        )
        if (
            positions["eager"]["coordinates"] != positions["compiled"]["coordinates"]
            or positions["eager"]["selected_index"] != positions["compiled"]["selected_index"]
        ):
            raise AssertionError(
                "Independent default pipelines selected different discrete path positions"
            )
        errors = compare_fit_outputs(actual, ref)
        record(
            "pipeline_parity",
            dict(
                fixture=name,
                errors=errors,
                labels_equal=True,
                raw_and_refit_multiplicities_equal=True,
                search_status=actual.search_status,
                selected_path_index=positions["eager"]["selected_index"],
                selected_path_coordinate=positions["eager"]["selected_coordinate"],
                no_exact_one_feasible=bool(np.all(host.upper < 1.0)),
            ),
        )
        matched_graph_cases(name, ref, actual, device, artifacts, record)


def validate_public_search(result, receipt, device):
    """Independently reconcile the published default path with its public result."""
    from clipp1d.cuda_api import SCHEMA as run_schema

    if (receipt.get("schema") != run_schema or receipt.get("status") != "success" or
            receipt.get("search_status") != "complete" or result.search_status != "complete"):
        raise AssertionError("Public default qualification path is incomplete or has an invalid receipt schema")
    n = len(result.mutation_ids)
    if (n == 0 or tuple(sorted(set(result.mutation_ids))) != result.mutation_ids or
            any(np.asarray(value).shape != (n,) or not np.all(np.isfinite(value))
                for value in (result.raw_phi, result.refitted_phi))):
        raise AssertionError("Public qualification requires finite canonical retained raw/refit arrays")
    source = receipt.get("provenance", {})
    if (source.get("source_sha256") != result.provenance.get("source_sha256") or
            source.get("model_sha256") != result.provenance.get("model_sha256") or
            receipt.get("graph_sha256") != result.graph_sha256 or
            receipt.get("selected_lambda") != result.selected_lambda or
            receipt.get("raw_objective") != result.raw_objective or
            receipt.get("selection_score") != result.selection_score or
            receipt.get("candidate_provenance") != clean(result.candidate_provenance) or
            receipt.get("raw_diagnostics") != clean(result.raw_diagnostics)):
        raise AssertionError("Public receipt/result scientific identities or selected states differ")
    lineage, raw = result.candidate_provenance, result.raw_diagnostics
    if (lineage.get("raw_qualified") is not True or lineage.get("refit_qualified") is not True or
            raw.get("box_feasible") is not True or raw.get("clonal_constraint") is not False or
            not (raw.get("raw_branch_stationarity_qualified") is True or
                 raw.get("separable_scalar_gap_qualified") is True)):
        raise AssertionError("Public selected raw/refit arrays lack their own qualification")
    if source.get("policy") != asdict(CudaPolicy()) or result.provenance.get("policy") != source["policy"]:
        raise AssertionError("Public fixture did not execute the full default inference policy")
    records = receipt.get("search")
    if not isinstance(records, list) or not records or any(
            row.get("raw_status") != "qualified" or row.get("refit_status") != "qualified" or
            row.get("search_complete") is not True for row in records):
        raise AssertionError("Public default path contains unresolved or omitted qualification records")
    # Reuse the separately tested exact-device grid oracle; this does not rerun
    # any fit or infer completion from the receipt's summary status alone.
    public_path = SimpleNamespace(
        lambda_value=torch.tensor(result.selected_lambda, dtype=torch.float64, device=device),
        records=records, timings=source["numerical_stages"], model=SimpleNamespace(n=n),
        policy=CudaPolicy(**source["policy"]),
    )
    return path_position(public_path, "default")


def validate_public_measurements(result, receipt):
    """A durable receipt ends before serializing itself; return metrics end later."""
    source = receipt["provenance"]
    phases = source.get("phase_seconds", {})
    required = ("input_preparation_seconds", "device_upload_and_compile_seconds", "pilot_seconds",
                "graph_build_seconds", "path_seconds", "refit_seconds",
                "final_device_qualification_seconds", "device_export_seconds", "output_preparation_seconds")
    if any(key not in phases or not math.isfinite(phases[key]) or phases[key] < 0 for key in required):
        raise AssertionError("Public timing phases are missing, negative, or nonfinite")
    metrics = result.operation_metrics
    if not isinstance(metrics, dict) or any(
            key not in metrics or not math.isfinite(metrics[key]) or metrics[key] < 0
            for key in ("publication_seconds", "elapsed_seconds")):
        raise AssertionError("Return-only publication completion metrics are missing or invalid")
    numerical = datetime.fromisoformat(source["numerical_completed_utc"])
    prepared = datetime.fromisoformat(source["output_prepared_utc"])
    published = datetime.fromisoformat(metrics["publication_completed_utc"])
    if any(value.tzinfo is None for value in (numerical, prepared, published)) or not numerical <= prepared <= published:
        raise AssertionError("Numerical/preparation/publication timestamps are not ordered aware times")
    if (source.get("device_measurement_scope") != "upload through final qualification and completed device export"
            or "excludes receipt serialization" not in source.get("elapsed_scope", "")
            or "return-only operation_metrics" not in source.get("publication_completion_scope", "")
            or source["elapsed_seconds"] > metrics["elapsed_seconds"]
            or source["through_device_export_seconds"] > source["elapsed_seconds"]
            or "operation_metrics" in receipt):
        raise AssertionError("Public measurement scope improperly claims self-timed durable completion")
    if any(not isinstance(source.get(key), int) or source[key] < 0
           for key in ("peak_allocated_bytes", "peak_reserved_bytes")):
        raise AssertionError("Final CUDA memory peaks are missing or invalid")
    return dict(phase_seconds=phases, numerical_completed_utc=source["numerical_completed_utc"],
                output_prepared_utc=source["output_prepared_utc"],
                return_only_operation_metrics=metrics,
                device_measurement_scope=source["device_measurement_scope"],
                receipt_elapsed_scope=source["elapsed_scope"],
                peak_allocated_bytes=source["peak_allocated_bytes"],
                peak_reserved_bytes=source["peak_reserved_bytes"],
                scalar_work_counters=source.get("scalar_work_counters", {}),
                integrity_counters=source.get("integrity_counters", {}))


def public_case(device, artifacts, record):
    """Exercise the actual input/fit/default path/publication/readback contract."""
    input_file = artifacts / "public-input.tsv"
    rows = []
    # Noncanonical input order tests the public mutation-ID canonicalization.
    for i in (2, 0, 3, 1):
        alt = (20, 21, 38, 39)[i]
        rows.append([f"m{i:04}", "sample1", alt, 100 - alt, 1, 1.0, 2, f"seg{i}", "cn1", 1.0, 2, 0])
    with input_file.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(SCHEMA_COLUMNS)
        writer.writerows(rows)
    destination = artifacts / "public-output"
    result, elapsed = timed(device, lambda: public_fit(input_file, destination, device=str(device)))
    receipt = json.loads((destination / "run.json").read_text())
    public_path = validate_public_search(result, receipt, device)
    measurements = validate_public_measurements(result, receipt)
    if (
        receipt["status"] != "success"
        or result.raw_witness_mutation_id is not None
        or result.provenance.get("clonal_constraint") is not False
        or not result.provenance.get("compiled_inference")
        or not np.all(result.original_upper_bounds < 1.0)
        or not np.all(result.raw_phi <= result.original_upper_bounds)
        or receipt["provenance"].get("input_sha256") != sha(input_file)
        or receipt["provenance"].get("numerical_device") != str(device)
        or receipt["provenance"].get("execution_scope") != "cuda_inference"
    ):
        raise AssertionError("Public CUDA/unconstrained/original-domain contract failed")
    names = ("mutation_clusters.tsv", "cluster_centers.tsv", "mutation_multiplicity.tsv")
    if set(receipt["table_sha256"]) != set(names) or {p.name for p in destination.iterdir()} != set(names) | {"run.json"}:
        raise AssertionError("Public output file schema differs from the declared minimal bundle")
    tables = {}
    for name in names:
        path = destination / name
        if sha(path) != receipt["table_sha256"][name]:
            raise AssertionError(f"Public table hash mismatch: {name}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            tables[name] = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in tables[name]):
            raise AssertionError(f"Public TSV has inconsistent column counts: {name}")
    lookup = {mid: i for i, mid in enumerate(result.mutation_ids)}
    mutations = tables[names[0]]
    if len(mutations) != 4 or {row["mutation_id"] for row in mutations} != set(lookup):
        raise AssertionError("Public mutation identity coverage differs")
    for row in mutations:
        i = lookup[row["mutation_id"]]
        if (
            float(row["raw_ccf"]) != result.raw_phi[i]
            or float(row["refitted_ccf"]) != result.refitted_phi[i]
            or int(row["cluster_label"]) != result.cluster_labels[i]
            or int(row["designated_clonal"]) != int(result.cluster_labels[i] == 0)
        ):
            raise AssertionError("Public mutation/raw/refit/label readback mismatch")
    if len(tables[names[1]]) != len(result.cluster_centers) or len(tables[names[2]]) != 4:
        raise AssertionError("Public center/multiplicity table coverage differs")
    for row in tables[names[1]]:
        label = int(row["cluster_label"])
        if (
            float(row["refitted_ccf"]) != result.cluster_centers[label]
            or int(row["cluster_size"]) != int(np.count_nonzero(result.cluster_labels == label))
            or int(row["designated_clonal"]) != int(label == 0)
        ):
            raise AssertionError("Public center readback mismatch")
    for row in tables[names[2]]:
        i = lookup[row["mutation_id"]]
        if (
            int(row["raw_multiplicity_call"]) != result.multiplicity_calls[i]
            or int(row["refitted_multiplicity_call"]) != result.refitted_multiplicity_calls[i]
        ):
            raise AssertionError("Public multiplicity readback mismatch")
    if abs(result.cluster_centers[0] - 1.0) != np.min(np.abs(result.cluster_centers - 1.0)):
        raise AssertionError("Public label0 does not designate the closest refitted center")
    record(
        "public_publication",
        dict(
            input_sha256=sha(input_file),
            directory=str(destination),
            run_sha256=sha(destination / "run.json"),
            table_sha256=receipt["table_sha256"],
            search_status=result.search_status,
            seconds=elapsed,
            selected_lambda=result.selected_lambda,
            clonal_constraint=False,
            every_original_upper_bound_below_one=True,
            default_path_recipe=public_path,
            measurements=measurements,
        ),
    )


def resource_probe(device, compiled, n, iterations, record, artifacts=None):
    """Both synthetic QP repetitions must qualify under the production budget."""
    if iterations != CudaPolicy().inner_max_iterations or n < 1:
        raise ValueError("Resource qualification requires positive N and the unchanged production QP budget")
    dtype = torch.float64
    i = torch.arange(n, device=device, dtype=dtype)
    target = 0.15 + 0.35 * (i % 3) + 0.003 * torch.sin(i)
    h = 80 + 40 * (i % 7)
    lower, upper = torch.full_like(i, 1e-6), torch.ones_like(i)
    graph, build_seconds = timed(device, lambda: build_graph(target))
    caps = graph.weights * 0.002
    policy = CudaPolicy()
    baseline = torch.cuda.memory_allocated(device)
    results = []
    for repeat in range(2):
        torch.cuda.reset_peak_memory_stats(device)
        result, elapsed = timed(
            device, lambda: solve_qp(h, target, lower, upper, caps, compiled, policy)
        )
        results.append(
            dict(
                repeat=repeat,
                seconds=elapsed,
                qp_qualified=result.qualified,
                iterations=result.iterations,
                polish_iterations=result.polish_iterations,
                gap=float(result.gap),
                gap_scale=float(result.scale),
                kkt=float(result.kkt),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        )
        record("resource_qp_attempt", dict(nodes=n, **results[-1]))
        if not result.qualified and artifacts is not None:
            # Preserve the literal failed QP for diagnosis; never promote it by
            # increasing the iteration budget or substituting another fixture.
            state = {name: value.cpu().tolist() for name, value in
                     (("h", h), ("target", target), ("lower", lower), ("upper", upper),
                      ("caps", caps), ("x", result.x), ("dual", result.dual))}
            failure_file = artifacts / f"resource-{n}-repeat{repeat}-unqualified.json"
            write_json(failure_file, dict(state=state, certificate=results[-1]))
            record("resource_failure_state", dict(artifact=str(failure_file),
                   artifact_sha256=sha(failure_file), nodes=n, repeat=repeat))
        qp_certificate(result, h, target, lower, upper, caps, compiled)
    record(
        "resource_probe",
        dict(
            status="qualified",
            nodes=n,
            edges=n * (n - 1) // 2,
            one_dense_float64_matrix_bytes=8 * n * n,
            graph_build_seconds=build_seconds,
            baseline_allocated_bytes=baseline,
            bounded_qp_iteration_budget=iterations,
            repeats=results,
            scope="synthetic complete-graph QP only; no cohort accuracy or full-fit speed claim",
            timing_scope="repeat0 may include compilation; repeat1 uses warmed kernel shapes",
        ),
    )


def scaling_cases(device, compiled, artifacts, record):
    """Increasing complete default fits; require final device and host validation."""
    from clipp1d.cuda_api import _export, _validate_result

    for n in (16, 64, 256):
        host = scaling_fixture(n)
        record("scaling_started", dict(nodes=n, execution="compiled", path_grid="default",
                                       fixture=host_identity(host)))
        torch.cuda.reset_peak_memory_stats(device)
        began = perf_counter()
        model, upload_seconds = timed(device, lambda: upload(host, device, compiled))
        fitted, fit_seconds = timed(device, lambda: fit_tensor_model(model))
        before_export = artifacts / f"scaling-{n}-path.json"
        summary = fit_summary(fitted)
        summary.update(fit_seconds=fit_seconds, path_position=path_position(fitted, "default"))
        write_json(before_export, summary)
        record("scaling_path_complete", dict(nodes=n, search_status=fitted.search_status,
               seconds=fit_seconds, artifact=str(before_export), artifact_sha256=sha(before_export)))
        if fitted.search_status != "complete":
            raise AssertionError(f"Scaling default path N={n} contains unresolved states")
        source = source_provenance()
        phases = dict(input_preparation_seconds=0.0, device_upload_and_compile_seconds=upload_seconds)
        phases.update({key: fitted.timings[key] for key in
                       ("pilot_seconds", "graph_build_seconds", "path_seconds", "refit_seconds")})
        source.update(backend="cuda", clonal_constraint=False, policy=asdict(CudaPolicy()),
                      numerical_stages=fitted.timings, phase_seconds=phases)
        exported = _export(fitted, source, wall_started=began)
        data = SimpleNamespace(mutations=[SimpleNamespace(mutation_id=mid, exclusion=None)
                                          for mid in host.mutation_ids])
        _validate_result(exported, data, fitted.records)
        detail = dict(nodes=n, qualified=True, search_status=exported.search_status,
                      selected_lambda=exported.selected_lambda, raw_objective=exported.raw_objective,
                      score=exported.selection_score, raw_ccf=exported.raw_phi.tolist(),
                      refitted_ccf=exported.refitted_phi.tolist(), labels=exported.cluster_labels.tolist(),
                      cluster_centers=exported.cluster_centers.tolist(), provenance=exported.provenance,
                      phase_seconds=exported.provenance["phase_seconds"], numerical_stages=fitted.timings,
                      model_integrity_counters=dict(model.integrity_counters),
                      graph_integrity_counters=dict(fitted.graph.integrity_counters),
                      scalar_work_counters=dict(model.scalar_work_counters),
                      peak_allocated_bytes=exported.provenance["peak_allocated_bytes"],
                      peak_reserved_bytes=exported.provenance["peak_reserved_bytes"],
                      final_device_qualification_and_export_complete=True,
                      seconds=perf_counter() - began,
                      scope="synthetic complete default path plus final qualification/export; no cohort accuracy claim")
        detail_file = artifacts / f"scaling-{n}-qualified.json"
        write_json(detail_file, detail)
        record("scaling_qualified", {key: value for key, value in detail.items()
               if key not in ("provenance", "raw_ccf", "refitted_ccf", "labels", "cluster_centers")}
               | dict(artifact=str(detail_file), artifact_sha256=sha(detail_file)))


@torch.no_grad()
def run(args, receipt, record, artifacts):
    device = require_cuda(args.device)
    compiler = os.environ.get("CC")
    if not compiler or shutil.which(compiler) is None:
        raise RuntimeError("Set CC to an available C compiler before CUDA qualification")
    props = torch.cuda.get_device_properties(device)
    receipt.update(
        device=str(device),
        gpu=props.name,
        capability=list(torch.cuda.get_device_capability(device)),
        total_device_bytes=props.total_memory,
        cuda_runtime=torch.version.cuda,
        cc=compiler,
        cc_resolved=shutil.which(compiler),
        cuda_available=True,
    )
    eager = Kernels(device, compiled=False)
    compiled, compile_seconds = timed(device, lambda: Kernels(device, compiled=True))
    if not compiled.compiled:
        raise AssertionError("Actual CUDA compiled-kernel admission did not occur")
    record(
        "compiler_admission",
        dict(
            compiled=True,
            seconds=compile_seconds,
            backend="inductor",
            dynamic_tensor_shapes=True,
            fullgraph=True,
            dtype="float64",
        ),
    )
    record("stage_started", dict(stage="independent_likelihood_and_qp"))
    kernel_cases(device, eager, compiled, record)
    warm_qp_cases(device, eager, compiled, record)
    record(
        "compilation_banks",
        dict(stage="after_kernel_cases", banks=compiled.compilation_diagnostics()),
    )
    # Exercise the observed convergence boundary before the longer complete
    # pipeline checks. This reorders the same acceptance inventory only.
    record("stage_started", dict(stage="bounded_resource_probe"))
    resource_probe(device, compiled, args.resource_n, args.resource_iterations, record, artifacts)
    record("stage_started", dict(stage="increasing_complete_default_fits"))
    scaling_cases(device, compiled, artifacts, record)
    record("stage_started", dict(stage="eager_compiled_complete_pipeline"))
    pipeline_cases(device, eager, compiled, args.path_grid, artifacts, record)
    record("stage_started", dict(stage="public_default_path_and_publication"))
    public_case(device, artifacts, record)
    record("compilation_banks", dict(stage="final", banks=compiled.compilation_diagnostics()))
    current = source_provenance()
    if (
        current["source_sha256"] != receipt["source"]["source_sha256"]
        or sha(__file__) != receipt["qualification_script_sha256"]
    ):
        raise RuntimeError("Source changed during qualification; the evidence is not source-bound")
    receipt.update(
        status="passed",
        finished_utc=utc_now(),
        scope="actual CUDA eager/compiled and warm-QP parity, complete small paths, public publication, qualified production-budget QPs and complete synthetic N16/64/256 paths; no cohort accuracy claim",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--path-grid",
        choices=("default", "short"),
        default="default",
        help="Numerical comparison grid; public fit always uses the default policy path",
    )
    parser.add_argument(
        "--resource-n",
        type=int,
        default=256,
        help="Required qualified synthetic QP nodes",
    )
    parser.add_argument("--resource-iterations", type=int, default=CudaPolicy().inner_max_iterations,
                        help="Must equal the unchanged production QP budget")
    parser.add_argument("--timeout-seconds", type=int, default=2700)
    args = parser.parse_args()
    if (
        args.resource_n < 1
        or args.resource_n > 4096
        or args.resource_iterations != CudaPolicy().inner_max_iterations
        or args.timeout_seconds < 1
    ):
        parser.error(
            "Require 1<=resource-n<=4096, production resource-iterations and positive timeout-seconds"
        )
    out = args.out.resolve()
    events = out.with_suffix(".events.jsonl")
    artifacts = out.with_suffix(".artifacts")
    if out.exists() or events.exists() or artifacts.exists():
        raise FileExistsError("Qualification evidence is never overwritten or silently resumed")
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    receipt = dict(
        schema=SCHEMA,
        status="running",
        started_utc=utc_now(),
        source=source_provenance(),
        qualification_script_sha256=sha(__file__),
        policy=asdict(CudaPolicy()),
        requested_device=args.device,
        command=sys.argv,
        torch=torch.__version__,
        python=platform.python_version(),
        interpreter=sys.executable,
        lsf_job_id=os.environ.get("LSB_JOBID"),
        lsf_queue=os.environ.get("LSB_QUEUE"),
        cuda_available=torch.cuda.is_available(),
        cuda_runtime=torch.version.cuda,
        path_grid=args.path_grid,
        dynamo_recompile_limit=torch._dynamo.config.recompile_limit,
        dynamo_accumulated_recompile_limit=torch._dynamo.config.accumulated_recompile_limit,
        cases=[],
    )
    write_json(artifacts / "source-and-plan.json", receipt)
    begin = perf_counter()
    with events.open("x", encoding="utf-8") as log:

        def record(kind, detail):
            entry = dict(kind=kind, utc=utc_now(), elapsed_seconds=perf_counter() - begin, **detail)
            receipt["cases"].append(entry)
            log.write(json.dumps(clean(entry), sort_keys=True, allow_nan=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            print(json.dumps(clean(entry), sort_keys=True, allow_nan=False), flush=True)

        def interrupted(signum, frame):
            raise TimeoutError(
                f"Qualification interrupted by signal {signum}; partial evidence preserved"
            )

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(args.timeout_seconds)
        code = 0
        try:
            run(args, receipt, record, artifacts)
        except BaseException as error:
            receipt.update(
                status="failed",
                finished_utc=utc_now(),
                error_type=type(error).__name__,
                error=str(error),
                diagnostics=getattr(error, "diagnostics", {}),
                traceback=traceback.format_exc(),
                cuda_available=torch.cuda.is_available(),
            )
            record(
                "failure",
                dict(
                    error_type=type(error).__name__,
                    error=str(error),
                    diagnostics=getattr(error, "diagnostics", {}),
                ),
            )
            code = 1
        finally:
            signal.alarm(0)
            receipt["elapsed_seconds"] = perf_counter() - begin
            receipt["events_sha256"] = sha(events)
            receipt["source_and_plan_sha256"] = sha(artifacts / "source-and-plan.json")
            write_json(out, receipt)
    return code


if __name__ == "__main__":
    sys.exit(main())
