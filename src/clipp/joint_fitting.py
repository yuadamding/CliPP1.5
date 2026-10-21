"""Experimental CPU joint mixture fitting; never used by the production fit.

The chain supplies starting centers only. This float64 reference maximizes the
existing uniform-multiplicity observed likelihood without a clonal anchor. EM
center steps use frozen joint responsibilities; fixed-center weight steps may
revive zero weights. Numerical stationarity establishes only a local solution.
"""

import numpy as np
from scipy.special import logsumexp, xlog1py, xlogy

from .scoring import fit_cluster_weights


EXPERIMENTAL_VERSION = "joint-observed-mixture-cpu-v1"


def _parameters(model, centers, weights):
    centers = np.asarray(centers, dtype=np.float64)
    weights = np.full(centers.size, 1.0 / max(1, centers.size)) if weights is None else np.asarray(
        weights, dtype=np.float64
    )
    if (
        centers.ndim != 1 or not centers.size or weights.shape != centers.shape
        or np.any(~np.isfinite(centers)) or np.any(centers < 0)
        or np.any(centers > model.purity) or np.any(~np.isfinite(weights))
        or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0
    ):
        raise ValueError("Expected CP centers in [0, purity] and finite nonnegative weights with positive sum")
    return centers.copy(), weights.copy() / weights.sum()


def _state(model, centers, weights):
    kernel = np.column_stack([model.log_likelihood(cp) for cp in centers])
    log_weights = np.full(len(weights), -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)
    log_mass = logsumexp(kernel + log_weights, axis=1)
    if np.any(~np.isfinite(log_mass)):
        raise ValueError("Starting parameters must give every mutation finite positive likelihood")
    return {
        "log_kernel": kernel, "log_mass": log_mass, "log_weights": log_weights,
        "log_likelihood": float(log_mass.sum()),
    }


def _center_gradient(model, centers, weights, state):
    """Observed derivative, including nonzero limits of zero-probability states.

The derivative uses d Bin(a;n,p)/dc = n*s*(Bin(a-1;n-1,p) -
Bin(a;n-1,p)). Computing the two terms in log space avoids 0 * infinity at
CP zero and at p one. In particular, an ALT-one row can contribute a positive
derivative at CP zero even though its responsibility for that component is zero.
"""
    result = np.zeros(len(centers))
    for k, cp in enumerate(centers):
        if weights[k] == 0:
            continue
        for rows, scale in model._chunks(1):
            p = np.minimum(1.0, cp * scale)
            constant = model.log_constant[rows, None] + np.log(scale)
            normalizer = state["log_weights"][k] - state["log_mass"][rows, None]
            for positive, count in (
                (True, model.r[rows]),
                (False, model.reference[rows]),
            ):
                valid = count > 0
                if not np.any(valid):
                    continue
                term = (
                    xlogy(model.r[rows[valid], None] - int(positive), p[valid])
                    + xlog1py(model.reference[rows[valid], None] - int(not positive), -p[valid])
                    + constant[valid] + np.log(count[valid, None]) + normalizer[valid]
                )
                result[k] += (1.0 if positive else -1.0) * float(np.exp(term).sum())
    return result


def joint_diagnostics(model, centers, weights):
    """Independently recompute observed score and box/simplex stationarity.

Center gaps are derivatives per mutation in CCF coordinates. Weight active
gaps are per mutation; the Frank-Wolfe weight gap has total-likelihood units.
These are numerical checks, not certificates of global optimality.
"""
    centers, weights = _parameters(model, centers, weights)
    state = _state(model, centers, weights)
    score = np.exp(state["log_kernel"] - state["log_mass"][:, None]).mean(axis=0)
    gradient = _center_gradient(model, centers, weights, state)
    feasible = np.abs(gradient)
    feasible[centers == 0] = np.maximum(gradient[centers == 0], 0)
    feasible[centers == model.purity] = np.maximum(-gradient[centers == model.purity], 0)
    gaps = feasible * model.purity / len(model)
    expected = np.exp(state["log_kernel"] + state["log_weights"] - state["log_mass"][:, None]).sum(
        axis=0
    )
    return {
        "log_likelihood": state["log_likelihood"],
        "weight_optimality_gap": len(model) * max(0.0, float(score.max() - 1.0)),
        "weight_active_score_gap": float(score.max() - score[weights > 0].min()),
        "center_gradient": gradient,
        "center_stationarity": gaps,
        "center_stationarity_gap": float(gaps.max()),
        "posterior_expected_num_SNV": expected,
        "weight_expected_count_max_error": float(np.max(np.abs(expected - len(model) * weights))),
    }


def _frozen_chunks(model, old_cp, old_log_weight, old_log_mass):
    for rows, scale in model._chunks(1):
        # These parameters stay fixed throughout all scalar trial-center calls.
        responsibility = np.exp(
            model._kernel(old_cp, rows, scale) + old_log_weight - old_log_mass[rows, None]
        )
        yield rows, scale, responsibility


def _center_m_step(model, old_cp, old_log_weight, old_log_mass):
    if not np.isfinite(old_log_weight):
        return float(old_cp)
    alternate_mass = sum(
        float(np.sum(responsibility * model.r[rows, None]))
        for rows, _, responsibility in _frozen_chunks(model, old_cp, old_log_weight, old_log_mass)
    )
    if alternate_mass == 0:
        return 0.0

    def derivative(cp):
        first = alternate_mass / cp
        second = alternate_mass / (cp * cp)
        for rows, scale, responsibility in _frozen_chunks(
            model, old_cp, old_log_weight, old_log_mass
        ):
            reference_mass = responsibility * model.reference[rows, None]
            positive = reference_mass > 0
            denominator = np.maximum(0.0, 1.0 - cp * scale)
            if np.any(positive & (denominator == 0)):
                return -np.inf, -np.inf
            ratio = np.zeros_like(scale)
            np.divide(scale, denominator, out=ratio, where=positive)
            first -= float(np.sum(reference_mass * ratio))
            second += float(np.sum(reference_mass * ratio * ratio))
        return first, -second

    if derivative(model.purity)[0] >= 0:
        return model.purity
    lower, upper = 0.0, model.purity
    cp = old_cp if 0 < old_cp < model.purity else model.purity / 2
    # A concave scalar objective has one derivative root (or an endpoint).
    for _ in range(100):
        gradient, hessian = derivative(cp)
        if gradient > 0:
            lower = cp
        else:
            upper = cp
        if upper - lower <= 4 * np.finfo(float).eps * model.purity:
            break
        if np.isfinite(gradient) and abs(gradient) <= 1e-11 * max(1.0, alternate_mass):
            return float(cp)
        newton = cp - gradient / hessian if np.isfinite(hessian) else np.nan
        cp = newton if np.isfinite(newton) and lower < newton < upper else (lower + upper) / 2
    return float((lower + upper) / 2)


def _escape_endpoint(model, centers, weights, state, center_tolerance):
    """A monotone one-sided step escapes absorbing zero-responsibility endpoints.

EM alone can stall at a nonstationary endpoint: a currently impossible latent
state has zero responsibility but a nonzero observed derivative. This bounded
backtracking step uses the observed objective, never fabricated responsibilities.
"""
    centers = centers.copy()
    gradient = _center_gradient(model, centers, weights, state)
    for k, cp in enumerate(centers):
        direction = 1.0 if cp == 0 else -1.0 if cp == model.purity else 0.0
        if direction * gradient[k] * model.purity / len(model) <= center_tolerance:
            continue
        step = model.purity / 2
        for _ in range(60):
            trial = centers.copy()
            trial[k] = cp + direction * step
            candidate = _state(model, trial, weights)
            gain = candidate["log_likelihood"] - state["log_likelihood"]
            if gain >= 1e-4 * direction * gradient[k] * step:
                centers, state = trial, candidate
                break
            step /= 2
    return centers, state


def canonicalize_components(centers, weights):
    """Remove exact zero mass and combine exactly equal CPs; never near-equals.

Returns descending centers, weights and original-component index groups. No
minimum mass or distance threshold is imposed. Positive groups with no MAP
assignments remain valid.
"""
    centers = np.asarray(centers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        centers.ndim != 1 or weights.shape != centers.shape or not centers.size
        or np.any(~np.isfinite(centers)) or np.any(~np.isfinite(weights))
        or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-12)
    ):
        raise ValueError("Invalid mixture for exact component reduction")
    groups = {}
    for k in np.flatnonzero(weights > 0):
        groups.setdefault(float(centers[k]), []).append(int(k))
    ordered = sorted(groups, reverse=True)
    return {
        "centers": np.asarray(ordered),
        "cluster_weights": np.asarray([weights[groups[cp]].sum() for cp in ordered]),
        "component_groups": [groups[cp] for cp in ordered],
        "discarded_zero_components": np.flatnonzero(weights == 0).tolist(),
    }


def fit_joint(
    model, centers, weights=None, *, max_iterations=500, likelihood_tolerance=1e-9,
    center_tolerance=1e-7, weight_tolerance=1e-8,
):
    """Return an explicitly qualified or failed local experimental fit.

Failures and iteration limits return their last accepted parameters with
``converged=False``; a caller must not publish them as qualified joint fits.
Input errors raise ValueError. All likelihood calculations use the complete
input population and full multiplicity support. Only grouped state blocks and
N-by-q marginals are held, never a global N-by-q-by-major tensor.
"""
    if (
        not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1
        or any(not np.isfinite(x) or x <= 0 for x in (
            likelihood_tolerance, center_tolerance, weight_tolerance
        ))
    ):
        raise ValueError("Iteration count and numerical tolerances must be positive")
    centers, weights = _parameters(model, centers, weights)
    initial_centers, initial_weights = centers.copy(), weights.copy()
    state = _state(model, centers, weights)
    initial_likelihood = state["log_likelihood"]
    history = [{"iteration": 0, "phase": "initial", "log_likelihood": initial_likelihood}]
    status, failure = "iteration_limit", None
    iteration = 0
    try:
        weight_fit = fit_cluster_weights(
            model, centers, weights, log_kernel=state["log_kernel"], allow_boundary=True
        )
        weights = weight_fit["cluster_weights"]
        state = _state(model, centers, weights)
        history.append({"iteration": 0, "phase": "initial_weights", "log_likelihood": state["log_likelihood"]})
        for iteration in range(1, max_iterations + 1):
            old_likelihood = state["log_likelihood"]
            trial = np.asarray([
                _center_m_step(model, cp, state["log_weights"][k], state["log_mass"])
                for k, cp in enumerate(centers)
            ])
            # The preceding center steps all used the same old parameter state.
            trial_state = _state(model, trial, weights)
            trial, trial_state = _escape_endpoint(
                model, trial, weights, trial_state, center_tolerance
            )
            weight_fit = fit_cluster_weights(
                model, trial, weights, log_kernel=trial_state["log_kernel"], allow_boundary=True
            )
            trial_weights = weight_fit["cluster_weights"]
            trial_state = _state(model, trial, trial_weights)
            numerical_slack = 64 * np.finfo(float).eps * max(1.0, abs(old_likelihood))
            if trial_state["log_likelihood"] < old_likelihood - numerical_slack:
                status = "nonmonotone_step"
                break
            centers, weights, state = trial, trial_weights, trial_state
            diagnostics = joint_diagnostics(model, centers, weights)
            improvement = state["log_likelihood"] - old_likelihood
            history.append({
                "iteration": iteration, "phase": "joint", "log_likelihood": state["log_likelihood"],
                "relative_improvement": improvement / max(1.0, abs(old_likelihood)),
                "center_stationarity_gap": diagnostics["center_stationarity_gap"],
                "weight_optimality_gap": diagnostics["weight_optimality_gap"],
                "weight_active_score_gap": diagnostics["weight_active_score_gap"],
            })
            if (
                improvement <= likelihood_tolerance * max(1.0, abs(old_likelihood))
                and diagnostics["center_stationarity_gap"] <= center_tolerance
                and diagnostics["weight_optimality_gap"] <= len(model) * weight_tolerance
                and diagnostics["weight_active_score_gap"] <= weight_tolerance
            ):
                status = "locally_converged"
                break
    except (RuntimeError, FloatingPointError, ValueError) as error:
        status, failure = "optimization_failed", str(error)
    reduction = canonicalize_components(centers, weights)
    diagnostics = joint_diagnostics(model, reduction["centers"], reduction["cluster_weights"])
    if abs(diagnostics["log_likelihood"] - state["log_likelihood"]) > (
        128 * np.finfo(float).eps * max(1.0, abs(state["log_likelihood"]))
    ):
        status, failure = "reduction_failed", "Exact component reduction changed predictive likelihood"
    if status == "locally_converged" and (
        diagnostics["center_stationarity_gap"] > center_tolerance
        or diagnostics["weight_optimality_gap"] > len(model) * weight_tolerance
        or diagnostics["weight_active_score_gap"] > weight_tolerance
        or diagnostics["log_likelihood"] < initial_likelihood - (
            128 * np.finfo(float).eps * max(1.0, abs(initial_likelihood))
        )
    ):
        status, failure = "final_verification_failed", "Final arithmetic did not satisfy qualification checks"
    return {
        "experimental_version": EXPERIMENTAL_VERSION,
        "status": status, "converged": status == "locally_converged", "failure": failure,
        **reduction, **diagnostics,
        "bic": float(-2 * diagnostics["log_likelihood"] + (2 * len(reduction["centers"]) - 1) * np.log(len(model))),
        "initial_centers": initial_centers, "initial_weights": initial_weights,
        "unreduced_centers": centers.copy(), "unreduced_weights": weights.copy(),
        "initial_log_likelihood": initial_likelihood,
        "initial_component_count": len(initial_centers), "effective_component_count": len(reduction["centers"]),
        "iterations": iteration, "history": history,
        "tolerances": {
            "likelihood": likelihood_tolerance, "center": center_tolerance, "weight": weight_tolerance,
        },
    }


def reduced_candidates(model, fit, **fit_options):
    """Refit each adjacent-center merge as an explicit lower-component candidate.

No distance threshold decides validity or acceptance. Each candidate receives
its own full joint fit, effective count, BIC and qualification status. Selection
is left to the caller so the unreduced fit always remains in its candidate bank.
"""
    if not fit["converged"]:
        raise ValueError("Component-reduction proposals require a qualified parent fit")
    centers = np.asarray(fit["centers"])
    weights = np.asarray(fit["cluster_weights"])
    candidates = []
    for k in range(len(centers) - 1):
        mass = weights[k] + weights[k + 1]
        merged = (centers[k] * weights[k] + centers[k + 1] * weights[k + 1]) / mass
        candidate = fit_joint(
            model, np.concatenate((centers[:k], [merged], centers[k + 2:])),
            np.concatenate((weights[:k], [mass], weights[k + 2:])), **fit_options,
        )
        candidate["reduction_proposal"] = {"merged_parent_components": [k, k + 1], "parent_bic": fit["bic"]}
        candidates.append(candidate)
    return candidates
