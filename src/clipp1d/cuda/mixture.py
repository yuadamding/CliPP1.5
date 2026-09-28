"""Experimental soft-mixture estimator, separate from certified fusion/refits.

The original complete-graph fit supplies optional starts, never certificates.
No truth, cohort identity, or clonal designation enters inference. Production
entry is compiled CUDA only; ``fit_mixture_reference`` is an explicit CPU
component-test adapter. This estimator is NOT an adopted production default.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .kernels import StructuralCompileBank


@dataclass(frozen=True)
class MixturePolicy:
    model_id: str = "soft_uniform_or_m1_enriched_v1"
    criterion_id: str = "observed_mixture_map_bic_v1"
    max_clusters: int = 8
    max_iterations: int = 500
    tolerance: float = 1e-7  # Absolute objective change per mutation.
    enrichment_shrinkage: float = 1.0  # Beta(1, 1 + shrinkage).
    adaptive_multiplicity: bool = True

    def __post_init__(self):
        if self.model_id != "soft_uniform_or_m1_enriched_v1" or \
                self.criterion_id != "observed_mixture_map_bic_v1":
            raise ValueError("Unknown experimental model/criterion identity")
        for name in ("max_clusters", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_clusters > 20:
            raise ValueError("The experimental component bank is bounded at 20")
        for name in ("tolerance", "enrichment_shrinkage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.adaptive_multiplicity, bool):
            raise ValueError("adaptive_multiplicity must be boolean")


def expectation(alt, ref, slope, valid, centers, weights, enrichment, eps):
    """Observed mixture likelihood and joint cluster/multiplicity posterior.

    Multiplicity prior: (1-w)/A + w*I(m=1), on the original integer support.
    The binomial coefficient is omitted, identically for every candidate.
    """
    support = valid.sum(-1).to(torch.float64)
    first = torch.arange(slope.shape[1], device=slope.device) == 0
    prior = (1 - enrichment)[None, :, None] / support[:, None, None]
    prior = prior + enrichment[None, :, None] * first[None, None, :]
    prior = torch.where(valid[:, None, :], prior, 0.0)
    p = (slope[:, None, :] * centers[None, :, None]).clamp(eps, 1 - eps)
    joint = (alt[:, None, None] * p.log() + ref[:, None, None] * torch.log1p(-p)
             + prior.log() + weights.log()[None, :, None])
    norm = torch.logsumexp(joint.flatten(1), dim=1)
    posterior = torch.exp(joint - norm[:, None, None])
    return norm.sum(), posterior, prior


def slope_groups(slope):
    """One device-side grouping of identical represented binomial slopes."""
    ordered, order = torch.sort(slope.flatten(), stable=True)
    values, lengths = torch.unique_consecutive(ordered, return_counts=True)
    return values, order, lengths


def maximize_centers(alt, ref, slope, posterior, lower, upper, eps):
    """Reference-facing entry; a complete fit prepares slope groups only once."""
    return _center_step(alt, ref, slope, posterior, lower, upper, eps, *slope_groups(slope))


def _center_step(alt, ref, slope, posterior, lower, upper, eps, values, order, lengths):
    """Safeguarded concave binomial M-step, in the shared feasible interval.

    All soft components can explain every mutation. Their domain is therefore
    the intersection of the ORIGINAL row bounds; no bound is enlarged. At the
    lower clipping plateaus the endpoint is also compared explicitly below.
    """
    lo = lower.expand(posterior.shape[1]).clone()
    hi = upper.expand_as(lo).clone()
    successes = (posterior * alt[:, None, None]).sum((0, 2))
    failures = posterior * ref[:, None, None]
    # The derivative depends on a mutation only through its slope and posterior
    # expected reference count. Aggregate those counts once, rather than doing
    # 36 repeated full-population reductions. Sorting/lengths are prepared once
    # per fit; deterministic segment reductions avoid floating scatter atomics.
    ordered_failures = failures.permute(1, 0, 2).flatten(1)[:, order]
    grouped_failures = torch.segment_reduce(ordered_failures, 'sum',
                                            lengths=lengths.expand(posterior.shape[1], -1), axis=1)
    for _ in range(36):
        mid = (lo + hi) * .5
        # In the shared feasible domain all upper clipping kinks are endpoints.
        p = (values[None, :] * mid[:, None]).clamp(eps, 1 - eps)
        gradient = successes / mid - (grouped_failures * values[None, :] / (1 - p)).sum(1)
        lo = torch.where(gradient > 0, mid, lo)
        hi = torch.where(gradient > 0, hi, mid)
    root = (lo + hi) * .5
    proposals = torch.stack((root, lower.expand_as(root), upper.expand_as(root)))
    # Evaluate the actual clipped Q, including boundary/zero-count cases.
    values = []
    for index in range(3):
        p = (slope[:, None, :] * proposals[index][None, :, None]).clamp(eps, 1 - eps)
        values.append((posterior * (alt[:, None, None] * p.log()
                                   + ref[:, None, None] * torch.log1p(-p))).sum((0, 2)))
    choice = torch.stack(values).argmax(0)
    return proposals.gather(0, choice[None, :])[0]


def _center_statistics(alt, ref, posterior, order, lengths):
    successes = (posterior * alt[:, None, None]).sum((0, 2))
    failures = (posterior * ref[:, None, None]).permute(1, 0, 2).flatten(1)[:, order]
    grouped = torch.segment_reduce(failures, 'sum',
                                   lengths=lengths.expand(posterior.shape[1], -1), axis=1)
    return successes, grouped


def _center_bisection_block(successes, grouped, values, lo, hi, eps):
    # Six blocks retain the exact original 36 updates. Keeping the full unrolled
    # reduction in one graph triggers a Triton dominance error for a singleton
    # slope/component on the observed PyTorch 2.9.1 / sm89 stack.
    for _ in range(6):
        mid = (lo + hi) * .5
        p = (values[None, :] * mid[:, None]).clamp(eps, 1 - eps)
        gradient = successes / mid - (grouped * values[None, :] / (1 - p)).sum(1)
        lo = torch.where(gradient > 0, mid, lo)
        hi = torch.where(gradient > 0, hi, mid)
    return lo, hi


def _center_endpoints(alt, ref, slope, posterior, lower, upper, lo, hi, eps):
    root = (lo + hi) * .5
    proposals = torch.stack((root, lower.expand_as(root), upper.expand_as(root)))
    values = []
    for index in range(3):
        p = (slope[:, None, :] * proposals[index][None, :, None]).clamp(eps, 1 - eps)
        values.append((posterior * (alt[:, None, None] * p.log()
                                   + ref[:, None, None] * torch.log1p(-p))).sum((0, 2)))
    return proposals.gather(0, torch.stack(values).argmax(0)[None, :])[0]


class _BlockedCenterStep:
    """Strictly compiled CUDA blocks; no catch, eager retry or CPU fallback."""

    def __init__(self):
        self.statistics = StructuralCompileBank(_center_statistics)
        self.bisect = StructuralCompileBank(_center_bisection_block)
        self.endpoints = StructuralCompileBank(_center_endpoints)

    def __call__(self, alt, ref, slope, posterior, lower, upper, eps, values, order, lengths):
        successes, grouped = self.statistics(alt, ref, posterior, order, lengths)
        lo = lower.expand(posterior.shape[1]).clone()
        hi = upper.expand_as(lo).clone()
        for _ in range(6):
            lo, hi = self.bisect(successes, grouped, values, lo, hi, eps)
        return self.endpoints(alt, ref, slope, posterior, lower, upper, lo, hi, eps)


def maximize_masses(posterior, prior, enrichment, informative, shrinkage, adaptive):
    responsibilities = posterior.sum(-1)
    weights = responsibilities.sum(0) / posterior.shape[0]
    # Underflowed components have exactly zero mass and remain inactive.
    if adaptive:
        enriched = posterior[:, :, 0] * enrichment[None, :] / prior[:, :, 0]
        numerator = (enriched * informative[:, None]).sum(0)
        denominator = (responsibilities * informative[:, None]).sum(0) + shrinkage
        enrichment = numerator / denominator
    return weights, enrichment


def criterion(log_likelihood, enrichment, n, k, adaptive, shrinkage):
    """A declared heuristic, not a calibrated mixture-model evidence bound.

    Count K CCFs, K-1 mixture masses, and K enrichment parameters on the adaptive
    branch. Boundary parameters still pay their cost. Diploid-only input never
    instantiates the adaptive branch. No hard-assignment allocation term occurs.
    """
    penalty = -shrinkage * torch.log1p(-enrichment).sum()
    dimensions = 2 * k - 1 + (k if adaptive else 0)
    objective = -log_likelihood + penalty
    return objective, 2 * objective + dimensions * math.log(n)


@dataclass
class MixtureResult:
    centers: torch.Tensor  # ALL latent components, including hard-empty ones.
    weights: torch.Tensor
    enrichment: torch.Tensor
    posterior: torch.Tensor  # N x K x original integer support.
    log_likelihood: float
    score: float
    adaptive: bool
    status: str
    records: list
    policy: MixturePolicy
    shared_lower: float
    shared_upper: float
    execution: str

    def public_arrays(self, mutation_ids):
        """Occupied MAP groups; closest-to-one label 0 without altering centers.

        Exact CCF ties use the smallest member mutation ID. Multiplicity is
        conditional on the hard selected group (not averaged over groups).
        """
        if len(mutation_ids) != self.posterior.shape[0] or len(set(mutation_ids)) != len(mutation_ids):
            raise ValueError("Unique mutation IDs must match the fitted population")
        labels = self.posterior.sum(-1).argmax(-1)
        occupied = torch.unique(labels).tolist()
        keys = {k: min(mutation_ids[i] for i in torch.where(labels == k)[0].tolist())
                for k in occupied}
        clonal = min(occupied, key=lambda k: (abs(float(self.centers[k]) - 1), keys[k]))
        order = [clonal] + sorted((k for k in occupied if k != clonal),
                                 key=lambda k: (-float(self.centers[k]), keys[k]))
        mapping = torch.full_like(self.centers, -1, dtype=torch.long)
        index = torch.tensor(order, dtype=torch.long, device=labels.device)
        mapping[index] = torch.arange(len(order), device=labels.device)
        rows = torch.arange(labels.numel(), device=labels.device)
        conditional = self.posterior[rows, labels]
        conditional = conditional / conditional.sum(-1, keepdim=True)
        return dict(labels=mapping[labels], centers=self.centers[index],
                    ccf=self.centers[labels], multiplicity=conditional.argmax(-1) + 1,
                    multiplicity_posterior=conditional, latent_component=index)

    def metadata(self):
        return dict(schema="clipp1d.experimental_mixture.v1", policy=asdict(self.policy),
                    execution=self.execution, score=self.score, log_likelihood=self.log_likelihood,
                    adaptive=self.adaptive, status=self.status,
                    selected_parameters_converged=self.status == "em_fixed_point",
                    search_status="bounded_multistart_only", global_optimality_certified=False,
                    raw_certificate_inherited=False, scalar_refit_certificate_inherited=False,
                    allocated_cuda_qualified=False, production_adopted=False, truth_used=False,
                    shared_feasible_interval=[self.shared_lower, self.shared_upper],
                    ccf_estimate="soft_mixture_component_mode_with_MAP_memberships",
                    candidates=self.records)


def _starts(model, policy, seed_centers):
    lower, upper = model.lower.max(), model.upper.min()
    # A data-only initializer, not a forced multiplicity-one model.
    naive = (model.alt / (model.alt + model.ref) / model.slope[:, 0]).clamp(lower, upper)
    for k in range(1, min(policy.max_clusters, model.n) + 1):
        fractions = (torch.arange(k, device=model.device, dtype=torch.float64) + .5) / k
        yield f"spread_k{k}", lower + (upper - lower) * fractions
        yield f"dosage_quantiles_k{k}", torch.quantile(naive, fractions)
    if seed_centers is not None:
        if (seed_centers.ndim != 1 or not 1 <= seed_centers.numel() <= 20 or
                seed_centers.device != model.device or seed_centers.dtype != torch.float64 or
                not bool(torch.isfinite(seed_centers).all())):
            raise ValueError("Seed centers must be 1..20 finite float64 values on the model device")
        # A baseline with more groups than the exploratory K cap is retained as a start.
        yield "preserved_partition_centers", seed_centers.sort().values.clamp(lower, upper)


def _fit(model, policy, seed_centers, compiled):
    model.validate(full=True)
    valid = torch.isfinite(model.log_prior)
    expected = -valid.sum(-1).to(torch.float64).log()[:, None].expand_as(model.log_prior)
    if not bool((valid == (model.slope > 0)).all() and valid[:, 0].all() and
                (torch.where(valid, model.log_prior - expected, 0).abs() < 1e-12).all()):
        raise ValueError("Experimental nested model requires canonical uniform integer support")
    support = valid.sum(-1)
    consecutive = torch.arange(valid.shape[1], device=model.device)[None, :] < support[:, None]
    if not torch.equal(consecutive, valid):
        raise ValueError("Integer support must start at one and be contiguous")
    lower, upper = model.lower.max(), model.upper.min()
    if not bool((lower > 0) & (lower <= upper) & (upper <= 1)):
        raise ValueError("No shared soft-component feasible CCF interval")
    informative = support > 1
    grouped_slopes = slope_groups(model.slope)
    kernels = [expectation, _center_step, maximize_masses]
    if compiled:
        kernels = [StructuralCompileBank(fn) for fn in kernels]
        if grouped_slopes[0].numel() == 1:
            kernels[1] = _BlockedCenterStep()
    estep, cstep, mstep = kernels
    records, best = [], None
    branches = [False, True] if policy.adaptive_multiplicity and bool(informative.any()) else [False]
    for origin, initial in _starts(model, policy, seed_centers):
        for adaptive in branches:
            centers = initial.clone()
            k = centers.numel()
            weights = centers.new_full((k,), 1 / k)
            enrichment = centers.new_full((k,), .5 if adaptive else 0.)
            ll, posterior, prior = estep(model.alt, model.ref, model.slope, valid,
                                        centers, weights, enrichment, model.eps)
            objective, score = criterion(ll, enrichment, model.n, k, adaptive,
                                         policy.enrichment_shrinkage)
            history = [float(objective)]
            status = "iteration_budget_exhausted"
            for iteration in range(policy.max_iterations):
                next_centers = cstep(model.alt, model.ref, model.slope, posterior,
                                    lower, upper, model.eps, *grouped_slopes)
                next_weights, next_enrichment = mstep(posterior, prior, enrichment, informative,
                                                      policy.enrichment_shrinkage, adaptive)
                next_ll, next_posterior, next_prior = estep(
                    model.alt, model.ref, model.slope, valid, next_centers, next_weights,
                    next_enrichment, model.eps)
                next_objective, next_score = criterion(next_ll, next_enrichment, model.n, k,
                                                        adaptive, policy.enrichment_shrinkage)
                margin = 512 * torch.finfo(torch.float64).eps * (1 + objective.abs())
                if not bool(torch.isfinite(next_objective) & (next_objective <= objective + margin)):
                    raise ArithmeticError("Mixture EM failed its independent monotonicity check")
                gain = float(objective - next_objective)
                step = max(float((next_centers-centers).abs().max()),
                           float((next_weights-weights).abs().max()),
                           float((next_enrichment-enrichment).abs().max()))
                centers, weights, enrichment = next_centers, next_weights, next_enrichment
                ll, posterior, prior = next_ll, next_posterior, next_prior
                objective, score = next_objective, next_score
                history.append(float(objective))
                if gain <= policy.tolerance * model.n and step <= math.sqrt(policy.tolerance):
                    status = "em_fixed_point"
                    break
            record = dict(origin=origin, adaptive=adaptive, components=k, status=status,
                          iterations=iteration+1, score=float(score), log_likelihood=float(ll),
                          objective_history=history, centers=centers.tolist(),
                          weights=weights.tolist(), enrichment=enrichment.tolist())
            records.append(record)
            # Unconverged candidates remain explicitly provisional, never certified.
            key = (float(score), k, adaptive, origin)
            if best is None or key < best[0]:
                best = key, (centers.clone(), weights.clone(), enrichment.clone(), posterior.clone(),
                             float(ll), float(score), adaptive, status)
    model.validate(full=True)
    return MixtureResult(*best[1], records, policy, float(lower), float(upper),
                         "compiled_cuda" if compiled else "cpu_component_reference")


@torch.no_grad()
def fit_mixture(model, policy=MixturePolicy(), *, seed_centers=None):
    """Explicit experimental CUDA entry; does not change fit() or its outputs."""
    if model.device.type != "cuda" or not model.kernels.compiled:
        raise ValueError("Experimental inference requires compiled PyTorch CUDA")
    with model.validated_stage():
        return _fit(model, policy, seed_centers, compiled=True)


@torch.no_grad()
def fit_mixture_reference(model, policy=MixturePolicy(), *, seed_centers=None):
    """Eager CPU component diagnostics only; never a production fallback."""
    if model.device.type != "cpu" or model.kernels.compiled:
        raise ValueError("Reference adapter requires an explicit eager CPU model")
    with model.validated_stage():
        return _fit(model, policy, seed_centers, compiled=False)
