"""Integer-multiplicity mixture, ported from CliPP2 77525a6 (AGPL-3.0).

Modified 2026-09-21: single sample, NumPy only; see UPSTREAM.md.
"""

import math

import numpy as np
from scipy.special import logsumexp

from .policy import Policy
from .types import CountModel, LikelihoodTerms, NoEligibleMutationsError


def compile_model(data, policy=Policy()):
    rows = data.retained
    if not rows:
        raise NoEligibleMutationsError("No informative eligible mutations remain",
                                       exclusions={m.mutation_id: m.exclusion for m in data.mutations})
    major = np.array([max(s[1] for s in m.states) for m in rows])
    copies = np.array([math.fsum(f * (a + b) for f, a, b in m.states) /
                       math.fsum(f for f, _, _ in m.states) for m in rows])
    normal = np.array([m.normal_cn for m in rows])
    denominator = (1 - data.purity) * normal + data.purity * copies
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise ValueError("Copy-number denominator must be finite and positive")
    scale = data.purity / denominator
    count = np.minimum(major, 4)
    candidates = np.arange(1, int(count.max()) + 1)
    valid = candidates <= count[:, None]
    slope = scale[:, None] * np.where(valid, candidates, 0)
    upper = np.clip(np.minimum(1, (1 - policy.eps) / np.maximum(scale * count, policy.eps)),
                    policy.eps, 1)
    return CountModel(tuple(m.mutation_id for m in rows), np.array([m.alt for m in rows]),
                      np.array([m.ref for m in rows]), np.full(len(rows), policy.eps), upper,
                      slope, np.where(valid, -np.log(count[:, None]), -np.inf), valid, policy.eps)


def evaluate(model, phi, *, derivatives=False):
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != (len(model),) or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a finite vector with one value per retained mutation")
    mass = model.slope * phi[:, None]
    p = np.clip(mass, model.eps, 1 - model.eps)
    joint = model.alt[:, None] * np.log(p) + model.ref[:, None] * np.log1p(-p) + model.log_prior
    normalizer = logsumexp(joint, axis=1)
    posterior = np.exp(joint - normalizer[:, None])
    gradient = curvature = None
    if derivatives:
        s = np.where((mass > model.eps) & (mass < 1 - model.eps), model.slope, 0)
        gradient = -np.sum(posterior * s * (model.alt[:, None] / p - model.ref[:, None] / (1 - p)), axis=1)
        curvature = np.maximum(np.sum(posterior * s**2 *
                               (model.alt[:, None] / p**2 + model.ref[:, None] / (1 - p)**2), axis=1), 1e-8)
    return LikelihoodTerms(-normalizer, gradient, curvature, posterior)


def posterior_multiplicity(model, phi):
    return evaluate(model, phi).posterior


def clipping_breakpoints(model, mutation_index=None):
    m = model if mutation_index is None else model.subset([mutation_index])
    slopes = np.where(m.valid, m.slope, np.nan)
    points = np.concatenate((m.eps / slopes, (1 - m.eps) / slopes), axis=1)
    valid = (points >= m.lower[:, None]) & (points <= m.upper[:, None])
    return np.unique(points[valid])


def one_sided_derivatives(model, phi, *, posterior=None):
    """Left and right loss derivatives, including the exact clipping kinks."""
    phi = np.asarray(phi)
    mass = model.slope * phi[:, None]
    p = np.clip(mass, model.eps, 1 - model.eps)
    if posterior is None:
        posterior = evaluate(model, phi).posterior
    score = model.alt[:, None] / p - model.ref[:, None] / (1 - p)
    # Compare in phi-space too: division followed by multiplication can round off the kink.
    s = np.where(model.valid, model.slope, np.nan)
    at_low = phi[:, None] == model.eps / s
    at_high = phi[:, None] == (1 - model.eps) / s
    inside = (mass > model.eps) & (mass < 1 - model.eps) & ~at_low & ~at_high
    left = np.where(inside | at_high, model.slope, 0)
    right = np.where(inside | at_low, model.slope, 0)
    return -np.sum(posterior * left * score, axis=1), -np.sum(posterior * right * score, axis=1)
