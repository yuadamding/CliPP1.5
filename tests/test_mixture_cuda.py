"""Allocated-CUDA agreement for the NEW estimator, never an old raw certificate."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.mixture import MixturePolicy, fit_mixture, fit_mixture_reference
from test_mixture_estimator import model_from_counts

sys.path.insert(0, str(Path(__file__).parents[1]/'benchmarks'))
from mixture_structural_guard import choose


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Allocated CUDA is required')
@pytest.mark.parametrize('enriched', [False, True])
def test_compiled_cuda_matches_fresh_reference(enriched):
    torch.set_num_threads(1)
    rng = np.random.default_rng(738)
    major = np.tile([1, 2, 3, 4], 24)
    phi = np.repeat([.42, .92], 48)
    multiplicity = np.ones_like(major) if enriched else rng.integers(1, major+1)
    alt = rng.binomial(180, .7*phi*multiplicity/(.6+.7*(major+1)))
    cpu = model_from_counts(alt, 180, major)
    gpu = replace(cpu, **{name: getattr(cpu, name).to('cuda') for name in
                          ('alt', 'ref', 'slope', 'log_prior', 'lower', 'upper')},
                  kernels=Kernels('cuda', True))
    policy = MixturePolicy(max_clusters=3, max_iterations=500)
    reference = fit_mixture_reference(cpu, policy)
    observed = fit_mixture(gpu, policy)
    assert observed.execution == 'compiled_cuda'
    assert observed.adaptive == reference.adaptive
    assert observed.status == reference.status
    assert observed.score == pytest.approx(reference.score, abs=1e-7)
    actual = observed.public_arrays(cpu.mutation_ids)
    expected = reference.public_arrays(cpu.mutation_ids)
    torch.testing.assert_close(actual['labels'].cpu(), expected['labels'], atol=0, rtol=0)
    torch.testing.assert_close(actual['multiplicity'].cpu(), expected['multiplicity'], atol=0, rtol=0)
    torch.testing.assert_close(actual['ccf'].cpu(), expected['ccf'], atol=1e-7, rtol=0)
    assert observed.metadata()['raw_certificate_inherited'] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Allocated CUDA is required')
def test_full_fusion_path_remains_unchanged_after_mixture_and_guard():
    from clipp1d.cuda.refinement import PartitionSearchPolicy
    from clipp1d.cuda.selection import fit_tensor_model
    from clipp1d.cuda_api import _validate_device_fit

    torch.set_num_threads(1)
    cpu = model_from_counts([7, 8, 28, 29], 100, [1, 1, 1, 1])
    gpu = replace(cpu, **{name: getattr(cpu, name).to('cuda') for name in
                          ('alt', 'ref', 'slope', 'log_prior', 'lower', 'upper')},
                  kernels=Kernels('cuda', True))
    original = fit_tensor_model(gpu, partition_search=PartitionSearchPolicy())
    _validate_device_fit(original)
    snapshot = [v.clone() for v in (original.raw.x, original.raw.dual, original.graph.weights,
                                   original.refit.phi, original.partition_estimate.candidate.refit.phi)]
    partition = original.partition_estimate.candidate.refit
    candidate = fit_mixture(gpu, MixturePolicy(max_clusters=3), seed_centers=partition.centers)
    _, decision = choose(gpu, candidate.records, candidate.policy, len(partition.centers))
    _validate_device_fit(original)
    for expected, actual in zip(snapshot, (original.raw.x, original.raw.dual, original.graph.weights,
                                          original.refit.phi, partition.phi)):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert not decision['raw_certificate_inherited']
