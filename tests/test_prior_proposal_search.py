"""Experimental-start boundary, baseline preservation and output separation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from clipp1d.api import source_provenance
from clipp1d.cuda.experimental import solve_extra_start
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.policy import QualificationError
from clipp1d.cuda.refinement import PartitionSearchPolicy
from clipp1d.cuda.selection import fit_tensor_model
from clipp1d.cuda_api import _export
from clipp1d.partition_output import validate_partition_result

SPEC = importlib.util.spec_from_file_location('prior_experiment_search', Path(__file__).parents[1]/'benchmarks/prior_perturbation.py')
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def model():
    alt = torch.tensor([20., 48., 21., 46.], dtype=torch.float64)
    slope = torch.full((4, 1), .5, dtype=torch.float64)
    return TensorModel(tuple('abcd'), alt, 100-alt, slope, torch.zeros_like(slope),
                       torch.full_like(alt, 1e-6), torch.ones_like(alt), 1e-6, Kernels('cpu', False))


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def baseline():
    return fit_tensor_model(model(), lambda_values=[0., .5, 8.], partition_search=PartitionSearchPolicy())


def test_all_single_candidate_and_zero_have_identical_selection(baseline, tmp_path):
    for eta in (0., .03):
        starts, receipts = experiment.pilot_proposals(baseline.model, baseline.pilot.phi,
            'negative', tmp_path/f'pilots-{eta}', eta)
        assert starts == []
        assert receipts['status'] == 'complete'
        assert all(r['substantial_shift_fraction'] == 0 for r in receipts['records'])
        result = experiment.additional_search(baseline, starts, [.5, 8.], tmp_path/f'P-{eta}', 'P')
        assert result.candidate is baseline.partition_estimate.candidate
        public = _export(replace(baseline, partition_estimate=result), source_provenance())
        reference = _export(baseline, source_provenance())
        np.testing.assert_array_equal(public.raw_phi, reference.raw_phi)
        np.testing.assert_array_equal(public.partition_estimate.memberships, reference.partition_estimate.memberships)
        np.testing.assert_array_equal(public.partition_estimate.ccf, reference.partition_estimate.ccf)
        assert public.partition_estimate.score == reference.partition_estimate.score
        validate_partition_result(public)


def test_extra_start_result_rejects_other_prior_and_graph(baseline):
    model = baseline.model
    penalty = model.alt.new_tensor(.5)
    result = solve_extra_start(model, baseline.graph, penalty, baseline.pilot.phi)
    assert result.raw.qualified
    result.validate(model, baseline.graph, penalty, baseline.policy)
    other = experiment.auxiliary_model(model, torch.zeros_like(model.log_prior), .03)
    with pytest.raises(ValueError, match='exact model'):
        result.validate(other, baseline.graph, penalty, baseline.policy)
    with pytest.raises(ValueError, match='absolute penalty'):
        result.validate(model, baseline.graph, penalty*2, baseline.policy)
    with pytest.raises(ValueError, match='feasible'):
        solve_extra_start(model, baseline.graph, penalty, model.upper+1)
    with pytest.raises(ValueError, match='positive'):
        solve_extra_start(model, baseline.graph, penalty*0, baseline.pilot.phi)
    result.raw.objective.data.add_(1)
    with pytest.raises(ValueError, match='modified'):
        result.validate(model, baseline.graph, penalty, baseline.policy)


@pytest.mark.parametrize('error_type', [QualificationError, torch.OutOfMemoryError])
def test_failed_auxiliary_search_preserves_B_and_records_coverage(baseline, tmp_path, monkeypatch, error_type):
    def fail(*args, **kwargs):
        raise error_type('forced isolated failure')
    monkeypatch.setattr(experiment, 'solve_extra_start', fail)
    starts = [(0, baseline.pilot.phi.clone())]
    before = baseline.raw.x.clone()
    result = experiment.additional_search(baseline, starts, [.5, 8.], tmp_path/'failure', 'P')
    assert result.status == 'incomplete'
    assert result.candidate is baseline.partition_estimate.candidate
    failures = [r for r in result.records if r['origin'].startswith('unresolved:P:')]
    assert len(failures) == 2
    assert all(r['status'] == 'unresolved' for r in failures)
    torch.testing.assert_close(before, baseline.raw.x, atol=0, rtol=0)


def test_failed_auxiliary_pilots_preserve_original_and_empty_bank(baseline, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError('forced isolated pilot allocation failure')
    monkeypatch.setattr(experiment, 'auxiliary_model', fail)
    before = baseline.model.log_prior.clone()
    starts, record = experiment.pilot_proposals(baseline.model, baseline.pilot.phi, 'oom', tmp_path/'oom')
    assert starts == [] and record['status'] == 'incomplete'
    assert len(record['records']) == 8
    assert torch.equal(before, baseline.model.log_prior)


def test_tsv_permutation_gives_identical_canonical_extra_search(tmp_path):
    # Use the canonical schema order/names, then reorder rows only.
    from clipp1d.io import SCHEMA_COLUMNS
    header = '\t'.join(SCHEMA_COLUMNS)+'\n'
    values = [f'{mid}\ts1\t{alt}\t{100-alt}\t1\t1\t2\tseg{mid}\tx\t1\t1\t1\n'
              for mid, alt in zip('abcd', (20, 48, 21, 46))]
    left, right = tmp_path/'a.tsv', tmp_path/'b.tsv'
    left.write_text(header+''.join(values))
    right.write_text(header+''.join(values[::-1]))
    outputs = []
    for path in (left, right):
        _, data = experiment.prepare_model(path, 'cpu', compiled=False)
        outputs.append(fit_tensor_model(data, lambda_values=[0., .5, 8.], partition_search=PartitionSearchPolicy()))
    first, second = outputs
    assert torch.equal(first.raw.x, second.raw.x)
    assert torch.equal(first.graph.weights, second.graph.weights)
    assert torch.equal(first.partition_estimate.candidate.refit.labels, second.partition_estimate.candidate.refit.labels)


def test_fresh_refits_and_same_refinement_budget_without_birth(baseline, tmp_path, monkeypatch):
    calls = []
    refit = experiment.refit_labels
    refine = experiment.refine_memberships
    def observed_refit(*args, **kwargs):
        assert 'cache' not in kwargs and 'pilot_reuse' not in kwargs
        calls.append('fresh_refit')
        return refit(*args, **kwargs)
    def observed_refine(model, fitted, policy, search_policy):
        assert search_policy.max_rounds == 4
        calls.append('four_round_refinement')
        return refine(model, fitted, policy, search_policy)
    monkeypatch.setattr(experiment, 'refit_labels', observed_refit)
    monkeypatch.setattr(experiment, 'refine_memberships', observed_refine)
    result = experiment.additional_search(baseline, [(0, baseline.pilot.phi)], [.5], tmp_path/'D', 'D')
    assert calls == ['fresh_refit', 'four_round_refinement']
    assert result.candidate.refit.score <= baseline.partition_estimate.candidate.refit.score
    exported = _export(replace(baseline, partition_estimate=result), source_provenance())
    validate_partition_result(exported)
    assert exported.partition_estimate.provenance['model_sha256'] == exported.provenance['model_sha256']
    assert exported.partition_estimate.provenance['preserved_partition']['score'] == float(baseline.partition_estimate.candidate.refit.score)


def test_zero_prior_same_environment_full_fit(baseline):
    original = baseline.model
    aux = experiment.auxiliary_model(original, torch.ones_like(original.log_prior), 0)
    repeated = fit_tensor_model(aux, lambda_values=[0., .5, 8.], partition_search=PartitionSearchPolicy())
    for before, after in [(baseline.raw.x, repeated.raw.x),
                          (baseline.refit.labels, repeated.refit.labels),
                          (baseline.partition_estimate.candidate.refit.phi, repeated.partition_estimate.candidate.refit.phi)]:
        assert torch.equal(before, after)
    assert baseline.lambda_value == repeated.lambda_value
