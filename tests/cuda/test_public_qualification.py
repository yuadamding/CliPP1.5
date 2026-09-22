"""Publication qualification rejects incomplete or altered evidence without CUDA."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from clipp1d.cuda.policy import CudaPolicy
from clipp1d.cuda_api import SCHEMA


@pytest.fixture
def qualification_module():
    source = Path(__file__).resolve().parents[2] / 'benchmarks' / 'qualify_cuda.py'
    spec = importlib.util.spec_from_file_location('public_qualification_reference', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def public_evidence():
    policy = CudaPolicy()
    lambdas = [0.] + [float(torch.ldexp(torch.tensor(1., dtype=torch.float64), torch.tensor(i)))
                     for i in range(policy.path_min_exponent, policy.path_max_exponent + 1)]
    records = [dict(lambda_value=value, raw_status='qualified', refit_status='qualified', search_complete=True)
               for value in lambdas]
    raw = dict(box_feasible=True, clonal_constraint=False, separable_scalar_gap_qualified=True)
    candidate = dict(raw_qualified=True, refit_qualified=True)
    source = dict(source_sha256='source', model_sha256='model', policy=asdict(policy),
                  numerical_stages=dict(lambda_reference=1., extensions=0))
    result = SimpleNamespace(mutation_ids=('a', 'b'), raw_phi=np.array([.2, .7]),
                             refitted_phi=np.array([.2, .7]), search_status='complete',
                             provenance=deepcopy(source), graph_sha256='graph', selected_lambda=0.,
                             raw_objective=10., selection_score=25., raw_diagnostics=deepcopy(raw),
                             candidate_provenance=deepcopy(candidate))
    receipt = dict(schema=SCHEMA, status='success', search_status='complete',
                   provenance=source, graph_sha256='graph', selected_lambda=0., raw_objective=10.,
                   selection_score=25., candidate_provenance=candidate, raw_diagnostics=raw,
                   search=records)
    return result, receipt


def test_complete_public_default_path_validates(qualification_module, public_evidence):
    result, receipt = public_evidence
    position = qualification_module.validate_public_search(result, receipt, torch.device('cpu'))
    assert position['recipe_recomputed_exactly']
    assert position['selected_index'] == 0
    assert len(position['coordinates']) == 26


@pytest.mark.parametrize('where', ['result', 'receipt', 'record'])
def test_public_incomplete_work_cannot_qualify(qualification_module, public_evidence, where):
    result, receipt = public_evidence
    if where == 'result':
        result.search_status = 'incomplete'
    elif where == 'receipt':
        receipt['search_status'] = 'incomplete'
    else:
        receipt['search'][5]['search_complete'] = False
    with pytest.raises(AssertionError, match='incomplete|unresolved'):
        qualification_module.validate_public_search(result, receipt, torch.device('cpu'))


@pytest.mark.parametrize('field', ['schema', 'graph_sha256', 'selected_lambda', 'raw_objective', 'selection_score'])
def test_public_receipt_identity_mismatch_rejected(qualification_module, public_evidence, field):
    result, receipt = public_evidence
    receipt[field] = 'tampered'
    with pytest.raises(AssertionError, match='schema|identities'):
        qualification_module.validate_public_search(result, receipt, torch.device('cpu'))


@pytest.mark.parametrize('damage', ['shortened', 'changed_penalty'])
def test_public_complete_label_cannot_hide_wrong_default_grid(qualification_module, public_evidence, damage):
    result, receipt = public_evidence
    if damage == 'shortened':
        receipt['search'].pop()
    else:
        receipt['search'][5]['lambda_value'] *= 1.01
    with pytest.raises(AssertionError, match='path|recipe'):
        qualification_module.validate_public_search(result, receipt, torch.device('cpu'))


def test_public_nonfinite_array_rejected(qualification_module, public_evidence):
    result, receipt = public_evidence
    result.raw_phi[0] = np.nan
    with pytest.raises(AssertionError, match='finite'):
        qualification_module.validate_public_search(result, receipt, torch.device('cpu'))


def test_public_unqualified_arrays_rejected_despite_matching_receipt(qualification_module, public_evidence):
    result, receipt = public_evidence
    result.candidate_provenance['raw_qualified'] = False
    receipt['candidate_provenance']['raw_qualified'] = False
    with pytest.raises(AssertionError, match='qualification'):
        qualification_module.validate_public_search(result, receipt, torch.device('cpu'))
