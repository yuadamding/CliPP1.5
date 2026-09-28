from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'benchmarks'))
from mixture_structural_guard import choose
from clipp1d.cuda.mixture import MixturePolicy, fit_mixture_reference
from test_mixture_estimator import model_from_counts


@pytest.fixture
def split():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model_from_counts(np.repeat([7, 32], 30), 100, np.ones(60, dtype=int))
    policy = MixturePolicy(max_clusters=2, max_iterations=50)
    result = fit_mixture_reference(model, policy)
    yield model, policy, result
    torch.set_num_threads(previous)


def test_unchanged_structure_preserves_baseline_and_supported_birth_can_enter(split):
    model, policy, result = split
    candidate, decision = choose(model, result.records, policy, baseline_k=2)
    assert candidate is None
    assert decision['selected_family'] == 'preserved_complete_graph_partition'
    candidate, decision = choose(model, result.records, policy, baseline_k=1)
    assert candidate is not None
    assert len(candidate.public_arrays(model.mutation_ids)['centers']) == 2
    assert decision['selected']['score'] < decision['reference']['score']
    assert decision['raw_certificate_inherited'] is False


def test_guard_checks_scores_and_reference_instead_of_trusting_metadata(split):
    model, policy, result = split
    records = deepcopy(result.records)
    records[0]['score'] -= 1
    with pytest.raises(ValueError, match='score'):
        choose(model, records, policy, baseline_k=1)
    with pytest.raises(ValueError, match='reference'):
        choose(model, result.records, policy, baseline_k=3)


def test_unconverged_new_structure_is_not_admitted(split):
    model, policy, result = split
    records = deepcopy(result.records)
    for record in records:
        record['status'] = 'iteration_budget_exhausted'
    candidate, decision = choose(model, records, policy, baseline_k=1)
    assert candidate is None
    assert decision['eligible'] == 0
