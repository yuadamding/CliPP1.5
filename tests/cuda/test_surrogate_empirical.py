from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import evaluate_surrogate_cohort as empirical
from clipp1d.cuda.policy import QualificationError


def path(lam, objective, *, qualified=True, complete=True):
    return dict(lambda_value=lam, raw_status='qualified' if qualified else 'unresolved',
                raw_objective=objective, search_complete=complete, starts_attempted=4,
                starts_unresolved=0 if complete else 1)


def test_positive_exact_lambda_and_incomplete_coverage_are_distinct():
    control = [path(0., 20.), path(1., 12., complete=False), path(2., 15.), path(3., 18.)]
    candidate = [path(0., 20.), path(1., 11.), path(np.nextafter(2., 3.), 14.),
                 path(3., None, qualified=False)]
    r = empirical.fixed_lambda_comparisons(candidate, control)
    first, second = r['common_positive_literal_lambdas']
    assert first['lambda_value'] == 1. and first['raw_objective_difference'] == -1.
    assert first['qualified_objectives_comparable']
    assert first['control']['search_complete'] is False
    assert second['lambda_value'] == 3. and second['raw_objective_difference'] is None
    assert not second['qualified_objectives_comparable']
    assert r['control_only_lambdas'] == [2.]
    assert r['candidate_only_lambdas'] == [np.nextafter(2., 3.)]


@pytest.mark.parametrize('lambdas', [[1., 1.], [-1.], [float('nan')]])
def test_invalid_or_duplicate_penalties_rejected(lambdas):
    with pytest.raises(AssertionError):
        empirical.fixed_lambda_comparisons([path(lam, 0.) for lam in lambdas], [])


def fitted(lam, status):
    return dict(publication_validated=True, summary=dict(
        selected_lambda=lam, search_status=status, raw_objective=lam+10., score=3.,
        raw_ccf=[.3, .8], refitted_ccf=[.3, .8], cluster_labels=[1, 0],
        path_records=[path(0., 10.), path(1., 11.)],
        pilot_and_graph=dict(pilot_sha256='pilot', weights_sha256='graph', weight_rule='fixed',
                             gap_floor=.1, normalization=2.)))


def test_selected_comparison_requires_same_lambda_and_graph():
    a, b = fitted(1., 'complete'), fitted(0., 'incomplete')
    r = empirical.compare(a, b)
    assert r['selected_raw_objective_difference'] is None and not r['both_searches_complete']
    assert r['selected_positive_penalty'] == dict(candidate=True, control=False)
    assert len(r['fixed_lambda_comparisons']['common_positive_literal_lambdas']) == 1
    changed = deepcopy(b)
    changed['summary']['pilot_and_graph']['weights_sha256'] = 'different'
    with pytest.raises(AssertionError, match='graph differ'):
        empirical.compare(a, changed)
    assert empirical.compare(a, dict(publication_validated=False))['comparable'] is False


def test_path_failure_retains_attempt_ledger_and_failure_class(tmp_path, monkeypatch):
    monkeypatch.setattr(empirical.torch.cuda, 'reset_peak_memory_stats', lambda _: None)
    monkeypatch.setattr(empirical.common, 'upload', lambda *a: 'model')
    monkeypatch.setattr(empirical.common, 'timed', lambda device, f: (f(), .1))
    def fail(*args, **kwargs):
        raise QualificationError('Controlled unresolved path', reason='unchanged admission gate')
    monkeypatch.setattr(empirical.study.solver, 'solve_start', fail)
    def fit(model):
        empirical.study.solver.solve_start(None, None, None, None, None)
    monkeypatch.setattr(empirical.timing.selection, 'fit_tensor_model', fit)
    out = tmp_path/'trial.json'
    result = empirical.trial(out, empirical.study.CANDIDATE, SimpleNamespace(), None, 'cpu', None, {}, None)
    assert result['status'] == 'numerical_failure' and not result['publication_validated']
    assert result['starts_attempted'] == result['starts_raised'] == 1
    assert result['starts_returned'] == 0 and out.is_file()
    assert empirical.study.solver.solve_start is fail
