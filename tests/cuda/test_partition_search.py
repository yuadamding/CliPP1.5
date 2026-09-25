"""Numerical references and provenance guards; CPU evidence is not CUDA qualification."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math
import json
import numpy as np
import pytest
import torch

from clipp1d.api import source_provenance
from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import canonical_labels, grouping, partition_score, refit_labels
from clipp1d.cuda.policy import QualificationError
from clipp1d.cuda.refinement import PartitionSearchPolicy, center_costs, move_deltas, reassign_fixed_centers
from clipp1d.cuda.selection import fit_tensor_model
from clipp1d.cuda_api import _export, _publish, _validate_result, PARTITION_SCHEMA


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def model(alt=(20., 48., 21., 46.), lower=None, upper=None):
    alt = tensor(alt)
    slope = torch.full((alt.numel(), 1), .5, dtype=torch.float64)
    return TensorModel(tuple(f"m{i:03}" for i in range(alt.numel())), alt, 100 - alt, slope,
                       torch.zeros_like(slope), torch.full_like(alt, 1e-6) if lower is None else tensor(lower),
                       torch.ones_like(alt) if upper is None else tensor(upper), 1e-6, Kernels('cpu'))


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_explicit_labels_are_canonical_without_merging_equal_centers():
    m = model(alt=(20., 20., 20., 20.))
    result = refit_labels(m, torch.tensor([99, -8, 99, -8]))
    assert result.labels.tolist() == [0, 1, 0, 1]
    assert result.sizes.tolist() == [2, 2]
    torch.testing.assert_close(result.centers, tensor([.4, .4]))
    assert result.clonal == 0 and result.scalar.qualified.all()


def test_empty_intersection_rejects_partition_before_scalar_search():
    m = model(alt=(10., 30.), lower=[.1, .6], upper=[.3, .9])
    with pytest.raises(QualificationError, match='empty feasible interval'):
        refit_labels(m, torch.tensor([0, 0]))


def test_incumbent_is_preserved_without_inheriting_a_certificate():
    m = model()
    z = torch.tensor([0, 1, 0, 1])
    centers = tensor([.41, .94])
    incumbent = partition_score(m.loss(centers[z]).sum(), torch.tensor([2, 2]))
    fitted = refit_labels(m, z, incumbent_centers=centers)
    assert fitted.score <= incumbent
    assert fitted.scalar.qualified.all()
    with pytest.raises(ValueError, match='Incumbent centers'):
        refit_labels(m, z, incumbent_centers=tensor([2., .5]))


def test_exact_one_exception_is_an_explicit_separate_ablation():
    assert grouping(tensor([.49999, .5]), 2e-5)[2].tolist() == [0, 0]
    assert grouping(tensor([.99999, 1.]), 2e-5)[2].tolist() == [0, 1]
    assert grouping(tensor([.99999, 1.]), 2e-5, separate_clonal=False)[2].tolist() == [0, 0]


def independent_score(costs, labels):
    n = len(labels)
    _, sizes = np.unique(labels, return_counts=True)
    k = len(sizes)
    mass = math.lgamma(k) - math.lgamma(n + k) + sum(math.lgamma(int(s) + 1) for s in sizes) + math.lgamma(k + 1)
    return 2 * costs[np.arange(n), labels].sum() + k * math.log(n) - 1.4 * mass


def test_move_formulas_match_full_independent_score_including_singletons():
    rng = np.random.default_rng(89235)
    singleton_checks = 0
    for _ in range(250):
        n = int(rng.integers(2, 30))
        k = int(rng.integers(2, n + 1))
        y = np.r_[np.arange(k), rng.integers(k, size=n-k)]
        rng.shuffle(y)
        c = rng.normal(size=(n, k))
        sizes = np.bincount(y, minlength=k)
        deltas = move_deltas(tensor(c), torch.tensor(y), torch.tensor(sizes), torch.ones((n, k), dtype=torch.bool)).numpy()
        baseline = independent_score(c, y)
        for node in range(n):
            for dest in range(k):
                if dest == y[node]:
                    assert np.isinf(deltas[node, dest])
                    continue
                trial = y.copy()
                trial[node] = dest
                assert deltas[node, dest] == pytest.approx(independent_score(c, trial)-baseline, abs=3e-12)
                singleton_checks += sizes[y[node]] == 1
    assert singleton_checks > 1000


def test_stale_two_singleton_batch_counterexample_is_not_accepted(monkeypatch):
    import clipp1d.cuda.refinement as refinement
    costs = tensor([[0., 1.], [1., 0.]])
    monkeypatch.setattr(refinement, 'center_costs', lambda *args: costs)
    m = model(alt=(20., 30.))
    z = torch.tensor([0, 1])
    before = partition_score(tensor(0.), torch.tensor([1, 1]))
    result = reassign_fixed_centers(m, z, tensor([.4, .6]))
    assert len(result.moves) == 1 and result.labels.tolist() == [0, 0]
    assert result.score < before
    swapped = independent_score(costs.numpy(), np.array([1, 0]))
    assert swapped == pytest.approx(float(before) + 4.)


def test_destinations_use_original_bounds_and_ties_are_deterministic(monkeypatch):
    import clipp1d.cuda.refinement as refinement
    m = model(alt=(20., 30.), lower=[.1, .6], upper=[.5, .9])
    monkeypatch.setattr(refinement, 'center_costs', lambda *args: tensor([[10., 0.], [0., 10.]]))
    result = reassign_fixed_centers(m, torch.tensor([0, 1]), tensor([.4, .8]))
    assert not result.moves and result.labels.tolist() == [0, 1]
    # Both singleton moves tie; canonical mutation zero wins in the unconstrained model.
    monkeypatch.setattr(refinement, 'center_costs', lambda *args: tensor([[0., 1.], [1., 0.]]))
    result = reassign_fixed_centers(model(alt=(20., 30.)), torch.tensor([0, 1]), tensor([.4, .8]))
    assert result.moves[0]['node'] == 0 and result.moves[0]['destination'] == 1


def test_blocked_costs_match_existing_likelihood():
    m = model()
    centers = tensor([.1, .4, .8, 1.])
    actual = center_costs(m, centers, block_size=2)
    expected = torch.stack([m.loss(c.expand(m.n)) for c in centers], dim=1)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


def test_search_keeps_original_raw_trajectory_and_scores(monkeypatch):
    import clipp1d.cuda.solver as solver
    old = solver.solve_start
    starts = []

    def capture(model, graph, lam, initial, policy):
        starts.append((float(lam), initial.clone()))
        return old(model, graph, lam, initial, policy)

    monkeypatch.setattr(solver, 'solve_start', capture)
    m = model()
    baseline = fit_tensor_model(m, lambda_values=[0., .1, 1.])
    expected = starts.copy()
    starts.clear()
    actual = fit_tensor_model(m, lambda_values=[0., .1, 1.], partition_search=PartitionSearchPolicy())
    assert len(starts) == len(expected)
    for (lam, initial), (other, old_initial) in zip(starts, expected):
        assert lam == other
        torch.testing.assert_close(initial, old_initial, rtol=0., atol=0.)
    for field in ('x', 'objective'):
        torch.testing.assert_close(getattr(baseline.raw, field), getattr(actual.raw, field), rtol=0., atol=0.)
    torch.testing.assert_close(baseline.refit.score, actual.refit.score, rtol=0., atol=0.)
    assert baseline.search_status == actual.search_status
    assert actual.partition_estimate.candidate.refit.score <= baseline.refit.score
    for old_record, new_record in zip(baseline.records, actual.records):
        for key in ('lambda_value', 'raw_objective', 'score', 'clusters', 'search_complete'):
            assert old_record[key] == new_record[key]


@pytest.fixture
def repaired_bundle():
    m = model()
    d = fit_tensor_model(m, lambda_values=[0.], partition_search=PartitionSearchPolicy())
    result = _export(d, source_provenance())
    data = SimpleNamespace(tumor_id='test', sample_id='sample',
                           mutations=[SimpleNamespace(mutation_id=mid, exclusion=None) for mid in m.mutation_ids])
    assert result.partition_estimate.provenance['candidate_family'] == 'direct_partition'
    return d, result, data


def test_direct_partition_publishes_separately_with_no_inherited_certificate(repaired_bundle, tmp_path):
    d, result, data = repaired_bundle
    _validate_result(result, data, d.records)
    p = result.partition_estimate
    assert p.provenance['raw_certificate'] is None and p.provenance['selected_lambda'] is None
    assert p.provenance['raw_reference']['certificate']['separable_scalar_gap_qualified']
    assert result.provenance['primary_estimator'] == 'selected_raw_complete_graph_penalized_candidate'
    _publish(result, data, tmp_path, d.records)
    receipt = json.loads((tmp_path/'run.json').read_text())
    assert receipt['schema'] == PARTITION_SCHEMA and len(receipt['table_sha256']) == 5
    assert receipt['partition_estimate']['provenance']['raw_qualified'] is False
    assert (tmp_path/'partition_mutation_clusters.tsv').exists()
    assert p.score < result.selection_score


@pytest.mark.parametrize('field,value', [('raw_certificate', {}), ('raw_qualified', True), ('selected_lambda', .1)])
def test_host_rejects_direct_partition_inheriting_raw_claims(repaired_bundle, field, value):
    d, result, data = repaired_bundle
    p = result.partition_estimate
    meta = deepcopy(p.provenance)
    meta[field] = value
    forged = replace(result, partition_estimate=replace(p, provenance=meta))
    with pytest.raises(QualificationError, match='inherited'):
        _validate_result(forged, data, d.records)


def test_direct_partition_cannot_be_relabelled_as_raw_family(repaired_bundle):
    d, result, data = repaired_bundle
    p = result.partition_estimate
    meta = deepcopy(p.provenance)
    meta.update(candidate_family='raw_fusion_path', raw_qualified=True,
                raw_certificate=meta['raw_reference']['certificate'], selected_lambda=0.)
    forged = replace(result, partition_estimate=replace(p, provenance=meta))
    with pytest.raises(QualificationError, match='raw-derived'):
        _validate_result(forged, data, d.records)


def test_device_rejects_changed_explicit_partition_identity(repaired_bundle):
    d, _, _ = repaired_bundle
    d.partition_estimate.candidate.refit.labels.data[0] = 0
    # Ensure a real alteration even if this node already had zero.
    d.partition_estimate.candidate.refit.centers.data[0] += .001
    with pytest.raises((ValueError, QualificationError), match='modified'):
        _export(d, source_provenance())


def test_fixed_center_move_budget_is_reported():
    m = model()
    result = reassign_fixed_centers(m, torch.arange(m.n), tensor([.4, .96, .42, .92]),
                                   PartitionSearchPolicy(max_moves=1))
    assert result.status == 'move_budget_exhausted' and len(result.moves) == 1


def test_roundoff_stop_without_moves_is_not_a_fixed_point(monkeypatch):
    import clipp1d.cuda.refinement as refinement
    m = model()
    initial = refit_labels(m, torch.arange(m.n))
    stopped = refinement.Reassignment(initial.labels, initial.centers, initial.loss,
                                      initial.score, [], 'roundoff_stop')
    monkeypatch.setattr(refinement, 'reassign_fixed_centers', lambda *args: stopped)
    fitted, records, status = refinement.refine_memberships(m, initial)
    assert fitted is initial and status == 'roundoff_stop'
    assert records[0]['status'] == 'roundoff_stop'
    d = fit_tensor_model(m, lambda_values=[0.], partition_search=PartitionSearchPolicy())
    result = _export(d, source_provenance())
    assert result.partition_estimate.search_status == 'incomplete'
    data = SimpleNamespace(mutations=[SimpleNamespace(mutation_id=x, exclusion=None) for x in m.mutation_ids])
    _validate_result(result, data, d.records)


def test_label_canonicalization_is_independent_of_label_names():
    a = canonical_labels(torch.tensor([7, 3, 7, -1, 3]))[0]
    b = canonical_labels(torch.tensor([0, 88, 0, 102, 88]))[0]
    assert torch.equal(a, b)


def test_failed_proposal_preserves_qualified_baseline_and_reports_incomplete(monkeypatch):
    import clipp1d.cuda.partition_search as search

    def fail(*args):
        raise QualificationError('proposal budget unresolved')

    monkeypatch.setattr(search, 'refine_memberships', fail)
    m = model()
    d = fit_tensor_model(m, lambda_values=[0.], partition_search=PartitionSearchPolicy())
    result = _export(d, source_provenance())
    data = SimpleNamespace(mutations=[SimpleNamespace(mutation_id=x, exclusion=None) for x in m.mutation_ids])
    _validate_result(result, data, d.records)
    assert d.search_status == 'complete'
    assert result.partition_estimate.search_status == 'incomplete'
    assert result.partition_estimate.score <= result.selection_score


def test_device_rejects_tampered_scalar_bounds(repaired_bundle):
    d, _, _ = repaired_bundle
    d.partition_estimate.candidate.refit.scalar.lower_bound.data[0] += 1e-9
    with pytest.raises((ValueError, QualificationError), match='modified'):
        _export(d, source_provenance())


def test_device_rejects_forged_partition_search_completeness(repaired_bundle):
    d, _, _ = repaired_bundle
    d.partition_estimate.status = 'incomplete' if d.partition_estimate.status == 'complete' else 'complete'
    with pytest.raises(QualificationError, match='evidence was modified'):
        _export(d, source_provenance())


def test_every_qualified_start_is_streamed_even_when_raw_objective_loses(monkeypatch):
    from clipp1d.cuda.graph import build_graph
    from clipp1d.cuda.scalar import pilot
    from clipp1d.cuda.solver import RawFit, fit_lambda
    import clipp1d.cuda.solver as solver
    m = model()
    pilots = pilot(m)
    graph = build_graph(pilots.phi)
    seen, scores = [], []
    count = 0

    def synthetic_start(model, graph, lam, initial, policy):
        nonlocal count
        x = tensor([.5, .5, .5, .5]) if count == 0 else tensor([.4, .94, .4, .94])
        work = {k: 0 for k in ('qp_calls', 'qp_admm_iterations', 'qp_dual_warm_starts',
                               'qp_dual_warm_resets', 'qp_polish_iterations', 'qp_seconds',
                               'audit_calls', 'audit_seconds')}
        answer = RawFit(x, torch.zeros_like(graph.weights), tensor(float(count)), None, True, work)
        count += 1
        return answer

    def observe(raw, name):
        seen.append(name)
        scores.append(float(refit_labels(m, grouping(raw.x, 2e-5)[2]).score))

    monkeypatch.setattr(solver, 'solve_start', synthetic_start)
    winner = fit_lambda(m, graph, pilots, .1, on_qualified=observe)
    assert len(seen) == count >= 2
    assert winner.objective == 0.  # Raw continuation retains its original decision.
    assert min(scores[1:]) < scores[0]  # The raw loser still reached partition scoring.
