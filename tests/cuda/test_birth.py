"""Independent algebra, exhaustive assignments, ancestry and recovery references."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from clipp1d.cuda.birth import birth_candidates, conditional_split, split_complexity
from clipp1d.cuda.partition import refit_labels
from clipp1d.cuda.ancestry import ancestry_step, refit_identity
from clipp1d.cuda.partition_search import PartitionSearch, PartitionCandidate
from clipp1d.cuda.refinement import PartitionSearchPolicy, refine_memberships
from clipp1d.cuda.policy import CudaPolicy, QualificationError
from clipp1d.partition_output import validate_ancestry
from test_partition_search import model, tensor


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def complexity(sizes):
    n, k = sum(sizes), len(sizes)
    return k*math.log(n)-1.4*(math.lgamma(k)-math.lgamma(n+k)+
                             sum(math.lgamma(x+1) for x in sizes)+math.lgamma(k+1))


def test_global_split_increment_matches_full_score_5000_cases():
    rng = np.random.default_rng(8428)
    for _ in range(5000):
        sizes = rng.integers(1, 60, size=int(rng.integers(1, 15))).tolist()
        group = int(rng.integers(len(sizes)))
        sizes[group] = max(sizes[group], 2)
        n = sizes[group]
        t = int(rng.integers(1, n))
        split = sizes.copy()
        split[group] -= t
        split.append(t)
        actual = float(split_complexity(sum(sizes), len(sizes), n, tensor(t)))
        assert actual == pytest.approx(complexity(split)-complexity(sizes), abs=5e-11)


def test_conditional_split_matches_exhaustive_forced_and_tied_assignments():
    rng = np.random.default_rng(83897)
    for i in range(1000):
        n = int(rng.integers(2, 9))
        costs = rng.normal(size=(n, 2)) if i % 3 else rng.integers(0, 3, (n, 2)).astype(float)
        feasible = rng.random((n, 2)) > .2
        others = rng.integers(1, 10, size=int(rng.integers(0, 4))).tolist()
        nt, k = n+sum(others), 1+len(others)
        loss = float(rng.uniform(0, 10))
        expected = []
        for bits in product((0, 1), repeat=n):
            if len(set(bits)) < 2 or not all(feasible[j, b] for j, b in enumerate(bits)):
                continue
            t = sum(bits)
            expected.append(2*(sum(costs[j, b] for j, b in enumerate(bits))-loss)+
                            complexity(others+[t, n-t])-complexity(others+[n]))
        actual = conditional_split(tensor(costs), torch.tensor(feasible), n_total=nt, k=k, parent_loss=loss)
        if not expected:
            assert actual is None
        else:
            y, delta = actual
            assert float(delta) == pytest.approx(min(expected), abs=1e-11)
            assert bool(torch.tensor(feasible)[torch.arange(n), y].all())
            # Repeated evaluation preserves tie identities, not merely scores.
            assert torch.equal(y, conditional_split(tensor(costs), torch.tensor(feasible),
                               n_total=nt, k=k, parent_loss=loss)[0])


def test_birth_evaluates_whole_group_and_never_forces_ccf_one():
    m = model(alt=(12., 13., 38., 39.))
    parent = refit_labels(m, torch.zeros(4, dtype=torch.long))
    proposals = [c for c, _ in birth_candidates(m, parent) if c is not None]
    best = min(proposals, key=lambda c: float(c.refit.score))
    assert best.refit.centers.numel() == 2 and best.refit.score < parent.score
    assert max(best.refit.centers) < .9  # Initializer=1 does not impose a constraint.
    assert best.refit.labels.tolist() == [0, 0, 1, 1]


def test_general_split_respects_bounds_and_other_memberships():
    m = model(alt=(10., 11., 36., 37., 25., 25.))
    parent = refit_labels(m, torch.tensor([0, 0, 0, 0, 1, 1]))
    proposals = [c for c, _ in birth_candidates(m, parent, search_policy=PartitionSearchPolicy(birth_mode='any_cluster'))
                 if c is not None and c.proposal['parent_group'] == 0]
    best = min(proposals, key=lambda c: float(c.refit.score))
    assert best.refit.centers.numel() == 3
    assert best.refit.labels[4] == best.refit.labels[5]
    assert bool((best.refit.phi >= m.lower).all() & (best.refit.phi <= m.upper).all())
    assert best.proposal['tumor_n'] == 6 and best.proposal['occupied_k'] == 2


def test_conditional_pair_with_no_shared_feasible_center_is_rejected():
    assert conditional_split(tensor([[1, 2], [2, 1]]), torch.tensor([[False, False], [True, True]]),
                             n_total=2, k=1, parent_loss=0.) is None


def test_birth_disabled_or_k_greater_than_one_is_unchanged_in_default_mode():
    m = model()
    parent = refit_labels(m, torch.tensor([0, 1, 0, 1]))
    assert list(birth_candidates(m, parent)) == []
    assert list(birth_candidates(m, parent, search_policy=PartitionSearchPolicy(birth_mode='off'))) == []


def test_refinement_keeps_accepted_history_after_later_move_failure(monkeypatch):
    import clipp1d.cuda.refinement as module
    original = module.reassign_fixed_centers
    calls = 0
    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise QualificationError('later move failure')
        return original(*args)
    monkeypatch.setattr(module, 'reassign_fixed_centers', failing)
    m = model()
    initial = refit_labels(m, torch.arange(4))
    fitted, history, status = refine_memberships(m, initial)
    assert calls == 2 and status == 'unresolved'
    assert fitted.score < initial.score
    assert history[0]['status'] == 'accepted' and history[1]['status'] == 'unresolved'
    assert history[1]['seed_score'] == float(fitted.score)


def test_ancestry_binds_immediate_parent_and_exposes_score_uncertainty():
    m = model()
    parent = refit_labels(m, torch.zeros(4, dtype=torch.long))
    child = refit_labels(m, torch.tensor([0, 1, 0, 1]))
    details = dict(algorithm='grid_k1_birth_v1', tumor_n=4, occupied_k=1)
    step = ancestry_step('birth', parent, child, details, 1e-8)
    proposal = dict(algorithm='partition_ancestry_v2', truth_used=False, raw_certificate_inherited=False,
                    ancestry=[step])
    validate_ancestry(proposal, refit_identity(parent), refit_identity(child))
    assert step['published_score_improved'] and step['refit_order_certified']
    forged = deepcopy(proposal)
    forged['ancestry'][0]['parent']['membership_sha256'] = '0'*64
    with pytest.raises(QualificationError, match='parent identity'):
        validate_ancestry(forged, refit_identity(parent), refit_identity(child))
    # A feasible improvement need not prove ordering of ideal fixed-partition minima.
    uncertain = replace(parent, gap=tensor(float(parent.score-child.score)))
    ambiguous = ancestry_step('birth', uncertain, child, details, 1e-8)
    assert ambiguous['published_score_improved'] and not ambiguous['refit_order_certified']


def candidate(m, labels, origin):
    fitted = refit_labels(m, torch.tensor(labels))
    raw = SimpleNamespace(x=fitted.phi, objective=fitted.loss)
    return PartitionCandidate('raw_fusion_path', fitted, raw, fitted, tensor(0.), origin, True, {})


def test_seed_bank_preserves_cluster_count_diversity_and_membership_distinction():
    m = model()
    search = PartitionSearch(m, None, None, CudaPolicy(), PartitionSearchPolicy(seed_bank_size=3))
    pool = [candidate(m, y, str(i)) for i, y in enumerate([
        [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 2, 2], [8, 9, 8, 9]])]
    for c in pool:
        search._retain_seed(c)
    assert len(search.bank) == 3
    assert {c.refit.centers.numel() for c in search.bank} == {1, 2, 3}
    assert len({tuple(c.refit.labels.tolist()) for c in search.bank}) == 3


def test_multiple_seeds_beat_refinement_of_best_initial_score():
    m = model(alt=(5., 8., 15., 23., 34., 40., 47., 48.))
    first = candidate(m, [0, 0, 0, 0, 1, 1, 2, 1], 'first')
    alternative = candidate(m, [0, 0, 1, 1, 1, 2, 3, 1], 'alternative')
    assert first.refit.score < alternative.refit.score
    results = []
    for count in (1, 3):
        search = PartitionSearch(m, None, None, CudaPolicy(),
                                 PartitionSearchPolicy(birth_mode='off', seed_bank_size=count))
        for c in (first, alternative):
            search._accept(c)
            search._retain_seed(c)
        results.append(search.finish(first.raw_reference, first.refit, tensor(0.)))
    assert results[1].candidate.refit.score < results[0].candidate.refit.score - 1
    assert results[1].preserved.refit.score == results[0].candidate.refit.score
    assert results[1].candidate.proposal['root_origin'] == 'alternative'


def test_birth_failure_retains_already_qualified_better_candidate(monkeypatch):
    import clipp1d.cuda.partition_search as module
    from clipp1d.cuda.birth import BirthCandidate
    m = model(alt=(12., 13., 38., 39.))
    parent = candidate(m, [0, 0, 0, 0], 'parent')
    fitted = refit_labels(m, torch.tensor([0, 0, 1, 1]))
    def fail_later(*args):
        yield BirthCandidate(fitted, dict(algorithm='grid_k1_birth_v1', tumor_n=4, occupied_k=1)), None
        raise QualificationError('later birth failure')
    monkeypatch.setattr(module, 'birth_candidates', fail_later)
    search = PartitionSearch(m, None, None, CudaPolicy(), PartitionSearchPolicy())
    result = search.finish(parent.raw_reference, parent.refit, tensor(0.))
    assert result.candidate.refit.score <= fitted.score < result.preserved.refit.score
    assert result.status == 'incomplete'
    assert any(r['status'] == 'unresolved' for r in result.records)


@pytest.mark.parametrize('kwargs', [dict(seed_bank_size=9), dict(birth_mode='mystery'),
                                   dict(birth_generations=0), dict(birth_refine_seeds=True)])
def test_invalid_budgets_fail(kwargs):
    with pytest.raises(ValueError):
        PartitionSearchPolicy(**kwargs)
