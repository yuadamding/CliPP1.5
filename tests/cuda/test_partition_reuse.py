"""CPU tensor reference tests; allocated CUDA qualification is separate."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from clipp1d.cuda.kernels import Kernels
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import (
    LastPartitionCache, QualifiedPilot, grouping, polish_quadratic, refit,
)
from clipp1d.cuda.policy import CudaPolicy, QualificationError
from clipp1d.cuda.scalar import pilot
from clipp1d.cuda.selection import fit_tensor_model


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def model(alt=(10., 30., 12., 16.)):
    a = tensor(alt)
    n = len(alt)
    return TensorModel(tuple(f"m{i:03}" for i in range(n)), a, 100. - a,
                       torch.full((n, 1), .4, dtype=torch.float64),
                       torch.zeros((n, 1), dtype=torch.float64), torch.full_like(a, 1e-6),
                       torch.ones_like(a), 1e-6, Kernels("cpu"))


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("values,expected", [
    ([.3, .3, .300015, .300030], [0, 0, 1, 2]),
    ([.3, .3, .300015, .300030, .300030], [0, 0, 1, 2, 2]),
    ([.300030, .3, .300015, .3, .300030], [0, 1, 2, 1, 0]),
])
def test_exact_fusions_survive_overwide_runs(values, expected):
    assert grouping(tensor(values), 2e-5)[2].tolist() == expected


def test_randomized_exact_equality_diameter_and_canonical_membership():
    rng = np.random.default_rng(48102)
    tolerance = 2e-5
    for _ in range(200):
        values = .3 + rng.integers(0, 30, size=int(rng.integers(4, 64))) * 1.5e-5
        values = np.concatenate((values, [1., 1. - 1e-8, 1.]))
        rng.shuffle(values)
        order, sorted_labels, labels, counts = grouping(tensor(values), tolerance)
        actual = labels.numpy()
        assert np.all((values[:, None] != values[None, :]) |
                      (actual[:, None] == actual[None, :]))
        blocks = [np.flatnonzero(actual == group) for group in range(int(actual.max()) + 1)]
        assert [int(b[0]) for b in blocks] == sorted(int(b[0]) for b in blocks)
        assert all(np.ptp(values[b]) <= tolerance for b in blocks)
        assert all(not (np.any(values[b] == 1.) and np.any(values[b] != 1.)) for b in blocks)
        assert sorted_labels.tolist() == labels[order].tolist()
        assert counts[:len(blocks)].tolist() == [len(b) for b in blocks]


def test_quadratic_polish_preserves_equal_pairs_in_overwide_runs():
    x = tensor([.3, .3, .300015, .300030, .300030])
    h, target = tensor([1., 3., 1., 1., 2.]), tensor([.1, .4, .8, .2, .5])
    caps = torch.zeros((5, 5), dtype=torch.float64)
    proposal = polish_quadratic(x, caps, h, target, torch.zeros_like(x),
                               torch.ones_like(x), caps, 2e-5)
    torch.testing.assert_close(proposal, tensor([.325, .325, .8, .4, .4]), atol=1e-15, rtol=0.)


def test_last_partition_cache_reuses_membership_and_isolates_returned_values(monkeypatch):
    import clipp1d.cuda.partition as partition
    m = model()
    cache = LastPartitionCache(m)
    original = partition.solve_scalar
    calls = []

    def counted(problems, policy):
        calls.append(problems.count)
        return original(problems, policy)

    monkeypatch.setattr(partition, "solve_scalar", counted)
    first = refit(m, tensor([.2, .8, .2, .8]), cache=cache)
    expected_phi, expected_score = first.phi.clone(), first.score.clone()
    first.phi.data[0] = .99
    first.score.fill_(0.)
    second = refit(m, tensor([.1, .9, .1, .9]), cache=cache)
    assert calls == [2]
    assert (cache.refits_computed, cache.refits_reused) == (1, 1)
    torch.testing.assert_close(second.phi, expected_phi, atol=0., rtol=0.)
    torch.testing.assert_close(second.score, expected_score, atol=0., rtol=0.)
    assert second.labels.tolist() == [0, 1, 0, 1]


def test_cache_retains_only_last_partition():
    m = model()
    cache = LastPartitionCache(m)
    a, b = tensor([.2, .8, .2, .8]), tensor([.2, .2, .8, .8])
    first = refit(m, a, cache=cache)
    refit(m, b, cache=cache)
    third = refit(m, a, cache=cache)
    assert (cache.refits_computed, cache.refits_reused) == (3, 0)
    torch.testing.assert_close(first.score, third.score, atol=0., rtol=0.)


def test_unqualified_refit_never_populates_cache(monkeypatch):
    import clipp1d.cuda.partition as partition
    m = model()
    cache = LastPartitionCache(m)
    original = partition.solve_scalar

    def unresolved(problems, policy):
        result = original(problems, policy)
        return replace(result, qualified=torch.zeros_like(result.qualified))

    monkeypatch.setattr(partition, "solve_scalar", unresolved)
    x = tensor([.2, .8, .2, .8])
    with pytest.raises(QualificationError, match="refit unresolved"):
        refit(m, x, cache=cache)
    assert cache.result is None
    assert (cache.refits_computed, cache.refits_reused) == (1, 0)
    monkeypatch.setattr(partition, "solve_scalar", original)
    actual = refit(m, x, cache=cache)
    assert actual.scalar.qualified.all()
    assert (cache.refits_computed, cache.refits_reused) == (2, 0)


@pytest.mark.parametrize("mismatch", ["model", "policy", "source_data", "cached_data", "metadata"])
def test_cache_rejects_changed_identity_or_immutable_state(mismatch):
    m, policy = model(), CudaPolicy()
    cache = LastPartitionCache(m, policy)
    x = tensor([.2, .8, .2, .8])
    refit(m, x, policy, cache=cache)
    if mismatch == "model":
        m = model()
    elif mismatch == "policy":
        policy = replace(policy, scalar_atol=2e-7)
    elif mismatch == "source_data":
        m.alt.data[0] += 1.
    elif mismatch == "cached_data":
        cache.result.centers.data[0] += .1
    else:
        cache.result.clonal = 19
    with pytest.raises(ValueError, match="modified|bound"):
        refit(m, x, policy, cache=cache)


def test_singleton_reuse_resolves_canonical_nodes_not_group_positions(monkeypatch):
    import clipp1d.cuda.partition as partition
    m = model()
    p = QualifiedPilot(m, pilot(m))
    cache = LastPartitionCache(m)
    raw = tensor([.2, .8, .2, .6])
    expected = refit(m, raw)
    original = partition.solve_scalar
    observations = []

    def counted(problems, policy):
        observations.append((problems.model.n, problems.count))
        return original(problems, policy)

    monkeypatch.setattr(partition, "solve_scalar", counted)
    actual = refit(m, raw, pilot_reuse=p, cache=cache)
    assert observations == [(2, 1)]
    assert actual.labels.tolist() == [0, 1, 0, 2]
    assert cache.singleton_pilots_reused == 2
    torch.testing.assert_close(actual.centers, tensor([.275, .75, .4]), atol=2e-15, rtol=0.)
    torch.testing.assert_close(actual.score, expected.score, atol=1e-12, rtol=0.)
    torch.testing.assert_close(actual.phi, expected.phi, atol=2e-15, rtol=0.)


def test_all_singletons_reuse_pilot_without_scalar_search(monkeypatch):
    import clipp1d.cuda.partition as partition
    m = model()
    p = pilot(m)
    reusable = QualifiedPilot(m, p)

    def forbidden(*args, **kwargs):
        raise AssertionError("A qualified singleton pilot should not be solved again")

    monkeypatch.setattr(partition, "solve_scalar", forbidden)
    cache = LastPartitionCache(m)
    actual = refit(m, p.phi, pilot_reuse=reusable, cache=cache)
    assert cache.singleton_pilots_reused == m.n
    torch.testing.assert_close(actual.phi, p.phi, atol=0., rtol=0.)
    assert actual.scalar.subdivisions == 0


@pytest.mark.parametrize("mismatch", ["model", "policy", "data"])
def test_pilot_reuse_rejects_changed_identity_or_state(mismatch):
    m, policy = model(), CudaPolicy()
    reusable = QualifiedPilot(m, pilot(m), policy)
    if mismatch == "model":
        m = model()
    elif mismatch == "policy":
        policy = replace(policy, scalar_atol=2e-7)
    else:
        reusable.scalar.phi.data[0] += .1
    with pytest.raises(ValueError, match="bound|modified"):
        refit(m, tensor([.1, .2, .3, .4]), policy, pilot_reuse=reusable)


@pytest.mark.parametrize("field", ["loss", "gap", "qualified"])
def test_pilot_reuse_requires_actual_qualified_model_loss(field):
    m = model()
    p = pilot(m)
    if field == "loss":
        bad = replace(p, loss=p.loss + 1., lower_bound=p.lower_bound + 1.)
    elif field == "gap":
        bad = replace(p, gap=p.gap + 1.)
    else:
        bad = replace(p, qualified=torch.zeros_like(p.qualified))
    with pytest.raises(QualificationError, match="pilot|Singleton"):
        QualifiedPilot(m, bad)


def test_selection_receipts_count_reuse_and_nonoverlapping_phase_times():
    m = model()
    result = fit_tensor_model(m, lambda_values=[0., 0.])
    assert result.search_status == "complete"
    timings = result.timings
    assert timings["refits_computed"] == 1
    assert timings["refits_reused"] == 1
    assert timings["singleton_pilots_reused"] == m.n
    assert result.records[0]["singleton_pilots_reused"] == m.n
    assert result.records[1]["refits_reused"] == 1
    phases = [timings[key] for key in ("pilot_seconds", "graph_build_seconds", "path_seconds",
                                      "refit_seconds", "stage_integrity_seconds")]
    assert all(value >= 0. for value in phases)
    assert sum(phases) == pytest.approx(timings["numerical_wall_seconds"], abs=1e-12)
