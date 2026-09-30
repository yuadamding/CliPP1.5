"""Allocated-CUDA qualification of the optional exact-refit neighborhood.

These tests deliberately have no CPU fallback. Metadata ledger rejection tests
live separately and are not numerical qualification.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from clipp1d.cuda.ancestry import refit_identity
from clipp1d.cuda.model import TensorModel
from clipp1d.cuda.partition import refit_labels
from clipp1d.cuda.partition_search import PartitionCandidate, PartitionSearch
from clipp1d.cuda.policy import CudaPolicy, QualificationError
from clipp1d.cuda.refinement import PartitionSearchPolicy, refine_memberships
from clipp1d.cuda.refit_relocation import scan_refit_relocations
from clipp1d.partition_output import validate_ancestry, validate_relocation_summary
from clipp1d.types import CountModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Allocated CUDA required")


def make_model(alt, upper=None, lower=None):
    n = len(alt)
    host = CountModel(tuple(str(i) for i in range(n)), np.asarray(alt), 100-np.asarray(alt),
        np.full(n, 1e-6) if lower is None else np.asarray(lower),
        np.ones(n) if upper is None else np.asarray(upper), np.ones((n, 1)),
        np.zeros((n, 1)), np.ones((n, 1), dtype=bool), 1e-6)
    return TensorModel.from_host(host, "cuda:0", compiled=True)


def fit(model, labels):
    return refit_labels(model, torch.tensor(labels, dtype=torch.long, device=model.device))


def brute_candidates(model, initial):
    values = []
    for node in range(model.n):
        for destination in range(initial.centers.numel()):
            if int(initial.labels[node]) == destination:
                continue
            labels = initial.labels.clone()
            labels[node] = destination
            candidate = refit_labels(model, labels)
            values.append((float(candidate.score), node, destination, candidate))
    return sorted(values, key=lambda x: x[:3])


def search_from(model, initial, **budgets):
    policy = PartitionSearchPolicy(refit_relocations=True, birth_mode="off", **budgets)
    search = PartitionSearch(model, None, None, CudaPolicy(), policy)
    raw = SimpleNamespace(x=initial.phi.clone(), objective=initial.loss.clone())
    search.best = PartitionCandidate("raw_fusion_path", initial, raw, initial,
        initial.loss.new_tensor(0.), "baseline", True, {})
    return search


def test_old_center_infeasible_move_matches_independent_whole_refit():
    model = make_model([20, 20, 20, 70, 80, 80, 80], [1., 1., 1., .75, 1., 1., 1.])
    initial = fit(model, [0, 0, 0, 0, 1, 1, 1])
    unchanged, _, status = refine_memberships(model, initial, CudaPolicy(),
                                             PartitionSearchPolicy(max_rounds=20))
    assert status == "fixed_point" and torch.equal(unchanged.labels, initial.labels)
    assert initial.centers[1] > model.upper[3]
    expected = brute_candidates(model, initial)[0]
    result, record = scan_refit_relocations(model, initial, max_candidates=7, batch_candidates=3)
    assert expected[1:3] == (3, 1)
    assert expected[0] < float(initial.score)-1
    assert record["accepted"] and not record["selected_endpoint_complete"]
    assert record["best_proposal"]["node"] == 3 and record["evaluated_candidates"] == 7
    assert record["certified_improvement_margin"] > 0
    assert abs(float(result.score)-expected[0]) < 1e-6
    assert torch.equal(result.labels, expected[3].labels)
    assert bool((result.phi <= model.upper).all())
    assert initial.labels.tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_singleton_deletion_canonical_reindex_and_batch_width_parity():
    model = make_model([15, 15, 55, 55, 54])
    initial = fit(model, [0, 0, 1, 1, 2])
    expected = brute_candidates(model, initial)[0]
    results = []
    for width in (1, 7):
        result, record = scan_refit_relocations(model, initial, max_candidates=10,
                                               batch_candidates=width)
        assert record["accepted"] and record["evaluated_candidates"] == 10
        assert record["best_proposal"]["source_deleted"]
        assert result.centers.numel() == 2
        assert torch.equal(result.labels, expected[3].labels)
        assert abs(float(result.score)-expected[0]) < 1e-6
        results.append(result)
    assert torch.equal(results[0].labels, results[1].labels)
    torch.testing.assert_close(results[0].phi, results[1].phi, atol=1e-12, rtol=0)


def test_independently_unqualified_child_preserves_parent(monkeypatch):
    import clipp1d.cuda.refit_relocation as module
    model = make_model([15, 15, 55, 55, 54])
    initial = fit(model, [0, 0, 1, 1, 2])
    identity = refit_identity(initial)

    def reject(*args, **kwargs):
        raise QualificationError("Injected independent child rejection")

    monkeypatch.setattr(module, "refit_labels", reject)
    result, record = module.scan_refit_relocations(model, initial, max_candidates=10)
    assert result is initial and refit_identity(result) == identity
    assert record["status"] == "unresolved" and not record["accepted"]
    assert record["scan_complete"] and not record["selected_endpoint_complete"]
    assert record["independent_child_refits"] == 1


def test_candidate_budget_rejects_before_any_scalar_candidate_work(monkeypatch):
    import clipp1d.cuda.refit_relocation as module
    model = make_model([20, 20, 80, 80])
    initial = fit(model, [0, 0, 1, 1])

    def unexpected(*args, **kwargs):
        raise AssertionError("Insufficient whole-scan budget entered scalar candidates")

    monkeypatch.setattr(module, "solve_scalar", unexpected)
    result, record = module.scan_refit_relocations(model, initial, max_candidates=3)
    assert result is initial and record["status"] == "candidate_budget_exhausted"
    assert record["evaluated_candidates"] == 0 and record["scalar_batches"] == 0
    assert record["remaining_candidates"] == 4 and not record["selected_endpoint_complete"]


def test_accepted_last_scan_is_incomplete_and_preserves_exact_raw_ancestry():
    model = make_model([15, 15, 55, 55, 54])
    initial = fit(model, [0, 0, 1, 1, 2])
    search = search_from(model, initial, refit_max_scans=1, refit_max_candidates=10)
    original = search.best
    summary = search._relocate_selected()
    assert summary["status"] == "scan_budget_exhausted"
    assert not summary["selected_endpoint_complete"]
    assert search.best.raw_reference is original.raw_reference
    assert search.best.reference_refit is initial
    assert search.best.ancestors == (initial,)
    selected = refit_identity(search.best.refit)
    validate_ancestry(search.best.proposal, refit_identity(initial), selected)
    validate_relocation_summary(summary, asdict(search.search_policy), selected, search.records)
    bad = deepcopy(summary)
    bad["selected_endpoint_complete"] = True
    with pytest.raises(QualificationError, match="final phase ledger"):
        validate_relocation_summary(bad, asdict(search.search_policy), selected, search.records)


def test_global_budget_tracks_new_k_and_requires_fresh_child_scan():
    model = make_model([15, 15, 55, 55, 54])
    initial = fit(model, [0, 0, 1, 1, 2])
    limited = search_from(model, initial, refit_max_scans=2, refit_max_candidates=14)
    incomplete = limited._relocate_selected()
    assert incomplete["status"] == "candidate_budget_exhausted"
    assert [s["planned_candidates"] for s in incomplete["scans"]] == [10, 5]
    assert incomplete["charged_candidates"] == 10
    assert not incomplete["selected_endpoint_complete"]
    complete = search_from(model, initial, refit_max_scans=2, refit_max_candidates=15)
    summary = complete._relocate_selected()
    assert summary["status"] == "fixed_point" and summary["selected_endpoint_complete"]
    assert summary["charged_candidates"] == 15 and summary["remaining_candidate_budget"] == 0
    assert torch.equal(complete.best.refit.labels, limited.best.refit.labels)
    validate_relocation_summary(summary, asdict(complete.search_policy),
                                refit_identity(complete.best.refit), complete.records)


def test_scalar_unresolved_candidates_preclude_complete_neighborhood(monkeypatch):
    import clipp1d.cuda.refit_relocation as module
    model = make_model([20, 20, 80, 80])
    initial = fit(model, [0, 0, 1, 1])
    original = module.solve_scalar

    def unqualified(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, qualified=torch.zeros_like(result.qualified))

    monkeypatch.setattr(module, "solve_scalar", unqualified)
    result, record = module.scan_refit_relocations(model, initial, max_candidates=4)
    assert result is initial and record["status"] == "unresolved"
    assert record["scan_complete"] and len(record["unresolved_candidates"]) == 4
    assert not record["selected_endpoint_complete"]


def test_empty_original_intersections_are_counted_without_forced_fit():
    model = make_model([20, 80], upper=[.3, .9], lower=[.1, .7])
    initial = fit(model, [0, 1])
    result, record = scan_refit_relocations(model, initial, max_candidates=2)
    assert result is initial and record["infeasible_candidates"] == 2
    assert record["scalar_batches"] == 0 and record["evaluated_candidates"] == 0
    assert record["status"] == "fixed_point" and record["selected_endpoint_complete"]


def test_default_off_finish_preserves_existing_result_and_never_scans(monkeypatch):
    import clipp1d.cuda.partition_search as module
    model = make_model([20, 20, 80, 80])
    initial = fit(model, [0, 0, 1, 1])
    raw = SimpleNamespace(x=initial.phi.clone(), objective=initial.loss.clone())

    def unexpected(*args, **kwargs):
        raise AssertionError("Default-off partition estimator entered new neighborhood")

    monkeypatch.setattr(module, "scan_refit_relocations", unexpected)
    search = PartitionSearch(model, None, None, CudaPolicy(), PartitionSearchPolicy(birth_mode="off"))
    result = search.finish(raw, initial, initial.loss.new_tensor(0.))
    assert result.refit_relocation is None
    assert torch.equal(result.candidate.refit.labels, initial.labels)
    assert result.candidate.raw_reference is raw
    result.validate_identity()
