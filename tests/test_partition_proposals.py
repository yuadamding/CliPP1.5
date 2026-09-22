"""Independent checks of proposal geometry, refit reuse and certificate separation."""
from dataclasses import replace
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d import fit
from clipp1d.partitions import IntervalRefitter, improves, propose_partitions
from clipp1d.policy import Policy
from clipp1d.selection import refit_partition
from clipp1d.types import NumericalQualificationError
from clipp1d.ward import adjacent_ward_cuts
from conftest import count_model


def test_ward_matches_naive_adjacent_weighted_geometry():
    rng = np.random.default_rng(104)
    for n in (1, 2, 7, 30):
        x, w = rng.random(n), rng.uniform(.1, 20, n)
        actual, work = adjacent_ward_cuts(x, w, 12)
        blocks = [(i, i + 1) for i in range(n)]
        expected = []
        while True:
            if len(blocks) <= 12:
                expected.append(tuple([0] + [b for _, b in blocks]))
            if len(blocks) == 1:
                break
            choices = []
            for i in range(len(blocks) - 1):
                a, b = blocks[i]
                _, c = blocks[i + 1]
                h1, h2 = w[a:b].sum(), w[b:c].sum()
                delta = np.average(x[a:b], weights=w[a:b]) - np.average(x[b:c], weights=w[b:c])
                choices.append((h1 * h2 / (h1 + h2) * delta**2, a, c, i))
            _, a, c, i = min(choices)
            blocks[i:i + 2] = [(a, c)]
        assert actual == tuple(expected)
        assert work['heap_pushes'] <= 3 * n
        assert work['node_capacity'] == 2 * n - 1
    ties, _ = adjacent_ward_cuts(np.zeros(4), np.ones(4))
    assert ties[1] == (0, 2, 3, 4)


def test_cached_refit_matches_original_and_rejects_fractional_cuts():
    model = count_model([3, 21, 4, 37], [97, 79, 96, 63])
    fitter = IntervalRefitter(model)
    for cuts in ((0, 4), (0, 1, 4), (0, 2, 4), (0, 1, 2, 4)):
        actual, expected = fitter.fit(cuts), refit_partition(model, cuts)
        assert_allclose(actual.centers, expected.centers, rtol=0, atol=0)
        assert (actual.loss, actual.score, actual.gap) == (expected.loss, expected.score, expected.gap)
    assert fitter.counters['interval_cache_hits'] > 0
    for cuts in ((0, 1.2, 4), (0, True, 4), (0, 2, 2, 4), ()):
        with pytest.raises(ValueError):
            fitter.fit(cuts)


def test_failed_scalar_is_counted_and_never_selected(monkeypatch):
    import clipp1d.partitions as module
    def fail(*args):
        raise NumericalQualificationError('test unqualified scalar')
    monkeypatch.setattr(module, 'minimize_block', fail)
    fitter = IntervalRefitter(count_model([10], [90]))
    assert fitter.fit((0, 1)) is None
    assert fitter.counters['interval_fits'] == 1
    assert fitter.counters['scalar_rows'] == 1
    assert fitter.failures[0]['error'] == 'NumericalQualificationError'


def test_qualified_original_is_retained_and_acceptance_accounts_for_gap():
    model = count_model([10, 20, 40], [90, 80, 60])
    baseline = refit_partition(model, (0, 1, 2, 3))
    result, diagnostics = propose_partitions(model, np.array([.25, .5, 1]),
                                             np.array([.25, .5, 1]), baseline, Policy())
    assert result.score <= baseline.score
    assert any(tuple(c['cuts']) == baseline.cuts for c in diagnostics['candidates'])
    uncertain = replace(baseline, score=baseline.score - .01, gap=1.)
    assert not improves(uncertain, baseline)


def test_direct_winner_has_no_selected_raw_certificate(make_input, tmp_path, monkeypatch):
    import clipp1d.api as api
    original = api.propose_partitions
    def forced(model, pilot, raw, baseline, policy):
        # Exercise publication of a valid direct partition independently of whether
        # this small fixture happens to prefer it under model selection.
        result, diagnostics = original(model, pilot, raw, baseline, policy)
        cuts = (0, len(model)) if len(baseline.cuts) > 2 else tuple(range(len(model) + 1))
        direct = refit_partition(model, cuts, policy)
        diagnostics['selected_origin'] = 'adjacent_ward_pilot'
        return direct, diagnostics
    monkeypatch.setattr(api, 'propose_partitions', forced)
    out = tmp_path / 'direct'
    result = fit(make_input([{'alt_count': 4}, {'alt_count': 40}]), out)
    run = json.loads((out / 'run.json').read_text())
    assert run['schema'] == 'clipp1d.run.v4'
    assert result.selected_lambda is run['selected_lambda'] is None
    assert run['candidate_provenance']['selected_raw_certificate'] is None
    assert not run['candidate_provenance']['selected_partition_certified']
    assert run['candidate_provenance']['raw_reference']['lambda'] is not None
    assert run['raw_diagnostics']['clonal_feasible']
    assert 'raw_reference_ccf' in (out / 'mutation_clusters.tsv').read_text().splitlines()[0]
