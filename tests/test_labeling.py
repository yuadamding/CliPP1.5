"""Historical chain-reference regression; public CUDA contracts live in tests/cuda."""
import numpy as np
import pytest

from clipp1d.labeling import clonality_summary, public_cluster_order
from legacy_chain_api import fit
from clipp1d.report import write_result


def test_nearest_one_uses_distance_and_deterministic_chain_ties():
    assert public_cluster_order([.25, .9, 1.3], [0, 4, 7, 9]) == [1, 2, 0]
    assert public_cluster_order([.8, .8, .2], [0, 5, 8, 10]) == [0, 1, 2]


def test_exact_one_ties_designate_only_one_cluster():
    summary = clonality_summary([1., 1., .3], np.array([0, 0, 1, 1, 1, 2]))
    assert summary['tie_count'] == 2 and summary['tied_cluster_labels'] == [0, 1]
    assert summary['clonal_mutations'] == 2
    assert summary['subclonal_mutations'] == 4
    assert summary['subclonal_mutation_fraction'] == 4 / 6


def test_no_center_is_changed_to_one(make_input, tmp_path):
    result = fit(make_input([dict(alt_count=10, ref_count=90)] * 8 +
                            [dict(alt_count=30, ref_count=70)] * 8), tmp_path / 'fit')
    np.testing.assert_allclose(result.cluster_centers, [.75, .25], atol=1e-12)
    summary = clonality_summary(result.cluster_centers, result.cluster_labels)
    assert summary['clonal_cluster_label'] == 0 and summary['clonal_mutations'] == 8
    assert summary['subclonal_mutation_fraction'] == .5
    assert summary['distance_to_one'] == pytest.approx(.25)
    assert result.raw_witness_index is None
    assert result.provenance['clonal_constraint'] is False
    result.designated_clonal_block = None
    with pytest.raises(ValueError, match='clonal designation'):
        write_result(result, tmp_path / 'invalid')
