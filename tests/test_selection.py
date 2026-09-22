import math

import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d.selection import partition_score, refit_partition
from conftest import count_model


def test_refit_independently_optimizes_every_block():
    model = count_model([1, 300], [4, 700])
    result = refit_partition(model, (0, 1, 2))
    assert result.designated_clonal_block is None
    assert_allclose(result.centers, [.5, .75], atol=1e-7)
    assert result.gap < 1e-6


def test_partition_score_arithmetic():
    value, components = partition_score(12, np.array([2, 3]))
    mass = math.factorial(2) * math.factorial(1) * math.factorial(2) * math.factorial(3) / math.factorial(6)
    expected = 24 + 2 * math.log(5) - 1.4 * math.log(mass)
    assert_allclose(value, expected, atol=1e-13)
    assert_allclose(components["partition_log_mass"], math.log(mass), atol=1e-13)


@pytest.mark.parametrize('sizes', [[np.inf], [np.nan], [0], [.5]])
def test_partition_score_rejects_invalid_sizes(sizes):
    with pytest.raises(ValueError, match='integer block sizes'):
        partition_score(12, sizes)


def test_refit_rejects_boolean_boundary():
    with pytest.raises(ValueError, match='nonempty intervals'):
        refit_partition(count_model([10, 20], [90, 80]), (0, True, 2))
