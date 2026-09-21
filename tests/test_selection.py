import math

import numpy as np
from numpy.testing import assert_allclose

from clipp1d.selection import partition_score, refit_partition
from conftest import count_model


def test_refit_profiles_clonal_block():
    model = count_model([1, 300], [4, 700])
    result = refit_partition(model, (0, 1, 2))
    assert result.designated_clonal_block == 0
    assert_allclose(result.centers, [1, .75], atol=1e-7)
    assert result.gap < 1e-6


def test_partition_score_arithmetic():
    value, components = partition_score(12, np.array([2, 3]))
    mass = math.factorial(2) * math.factorial(1) * math.factorial(2) * math.factorial(3) / math.factorial(6)
    expected = 24 + 2 * math.log(5) - 1.4 * math.log(mass)
    assert_allclose(value, expected, atol=1e-13)
    assert_allclose(components["partition_log_mass"], math.log(mass), atol=1e-13)
