import numpy as np
from numpy.testing import assert_allclose
import pytest

from clipp1d.chain import adjoint, build_chain, difference, extract_blocks
from clipp1d.types import PilotResult


def pilot(phi):
    phi = np.asarray(phi)
    return PilotResult(phi, phi * 0, phi * 0, phi * 0 + 1, phi, phi * 0)


def test_chain_ties_and_weights():
    p = pilot([.3, .3, .9])
    c = build_chain(p, ("z", "a", "b"))
    assert c.order.tolist() == [1, 0, 2]
    assert c.weights.shape == (2,) and np.mean(c.weights) == 1
    assert c.weights[0] > c.weights[1]
    assert_allclose(build_chain(pilot([.5] * 3), ("a", "b", "c")).weights, 1)
    assert build_chain(pilot([.5]), ("a",)).weights.size == 0
    with pytest.raises(ValueError):
        c.weights[0] = 2


def test_adjoint():
    rng = np.random.default_rng(8)
    for n in (1, 2, 9):
        x, q = rng.normal(size=n), rng.normal(size=n - 1)
        assert_allclose(np.dot(difference(x), q), np.dot(x, adjoint(q)), atol=1e-14)


def test_partition_ranges_and_clonal_exactness():
    assert extract_blocks([.1, .10009, .10018], .0001) == (0, 2, 3)
    assert extract_blocks([.2, .8, .2], .001) == (0, 1, 2, 3)
    assert extract_blocks([1, 1 - 1e-9, 1], .001) == (0, 1, 2, 3)
