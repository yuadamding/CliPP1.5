from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from clipp1d.io import read_tumor
from clipp1d.model import compile_model, evaluate
from clipp1d.policy import Policy
from clipp1d.scalar import interval_lower_bound, minimize_block
from conftest import count_model


def test_scalar_boundary_and_plateau():
    zero = minimize_block(count_model([0], [100]))
    assert zero.qualified and zero.argmin == 1e-6
    high = minimize_block(count_model([90], [10]))
    assert high.qualified and high.argmin == 1
    interior = minimize_block(count_model([10], [90]))
    assert interior.qualified
    assert_allclose(interior.argmin, .25, atol=1e-7)


def test_multimode_lower_bound(fixtures):
    model = compile_model(read_tumor(fixtures / "mixed_cn.tsv")).subset([1])
    result = minimize_block(model)
    assert result.qualified and result.optimality_gap <= 2e-7
    grid = np.linspace(model.lower[0], model.upper[0], 4001)
    dense_best = min(float(evaluate(model, np.array([x])).loss[0]) for x in grid)
    assert result.lower_bound <= dense_best + 1e-10
    assert result.attained_loss <= dense_best + 1e-8
    assert result.alternatives
    for lo, hi in ((1e-6, 1e-4), (.2, .6), (.7, .8)):
        lb = interval_lower_bound(model, lo, hi)
        assert lb <= min(float(evaluate(model, np.array([x])).loss[0]) for x in np.linspace(lo, hi, 50)) + 1e-9


def test_block_minimum_and_gap():
    m = count_model([10, 20], [90, 80])
    result = minimize_block(m)
    assert result.qualified
    assert_allclose(result.argmin, .375, atol=2e-7)
    # A bounded search must keep its gap honest even if its budget is exhausted.
    coarse = minimize_block(m, replace(Policy(), scalar_max_intervals=0))
    assert coarse.lower_bound <= result.attained_loss + 1e-9
