from dataclasses import replace

import pytest

from clipp1d.policy import Policy


@pytest.mark.parametrize("field", ["scalar_atol", "scalar_rtol", "inner_atol", "inner_rtol",
                                  "inner_kkt_tol", "stationarity_tol", "fusion_tol"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_qualification_tolerance(field, value):
    with pytest.raises(ValueError, match=field):
        replace(Policy(), **{field: value})


@pytest.mark.parametrize("value", [0, -1e-6, 1e-30, .5, float("nan")])
def test_probability_bounds_remain_representable(value):
    with pytest.raises(ValueError, match="eps"):
        replace(Policy(), eps=value)


@pytest.mark.parametrize("field", ["inner_max_iterations", "outer_max_iterations", "max_backtracks"])
def test_iterations_require_work(field):
    with pytest.raises(ValueError, match=field):
        replace(Policy(), **{field: 0})


def test_zero_search_budgets_are_valid_but_reversed_path_is_not():
    replace(Policy(), scalar_max_intervals=0, path_extensions=0)
    with pytest.raises(ValueError, match="path_min_exponent"):
        replace(Policy(), path_min_exponent=1, path_max_exponent=0)
