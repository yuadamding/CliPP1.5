import importlib.util
from pathlib import Path

import pytest

from clipp1d.io import read_tumor
from clipp1d.model import compile_model


def module(name):
    path = Path(__file__).parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_simulation_input_and_truth(tmp_path):
    source, truth = module("simulate").simulate(tmp_path / "sim", 12, ambiguous=True, close_centers=True)
    data = read_tumor(source)
    model = compile_model(data)
    assert len(model) == 12
    assert (model.upper == 1).any()
    assert len(truth.read_text().splitlines()) == 13


def test_metrics_missing_call_and_cna_population():
    compare = module("compare_clipp2")
    f1 = compare.f1_metrics({"a": 1, "b": 2}, {"a": 1, "b": None}, ["a", "b"])
    assert f1["macro_f1"] == .5 and f1["coverage"] == .5
    assert f1["micro_f1"] == pytest.approx(2 / 3)
    assert compare.ari([1, 1, 2, 2], [3, 3, 4, 4]) == 1
    assert compare.ari([1, 1, 1], [1, 2, 3]) == 0
    assert compare.ari([1, 1], [3, 3]) == 1


def test_ccc_constants_and_balanced_cna():
    compare = module("compare_clipp2")
    rows = {i: {"phi": "0.1", "cluster_label": "0"} for i in ("a", "b", "c")}
    truth = {i: {"true_ccf": "0.1", "true_cluster": "0", "true_multiplicity": "1",
                 "major_cn": "2", "minor_cn": "2"} for i in rows}
    calls = {i: {"multiplicity_call": "1"} for i in rows}
    result = compare.metrics(rows, truth, calls, "phi")
    assert result["ccc_constant_vector_case"] and result["ccc"] == 1
    assert result["cna_only_multiplicity"]["eligible"] == 3
    rows["a"]["phi"] = "0.2"
    assert compare.metrics(rows, truth, calls, "phi")["ccc"] == 0


def test_unconstrained_label_zero_is_not_designated_clonal():
    compare = module("compare_clipp2")
    rows = {"a": dict(phi=".4", cluster_label="0")}
    truth = {"a": dict(true_ccf=".4", true_cluster="1", true_multiplicity="1",
                       major_cn="1", minor_cn="1")}
    result = compare.metrics(rows, truth, {}, "phi", designated_label=None)
    assert result["designated_clonal_fraction"] is None
    assert result["all_exact_one_fraction"] == 0
    assert result["ari"] == result["ccc"] == 1


def test_metrics_reject_empty_or_incompatible_populations():
    compare = module("compare_clipp2")
    with pytest.raises(ValueError, match="nonempty"):
        compare.metrics({}, {}, {}, "phi")
    with pytest.raises(ValueError, match="same number"):
        compare.ari([1, 2], [1])


@pytest.mark.parametrize("cn", ["nan", "inf", "-1", "1.5"])
def test_invalid_cn_is_not_silently_scored_as_cna(cn):
    compare = module("compare_clipp2")
    rows = {"a": dict(phi=".4", cluster_label="0")}
    truth = {"a": dict(true_ccf=".4", true_cluster="1", true_multiplicity="1",
                       major_cn=cn, minor_cn="1")}
    with pytest.raises(ValueError, match="CNA eligibility"):
        compare.metrics(rows, truth, {}, "phi")


def test_mixed_cn_requires_truth_target_even_if_primary_state_is_diploid():
    compare = module("compare_clipp2")
    rows = {"a": dict(phi=".4", cluster_label="0")}
    truth = {"a": dict(true_ccf=".4", true_cluster="1", true_multiplicity="1",
                       major_cn="1", minor_cn="1", mixed_cn="1")}
    with pytest.raises(ValueError, match="declared multiplicity truth target"):
        compare.metrics(rows, truth, {}, "phi")
