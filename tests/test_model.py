import hashlib
import json

import numpy as np
from numpy.testing import assert_allclose

from clipp1d.io import read_tumor
from clipp1d.model import compile_model, evaluate, one_sided_derivatives


def test_pinned_upstream_fixture(fixtures):
    metadata = json.loads((fixtures / "upstream_reference.json").read_text())
    assert metadata["commit"] == "77525a6875e698f6834cb73e6d2a1c27af2af8bf"
    assert hashlib.sha256((fixtures / "upstream_reference.npz").read_bytes()).hexdigest() == metadata["npz_sha256"]
    assert hashlib.sha256((fixtures / "mixed_cn.tsv").read_bytes()).hexdigest() == metadata["input_sha256"]
    model = compile_model(read_tumor(fixtures / "mixed_cn.tsv"))
    with np.load(fixtures / "upstream_reference.npz") as expected:
        assert_allclose(model.upper, expected["upper"], rtol=0, atol=0)
        assert_allclose(model.slope, expected["slope"], rtol=0, atol=0)
        for i, phi in enumerate(expected["phi"]):
            terms = evaluate(model, phi, derivatives=True)
            for field, upstream in (("loss", "loss"), ("gradient", "gradient"),
                                     ("curvature", "curvature"), ("posterior", "posterior")):
                assert_allclose(getattr(terms, field), expected[upstream][i], rtol=2e-12, atol=2e-10)


def test_derivative_finite_difference(fixtures):
    model = compile_model(read_tumor(fixtures / "mixed_cn.tsv"))
    phi = .45 * model.upper
    terms = evaluate(model, phi, derivatives=True)
    derivative = (evaluate(model, phi + 1e-6).loss - evaluate(model, phi - 1e-6).loss) / 2e-6
    assert_allclose(terms.gradient, derivative, rtol=1e-7, atol=1e-7)


def test_clipping_derivatives():
    from conftest import count_model
    model = count_model([1], [9])
    x = np.array([model.eps / .4])
    terms = evaluate(model, x, derivatives=True)
    left, right = one_sided_derivatives(model, x)
    assert terms.gradient[0] == left[0] == 0 and right[0] < 0
