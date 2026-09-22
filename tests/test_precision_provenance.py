"""Runtime precision receipts must distinguish storage from arithmetic precision."""

import json
import platform

import numpy as np
import pytest

from clipp1d.api import _floating_format, source_provenance


@pytest.mark.parametrize("dtype", [np.float64, np.longdouble])
def test_format_receipt_matches_actual_numpy_runtime(dtype):
    record = _floating_format(dtype)
    info = np.finfo(dtype)
    assert record["storage_bits"] == np.dtype(dtype).itemsize * 8
    assert record["nmant"] == info.nmant
    assert record["significand_bits"] == info.nmant + 1
    assert record["exponent_bits"] == info.iexp
    assert record["minexp"] == info.minexp
    assert record["maxexp"] == info.maxexp
    assert record["machep"] == info.machep
    for key, value in (("eps", info.eps), ("epsneg", info.epsneg),
                       ("smallest_normal", info.smallest_normal), ("max_finite", info.max)):
        assert isinstance(record[key], str)
        assert dtype(record[key]) == value
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_precision_is_not_inferred_from_storage_width():
    # This assertion also works where longdouble aliases binary64. It never
    # assumes that a 128-bit storage slot supplies a 113-bit significand.
    for dtype in (np.float64, np.longdouble):
        record = _floating_format(dtype)
        assert record["significand_bits"] < record["storage_bits"]
        assert record["significand_bits"] == np.finfo(dtype).nmant + 1


def test_source_provenance_binds_machine_and_precision_without_portability_claim():
    receipt = source_provenance()
    assert receipt["platform"] == platform.platform()
    assert receipt["machine"] == platform.machine()
    assert receipt["python"] == platform.python_version()
    assert receipt["numpy"] == np.__version__
    assert receipt["floating_point_formats"] == {
        "float64": _floating_format(np.float64), "longdouble": _floating_format(np.longdouble),
    }
    assert receipt["dtype"] == "float64"
    assert "does not qualify other platforms" in receipt["precision_scope"]
    json.dumps(receipt, allow_nan=False)
