import gzip

import numpy as np
import pytest

from clipp1d.io import read_tumor
from clipp1d.model import compile_model
from clipp1d.policy import Policy
from clipp1d.types import InputError, NoEligibleMutationsError


def test_identifiers_and_missing(make_input):
    p = make_input([{"mutation_id": "001"}, {"mutation_id": "002", "count_observed": 0,
                   "alt_count": ".", "ref_count": "."}, {"alt_count": 0, "ref_count": 0},
                   {"alt_count": 0}], metadata="##missing_value=.\n")
    data = read_tumor(p)
    assert data.sample_id == "01" and data.mutations[0].mutation_id == "001"
    assert [m.exclusion for m in data.mutations] == [None, "MISSING_COUNTS", "ZERO_DEPTH", None]
    model = compile_model(data)
    assert len(model) == 2 and model.alt[-1] == 0


@pytest.mark.parametrize("changes", [
    [{}, {"sample_id": "02"}], [{"cn_state_fraction": .7}], [{"allele_a_cn": 1.5}],
    [{"allele_b_cn": 2}], [{"purity": "nan"}], [{"purity": 0}], [{}, {"purity": .7}],
    [{"count_observed": "true"}], [{"normal_cn": -1}], [{"alt_count": "."}],
])
def test_rejections(make_input, changes):
    with pytest.raises(InputError):
        read_tumor(make_input(changes))


def test_repeated_fields_and_complete_states(make_input):
    rows = [{"mutation_id": "a", "segment_id": "seg", "cn_state_fraction": .5},
            {"mutation_id": "a", "segment_id": "seg", "cn_state_id": "s2", "cn_state_fraction": .5,
             "allele_a_cn": 2}]
    assert len(compile_model(read_tumor(make_input(rows)))) == 1
    rows[1]["alt_count"] = 11
    with pytest.raises(InputError, match="Repeated"):
        read_tumor(make_input(rows))
    rows[1]["alt_count"] = 10
    rows.append({"mutation_id": "b", "segment_id": "seg", "cn_state_fraction": .5})
    with pytest.raises(InputError, match="Incomplete"):
        read_tumor(make_input(rows))


def test_duplicate_and_conflicting_state(make_input):
    rows = [{"mutation_id": "a", "segment_id": "seg"}] * 2
    with pytest.raises(InputError, match="Duplicate"):
        read_tumor(make_input(rows))
    rows = [{"segment_id": "seg"}, {"segment_id": "seg", "allele_a_cn": 2}]
    with pytest.raises(InputError, match="Inconsistent"):
        read_tumor(make_input(rows))


def test_original_tiny_cn_state_excludes(make_input):
    p = make_input([{"mutation_id": "a", "segment_id": "s", "cn_state_fraction": 1 - 1e-9},
                    {"mutation_id": "a", "segment_id": "s", "cn_state_fraction": 1e-9,
                     "cn_state_id": "s2", "allele_a_cn": 5}, {}])
    data = read_tumor(p)
    assert next(m for m in data.mutations if m.mutation_id == "a").exclusion == "MAJOR_CN_ABOVE_LIMIT"
    assert len(compile_model(data)) == 1


def test_support_cap_and_original_bounds(make_input):
    p = make_input([{"allele_a_cn": 6, "allele_b_cn": 0},
                    {"allele_a_cn": 4, "allele_b_cn": 0, "normal_cn": 0}])
    model = compile_model(read_tumor(p, Policy(max_major_cn=6)), Policy(max_major_cn=6))
    assert np.array_equal(model.valid.sum(axis=1), [4, 4])
    assert model.upper[1] == 1 - 1e-6 and model.upper[1] < 1
    with pytest.raises(ValueError):
        model.upper.setflags(write=True)


def test_all_excluded_and_zero_copy(make_input):
    with pytest.raises(NoEligibleMutationsError):
        compile_model(read_tumor(make_input([{"count_observed": 0}])))
    with pytest.raises(InputError, match="NO_POSITIVE"):
        read_tumor(make_input([{"allele_a_cn": 0, "allele_b_cn": 0}]))


def test_gzip_matches_plain(make_input):
    path = make_input([{}])
    compressed = path.with_suffix(".tsv.gz")
    compressed.write_bytes(gzip.compress(path.read_bytes()))
    assert read_tumor(path).mutations == read_tumor(compressed).mutations


@pytest.mark.parametrize("value", ["1.000000001", "9007199254740993", "NaN", "Infinity"])
@pytest.mark.parametrize("field", ["alt_count", "allele_a_cn"])
def test_integer_text_is_validated_before_rounding(make_input, value, field):
    with pytest.raises(InputError, match="integer"):
        read_tumor(make_input([{field: value}]))


@pytest.mark.parametrize("value,expected", [("10.0", 10), ("1e1", 10),
                                            ("9007199254740992", 2**53)])
def test_exact_integer_notation(make_input, value, expected):
    assert read_tumor(make_input([{"alt_count": value}])).mutations[0].alt == expected


@pytest.mark.parametrize("value", [".", " bad", "bad ", "bad\tname"])
def test_invalid_tumor_metadata(make_input, value):
    with pytest.raises(InputError, match="identifier"):
        read_tumor(make_input([{}], metadata=f"##tumor_id={value}\n"))


def test_blank_extra_column_is_inert(make_input):
    path = make_input([{}])
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\tannotation\n" + lines[1] + "\t\n")
    assert len(read_tumor(path).mutations) == 1


@pytest.mark.parametrize("kind", ["truncated_gzip", "utf8", "tsv_quotes"])
def test_malformed_encoding_is_input_error(make_input, kind):
    path = make_input([{}])
    if kind == "truncated_gzip":
        payload = gzip.compress(path.read_bytes())[:-5]
        path = path.with_suffix(".tsv.gz")
    elif kind == "utf8":
        payload = b"\xff"
    else:
        payload = path.read_bytes() + b'"unterminated\n'
    path.write_bytes(payload)
    with pytest.raises(InputError):
        read_tumor(path)
