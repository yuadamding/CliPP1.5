"""Regression for filename-derived tumor IDs in allocated cohort workers."""
import hashlib
import io
import json
import zipfile

import pytest

from benchmarks.cohort_staging import stage_case_input, staged_input_path, verify_staged_input
from clipp1d.io import SCHEMA_COLUMNS, read_tumor


def fixture(metadata=b"", filename="tumor-007.tsv", tumor_id="tumor-007"):
    data = metadata + ("\t".join(SCHEMA_COLUMNS) + "\n"
        "m1\tregion1\t20\t80\t1\t0.8\t2\tseg1\tstate1\t1\t1\t1\n").encode()
    case = dict(input_filename=filename, input_member="inputs/000001.tsv",
        input_sha256=hashlib.sha256(data).hexdigest(), tumor_id=tumor_id, sample_id="region1",
        retained_mutations=1, retained_ids_sha256=hashlib.sha256(json.dumps(["m1"]).encode()).hexdigest())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(case["input_member"], data)
    return case, data, zipfile.ZipFile(buffer)


def test_filename_derived_id_survives_staging_without_changing_bytes(tmp_path):
    case, data, bundle = fixture()
    renamed = tmp_path / "input.tsv"
    renamed.write_bytes(data)
    assert read_tumor(renamed).tumor_id == "input"  # Original failure.
    actual = stage_case_input(tmp_path, case, bundle)
    assert actual.name == "tumor-007.tsv" and actual.read_bytes() == data
    assert verify_staged_input(tmp_path, case).tumor_id == "tumor-007"


def test_explicit_metadata_preserved_for_corrected_simclone_filename(tmp_path):
    case, data, bundle = fixture(b"##tumor_id=simht40nt\n", "input.normal-cn2.tsv", "simht40nt")
    actual = stage_case_input(tmp_path, case, bundle)
    assert actual.read_bytes() == data and verify_staged_input(tmp_path, case).tumor_id == "simht40nt"


@pytest.mark.parametrize("field,value", [("tumor_id", "wrong"), ("sample_id", "wrong"),
                                      ("retained_mutations", 2), ("retained_ids_sha256", "wrong")])
def test_wrong_identity_or_population_rejected_before_fit(tmp_path, field, value):
    case, _, bundle = fixture()
    case[field] = value
    with pytest.raises(ValueError, match="differ"):
        stage_case_input(tmp_path, case, bundle)


@pytest.mark.parametrize("name", ["../tumor.tsv", "/tumor.tsv", "x/y.tsv", "x\\y.tsv", "", ".", ".."])
def test_unsafe_input_filename_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="basename"):
        staged_input_path(tmp_path, dict(input_filename=name))


def test_changed_bytes_and_duplicate_stage_rejected(tmp_path):
    case, _, bundle = fixture()
    path = stage_case_input(tmp_path, case, bundle)
    with pytest.raises(FileExistsError):
        stage_case_input(tmp_path, case, bundle)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        verify_staged_input(tmp_path, case)
