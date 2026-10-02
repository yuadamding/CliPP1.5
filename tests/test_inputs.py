from pathlib import Path
import subprocess
import pandas as pd
import pytest


def preprocess(inputs, output):
    import clipp

    resource = Path(clipp.__file__).parent / "resources/preprocess.R"
    result = subprocess.run(
        ["Rscript", str(resource), *(str(p) for p in inputs), "fixture", str(output)],
        capture_output=True,
        text=True,
    )
    return result


@pytest.mark.parametrize("encoding", ["1", "chr1", "CHR1", "01"])
def test_normalization_and_ledger(tiny_inputs, tmp_path, encoding):
    snv, cna, _ = tiny_inputs
    for p in [snv, cna]:
        p.write_text(p.read_text().replace("\n1\t", "\n" + encoding + "\t"))
    result = preprocess(tiny_inputs, tmp_path / "pre")
    assert result.returncode == 0, result.stderr
    retained = pd.read_csv(tmp_path / "pre/retained.tsv", sep="\t")
    assert len(retained) == 4 and set(retained.chromosome_index) == {1}
    ledger = pd.read_csv(tmp_path / "pre/input_ledger.tsv", sep="\t")
    assert ledger.original_row.tolist() == [1, 2, 3, 4]
    assert set(ledger.matched_segment_id) == {1}


@pytest.mark.parametrize("encoding", ["X", "chrY", "23", "24", "MT", "NA", "1.5"])
def test_unsupported_chromosome(tiny_inputs, tmp_path, encoding):
    tiny_inputs[0].write_text(tiny_inputs[0].read_text().replace("\n1\t", "\n" + encoding + "\t"))
    result = preprocess(tiny_inputs, tmp_path / "pre")
    assert result.returncode != 0 and "Only autosomes" in result.stderr


@pytest.mark.parametrize(
    "change,needle",
    [
        ("purity", "exactly one"),
        ("overlap", "overlapping"),
        ("coordinate", "coordinates"),
        ("duplicate", "Duplicate"),
        ("cn", "copy numbers"),
        ("missing_cn", "copy numbers"),
    ],
)
def test_malformed_inputs(tiny_inputs, tmp_path, change, needle):
    snv, cna, purity = tiny_inputs
    if change == "purity":
        purity.write_text(".8\n.9\n")
    elif change == "overlap":
        cna.write_text(cna.read_text() + "1\t200\t1001\t2\t1\t3\n")
    elif change == "coordinate":
        snv.write_text(snv.read_text().replace("\t100\t", "\t100.5\t"))
    elif change == "duplicate":
        snv.write_text(snv.read_text() + "chr1\t100\t50\t1\n")
    elif change == "cn":
        cna.write_text(cna.read_text().replace("\t1\t1\t2", "\t1.5\t1\t2.5"))
    else:
        cna.write_text(cna.read_text().replace("\t1\t1\t2", "\t1\tNA\t2"))
    result = preprocess(tiny_inputs, tmp_path / "pre")
    assert result.returncode != 0 and needle in result.stderr


def test_exclusions_reconciled_and_empty_rejected(tiny_inputs, tmp_path):
    snv, cna, _ = tiny_inputs
    snv.write_text(snv.read_text().replace("\t80\t20", "\t0\t0") + "1\t2000\t10\t2\n")
    result = preprocess(tiny_inputs, tmp_path / "pre")
    assert result.returncode == 0, result.stderr
    ledger = pd.read_csv(tmp_path / "pre/input_ledger.tsv", sep="\t")
    assert ledger.reason.tolist() == [
        "invalid_counts_or_zero_depth",
        "retained",
        "retained",
        "retained",
        "no_cna_segment",
    ]
    cna.write_text(cna.read_text().replace("\t1\t1000\t", "\t9000\t10000\t"))
    result = preprocess(tiny_inputs, tmp_path / "empty")
    assert result.returncode != 0 and "No retained mutations" in result.stderr
    assert (tmp_path / "empty/input_ledger.tsv").is_file()


@pytest.mark.parametrize("extra", ["\textra", ""])
def test_inconsistent_row_width_fails(tiny_inputs, tmp_path, extra):
    snv = tiny_inputs[0]
    lines = snv.read_text().splitlines()
    lines[1] = lines[1] + extra if extra else "\t".join(lines[1].split("\t")[:-1])
    snv.write_text("\n".join(lines) + "\n")
    result = preprocess(tiny_inputs, tmp_path / "bad-width")
    assert result.returncode != 0 and "number of fields" in result.stderr


def test_quoted_purity_is_not_a_numeric_token(tiny_inputs, tmp_path):
    tiny_inputs[2].write_text('"0.9"\n')
    result = preprocess(tiny_inputs, tmp_path / "quoted")
    assert result.returncode != 0 and "Purity" in result.stderr
