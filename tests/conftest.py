from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def repository():
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def native():
    from clipp.native import load_native

    return load_native()[0]


@pytest.fixture
def tiny_inputs(tmp_path):
    snv, cna, purity = [tmp_path / name for name in ("snv.txt", "cna.txt", "purity.txt")]
    snv.write_text(
        "chromosome_index\tposition\tref_count\talt_count\n1\t100\t80\t20\n1\t200\t79\t21\n1\t300\t60\t40\n1\t400\t59\t41\n"
    )
    cna.write_text(
        "chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\n1\t1\t1000\t1\t1\t2\n"
    )
    purity.write_text("0.9\n")
    return snv, cna, purity


@pytest.fixture(scope="session")
def legacy_run(repository, tmp_path_factory):
    from clipp.api import fit
    from clipp.config import FitConfig

    out = tmp_path_factory.mktemp("regression") / "fit"
    fit(
        *(repository / "sample" / f"sample.{part}.txt" for part in ("snv", "cna", "purity")),
        out,
        config=FitConfig(device="cpu"),
    )
    return out
