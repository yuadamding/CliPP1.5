from pathlib import Path

import numpy as np
import pytest

from clipp1d.io import SCHEMA_COLUMNS
from clipp1d.types import CountModel


@pytest.fixture
def make_input(tmp_path):
    def make(rows, name="tumor.tsv", metadata=""):
        path = tmp_path / name
        normalized = []
        for i, changes in enumerate(rows):
            row = dict(zip(SCHEMA_COLUMNS, [f"m{i:03}", "01", 10, 90, 1, .8, 2,
                                           f"seg{i}", "s1", 1, 1, 1]))
            row.update(changes)
            normalized.append("\t".join(str(row[k]) for k in SCHEMA_COLUMNS))
        path.write_text(metadata + "\t".join(SCHEMA_COLUMNS) + "\n" + "\n".join(normalized) + "\n")
        return path
    return make


def count_model(alt, ref, slope=.4, upper=None):
    alt, ref = np.asarray(alt), np.asarray(ref)
    n = len(alt)
    return CountModel(tuple(f"m{i}" for i in range(n)), alt, ref, np.full(n, 1e-6),
                      np.ones(n) if upper is None else np.asarray(upper),
                      np.broadcast_to(np.asarray(slope), (n, 1)), np.zeros((n, 1)),
                      np.ones((n, 1), dtype=bool), 1e-6)


@pytest.fixture
def fixtures():
    return Path(__file__).parent / "fixtures"
