"""Representation-only I/O; no statistical or verification policy."""

import json
import os
from pathlib import Path

import pandas as pd


def read_table(path):
    return pd.read_csv(
        path,
        sep="\t",
        float_precision="round_trip",
        converters={
            name: str
            for name in (
                "mutation_id",
                "chromosome_index",
                "position",
                "original_chromosome",
                "original_position",
                "chain_cuts",
            )
        },
    )


def write_table(data, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    data.to_csv(temporary, sep="\t", index=False, float_format="%.17g")
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
