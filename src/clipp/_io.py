"""Representation-only I/O; no statistical or verification policy."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def partition_labels(order, cuts, block_labels=None):
    """Expand validated chain cuts and optional block labels to original rows."""
    labels = np.empty(len(order), dtype=int)
    blocks = np.searchsorted(cuts, np.arange(len(order)), side="right")
    labels[order] = blocks if block_labels is None else np.asarray(block_labels)[blocks]
    return labels


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
