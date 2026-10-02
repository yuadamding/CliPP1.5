import os
import numpy as np
from numpy import genfromtxt
import ctypes


CLIPP_CHAIN_ARGTYPES = [
    ctypes.c_int,
    *[np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")] * 4,
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_char_p,
]


def _get_clipp_entrypoint(clipp_lib):
    version = getattr(clipp_lib, "CliPPMultiplicityVersion", None)
    if version is None:
        raise RuntimeError("Rebuild CliPP: this library predates the chain distance-to-set model.")
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != 3:
        raise RuntimeError("Rebuild CliPP for the chain distance-to-set model.")
    clipp_lib.CliPPChainStatus.argtypes = CLIPP_CHAIN_ARGTYPES
    clipp_lib.CliPPChainStatus.restype = ctypes.c_int
    return clipp_lib.CliPPChainStatus


def _load_inputs(prefix):
    with open(os.path.join(prefix, "multiplicity_model.txt")) as handle:
        if handle.read().strip() != "uniform_1_to_major_v1":
            raise ValueError("Rerun preprocessing for the multiplicity mixture model.")
    arrays = [
        np.atleast_1d(genfromtxt(os.path.join(prefix, name + ".txt")))
        for name in ("r", "n", "major", "total")
    ]
    if not arrays[0].size or any(a.ndim != 1 or a.size != arrays[0].size for a in arrays):
        raise ValueError("Preprocessed count and copy-number vectors must have matching nonzero lengths.")
    for a in arrays:
        if (
            not np.all(np.isfinite(a))
            or np.any(a != np.floor(a))
            or np.any(a > np.iinfo(np.int32).max)
            or np.any(a < 0)
        ):
            raise ValueError("Counts and copy numbers must be nonnegative int32 integers.")
    r, n, major, total = arrays
    if np.any(n <= 0) or np.any(r > n) or np.any(major < 1) or np.any(major > total):
        raise ValueError("Invalid read counts or multiplicity support.")
    purity = float(np.loadtxt(os.path.join(prefix, "purity_ploidy.txt")))
    if not np.isfinite(purity) or not 0 < purity <= 1:
        raise ValueError("Purity must be in (0, 1].")
    return tuple(np.ascontiguousarray(a, dtype=np.int32) for a in arrays) + (purity,)


def _prepare_chain(prefix, preliminary_result):
    import json
    from .model import MultiplicityModel
    from .initialization import pooled_cp_initialization
    import pandas as pd

    inputs = _load_inputs(prefix)
    model = MultiplicityModel(*inputs)
    coordinates = pd.read_csv(os.path.join(prefix, "multiplicity.txt"), sep=r"\s+", header=None, dtype=str)
    if len(coordinates) != len(model) or coordinates[[0, 1]].duplicated().any():
        raise ValueError("Expected one unique mutation coordinate per input row.")
    pilot, diagnostics = pooled_cp_initialization(model)
    # This order is frozen for every K and every continuation step.
    order = np.lexsort((coordinates[1].to_numpy(), coordinates[0].to_numpy(), pilot))
    os.makedirs(preliminary_result, exist_ok=True)
    np.savetxt(os.path.join(preliminary_result, "chain_order.txt"), order, fmt="%d")
    np.savetxt(os.path.join(preliminary_result, "pilot_cp.txt"), pilot, fmt="%.17g")
    with open(os.path.join(preliminary_result, "chain_initialization.json"), "w") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return inputs, pilot, order


def _run_kernel(r, n, major, total, purity, pilot, preliminary_result, cluster_list):
    os.makedirs(preliminary_result, exist_ok=True)
    from .native import load_native

    clipp_lib, _ = load_native()
    entrypoint = _get_clipp_entrypoint(clipp_lib)
    clusters = np.asarray(cluster_list, dtype=float)
    count = len(r)
    if (
        clusters.ndim != 1
        or not clusters.size
        or not np.all(np.isfinite(clusters))
        or np.any(clusters != np.floor(clusters))
        or np.any(clusters < 1)
        or np.any(clusters > min(10, count))
        or len(np.unique(clusters)) != len(clusters)
    ):
        raise ValueError("Cluster capacities must be distinct integers in 1..min(10,N).")
    clusters = np.ascontiguousarray(clusters, dtype=np.int32)
    pilot = np.ascontiguousarray(pilot, dtype=np.float64)
    if pilot.shape != (count,) or np.any(~np.isfinite(pilot)) or np.any(pilot < 0) or np.any(pilot > purity):
        raise ValueError("Invalid marginal CP estimates for chain initialization.")
    status = entrypoint(
        count, r, n, major, total, purity, pilot, clusters, len(clusters), os.fsencode(preliminary_result)
    )
    if status != 0:
        raise RuntimeError("CliPP chain fit failed with status %s" % status)


def run_clipp_nosub(prefix, preliminary_result, cluster_list):
    (r, n, major, total, purity), pilot, order = _prepare_chain(prefix, preliminary_result)
    _run_kernel(
        r[order], n[order], major[order], total[order], purity, pilot[order], preliminary_result, cluster_list
    )
