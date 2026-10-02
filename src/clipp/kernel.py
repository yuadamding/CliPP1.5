import os
import numpy as np
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


def _prepare_chain(canonical, preliminary_result):
    import json
    from .model import MultiplicityModel
    from .initialization import pooled_cp_initialization

    inputs = canonical.arrays
    pilot, diagnostics = pooled_cp_initialization(MultiplicityModel(*inputs))
    coordinates = canonical.retained
    # Coordinate strings preserve the historical R spelling and tie policy.
    order = np.lexsort((coordinates.position.to_numpy(), coordinates.chromosome_index.to_numpy(), pilot))
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
