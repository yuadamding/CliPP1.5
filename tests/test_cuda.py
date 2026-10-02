"""Opt-in GPU qualification. Skips are explicitly not CUDA evidence."""

import ctypes
import os
import numpy as np
import pandas as pd
import pytest

from clipp.api import fit
from clipp.config import FitConfig
from test_likelihood_native import D, INT_VECTOR, evaluate

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        os.environ.get("CLIPP_TEST_CUDA") != "1",
        reason="requires authorized allocated CUDA and CLIPP_TEST_CUDA=1",
    ),
]


@pytest.mark.parametrize("count", [1, 255, 256, 257, 513])
def test_cuda_algebra_at_launch_boundaries(native, count):
    native.CliPPWarmupCUDA.restype = ctypes.c_int
    assert native.CliPPWarmupCUDA() == 0
    rng = np.random.default_rng(count)
    major = rng.integers(1, 12, count, dtype=np.int32)
    total = major + rng.integers(0, 2, count, dtype=np.int32)
    depth = rng.integers(20, 10000, count, dtype=np.int32)
    alt = np.array([rng.integers(0, int(n) + 1) for n in depth], dtype=np.int32)
    x = rng.uniform(0.001, 0.999, count)
    if count > 1:
        alt[:2] = [0, depth[1]]
        x[:2] = [0.0, 1.0]
    cpu = evaluate(native, alt, depth, major, total, 0.8, x)
    native.CliPPEvaluateCUDA.argtypes = [
        ctypes.c_int,
        INT_VECTOR,
        INT_VECTOR,
        INT_VECTOR,
        INT_VECTOR,
        ctypes.c_double,
        D,
        D,
    ]
    gpu = np.empty_like(cpu)
    assert native.CliPPEvaluateCUDA(count, alt, depth, major, total, 0.8, x, gpu) == 0
    np.testing.assert_allclose(cpu, gpu, rtol=1e-9, atol=2e-7)


def test_cuda_full_pipeline(native, tiny_inputs, tmp_path):
    for mode in ("cpu", "cuda"):
        fit(*tiny_inputs, tmp_path / mode, config=FitConfig(device=mode, max_clusters=3))
    for k in range(1, 4):
        cpu = pd.read_csv(tmp_path / f"cpu/final_result/mutation_assignments_K{k}.txt", sep="\t")
        gpu = pd.read_csv(tmp_path / f"cuda/final_result/mutation_assignments_K{k}.txt", sep="\t")
        np.testing.assert_array_equal(cpu.cluster_index, gpu.cluster_index)
        for column in ("cellular_prevalence", "mixture_weight"):
            a = pd.read_csv(tmp_path / f"cpu/final_result/subclonal_structure_K{k}.txt", sep="\t")[column]
            b = pd.read_csv(tmp_path / f"cuda/final_result/subclonal_structure_K{k}.txt", sep="\t")[column]
            np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-8)
