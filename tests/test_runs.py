import ctypes
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from clipp._flags import parse_flag
from clipp.api import fit
from clipp.config import FitConfig
from clipp.native import sha256
from clipp.verify import verify_run


@pytest.mark.parametrize(
    "value", [None, "", "0", "false", "False", "no", "off", "1", "TRUE", "yes", "on", "bad", " true "]
)
def test_boolean_contract(native, value):
    native.CliPPParseFlag.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    native.CliPPParseFlag.restype = ctypes.c_int
    output = ctypes.c_int()
    status = native.CliPPParseFlag(None if value is None else value.encode(), ctypes.byref(output))
    if value in {"bad", " true "}:
        with pytest.raises(ValueError):
            parse_flag(value)
        assert status == 1
    else:
        parsed = parse_flag(value)
        assert status == 0 and output.value == (-1 if parsed is None else int(parsed))


def test_conflicts_and_false_flags(monkeypatch):
    monkeypatch.setenv("CLIPP_FORCE_CPU", "0")
    monkeypatch.setenv("CLIPP_REQUIRE_CUDA", "false")
    assert FitConfig(device="cuda").resolved_device() == "cuda"
    monkeypatch.setenv("CLIPP_FORCE_CPU", "true")
    with pytest.raises(ValueError, match="Conflicting"):
        FitConfig(device="cuda").resolved_device()
    monkeypatch.setenv("CLIPP_REQUIRE_CUDA", "yes")
    with pytest.raises(ValueError, match="Conflicting"):
        FitConfig().resolved_device()
    for kwargs in (
        {"clusters": True},
        {"seed": -1},
        {"subsample_size": 0},
        {"sample_id": "a/b"},
        {"window_size": 0.0},
    ):
        with pytest.raises(ValueError):
            FitConfig(**kwargs)


def test_baseline_exact_membership_and_centers(repository, legacy_run):
    baseline = json.loads((repository / "tests/fixtures/legacy_cpu_baseline.json").read_text())
    for name, digest in baseline["inputs"].items():
        assert sha256(repository / "sample" / name) == digest
    np.testing.assert_array_equal(
        np.loadtxt(legacy_run / "preliminary_result/chain_order.txt", dtype=int), baseline["chain_order"]
    )
    np.testing.assert_allclose(
        np.loadtxt(legacy_run / "preliminary_result/pilot_cp.txt"),
        baseline["pilot_cp"],
        rtol=2e-10,
        atol=2e-12,
    )
    selection = pd.read_csv(
        legacy_run / "final_result/bic_selection.tsv", sep="\t", float_precision="round_trip"
    )
    for key, expected in baseline["capacities"].items():
        labels = pd.read_csv(legacy_run / f"final_result/mutation_assignments_K{key}.txt", sep="\t")
        structure = pd.read_csv(
            legacy_run / f"final_result/subclonal_structure_K{key}.txt",
            sep="\t",
            float_precision="round_trip",
        )
        np.testing.assert_array_equal(labels.cluster_index, expected["labels"])
        np.testing.assert_allclose(structure.cellular_prevalence, expected["centers"], rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(structure.mixture_weight, expected["weights"], rtol=2e-7, atol=2e-9)
        row = (
            selection.query("requested_k == @key")
            if False
            else selection[selection.requested_k == int(key)].iloc[0]
        )
        assert abs(row.bic - expected["bic"]) < 1e-5
        assert row.candidate_kind == expected["candidate_kind"]
    result = verify_run(legacy_run)
    assert result["selected_k"] == 3 and result["occupied_clusters"] == 3


def test_missing_diagnostics_rejected(tmp_path):
    from clipp.selection import _raw_diagnostics

    with pytest.raises(ValueError, match="Missing native diagnostics"):
        _raw_diagnostics(tmp_path / "missing.tsv")


@pytest.mark.parametrize("target", ["partial", "checksum", "stale", "raw", "partition", "chain"])
def test_corrupt_runs_fail(legacy_run, tmp_path, target):
    out = tmp_path / "copy"
    shutil.copytree(legacy_run, out)
    manifest = json.loads((out / "manifest.json").read_text())
    if target == "partial":
        (out / "COMPLETE.json").unlink()
    elif target == "stale":
        (out / "final_result/Best_K/stale.txt").write_text("stale")
    elif target == "checksum":
        (out / "inputs/snv.txt").write_text("different input")
    else:
        if target == "raw":
            name = "preliminary_result/K1_fit.tsv"
            table = pd.read_csv(out / name, sep="\t")
            table.loc[0, "objective"] += 100
            table.to_csv(out / name, sep="\t", index=False, float_format="%.17g")
        elif target == "partition":
            name = "final_result/mutation_assignments_K3.txt"
            table = pd.read_csv(out / name, sep="\t")
            table.loc[0, "cluster_index"] = 99
            table.to_csv(out / name, sep="\t", index=False)
        else:
            name = "preliminary_result/chain_order.txt"
            order = np.loadtxt(out / name, dtype=int)
            order[[0, 1]] = order[[1, 0]]
            np.savetxt(out / name, order, fmt="%d")
            manifest["chain_order_sha256"] = sha256(out / name)
        manifest["artifacts"][name] = sha256(out / name)
        (out / "manifest.json").write_text(json.dumps(manifest))
        completion = json.loads((out / "COMPLETE.json").read_text())
        completion["manifest_sha256"] = sha256(out / "manifest.json")
        (out / "COMPLETE.json").write_text(json.dumps(completion))
    with pytest.raises((ValueError, OSError, IndexError)):
        verify_run(out)


def test_output_reuse_and_write_failure(tiny_inputs, tmp_path, monkeypatch):
    out = tmp_path / "fit"
    out.mkdir()
    (out / "old.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        fit(*tiny_inputs, out, config=FitConfig(clusters=1, device="cpu"))
    assert (out / "old.txt").read_text() == "preserve"
    import clipp.output as output

    def fail(*args, **kwargs):
        raise OSError("deliberate publication failure")

    monkeypatch.setattr(output, "_write_table", fail)
    with pytest.raises(OSError, match="deliberate"):
        fit(*tiny_inputs, tmp_path / "failed", config=FitConfig(clusters=1, device="cpu"))
    assert not (tmp_path / "failed").exists()
    attempts = list(tmp_path.glob(".failed.inprogress.*"))
    assert len(attempts) == 1 and (attempts[0] / "FAILURE.json").is_file()
    assert not (attempts[0] / "COMPLETE.json").exists()
    assert not (tmp_path / ".failed.lock").exists()


def test_subsample_replay_and_full_n(tiny_inputs, tmp_path):
    from clipp.subsampling import run_clipp_sub
    from clipp.kernel import _prepare_chain
    from test_inputs import preprocess

    pre = tmp_path / "pre"
    assert preprocess(tiny_inputs, pre).returncode == 0
    first, second = tmp_path / "first", tmp_path / "second"
    prepared = _prepare_chain(pre, first)
    for out in (first, second):
        run_clipp_sub(pre, out, [1, 2], 3, 2, 0.3, 0.1, seed=71, prepared=prepared)
    for rep in (1, 2):
        a = np.loadtxt(first / f"sample_indices_rep{rep}.txt", dtype=int, ndmin=1)
        b = np.loadtxt(second / f"sample_indices_rep{rep}.txt", dtype=int, ndmin=1)
        np.testing.assert_array_equal(a, b)
        assert len(np.unique(a)) == 3
    out = tmp_path / "subsample"
    fit(
        *tiny_inputs,
        out,
        config=FitConfig(device="cpu", max_clusters=2, subsample_size=3, replicates=2, seed=71),
    )
    result = verify_run(out)
    assert result["num_mutations"] == 4
    manifest = json.loads((out / "manifest.json").read_text())
    assert [r["seed"] for r in manifest["subsamples"]] == [72, 73]


def test_import_safe_legacy_cli(repository, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("legacy_entry", repository / "run_clipp_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main) and not list(tmp_path.iterdir())


def test_package_identity_refuses_stale_source(tmp_path, monkeypatch):
    import clipp.native as native

    source = Path(native.__file__).parent
    destination = tmp_path / "package"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
    # The installed source may deliberately be read-only; only this disposable
    # copy is made writable for deliberate corruption tests.
    for path in [destination, *destination.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    monkeypatch.setattr(native, "__file__", str(destination / "native.py"))
    # Unrelated, newer library files cannot affect selection.
    (destination / "CliPP-newer.so").write_text("wrong library")
    native.load_native()
    (destination / "model.py").write_text("wrong source")
    with pytest.raises(RuntimeError, match="source does not match"):
        native.load_native()


def test_require_cuda_never_falls_back(native, tiny_inputs, tmp_path):
    native.CliPPCUDACompiled.restype = ctypes.c_int
    if native.CliPPCUDACompiled():
        pytest.skip("CPU-only build negative test")
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        fit(*tiny_inputs, tmp_path / "gpu", config=FitConfig(device="cuda", clusters=1))
    assert not (tmp_path / "gpu").exists()


def test_string_ids_are_never_missing_values(tiny_inputs, tmp_path):
    snv, _, _ = tiny_inputs
    lines = snv.read_text().splitlines()
    ids = ["NULL", "001", "None", "nan"]
    snv.write_text(
        "mutation_id\t"
        + lines[0]
        + "\n"
        + "\n".join(identifier + "\t" + row for identifier, row in zip(ids, lines[1:]))
        + "\n"
    )
    out = tmp_path / "string-ids"
    fit(*tiny_inputs, out, config=FitConfig(device="cpu", clusters=1))
    result = pd.read_csv(
        out / "final_result/Best_K/mutation_assignments_K1.txt", sep="\t", keep_default_na=False, dtype=str
    )
    assert result.mutation_id.tolist() == ids
    verify_run(out)


def test_missing_rscript_is_clear_failure(tiny_inputs, tmp_path, monkeypatch):
    import clipp.api as api

    monkeypatch.setattr(api.shutil, "which", lambda command: None)
    with pytest.raises(RuntimeError, match="Rscript is required"):
        fit(*tiny_inputs, tmp_path / "missing-r", config=FitConfig(device="cpu", clusters=1))
    assert not (tmp_path / "missing-r").exists()


def test_missing_parent_cuda_probe(repository, monkeypatch):
    # Execute the real build helper without invoking setup or requiring NVIDIA.
    import ast
    import importlib.util
    from pathlib import Path

    module = ast.parse((repository / "setup.py").read_text())
    node = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_nvidia_package_paths"
    )
    namespace = {"importlib": __import__("importlib"), "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<build-probe>", "exec"), namespace)

    def absent(name):
        raise ModuleNotFoundError("nvidia namespace absent")

    monkeypatch.setattr(importlib.util, "find_spec", absent)
    assert namespace["_nvidia_package_paths"]("nvidia.cuda_runtime") is None
