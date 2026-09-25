"""CN-first multiplicity, read-generation consistency and reproducible outputs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from clipp1d.simulation import SimulationConfig, generate_cohort
from clipp1d.simulation import generate_clippsim4k as generator


REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "clippsim4k_original_reference.json").read_text()
)


def run_module(*args):
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    return subprocess.run(
        [sys.executable, "-m", "clipp1d.simulation", *map(str, args)],
        env=env, capture_output=True, text=True, check=False, timeout=60,
    )


def test_copy_number_distribution_preserves_original_allele_draws():
    major, minor = generator.sample_copy_numbers(np.random.RandomState(123), 20_000, 1.0)
    # The original 20 equally likely allele pairs give probability 1/20 for
    # minor=0 or minor=major, and 2/20 for each strictly intermediate minor CN.
    assert set(zip(major, minor)) == {(a, b) for a in range(1, 5) for b in range(a + 1)}
    for a in range(1, 5):
        for b in range(a + 1):
            expected = 0.05 if b in (0, a) else 0.1
            observed = np.mean((major == a) & (minor == b))
            assert abs(observed - expected) < 0.01


def test_diploid_loci_have_multiplicity_one():
    rng = np.random.RandomState(123)
    major, minor = generator.sample_copy_numbers(rng, 1000, 0.0)
    assert np.all(major == 1) and np.all(minor == 1)
    assert np.all(generator.sample_multiplicity(rng, major) == 1)


def test_multiplicity_is_uniform_over_every_integer_through_major_cn():
    major = np.repeat(np.arange(1, 5), 20_000)
    original_major = major.copy()
    multiplicity = generator.sample_multiplicity(np.random.RandomState(123), major)
    np.testing.assert_array_equal(major, original_major)
    for a in range(1, 5):
        values, counts = np.unique(multiplicity[major == a], return_counts=True)
        np.testing.assert_array_equal(values, np.arange(1, a + 1))
        assert np.all(np.abs(counts / counts.sum() - 1 / a) < 0.015)


def test_simulated_reads_use_sampled_multiplicity_without_changing_cn(tmp_path, monkeypatch):
    class RecordingRNG(np.random.RandomState):
        def binomial(self, n, p, size=None):
            self.read_depth = np.asarray(n).copy()
            self.read_probability = np.asarray(p).copy()
            return super().binomial(n, p, size)

    def fixed_cn(rng, mutation_count, cna_rate):
        # Include balanced amplification and LOH, where the old rule forced
        # multiplicity to equal major CN. Arrays must survive unchanged in CNA.
        return (
            np.resize([1, 2, 3, 4, 4], mutation_count),
            np.resize([1, 2, 3, 4, 0], mutation_count),
        )

    monkeypatch.setattr(generator, "sample_copy_numbers", fixed_cn)
    rng = RecordingRNG(20260730)
    case_id, _ = generator.simulate_case(rng, tmp_path, 200, 0.6, 1.0, 0, (2, 3, 4))
    cna = pd.read_csv(tmp_path / case_id / "cna.txt", sep="\t")
    truth = pd.read_csv(tmp_path / case_id / "truth.txt", sep="\t")
    snv = pd.read_csv(tmp_path / case_id / "snv.txt", sep="\t")
    expected_major, expected_minor = fixed_cn(None, len(truth), None)
    np.testing.assert_array_equal(cna.major_cn, expected_major)
    np.testing.assert_array_equal(cna.minor_cn, expected_minor)
    np.testing.assert_array_equal(cna.total_cn, expected_major + expected_minor)
    for major_cn, minor_cn in ((2, 2), (3, 3), (4, 4), (4, 0)):
        mask = (cna.major_cn == major_cn) & (cna.minor_cn == minor_cn)
        assert set(truth.loc[mask, "multiplicity"]) == set(range(1, major_cn + 1))
    expected_vaf = 0.6 * truth.ccf * truth.multiplicity / (0.8 + 0.6 * cna.total_cn)
    np.testing.assert_allclose(rng.read_probability, expected_vaf, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(snv.alt_count + snv.ref_count, rng.read_depth)
    assert np.all((rng.read_probability >= 0) & (rng.read_probability <= 1))


def output_hashes(output):
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file() and path.name != "generation_manifest.json"
    }


@pytest.mark.parametrize("cna_rates", [[0.0, 0.1, 0.2], [0.1, 0.4, 0.7]])
def test_seeded_cohort_is_reproducible_through_api_and_module(tmp_path, cna_rates):
    api_output = tmp_path / "api"
    module_output = tmp_path / "module"
    config = {**REFERENCE["config"], "cna_rates": cna_rates}
    generate_cohort(api_output, SimulationConfig(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in config.items()
    }))
    args = ["--output-dir", str(module_output)]
    for key, value in config.items():
        args.extend([
            "--" + key.replace("_", "-"),
            ",".join(map(str, value)) if isinstance(value, list) else str(value),
        ])
    result = run_module(*args)
    assert result.returncode == 0, result.stderr
    assert len(output_hashes(api_output)) == 4 * config["sample_count"] + 1
    assert output_hashes(api_output) == output_hashes(module_output)

    manifests = []
    for output in (api_output, module_output):
        manifest = json.loads((output / "generation_manifest.json").read_text())
        assert manifest.pop("generated_at_utc")
        manifests.append(manifest)
    assert manifests[0] == manifests[1]
    manifest = manifests[0]
    assert manifest["model_version"] == "cn-first-uniform-multiplicity-v1"
    assert manifest["lineage"]["parent_generator_sha256"] == REFERENCE["source_sha256"]
    assert "DiscreteUniform[1, major_cn]" in manifest["distributions"]["mutation_multiplicity"]
    assert manifest["config"] == config

    summary = pd.read_csv(api_output / "generation_summary.tsv", sep="\t")
    assert len(summary) == config["sample_count"]
    assert len(summary.groupby(["read_depth", "purity", "cna_rate"])) == 27
    saw_intermediate_multiplicity = False
    for case in summary.itertuples():
        case_dir = api_output / case.sample
        cna = pd.read_csv(case_dir / "cna.txt", sep="\t")
        truth = pd.read_csv(case_dir / "truth.txt", sep="\t")
        snv = pd.read_csv(case_dir / "snv.txt", sep="\t")
        assert len(cna) == len(truth) == len(snv) == case.total_snvs
        np.testing.assert_array_equal(cna.total_cn, cna.major_cn + cna.minor_cn)
        np.testing.assert_array_equal(truth.position, snv.position)
        assert np.all((truth.multiplicity >= 1) & (truth.multiplicity <= cna.major_cn))
        if case.cna_rate == 0:
            assert np.all(cna.major_cn == 1) and np.all(cna.minor_cn == 1)
            assert np.all(truth.multiplicity == 1)
        saw_intermediate_multiplicity |= np.any(
            (truth.multiplicity != cna.major_cn) & (truth.multiplicity != cna.minor_cn)
        )
    assert saw_intermediate_multiplicity


def test_module_dry_run_reports_current_defaults_without_writes(tmp_path):
    output = tmp_path / "not-created"
    result = run_module("--output-dir", output, "--dry-run")
    assert result.returncode == 0, result.stderr
    text, case_count = result.stdout.rsplit("\n", 2)[:2]
    assert json.loads(text) == {
        **REFERENCE["config"], "output_dir": str(output), "sample_count": 4000,
        "cna_rates": [0.1, 0.4, 0.7],
    }
    assert case_count == "case_count: 4000"
    assert not output.exists()


def test_module_does_not_overwrite_existing_cohort(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "truth.txt"
    sentinel.write_bytes(b"existing data\n")
    result = run_module("--output-dir", output, "--sample-count", 1)
    assert result.returncode != 0
    assert "output directory is not empty" in result.stderr
    assert sorted(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing data\n"
