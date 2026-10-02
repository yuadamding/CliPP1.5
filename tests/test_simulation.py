import json
import numpy as np
import pandas as pd
import pytest

from clipp.simulation import generate, SimulationConfig


def test_generator_replay_truth_and_full_support(tmp_path):
    config = SimulationConfig(seed=21, mutations=1000, cn_states=((4, 2),))
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, config)
    generate(b, config)
    for filename in ("snv.tsv", "cna.tsv", "purity.txt", "truth.tsv", "simulation.json"):
        assert (a / filename).read_bytes() == (b / filename).read_bytes()
    truth = pd.read_csv(a / "truth.tsv", sep="\t")
    assert set(truth.multiplicity) == {1, 2, 3, 4}
    assert truth.multiplicity.between(1, truth.major_cn).all()
    np.testing.assert_allclose(truth.cp, truth.ccf * config.purity)
    assert (
        json.loads((a / "simulation.json").read_text())["generator_version"]
        == "autosomal_cnfirst_binomial_pcg64_v1"
    )
    with pytest.raises(FileExistsError):
        generate(a, config)


@pytest.mark.parametrize(
    "config",
    [
        SimulationConfig(mutations=0),
        SimulationConfig(purity=0),
        SimulationConfig(centers=(1.0,), proportions=(0.5,)),
        SimulationConfig(cn_states=((1, 2),)),
    ],
)
def test_generator_validates_before_creation(tmp_path, config):
    with pytest.raises(ValueError):
        generate(tmp_path / "bad", config)
    assert not (tmp_path / "bad").exists()
