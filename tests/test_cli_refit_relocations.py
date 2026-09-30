"""CLI wiring for the separately identified optional repository-wide search."""
from types import SimpleNamespace

import pytest

from clipp1d import cli


def test_refit_relocations_require_partition_search(monkeypatch):
    monkeypatch.setattr(cli, "fit", lambda *a, **kw: pytest.fail("Invalid CLI reached fit"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["fit", "--input-file", "input.tsv", "--outdir", "out",
                  "--partition-refit-relocations"])
    assert raised.value.code == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_refit_relocations_are_explicitly_passed_and_default_off(monkeypatch, enabled):
    seen = {}

    def fit(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(search_status="complete")

    monkeypatch.setattr(cli, "fit", fit)
    arguments = ["fit", "--input-file", "input.tsv", "--outdir", "out", "--partition-search"]
    if enabled:
        arguments.append("--partition-refit-relocations")
    assert cli.main(arguments) == 0
    assert seen["partition_policy"].refit_relocations is enabled


@pytest.mark.parametrize("partition,relocations,version", [(False, False, "v3"),
    (True, False, "v5"), (True, True, "v6")])
def test_new_ancestry_uses_a_distinct_schema_without_changing_default(partition, relocations, version):
    from clipp1d.cuda_api import _output_schema
    assert _output_schema(partition, relocations) == "clipp1d.cuda.run." + version
