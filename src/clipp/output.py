"""Serialization within an isolated, unpublished run directory."""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from ._io import partition_labels, read_table, write_table as _write_table

MUTATION_COLUMNS = [
    "chromosome_index", "position", "mutation_id", "original_row", "cluster_index",
    "major_cn", "multiplicity", "expected_multiplicity", "multiplicity_probability",
]
CLUSTER_COLUMNS = [
    "cluster_index", "num_SNV", "cellular_prevalence", "cancer_cell_fraction",
    "purity", "assignment_proportion",
]


def write_posterior_diagnostics(
    model, coordinates, result, directory, *, marginal_multiplicity=False, chain_to_component=None
):
    """Write separate diagnostics without changing the two production tables.

    ``directory`` is the diagnostics subdirectory, normally
    ``final_result/diagnostics``. No full membership matrix is written.
    The returned relative filenames can be bound by the caller's manifest.
    """
    from ._io import write_json
    from .posterior import posterior_diagnostics

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    values = posterior_diagnostics(
        model, result["centers"], result["cluster_weights"], result["labels"],
        marginal_multiplicity=marginal_multiplicity, chain_to_component=chain_to_component,
    )
    k = len(result["centers"])
    clusters = pd.DataFrame({
        "cluster_index": np.arange(k),
        "cellular_prevalence": result["centers"],
        "cancer_cell_fraction": result["centers"] / model.purity,
        **{name: values[name] for name in (
            "chain_num_SNV", "posterior_expected_num_SNV", "posterior_map_num_SNV",
            "mixture_weight", "fitted_mixture_num_SNV", "posterior_count_minus_mixture_count",
        )},
    })
    mutations = coordinates.copy()
    mutations["proposal_chain_cluster_index"] = result["labels"]
    for name in (
        "posterior_map_cluster_index", "chain_label_posterior", "max_cluster_posterior",
        "membership_entropy", "expected_cancer_cell_fraction", "marginal_multiplicity",
        "marginal_expected_multiplicity", "posterior_map_conditional_multiplicity",
        "posterior_map_conditional_expected_multiplicity", "mapped_proposal_cluster_index",
    ):
        if name in values:
            mutations[name] = values[name]
    flow = pd.DataFrame({
        "chain_cluster_index": np.repeat(np.arange(values["allocation_flow"].shape[0]), k),
        "posterior_cluster_index": np.tile(np.arange(k), values["allocation_flow"].shape[0]),
        "posterior_expected_num_SNV": values["allocation_flow"].ravel(),
    })
    names = {name: "posterior_" + name + suffix for name, suffix in (
        ("clusters", ".tsv"), ("mutations", ".tsv"), ("flow", ".tsv"), ("summary", ".json")
    )}
    for name, table in (("clusters", clusters), ("mutations", mutations), ("flow", flow)):
        _write_table(table, directory / names[name])
    write_json(directory / names["summary"], values["summary"])
    return names


def _write_result(model, coordinates, result, directory):
    if np.any(result["cluster_weights"] <= 0):
        raise ValueError("Published occupied clusters require positive fitted mixture weights")
    mutations = coordinates.copy()
    mutations["cluster_index"] = result["labels"]
    counts = np.bincount(result["labels"])
    structures = pd.DataFrame(
        {
            "cluster_index": np.arange(result["num_clusters"]),
            "num_SNV": counts,
            "cellular_prevalence": result["centers"],
        }
    )
    structures["cancer_cell_fraction"] = result["centers"] / model.purity
    structures["purity"] = model.purity
    structures["assignment_proportion"] = counts / len(model)
    posterior = model.posterior(result["centers"][result["labels"]])
    calls = np.argmax(posterior, axis=1)
    mutations["major_cn"] = model.major
    mutations["multiplicity"] = calls + 1
    mutations["expected_multiplicity"] = posterior @ model.m
    mutations["multiplicity_probability"] = posterior.max(axis=1)
    _write_table(mutations, os.path.join(directory, "mutations.tsv"))
    _write_table(structures, os.path.join(directory, "clusters.tsv"))


class Result:
    """Selected tables; historical evidence bundles also expose recorded fits."""

    def __init__(self, root, manifest=None):
        import json

        self.manifest = manifest
        if manifest is None:
            self.mutations = read_table(root / "final_result/mutations.tsv")
            self.clusters = read_table(root / "final_result/clusters.tsv")
            self.fits = None
            return
        self.mutations = read_table(root / manifest["selected_outputs"]["mutations"])
        self.clusters = read_table(root / manifest["selected_outputs"]["clusters"])
        self._parameters = {}
        if manifest["output_schema_version"] >= 4:
            self.fits = pd.DataFrame(manifest["fits"]).drop(columns="partition_parameters")
            self._parameters = {
                (record["replicate"], record["candidate_id"]): record["partition_parameters"]
                for record in manifest["fits"]
            }
            ranked = self.fits.sort_values(["bic", "num_clusters", "requested_k", "replicate"])
            self.fits["selected_for_k"] = self.fits.index.isin(
                ranked.drop_duplicates("requested_k").index
            )
            self.fits["selected"] = self.fits.index == ranked.index[0]
        else:
            self.fits = read_table(root / "final_result/bic_selection.tsv")
            referenced = set(zip(self.fits.replicate, self.fits.candidate_id))
            with pd.read_csv(
                root / "final_result/chain_candidates.tsv",
                sep="\t",
                chunksize=4096,
                usecols=["replicate", "candidate_id", "partition_parameters"],
            ) as chunks:
                for chunk in chunks:
                    self._parameters.update(
                        {
                            (int(row.replicate), int(row.candidate_id)): json.loads(row.partition_parameters)
                            for row in chunk.itertuples()
                            if (row.replicate, row.candidate_id) in referenced
                        }
                    )
                    if len(self._parameters) == len(referenced):
                        break
        self._order = np.loadtxt(root / "preliminary_result/chain_order.txt", dtype=int, ndmin=1)

    def partition(self, capacity, replicate=None):
        if self.manifest is None:
            raise ValueError("Two-table results do not retain the candidate bank")
        rows = self.fits[self.fits.requested_k == capacity]
        rows = rows[rows.selected_for_k] if replicate is None else rows[rows.replicate == replicate]
        if len(rows) != 1:
            raise ValueError("No unique fit for the requested capacity/replicate")
        record = rows.iloc[0]
        params = self._parameters[int(record.replicate), int(record.candidate_id)]
        return {
            "labels": partition_labels(self._order, params["cuts"], params["block_labels"]),
            "centers": np.asarray(params["centers"]),
            "weights": np.asarray(params["weights"]),
            "record": record.to_dict(),
        }


def load_result(directory):
    """Check and read selected tables, or replay a historical evidence bundle."""
    import json
    from pathlib import Path
    from .verify import verify_run

    root = Path(directory)
    verify_run(root)
    manifest = root / "manifest.json"
    return Result(root, json.loads(manifest.read_text()) if manifest.is_file() else None)
