"""Serialization within an isolated, unpublished run directory."""

import os
import numpy as np
import pandas as pd

from ._io import partition_labels, read_table, write_table as _write_table


def _write_result(model, coordinates, result, directory):
    if np.any(result["cluster_weights"] <= 0):
        raise ValueError("Published occupied clusters require positive fitted mixture weights")
    mutations = coordinates.copy()
    mutations["cluster_index"] = result["labels"]
    structures = pd.DataFrame(
        {
            "cluster_index": np.arange(result["num_clusters"]),
            "num_SNV": np.bincount(result["labels"]),
            "cellular_prevalence": result["centers"],
        }
    )
    structures["cancer_cell_fraction"] = result["centers"] / model.purity
    structures["purity"] = model.purity
    structures["assignment_proportion"] = np.bincount(result["labels"]) / len(model)
    posterior = model.posterior(result["centers"][result["labels"]])
    calls = np.argmax(posterior, axis=1)
    mutations["major_cn"] = model.major
    mutations["multiplicity"] = calls + 1
    mutations["expected_multiplicity"] = posterior @ model.m
    _write_table(mutations, os.path.join(directory, "mutations.tsv"))
    _write_table(structures, os.path.join(directory, "clusters.tsv"))


class Result:
    """Verified result tables and lazy O(N) reconstruction of any recorded fit."""

    def __init__(self, root, manifest):
        import json

        self.manifest = manifest
        self.mutations = read_table(root / manifest["selected_outputs"]["mutations"])
        self.clusters = read_table(root / manifest["selected_outputs"]["clusters"])
        self._parameters = {}
        if manifest["output_schema_version"] == 4:
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
    """Verify once, then read the selected result and reconstructable per-fit bank."""
    import json
    from pathlib import Path
    from .verify import verify_run

    root = Path(directory)
    verify_run(root)
    return Result(root, json.loads((root / "manifest.json").read_text()))
