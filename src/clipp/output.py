"""Serialization within an isolated, unpublished run directory."""

import os
import numpy as np
import pandas as pd


def _write_table(data, path):
    temporary = path + ".tmp"
    data.to_csv(temporary, sep="\t", index=False, float_format="%.17g")
    os.replace(temporary, path)


def _write_result(model, coordinates, result, directory):
    if np.any(result["cluster_weights"] <= 0):
        raise ValueError("Published occupied clusters require positive fitted mixture weights")
    assignments = coordinates.copy()
    assignments["cluster_index"] = result["labels"]
    structures = pd.DataFrame(
        {
            "cluster_index": np.arange(result["num_clusters"]),
            "num_SNV": np.bincount(result["labels"]),
            "cellular_prevalence": result["centers"],
            "mixture_weight": result["cluster_weights"],
        }
    )
    structures["cancer_cell_fraction"] = result["centers"] / model.purity
    structures["purity"] = model.purity
    structures["assignment_proportion"] = np.bincount(result["labels"]) / len(model)
    posterior = model.posterior(result["centers"][result["labels"]])
    calls = np.argmax(posterior, axis=1)
    multiplicity = assignments.copy()
    multiplicity["major_cn"] = model.major
    multiplicity["multiplicity"] = calls + 1
    multiplicity["multiplicity_probability"] = posterior[np.arange(len(model)), calls]
    multiplicity["expected_multiplicity"] = posterior @ model.m
    _write_table(multiplicity, os.path.join(directory, "mutations.tsv"))
    _write_table(structures, os.path.join(directory, "clusters.tsv"))


class Result:
    """Verified result tables and lazy O(N) reconstruction of any recorded fit."""

    def __init__(self, root, manifest):
        import json
        from .verify import _table

        self.manifest = manifest
        self.mutations = _table(root / manifest["selected_outputs"]["mutations"])
        self.clusters = _table(root / manifest["selected_outputs"]["clusters"])
        self.fits = _table(root / "final_result/bic_selection.tsv")
        candidates = _table(root / "final_result/chain_candidates.tsv")
        self._parameters = {
            (int(row.replicate), int(row.candidate_id)): json.loads(row.partition_parameters)
            for row in candidates.itertuples()
            if row.status == "scored"
        }
        self._order = np.loadtxt(root / "preliminary_result/chain_order.txt", dtype=int, ndmin=1)

    def partition(self, capacity, replicate=None):
        rows = self.fits[self.fits.requested_k == capacity]
        rows = rows[rows.selected_for_k] if replicate is None else rows[rows.replicate == replicate]
        if len(rows) != 1:
            raise ValueError("No unique fit for the requested capacity/replicate")
        record = rows.iloc[0]
        params = self._parameters[int(record.replicate), int(record.candidate_id)]
        labels = np.empty(len(self._order), dtype=int)
        labels[self._order] = np.asarray(params["block_labels"])[
            np.searchsorted(params["cuts"], np.arange(len(labels)), side="right")
        ]
        return {
            "labels": labels,
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
