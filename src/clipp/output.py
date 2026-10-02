"""Serialization within an isolated, unpublished run directory."""

import os
import numpy as np
import pandas as pd


def _write_table(data, path):
    temporary = path + ".tmp"
    data.to_csv(temporary, sep="\t", index=False, float_format="%.17g")
    os.replace(temporary, path)


def _write_result(model, coordinates, result, directory, requested_k):
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
    files = []
    for stem, frame in (
        ("mutation_assignments", assignments),
        ("subclonal_structure", structures),
        ("posterior_multiplicity", multiplicity),
    ):
        filename = "%s_K%d.txt" % (stem, requested_k)
        _write_table(frame, os.path.join(directory, filename))
        files.append(filename)
    return files
