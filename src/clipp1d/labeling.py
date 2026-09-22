"""Post-fit clonal designation; never changes the fitted CCFs or objective."""

import numpy as np

CLONAL_LABEL_RULE = "nearest_to_one_l2_v1"


def public_cluster_order(centers, cuts):
    """Closest-to-one block first, then descending CCF; ties use chain position."""
    values = np.asarray(centers, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Cluster centers must be a nonempty finite vector")
    if len(cuts) != len(values) + 1:
        raise ValueError("One chain interval is required per cluster")
    clonal = min(range(len(values)), key=lambda i: (abs(values[i] - 1), cuts[i]))
    return [clonal] + sorted((i for i in range(len(values)) if i != clonal),
                            key=lambda i: (-values[i], cuts[i]))


def clonality_summary(centers, labels):
    """Summarize occupied public labels after closest-to-one relabeling."""
    centers = np.asarray(centers, dtype=float)
    labels = np.asarray(labels)
    if (centers.ndim != 1 or not centers.size or not np.all(np.isfinite(centers)) or
            labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu" or
            not np.array_equal(np.unique(labels), np.arange(len(centers)))):
        raise ValueError("Expected finite centers and occupied integer public cluster labels")
    distances = np.abs(centers - 1)
    if distances[0] != distances.min():
        raise ValueError("Public cluster zero must be closest to CCF one")
    ties = np.flatnonzero(distances == distances[0]).tolist()
    clonal = int(np.count_nonzero(labels == 0))
    total = int(labels.size)
    return {"rule": CLONAL_LABEL_RULE, "clonal_cluster_label": 0,
            "clonal_ccf": float(centers[0]), "distance_to_one": float(distances[0]),
            "tie_count": len(ties), "tied_cluster_labels": ties,
            "tie_breaker": "earliest chain block (lowest public label after relabeling)",
            "mutation_filter": "all retained informative mutations after input filtering",
            "total_mutations": total, "clonal_mutations": clonal,
            "subclonal_mutations": total - clonal,
            "subclonal_mutation_fraction": (total - clonal) / total}
