"""Reproducible single-sample count simulations with explicit multiplicity truth."""

import argparse
import csv
from pathlib import Path

import numpy as np


def simulate(outdir, mutations=30, seed=17, *, ambiguous=False, close_centers=False):
    if mutations < 1:
        raise ValueError("mutations must be positive")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    centers = np.array([.35, .45, 1.0] if close_centers else [.2, .55, 1.0])
    labels = np.arange(mutations) % 3
    labels[-1] = 2  # every generated tumor contains at least one clonal mutation
    depths = rng.integers(80, 401, size=mutations)
    input_path = outdir / "tumor.tsv"
    truth_path = outdir / "truth.tsv"
    with input_path.open("x", newline="") as inputs, truth_path.open("x", newline="") as truths:
        writer, truth = csv.writer(inputs, delimiter="\t"), csv.writer(truths, delimiter="\t")
        writer.writerow(["mutation_id", "sample_id", "alt_count", "ref_count", "count_observed", "purity",
                         "normal_cn", "segment_id", "cn_state_id", "cn_state_fraction", "allele_a_cn", "allele_b_cn"])
        truth.writerow(["mutation_id", "true_ccf", "true_cluster", "true_multiplicity", "major_cn", "minor_cn",
                        "mixed_cn", "multiplicity_truth_target"])
        for i, (label, depth) in enumerate(zip(labels, depths)):
            states = [(1.0, 1, 1)]
            if ambiguous and label != 2:
                states = [(1., 2, 1)] if i % 2 else [(.8, 1, 1), (.2, 3, 1)]
            major, minor = max(s[1] for s in states), max(s[2] for s in states)
            multiplicity = int(rng.integers(1, min(4, major) + 1))
            mean_cn = sum(f * (a + b) for f, a, b in states)
            slope = .8 / (.4 + .8 * mean_cn)
            phi = centers[label]
            alt = int(rng.binomial(depth, slope * multiplicity * phi))
            for j, (fraction, a, b) in enumerate(states):
                writer.writerow([f"m{i:06}", "01", alt, depth - alt, 1, .8, 2, f"seg{i}", f"s{j}", fraction, a, b])
            truth.writerow([f"m{i:06}", phi, label, multiplicity, major, minor,
                            int(len(states) > 1), "generating_integer_candidate"])
    return input_path, truth_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mutations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ambiguous", action="store_true")
    parser.add_argument("--close-centers", action="store_true")
    args = parser.parse_args()
    simulate(args.outdir, args.mutations, args.seed, ambiguous=args.ambiguous, close_centers=args.close_centers)


if __name__ == "__main__":
    main()
