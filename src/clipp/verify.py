"""Independent arithmetic verifier. Does not invoke the fitter or optimizers.

Hashes establish artifact identity, not authenticity against a malicious writer.
Statistical checks use scipy.stats.binom and independently enumerated support.
Certificates remain scoped; verification does not establish global optimality.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import binom

from .native import sha256, build_identity
from .versions import IDENTITIES, NUMERICS


def _require(condition, message):
    if not condition:
        raise ValueError("Verification failed: " + message)


def _close(value, expected, name, *, atol=2e-7, rtol=2e-9):
    _require(np.allclose(value, expected, atol=atol, rtol=rtol, equal_nan=False), name)


def _table(path):
    return pd.read_csv(
        path,
        sep="\t",
        float_precision="round_trip",
        converters={
            name: str
            for name in (
                "mutation_id",
                "chromosome_index",
                "position",
                "original_chromosome",
                "original_position",
            )
        },
    )


def _kernel(data, cp, purity):
    values, posteriors = [], []
    cp = np.broadcast_to(np.asarray(cp), (len(data),))
    for i, row in enumerate(data.itertuples()):
        m = np.arange(1, int(row.major_cn) + 1)
        p = cp[i] * m / (2 * (1 - purity) + purity * row.total_cn)
        logs = binom.logpmf(row.alt_count, row.depth, np.minimum(1, p)) - np.log(row.major_cn)
        ll = logsumexp(logs)
        values.append(ll)
        posteriors.append(np.exp(logs - ll))
    return np.array(values), posteriors


def _cuts(labels):
    return np.flatnonzero(np.diff(labels)) + 1


def _partition_hash(order, cuts):
    return hashlib.sha256(
        np.asarray(order, dtype="<i8").tobytes() + np.asarray(cuts, dtype="<i8").tobytes()
    ).hexdigest()


def _verify_input_mapping(root, data, ledger, purity):
    """Independently reconstruct the R interval join and row-exclusion policy."""

    def read(name):
        return pd.read_csv(
            root / "inputs" / (name + ".txt"), sep=r"\s+", dtype=str, keep_default_na=False, quoting=3
        )

    def chrom(values):
        text = values.str.replace(r"(?i)^chr", "", regex=True)
        _require(text.str.fullmatch(r"[0-9]+").all(), "original chromosome encoding")
        number = pd.to_numeric(text)
        _require(number.between(1, 22).all(), "original autosomal scope")
        return number.to_numpy(dtype=int)

    def integer(values, name, minimum=0):
        numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        _require(
            np.isfinite(numbers).all()
            and np.all(numbers == np.floor(numbers))
            and np.all((numbers >= minimum) & (numbers <= np.iinfo(np.int32).max)),
            name,
        )
        return numbers.astype(np.int64)

    snv, cn = read("snv"), read("cna")
    tokens = (root / "inputs/purity.txt").read_text().split()
    _require(len(tokens) == 1 and float(tokens[0]) == purity, "original purity")
    _require(len(snv) == len(ledger), "original SNV count")
    chromosomes = chrom(snv.chromosome_index)
    positions = integer(snv.position, "original positions", 1)
    cn_chromosomes = chrom(cn.chromosome_index)
    starts = integer(cn.start_position, "original CNA start", 1)
    ends = integer(cn.end_position, "original CNA end", 1)
    major = integer(cn.major_cn, "original major CN", 1)
    minor = integer(cn.minor_cn, "original minor CN")
    total = integer(cn.total_cn, "original total CN", 1)
    _require(
        np.all(starts <= ends) and np.all(major >= minor) and np.all(total == major + minor),
        "original CN consistency",
    )
    ids = (
        snv.mutation_id.to_numpy()
        if "mutation_id" in snv
        else np.array([f"{c}:{p}" for c, p in zip(chromosomes, positions)])
    )
    _require(np.array_equal(ledger.mutation_id, ids), "original mutation IDs")
    _require(
        np.array_equal(ledger.original_chromosome, snv.chromosome_index)
        and np.array_equal(ledger.original_position, snv.position),
        "original encodings in ledger",
    )
    _require(
        np.array_equal(pd.to_numeric(ledger.chromosome_index), chromosomes)
        and np.array_equal(pd.to_numeric(ledger.position), positions),
        "normalized ledger coordinates",
    )
    matched = np.full(len(snv), -1, dtype=int)
    for chromosome in np.unique(cn_chromosomes):
        segments = np.flatnonzero(cn_chromosomes == chromosome)
        segments = segments[np.argsort(starts[segments], kind="stable")]
        _require(
            np.all(starts[segments[1:]] > np.maximum.accumulate(ends[segments])[:-1]),
            "original overlapping CNA",
        )
        rows = np.flatnonzero(chromosomes == chromosome)
        index = np.searchsorted(starts[segments], positions[rows], side="right") - 1
        covered = (index >= 0) & (positions[rows] <= ends[segments[np.maximum(index, 0)]])
        matched[rows[covered]] = segments[index[covered]]
    _require(
        np.array_equal(ledger.matched_segment_id.fillna(0).to_numpy(), matched + 1),
        "matched original segment IDs",
    )
    alt = pd.to_numeric(snv.alt_count, errors="coerce").to_numpy(dtype=float)
    ref = pd.to_numeric(snv.ref_count, errors="coerce").to_numpy(dtype=float)
    counts_ok = (
        np.isfinite(alt)
        & np.isfinite(ref)
        & (alt >= 0)
        & (ref >= 0)
        & (alt == np.floor(alt))
        & (ref == np.floor(ref))
        & (alt + ref > 0)
        & (alt + ref <= np.iinfo(np.int32).max)
    )
    reasons = np.where(
        ~counts_ok, "invalid_counts_or_zero_depth", np.where(matched < 0, "no_cna_segment", "retained")
    )
    _require(np.array_equal(ledger.reason, reasons), "input exclusion reasons")
    keep = reasons == "retained"
    _require(np.array_equal(ledger.status, np.where(keep, "retained", "excluded")), "input row status")
    _require(
        np.array_equal(data.alt_count, alt[keep])
        and np.array_equal(data.ref_count, ref[keep])
        and np.array_equal(data.depth, (alt + ref)[keep]),
        "original retained read counts",
    )
    for name, values in (("major_cn", major), ("minor_cn", minor), ("total_cn", total)):
        _require(np.array_equal(data[name], values[matched[keep]]), "original retained " + name)


def _raw_check(root, data, purity, order, k, rep, subsampling, actual_backend):
    raw = root / "preliminary_result"
    suffix = f"_rep{rep}" if subsampling else ""
    indices = np.loadtxt(raw / f"sample_indices_rep{rep}.txt", dtype=int, ndmin=1) if subsampling else order
    _require(
        len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(data))),
        "sample indices",
    )
    ranks = np.argsort(order)[indices]
    _require(np.all(np.diff(ranks) > 0), "sample chain order")
    frame = _table(raw / f"K{k}_fit{suffix}.tsv")
    _require(len(frame) == 1, "raw diagnostic row count")
    row = frame.iloc[0]
    cp = np.loadtxt(raw / f"K{k}_phi{suffix}.txt", ndmin=1)
    labels = np.loadtxt(raw / f"K{k}_label{suffix}.txt", dtype=int, ndmin=1)
    _require(cp.shape == labels.shape == (len(indices),), "raw shape")
    x = cp / purity
    _require(
        np.isfinite(x).all()
        and np.all((x >= row.ccf_lower_bound - 1e-15) & (x <= row.ccf_upper_bound + 1e-15)),
        "raw CCF bounds",
    )
    diff = np.diff(x)
    projection = np.zeros_like(diff)
    edges = np.argsort(-np.abs(diff), kind="stable")[: k - 1]
    projection[edges] = diff[edges]
    expected_labels = np.r_[0, np.cumsum(projection != 0)]
    _require(np.array_equal(labels, expected_labels), "raw projection/labels")
    subset = data.iloc[indices]
    ll, posterior = _kernel(subset, cp, purity)
    nll = -float(ll.sum())
    residual = diff - projection
    violation = float(np.linalg.norm(residual))
    _close(row.negative_log_likelihood, nll, "raw likelihood")
    _close(row.objective, nll + 0.5 * row.rho_internal * np.dot(residual, residual), "raw objective")
    _close(row.constraint_residual_CCF, violation, "raw CCF feasibility", atol=1e-13)
    _close(row.constraint_residual_CP, violation * purity, "raw CP feasibility", atol=1e-13)
    gradients = []
    for xi, probs, obs in zip(x, posterior, subset.itertuples()):
        scale = purity * np.arange(1, int(obs.major_cn) + 1) / (2 * (1 - purity) + purity * obs.total_cn)
        gradients.append(
            float(probs @ (-obs.alt_count / xi + (obs.depth - obs.alt_count) * scale / (1 - xi * scale)))
        )
    gradient = np.array(gradients)
    penalty = row.rho_internal * residual
    for i, contribution in enumerate(penalty):
        gradient[i] -= contribution
        gradient[i + 1] += contribution
    stationarity = np.max(np.abs(x - np.clip(x - gradient, row.ccf_lower_bound, row.ccf_upper_bound)))
    _close(row.stationarity_CCF, stationarity, "serialized-point stationarity", atol=2e-7)
    bounds = np.r_[0, _cuts(labels), len(labels)]
    ranges = [np.ptp(x[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    _close(row.max_block_range_CCF, max(ranges), "raw block range", atol=1e-14)
    _require(int(row.longest_block) == int(np.diff(bounds).max()), "raw longest block")
    _require(int(row.requested_K) == k and int(row.actual_blocks) == len(bounds) - 1, "raw capacity")
    expected_backend = "cuda_likelihood_host_chain" if actual_backend == "cuda" else "cpu"
    _require(row.backend == expected_backend, "actual backend")
    _require(not bool(row.constrained_optimum_certified), "raw must not claim global optimality")
    statuses = {
        "penalty_stationary_constraint_tolerance",
        "penalty_numerical_limit_candidate",
        "line_search_stalled_candidate",
        "finite_budget_candidate",
    }
    _require(row.status in statuses, "raw status")
    if row.status == "penalty_stationary_constraint_tolerance":
        _require(
            row.stationarity_CCF <= 1e-6 and violation <= 1e-6 and max(ranges) <= 1e-6, "raw tolerance claim"
        )
    full_cuts = _cuts(labels)
    if subsampling:
        full_cuts = (ranks[full_cuts - 1] + ranks[full_cuts]) // 2 + 1
    return _partition_hash(order, full_cuts)


def verify_run(directory, *, require_complete=True):
    """Verify an entire run, including input identities and all published K fits."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, value in IDENTITIES.items():
        _require(manifest.get(name) == value, "contract identity " + name)
    _require(manifest.get("numerics") == NUMERICS, "resolved numerical configuration")
    _require(
        build_identity(manifest["native"]) == manifest["native"]["native_build_id"], "native build record"
    )
    if require_complete:
        completion = json.loads((root / "COMPLETE.json").read_text())
        _require(
            completion.get("status") == "complete"
            and completion.get("manifest_sha256") == sha256(manifest_path),
            "completion/manifest identity",
        )
    artifacts = manifest["artifacts"]
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    _require(not any(p.is_symlink() for p in root.rglob("*")), "symlink in run")
    _require(files - {"manifest.json", "COMPLETE.json"} == set(artifacts), "missing or unexpected artifacts")
    for name, digest in artifacts.items():
        path = Path(name)
        _require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
        _require(sha256(root / path) == digest, "checksum " + name)
    for record in manifest["inputs"].values():
        _require(artifacts.get(record["path"]) == record["sha256"], "original input identity")
    pre, raw, final = (root / name for name in ("preprocess_result", "preliminary_result", "final_result"))
    for path, name in (
        (pre / "retained.tsv", "canonical_input_sha256"),
        (pre / "input_ledger.tsv", "mutation_identity_sha256"),
        (raw / "chain_order.txt", "chain_order_sha256"),
    ):
        _require(sha256(path) == manifest[name], name)
    data, ledger = _table(pre / "retained.tsv"), _table(pre / "input_ledger.tsv")
    n = len(data)
    _require(
        n == manifest["num_retained"] and n > 0 and len(ledger) == manifest["num_input_rows"], "input counts"
    )
    _require(int((ledger.status == "excluded").sum()) == manifest["num_excluded"], "excluded count")
    _require(np.array_equal(ledger.original_row, np.arange(1, len(ledger) + 1)), "original row ledger")
    retained = ledger[ledger.status == "retained"]
    _require(
        set(ledger.status) <= {"retained", "excluded"}
        and np.array_equal(retained.mutation_id, data.mutation_id)
        and np.array_equal(retained.original_row, data.original_row),
        "retention reconciliation",
    )
    _require(not data.mutation_id.duplicated().any(), "duplicate mutation identity")
    purity = float(np.loadtxt(pre / "purity_ploidy.txt"))
    _require(0 < purity <= 1, "purity")
    _verify_input_mapping(root, data, ledger, purity)
    for filename, column in (
        ("r.txt", "alt_count"),
        ("n.txt", "depth"),
        ("major.txt", "major_cn"),
        ("total.txt", "total_cn"),
    ):
        _require(np.array_equal(np.loadtxt(pre / filename, ndmin=1), data[column]), "canonical/native arrays")
    order = np.loadtxt(raw / "chain_order.txt", dtype=int, ndmin=1)
    pilot = np.loadtxt(raw / "pilot_cp.txt", ndmin=1)
    _require(np.array_equal(np.sort(order), np.arange(n)), "chain permutation")
    _require(
        pilot.shape == (n,) and np.isfinite(pilot).all() and np.all((pilot >= 0) & (pilot <= purity)),
        "pilot bounds",
    )
    _require(
        np.array_equal(
            order, np.lexsort((data.position.to_numpy(), data.chromosome_index.to_numpy(), pilot))
        ),
        "frozen chain tie policy",
    )
    config = manifest["config"]
    subsampling = config["subsample_size"] is not None
    _require(
        json.loads((raw / "chain_initialization.json").read_text()) == manifest["initializer"],
        "initializer evidence",
    )
    _require(not manifest["initializer"]["constrained_optimum_certified"], "initializer scope")
    expected_replicates = list(range(1, config["replicates"] + 1)) if subsampling else []
    _require(
        sorted(record["replicate"] for record in manifest["subsamples"]) == expected_replicates,
        "subsample receipt count",
    )
    parent_hashes = {}
    for rep in range(1, config["replicates"] + 1):
        for k in manifest["capacities"]:
            parent_hashes[k, rep] = _raw_check(
                root, data, purity, order, k, rep, subsampling, manifest["backend_actual"]
            )
    for record in manifest["subsamples"]:
        indices = np.loadtxt(root / record["indices_file"], dtype=int, ndmin=1)
        _require(
            data.iloc[indices].mutation_id.tolist() == record["mutation_ids"]
            and record["seed"] == config["seed"] + record["replicate"],
            "subsample identity",
        )
        size = min(n, config["subsample_size"])
        endings = np.minimum(
            np.arange(
                config["window_size"],
                1 + config["window_size"] - config["overlap"],
                config["window_size"] - config["overlap"],
            ),
            1.0,
        )
        bins = np.searchsorted(endings, data.alt_count.to_numpy() / data.depth.to_numpy(), side="left")
        groups = [np.flatnonzero(bins == value) for value in np.unique(bins)]
        quota = np.array([len(group) * size / n for group in groups])
        take = np.floor(quota).astype(int)
        take[np.argsort(-(quota - take), kind="stable")[: size - int(take.sum())]] += 1
        rng = np.random.RandomState(record["seed"])
        replay = np.concatenate(
            [rng.choice(group, amount, replace=False) for group, amount in zip(groups, take) if amount]
        )
        replay = replay[np.argsort(np.argsort(order)[replay])]
        _require(np.array_equal(indices, replay), "subsample seed/quota replay")
    selection = _table(final / "bic_selection.tsv")
    candidates = _table(final / "chain_candidates.tsv")
    selected_rows = selection[selection.selected]
    _require(len(selected_rows) == 1, "exactly one selected result")
    selected = selected_rows.iloc[0]
    valid = selection[selection.selected_for_k]
    _require(set(valid.requested_k) == set(manifest["capacities"]), "one result per requested capacity")
    best = valid.sort_values(["bic", "num_clusters", "requested_k", "replicate"], kind="stable").iloc[0]
    _require(
        int(best.requested_k) == int(selected.requested_k) == manifest["selected_k"]
        and int(best.replicate) == int(selected.replicate),
        "BIC-form selection",
    )
    for record in valid.itertuples():
        k = int(record.requested_k)
        assignments = _table(final / f"mutation_assignments_K{k}.txt")
        structure = _table(final / f"subclonal_structure_K{k}.txt")
        calls = _table(final / f"posterior_multiplicity_K{k}.txt")
        _require(
            np.array_equal(assignments.mutation_id, data.mutation_id)
            and np.array_equal(assignments.original_row, data.original_row),
            "assignment identities",
        )
        for column in ("chromosome_index", "position"):
            _require(np.array_equal(assignments[column], data[column]), "assignment coordinates")
        labels = assignments.cluster_index.to_numpy()
        q = len(structure)
        _require(1 <= q <= k and np.array_equal(np.unique(labels), np.arange(q)), "occupied q <= K")
        _require(len(_cuts(labels[order])) + 1 == q, "contiguous chain blocks")
        _require(
            np.array_equal(structure.cluster_index, np.arange(q))
            and np.array_equal(structure.num_SNV, np.bincount(labels)),
            "membership counts",
        )
        centers, weights = structure.cellular_prevalence.to_numpy(), structure.mixture_weight.to_numpy()
        _require(
            np.isfinite(centers).all()
            and np.all((centers >= 0) & (centers <= purity))
            and np.all(np.diff(centers) <= 0),
            "CP bounds/label order",
        )
        _require(np.isfinite(weights).all() and np.all(weights > 0), "positive mixture weights")
        _close(weights.sum(), 1, "weight normalization", atol=1e-12)
        _close(structure.cancer_cell_fraction, centers / purity, "CCF conversion")
        _close(structure.purity, purity, "output purity")
        _close(structure.assignment_proportion, np.bincount(labels) / n, "assignment proportions")
        kernel = np.column_stack([_kernel(data, cp, purity)[0] for cp in centers])
        mixture_ll = logsumexp(kernel + np.log(weights), axis=1)
        ll = float(mixture_ll.sum())
        conditional = float(kernel[np.arange(n), labels].sum())
        _close(record.log_likelihood, ll, "observed likelihood")
        _close(record.conditional_log_likelihood, conditional, "conditional likelihood")
        _require(
            record.num_parameters == 2 * q - 1 and record.num_clusters == q and record.num_mutations == n,
            "score dimension/full-data N",
        )
        _close(record.bic, -2 * ll + (2 * q - 1) * np.log(n), "BIC-form score")
        weight_score = np.exp(kernel - mixture_ll[:, None]).mean(axis=0)
        _require(np.max(np.abs(weight_score - 1)) <= 1.1e-8, "fixed-center weight optimality")
        _, posterior = _kernel(data, centers[labels], purity)
        _require(
            np.array_equal(calls.mutation_id, data.mutation_id)
            and np.array_equal(calls.cluster_index, labels)
            and np.array_equal(calls.major_cn, data.major_cn),
            "posterior identities",
        )
        _require(np.array_equal(calls.multiplicity, [np.argmax(p) + 1 for p in posterior]), "posterior modes")
        _close(calls.multiplicity_probability, [p.max() for p in posterior], "posterior mode probabilities")
        _close(
            calls.expected_multiplicity, [p @ np.arange(1, len(p) + 1) for p in posterior], "posterior means"
        )
        _require(record.partition_sha256 == _partition_hash(order, _cuts(labels[order])), "partition hash")
        _require(
            record.parent_partition_sha256
            == parent_hashes[int(record.parent_requested_k), int(record.parent_replicate)],
            "raw parent identity",
        )
        evidence = candidates[
            (candidates.requested_k == k)
            & (candidates.replicate == record.replicate)
            & (candidates.candidate_id == record.candidate_id)
        ]
        _require(
            len(evidence) == 1
            and evidence.iloc[0].partition_sha256 == record.partition_sha256
            and bool(evidence.iloc[0].publication_eligible)
            and bool(evidence.iloc[0].selected_for_k),
            "selected candidate provenance",
        )
        _close(evidence.iloc[0].bic, record.bic, "candidate score")
        if record.candidate_kind != "native":
            _require(pd.isna(record.raw_status), "derived candidate inherited raw status")
    k = manifest["selected_k"]
    expected_names = {
        f"{stem}_K{k}.txt"
        for stem in ("mutation_assignments", "subclonal_structure", "posterior_multiplicity")
    }
    _require({p.name for p in (final / "Best_K").iterdir()} == expected_names, "stale Best_K")
    for name in expected_names:
        _require(sha256(final / "Best_K" / name) == sha256(final / name), "selected output copy")
    return {
        "status": "verified",
        "selected_k": k,
        "occupied_clusters": int(selected.num_clusters),
        "num_mutations": n,
        "bic": float(selected.bic),
        "backend": manifest["backend_actual"],
        "scope": "artifact/statistical/structural consistency; not global optimality or scientific accuracy",
    }
