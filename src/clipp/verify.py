"""Independent arithmetic verifier. Does not invoke the fitter or optimizers.

Hashes establish artifact identity, not authenticity against a malicious writer.
Statistical checks use scipy.stats.binom and independently enumerated support.
Certificates remain scoped; verification does not establish global optimality.
"""

from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, xlog1py, xlogy
from scipy.stats import binom

from ._io import read_table as _table
from .config import FitConfig
from .native import sha256, build_identity
from .versions import IDENTITIES, NUMERICS, FIT_FIELDS, FIT_INTEGER_FIELDS, FIT_SCORE_FIELDS


def _require(condition, message):
    if not condition:
        raise ValueError("Verification failed: " + message)


def _close(value, expected, name, *, atol=2e-7, rtol=2e-9):
    _require(np.allclose(value, expected, atol=atol, rtol=rtol, equal_nan=False), name)


def _integer_vector(path):
    # Writers serialize decimal integers. Parse without float rounding/truncation.
    tokens = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    _require(
        bool(tokens) and all(token.isascii() and token.isdecimal() for token in tokens),
        "nonnegative integer vector " + str(path),
    )
    values = [int(token) for token in tokens]
    _require(max(values) < 2**63, "integer vector overflow")
    return np.asarray(values, dtype=np.int64)


def _kernel(data, cp, purity, *, posterior=True):
    # Independent scipy.stats arithmetic, grouped only to bound temporary storage.
    values = np.empty(len(data))
    posteriors = [None] * len(data) if posterior else None
    cp = np.broadcast_to(np.asarray(cp), (len(data),))
    alt, depth, total, major = (data[c].to_numpy() for c in ("alt_count", "depth", "total_cn", "major_cn"))
    for cn in np.unique(major):
        rows = np.flatnonzero(major == cn)
        size = max(1, 1_000_000 // int(cn))
        m = np.arange(1, int(cn) + 1)
        for start in range(0, len(rows), size):
            index = rows[start : start + size]
            p = cp[index, None] * m / (2 * (1 - purity) + purity * total[index, None])
            logs = binom.logpmf(alt[index, None], depth[index, None], np.minimum(1, p)) - np.log(cn)
            ll = logsumexp(logs, axis=1)
            values[index] = ll
            if posterior:
                for i, probability in zip(index, np.exp(logs - ll[:, None])):
                    posteriors[i] = probability
    return values, posteriors


def _verify_posterior_modes(data, cp, purity, multiplicity, posterior):
    """Accept independently maximal calls, including float64-indistinguishable ties."""
    calls = np.asarray(multiplicity)
    major = data.major_cn.to_numpy()
    _require(
        calls.shape == major.shape
        and np.isfinite(calls).all()
        and np.all(calls == np.floor(calls))
        and np.all((calls >= 1) & (calls <= major)),
        "posterior modes: integer support",
    )
    _require(
        len(posterior) == len(data)
        and all(np.isfinite(p).all() and np.max(p) > 0 for p in posterior),
        "posterior modes: defined posterior",
    )
    best = np.asarray([np.argmax(p) + 1 for p in posterior])
    rows = np.flatnonzero(calls != best)
    if not len(rows):
        return
    # Only disagreeing labels need a roundoff comparison. Uniform priors and
    # normalization cancel in the log odds; retain independent scipy arithmetic.
    m = np.column_stack((calls[rows], best[rows]))
    cp = np.broadcast_to(np.asarray(cp), (len(data),))[rows, None]
    alt = data.alt_count.to_numpy()[rows, None]
    depth = data.depth.to_numpy()[rows, None]
    denominator = 2 * (1 - purity) + purity * data.total_cn.to_numpy()[rows, None]
    p = np.minimum(1, cp * m / denominator)
    alternate_p = np.minimum(1, cp * (m / denominator))
    scores = binom.logpmf(alt, depth, p)
    alternate_scores = binom.logpmf(alt, depth, alternate_p)
    # Impossible states must never acquire an infinite acceptance tolerance.
    _require(
        np.isfinite(scores).all() and np.isfinite(alternate_scores).all(),
        "posterior modes: finite likelihood",
    )
    terms = np.abs(xlogy(alt, p)) + np.abs(xlog1py(depth - alt, -p))
    # Practical float64 budget for log-term evaluation/addition, separately
    # accounting for the two multiplication/division associations in p. The
    # combinatorial constant is bounded by |score| + |log terms|. Eight eps
    # per magnitude allows both arithmetic paths; this is not a libm interval
    # certificate or the much broader statistical _close tolerance.
    error = 8 * np.finfo(float).eps * (
        1 + np.abs(scores) + 2 * terms + np.log(major[rows, None])
    ) + np.abs(alternate_scores - scores)
    _require(np.all(scores[:, 1] - scores[:, 0] <= error.sum(axis=1)), "posterior modes")


def _cuts(labels):
    return np.flatnonzero(np.diff(labels)) + 1


def _partition_hash(order, cuts):
    return hashlib.sha256(
        np.asarray(order, dtype="<i8").tobytes() + np.asarray(cuts, dtype="<i8").tobytes()
    ).hexdigest()


def _verify_input_mapping(root, data, ledger, purity):
    """Independently reconstruct the R interval join and row-exclusion policy."""

    def read(name, columns):
        rows = [
            line.split()
            for line in (root / "inputs" / (name + ".txt")).read_text().splitlines()
            if line.strip()
        ]
        _require(
            len(rows) >= 2
            and len(set(rows[0])) == len(rows[0])
            and set(columns) <= set(rows[0])
            and all(len(row) == len(rows[0]) for row in rows),
            "original input header/row widths " + name,
        )
        return pd.DataFrame(rows[1:], columns=rows[0], dtype=str)

    def chrom(values):
        text = values.str.replace(r"(?i)^chr", "", regex=True)
        _require(text.str.fullmatch(r"[0-9]+").all(), "original chromosome encoding")
        number = pd.to_numeric(text)
        _require(number.between(1, 22).all(), "original autosomal scope")
        return number.to_numpy(dtype=int)

    def integer(values, name, minimum=0):
        numbers = numeric(values)
        _require(
            np.isfinite(numbers).all()
            and np.all(numbers == np.floor(numbers))
            and np.all((numbers >= minimum) & (numbers <= np.iinfo(np.int32).max)),
            name,
        )
        return numbers.astype(np.int64)

    def numeric(values):
        numbers = np.empty(len(values), dtype=float)
        for index, value in enumerate(values):
            try:
                if not value.isascii() or "_" in value:
                    raise ValueError("Invalid numeric literal")
                numbers[index] = float.fromhex(value) if "0x" in value.lower() else float(value)
            except (ValueError, OverflowError):
                numbers[index] = np.nan
        return numbers

    def coordinate_text(position):
        mantissa, exponent = format(float(position), ".14e").split("e")
        scientific = mantissa.rstrip("0").rstrip(".") + "e" + exponent
        return min((str(int(position)), scientific), key=len)

    snv = read("snv", ("chromosome_index", "position", "ref_count", "alt_count"))
    cn = read(
        "cna", ("chromosome_index", "start_position", "end_position", "major_cn", "minor_cn", "total_cn")
    )
    tokens = (root / "inputs/purity.txt").read_text().split()
    original_purity = numeric(tokens)[0] if len(tokens) == 1 else np.nan
    _require(
        len(tokens) == 1
        and np.isfinite(original_purity)
        and 0 < original_purity <= 1
        and float(format(original_purity, ".15g")) == purity,
        "original purity",
    )
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
        else np.array([f"{c}:{coordinate_text(p)}" for c, p in zip(chromosomes, positions)])
    )
    _require(np.array_equal(ledger.mutation_id, ids), "original mutation IDs")
    _require(
        len(set(ids)) == len(ids) and all(value not in {"", "NA", "NaN"} for value in ids),
        "unique original mutation IDs",
    )
    _require(len(set(zip(chromosomes, positions))) == len(snv), "unique original loci")
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
    _require(
        ledger.chromosome_index.tolist() == chromosomes.astype(str).tolist()
        and ledger.position.tolist() == [coordinate_text(p) for p in positions],
        "canonical coordinate spelling",
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
    alt, ref = numeric(snv.alt_count), numeric(snv.ref_count)
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
    for column in ("original_row", "mutation_id", "chromosome_index", "position", "matched_segment_id"):
        _require(np.array_equal(data[column], ledger.loc[keep, column]), "retained ledger " + column)
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
    indices = _integer_vector(raw / f"sample_indices_rep{rep}.txt") if subsampling else order
    _require(
        len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(data))),
        "sample indices",
    )
    ranks = np.argsort(order)[indices]
    _require(np.all(np.diff(ranks) > 0), "sample chain order")
    frame = _table(raw / f"K{k}_fit{suffix}.tsv")
    _require(len(frame) == 1, "raw diagnostic row count")
    row = frame.iloc[0]
    _require(
        [row.ccf_lower_bound, row.ccf_upper_bound] == NUMERICS["native"]["ccf_bounds"], "raw numerical bounds"
    )
    cp = np.loadtxt(raw / f"K{k}_phi{suffix}.txt", ndmin=1)
    labels = _integer_vector(raw / f"K{k}_label{suffix}.txt")
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
    full_cuts = _cuts(labels)
    bounds = np.r_[0, full_cuts, len(labels)]
    ranges = [np.ptp(x[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    _close(row.max_block_range_CCF, max(ranges), "raw block range", atol=1e-14)
    _require(row.longest_block == np.diff(bounds).max(), "raw longest block")
    _require(row.requested_K == k and row.actual_blocks == len(bounds) - 1, "raw capacity")
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
    if subsampling:
        full_cuts = (ranks[full_cuts - 1] + ranks[full_cuts]) // 2 + 1
    return {"partition_sha256": _partition_hash(order, full_cuts), "cuts": full_cuts, "diagnostics": row}


def _integer_columns(frame, names):
    for name in names:
        value = frame[name].to_numpy(dtype=float)
        _require(np.isfinite(value).all() and np.all(value == np.floor(value)), "integer column " + name)


def _boolean_columns(frame, names):
    for name in names:
        _require(frame[name].map(lambda x: isinstance(x, (bool, np.bool_))).all(), "Boolean column " + name)


def _candidate_parameters(row, order, purity):
    parameters = row.partition_parameters
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    _require(set(parameters) == {"cuts", "block_labels", "centers", "weights"}, "candidate parameter fields")
    cuts, visits, centers, weights = (
        np.asarray(parameters[key]) for key in ("cuts", "block_labels", "centers", "weights")
    )
    q = int(row.num_clusters)
    _require(
        cuts.ndim == visits.ndim == centers.ndim == weights.ndim == 1
        and len(cuts) == q - 1
        and len(visits) == len(centers) == len(weights) == q,
        "candidate parameter dimensions",
    )
    _require(
        np.all(cuts == np.floor(cuts))
        and np.all((cuts > 0) & (cuts < len(order)))
        and np.all(np.diff(cuts) > 0)
        and np.array_equal(np.sort(visits), np.arange(q)),
        "candidate cuts/block labels",
    )
    _require(
        np.isfinite(centers).all()
        and np.all((centers >= 0) & (centers <= purity))
        and np.all(np.diff(centers) <= 0),
        "candidate CP bounds/order",
    )
    _require(np.isfinite(weights).all() and np.all(weights >= 0), "candidate weights")
    _close(weights.sum(), 1, "candidate weight normalization", atol=1e-12)
    _require(row.partition_sha256 == _partition_hash(order, cuts), "candidate partition hash")
    labels = np.empty(len(order), dtype=int)
    labels[order] = visits[np.searchsorted(cuts, np.arange(len(order)), side="right")]
    return labels, centers, weights


def _verify_fit_statistics(row, labels, centers, weights, data, purity, cache):
    """Replay fit arithmetic, with bounded reusable independent likelihood columns."""
    columns, bytes_used = cache
    values = []
    for cp in centers:
        column = columns.pop(float(cp), None)
        if column is None:
            column = _kernel(data, cp, purity, posterior=False)[0]
            if column.nbytes <= 32 * 1024**2:
                while columns and bytes_used + column.nbytes > 32 * 1024**2:
                    bytes_used -= columns.popitem(last=False)[1].nbytes
                bytes_used += column.nbytes
        if column.nbytes <= 32 * 1024**2:
            columns[float(cp)] = column
        values.append(column)
    cache[1] = bytes_used
    kernel = np.column_stack(values)
    with np.errstate(divide="ignore"):
        mixture = logsumexp(kernel + np.log(weights), axis=1)
    likelihood = float(mixture.sum())
    _close(row.log_likelihood, likelihood, "candidate observed likelihood")
    _close(
        row.conditional_log_likelihood,
        kernel[np.arange(len(data)), labels].sum(),
        "candidate conditional likelihood",
    )
    _close(row.bic, -2 * likelihood + (2 * len(centers) - 1) * np.log(len(data)), "candidate BIC-form score")
    score = np.exp(kernel - mixture[:, None]).mean(axis=0)
    _close(row.weight_optimality_gap, len(data) * max(0, float(score.max() - 1)), "declared weight gap")
    _close(
        row.weight_active_score_gap,
        float(score.max() - score[weights > 0].min()),
        "declared active weight gap",
    )
    _require(max(0, float(score.max() - 1)) <= 1.1e-8, "candidate weight optimality")
    _require(np.max(np.abs(score[weights > 0] - 1)) <= 1.1e-8, "candidate active weight optimality")


@contextmanager
def _candidate_source(candidates):
    """Use a bounded indexed cursor, importing legacy evidence one row at a time."""
    if hasattr(candidates, "iter_rows") and hasattr(candidates, "find_ancestor"):
        yield candidates
        return
    from ._candidate_store import CandidateStore

    with CandidateStore() as store:
        if isinstance(candidates, pd.DataFrame):
            blocks = (candidates,)
        elif isinstance(candidates, (str, Path)):
            blocks = pd.read_csv(
                candidates,
                sep="\t",
                float_precision="round_trip",
                converters={"chain_cuts": str},
                chunksize=256,
            )
        else:
            raise ValueError("Verification failed: candidate cursor or legacy table required")
        for block in blocks:
            integer_fields = (
                "requested_k",
                "replicate",
                "candidate_id",
                "parent_requested_k",
                "parent_replicate",
            )
            boolean_fields = (
                "selected",
                "selected_for_k",
                "selected_for_replicate_k",
                "publication_eligible",
            )
            _require(set(integer_fields + boolean_fields) <= set(block.columns), "candidate schema columns")
            # Scratch append assigns missing IDs/flags for fitter convenience.
            # Imported evidence must already declare them; never repair it here.
            _integer_columns(block, integer_fields)
            _boolean_columns(block, boolean_fields)
            for row in block.itertuples(index=False):
                record = row._asdict()
                # Legacy DataFrames fill fields absent from failed rows with NA.
                if pd.isna(record.get("partition_parameters", np.nan)):
                    record.pop("partition_parameters", None)
                store.append(record)
        yield store


def _integer_record(record, names):
    for name in names:
        value = float(record[name])
        _require(np.isfinite(value) and value == np.floor(value), "integer column " + name)


def _verify_candidates(
    candidates, selection, capacities, replicates, parents, order, data, purity, *, _progress=None
):
    """Independently verify an ordered, disk-backed bank with bounded reductions.

    Only one row, per-attempt reductions and independent likelihood columns are
    retained. The returned records are the small set of verified finalists.
    Legacy DataFrames/TSVs are imported into the same indexed cursor contract.
    """
    expected = {(k, rep) for k in capacities for rep in range(1, replicates + 1)}
    _integer_columns(selection, ("requested_k", "replicate", "candidate_id", "num_clusters"))
    _boolean_columns(selection, ("selected", "selected_for_k", "publication_eligible"))
    _require(
        len(selection) == len(expected)
        and not selection.duplicated(["requested_k", "replicate"]).any()
        and set(zip(selection.requested_k, selection.replicate)) == expected,
        "configuration/selection attempts",
    )
    selected_attempts = {(row.requested_k, row.replicate): row for _, row in selection.iterrows()}
    counts, next_ids, natives, best = {}, {}, {}, {}
    kinds = {
        "native",
        "adjacent_coarsening",
        "capacity_reuse",
        "chain_boundary_polish",
        "unsupported_component_coarsening",
        "supported_single_block_fallback",
    }
    cache = [OrderedDict(), 0]
    verified_count = 0

    def progress(phase):
        if _progress is not None:
            _progress(
                {"stage": "candidate_verification", "phase": phase, "verified_candidates": verified_count}
            )

    def checked_row():
        nonlocal verified_count
        verified_count += 1
        if verified_count % 1024 == 0:
            progress("arithmetic")

    progress("arithmetic")
    with _candidate_source(candidates) as source:
        # Pass one independently checks every row and builds small reductions.
        for record in source.iter_rows():
            _integer_record(
                record, ("requested_k", "replicate", "candidate_id", "parent_requested_k", "parent_replicate")
            )
            for name in ("selected", "selected_for_k", "selected_for_replicate_k", "publication_eligible"):
                _require(isinstance(record[name], (bool, np.bool_)), "Boolean column " + name)
            row = SimpleNamespace(**record)
            attempt = (row.requested_k, row.replicate)
            _require(attempt in expected, "candidate capacities/replicates")
            _require(row.candidate_id == next_ids.get(row.replicate, 0), "ordered candidate IDs")
            next_ids[row.replicate] = row.candidate_id + 1
            counts[attempt] = counts.get(attempt, 0) + 1
            _require(
                row.candidate_search_version == IDENTITIES["candidate_search_version"],
                "candidate search contract",
            )
            _require(row.candidate_kind in kinds, "candidate kinds")
            _require(row.status in {"scored", "refit_failed"}, "candidate status")
            if row.status == "scored":
                _integer_record(record, ("num_clusters", "active_mixture_components"))
                _require(np.isfinite(row.bic), "finite candidate scores")
            eligible = row.status == "scored" and row.active_mixture_components == row.num_clusters
            _require(row.publication_eligible == eligible, "candidate eligibility")
            if eligible:
                key = (row.bic, row.num_clusters, row.candidate_id)
                if attempt not in best or key < best[attempt][0]:
                    best[attempt] = (key, record)
            if row.candidate_kind == "native":
                _require(attempt not in natives, "one native candidate per attempt")
                _require(row.status == "scored", "native candidate must be scored")
                natives[attempt] = record
            parent = parents.get((row.parent_requested_k, row.parent_replicate))
            _require(
                parent is not None and parent["partition_sha256"] == row.parent_partition_sha256,
                "candidate raw parent",
            )
            _require(row.parent_replicate == row.replicate, "candidate replicate parent")
            cut_text = str(row.chain_cuts)
            proposal = [] if not cut_text else [int(x) for x in cut_text.split(",")]
            _require(
                all(0 < x < len(data) for x in proposal) and proposal == sorted(set(proposal)),
                "proposal cuts",
            )
            _require(row.proposal_partition_sha256 == _partition_hash(order, proposal), "proposal hash")
            parent_cuts = set(parent["cuts"])
            if row.candidate_kind == "native":
                _require(
                    row.proposal_partition_sha256 == row.parent_partition_sha256, "native proposal parent"
                )
                _require(row.requested_k == row.parent_requested_k, "native capacity parent")
            elif row.candidate_kind in {"adjacent_coarsening", "capacity_reuse"}:
                relation = (
                    set(proposal) < parent_cuts
                    if row.candidate_kind == "adjacent_coarsening"
                    else set(proposal) == parent_cuts
                )
                _require(relation and len(proposal) + 1 == row.requested_k, "coarsening proposal parent")
            refinement_parent = None
            if not pd.isna(record.get("refinement_parent_partition_sha256", np.nan)):
                ancestor = source.find_ancestor(record)
                _require(ancestor is not None, "refinement parent")
                refinement_parent = set(json.loads(ancestor["partition_parameters"])["cuts"])
            for prefix in ("polish_", "boundary_"):
                for name, value_expected in (
                    ("fixed_chain", True),
                    ("unconstrained_reassignment", False),
                    ("constrained_optimum_certified", False),
                ):
                    value = record.get(prefix + name, np.nan)
                    if not pd.isna(value):
                        _require(
                            isinstance(value, (bool, np.bool_)) and value == value_expected,
                            "refinement scope " + prefix + name,
                        )
            if row.status != "scored":
                checked_row()
                continue
            _require(1 <= row.num_clusters <= row.requested_k, "candidate occupied q <= K")
            labels, centers, weights = _candidate_parameters(row, order, purity)
            final_cuts = set(_cuts(labels[order]))
            _require(row.num_input_blocks == len(proposal) + 1, "candidate input block count")
            if row.candidate_kind in {"native", "adjacent_coarsening", "capacity_reuse"}:
                _require(final_cuts <= set(proposal), "conditional refit cannot introduce cuts")
            else:
                _require(
                    refinement_parent is not None and final_cuts == set(proposal), "refined proposal identity"
                )
                if row.candidate_kind == "unsupported_component_coarsening":
                    _require(final_cuts < refinement_parent, "unsupported component coarsening")
                elif row.candidate_kind == "chain_boundary_polish":
                    _require(len(final_cuts) <= len(refinement_parent), "boundary polish capacity")
                else:
                    _require(not final_cuts, "single block fallback")
            _require(row.active_mixture_components == np.count_nonzero(weights), "candidate active weights")
            for name, value_expected in (
                ("minimum_mixture_weight", float(weights.min())),
                ("distinct_centers", len(np.unique(centers))),
                ("joint_mixture_center_mle", False),
            ):
                value = record.get(name, np.nan)
                if not pd.isna(value):
                    if name == "joint_mixture_center_mle":
                        _require(
                            isinstance(value, (bool, np.bool_)) and not value, "conditional center scope"
                        )
                    elif name == "distinct_centers":
                        _require(value == value_expected, "candidate distinct centers")
                    else:
                        _close(value, value_expected, "candidate minimum weight", atol=1e-14)
            _verify_fit_statistics(row, labels, centers, weights, data, purity, cache)
            checked_row()

        progress("reconciliation")
        _require(set(counts) == expected, "candidate capacities/replicates")
        _require(set(best) == expected, "supported candidate per attempt")
        _require(set(natives) == expected, "one native candidate per attempt")
        for attempt, (_, winner) in best.items():
            row = selected_attempts[attempt]
            _require(row.candidate_id == winner["candidate_id"], "selection/candidate winner")
            for name in (
                "partition_sha256",
                "proposal_partition_sha256",
                "parent_partition_sha256",
                "candidate_kind",
                "parent_requested_k",
                "parent_replicate",
                "num_clusters",
                "publication_eligible",
                "active_mixture_components",
                "candidate_search_version",
                "distinct_centers",
                "joint_mixture_center_mle",
            ):
                _require(row[name] == winner[name], "selection/candidate " + name)
            for name in (
                "bic",
                "log_likelihood",
                "conditional_log_likelihood",
                "weight_optimality_gap",
                "weight_active_score_gap",
                "minimum_mixture_weight",
            ):
                _close(row[name], winner[name], "selection/candidate " + name)
            _require(
                row.status == "numerical_multimode_refit"
                and row.center_refit == "conditional"
                and row.bic_definition == IDENTITIES["scoring_version"]
                and row.num_parameters == 2 * row.num_clusters - 1
                and row.num_mutations == len(data),
                "selection scientific contract",
            )
            _close(row.native_bic, natives[attempt]["bic"], "selection native score")
            _require(row.native_num_clusters == natives[attempt]["num_clusters"], "selection native size")
            parent = parents.get((row.parent_requested_k, row.parent_replicate))
            _require(parent is not None, "selection raw parent")
            for name, value in parent["diagnostics"].items():
                fields = ["parent_raw_" + name]
                if row.candidate_kind == "native":
                    fields.append("raw_" + name)
                else:
                    _require(
                        pd.isna(row.get("raw_" + name, np.nan)), "derived candidate inherited raw " + name
                    )
                for field in fields:
                    if isinstance(value, (bool, np.bool_)):
                        _require(
                            isinstance(row[field], (bool, np.bool_)) and row[field] == value,
                            "selection diagnostic " + field,
                        )
                    elif isinstance(value, str):
                        _require(row[field] == value, "selection diagnostic " + field)
                    else:
                        _close(row[field], value, "selection diagnostic " + field, atol=1e-13, rtol=1e-14)
            _require(row.candidate_count == counts[attempt], "candidate count")

        winners = (
            selection.sort_values(["bic", "num_clusters", "replicate"], kind="stable")
            .groupby("requested_k", sort=False)
            .head(1)
        )
        chosen = winners.sort_values(["bic", "num_clusters", "requested_k", "replicate"], kind="stable").iloc[
            0
        ]
        per_k = {row.requested_k: row for row in winners.itertuples()}
        for row in selection.itertuples():
            _require(
                row.selected_for_k == (row.replicate == per_k[row.requested_k].replicate),
                "replicate selection flags",
            )
            _require(
                row.selected == (row.requested_k == chosen.requested_k and row.replicate == chosen.replicate),
                "overall selection flags",
            )
        # Pass two reconciles flags for every candidate, including failed routes.
        flag_counts, flag_next_ids = {}, {}
        for record in source.iter_rows():
            row = SimpleNamespace(**record)
            attempt = (row.requested_k, row.replicate)
            _require(row.candidate_id == flag_next_ids.get(row.replicate, 0), "replayed candidate IDs")
            flag_next_ids[row.replicate] = row.candidate_id + 1
            flag_counts[attempt] = flag_counts.get(attempt, 0) + 1
            selected = selected_attempts[attempt]
            is_winner = row.candidate_id == selected.candidate_id
            _require(row.selected_for_replicate_k == is_winner, "candidate-bank ranking")
            _require(
                row.selected_for_k == (is_winner and selected.selected_for_k), "candidate selected-for-K flag"
            )
            _require(row.selected == (is_winner and selected.selected), "candidate selected flag")
        _require(flag_counts == counts and flag_next_ids == next_ids, "candidate cursor replay completeness")
    progress("complete")
    return {(record["replicate"], record["candidate_id"]): record for _, record in best.values()}


def _verify_finalists(
    fits, capacities, replicates, parents, order, data, purity, evidence, *, _progress=None
):
    """Check compact finalists; discarded proposal ancestry is checked before publication."""
    fields = set(FIT_FIELDS) | {"proposal_cuts", "partition_parameters"}
    _require(
        isinstance(fits, list) and all(isinstance(record, dict) and set(record) == fields for record in fits),
        "compact finalist fields",
    )
    for record in fits:
        _require(
            all(type(record[name]) is int for name in FIT_INTEGER_FIELDS)
            and all(
                type(record[name]) in (int, float) and np.isfinite(record[name]) for name in FIT_SCORE_FIELDS
            ),
            "compact finalist numeric values",
        )
        parameters = record["partition_parameters"]
        _require(
            isinstance(parameters, dict)
            and set(parameters) == {"cuts", "block_labels", "centers", "weights"},
            "compact finalist parameter fields",
        )
        for name, values in {"proposal_cuts": record["proposal_cuts"], **parameters}.items():
            allowed = (int, float) if name in {"centers", "weights"} else (int,)
            _require(
                isinstance(values, list) and all(type(value) in allowed for value in values),
                "compact finalist vector " + name,
            )
    expected = [(k, rep) for k in capacities for rep in range(1, replicates + 1)]
    _require(
        [(row["requested_k"], row["replicate"]) for row in fits] == expected,
        "configuration/finalist attempts",
    )
    selection = pd.DataFrame(fits)
    _require(
        not selection.duplicated(["replicate", "candidate_id"]).any() and (selection.candidate_id >= 0).all(),
        "finalist candidate IDs",
    )
    if evidence is not None:
        _require(isinstance(evidence, tuple) and len(evidence) == 2, "transient candidate evidence")
        transient_selection, candidates = evidence
        verified_candidates = _verify_candidates(
            candidates,
            transient_selection,
            capacities,
            replicates,
            parents,
            order,
            data,
            purity,
            _progress=_progress,
        )
    cache = [OrderedDict(), 0]
    for row in selection.itertuples():
        _require(1 <= row.num_clusters <= row.requested_k, "finalist occupied q <= K")
        labels, centers, weights = _candidate_parameters(row, order, purity)
        _require(np.all(weights > 0), "finalist occupied weights")
        parent = parents.get((row.parent_requested_k, row.parent_replicate))
        _require(
            parent is not None
            and parent["partition_sha256"] == row.parent_partition_sha256
            and row.parent_replicate == row.replicate,
            "finalist raw parent",
        )
        proposal = row.proposal_cuts
        _require(
            all(0 < cut < len(data) for cut in proposal)
            and proposal == sorted(set(proposal))
            and row.proposal_partition_sha256 == _partition_hash(order, proposal),
            "finalist proposal identity",
        )
        final_cuts, parent_cuts = set(row.partition_parameters["cuts"]), set(parent["cuts"])
        if row.candidate_kind in {"native", "adjacent_coarsening", "capacity_reuse"}:
            _require(final_cuts <= set(proposal), "finalist conditional refit cannot introduce cuts")
            if row.candidate_kind == "native":
                _require(
                    row.proposal_partition_sha256 == row.parent_partition_sha256
                    and row.requested_k == row.parent_requested_k,
                    "finalist native proposal parent",
                )
            else:
                relation = (
                    set(proposal) < parent_cuts
                    if row.candidate_kind == "adjacent_coarsening"
                    else set(proposal) == parent_cuts
                )
                _require(
                    relation and len(proposal) + 1 == row.requested_k, "finalist coarsening proposal parent"
                )
        else:
            _require(
                row.candidate_kind
                in {
                    "chain_boundary_polish",
                    "unsupported_component_coarsening",
                    "supported_single_block_fallback",
                }
                and final_cuts == set(proposal),
                "finalist refinement identity",
            )
            if row.candidate_kind == "supported_single_block_fallback":
                _require(not final_cuts, "finalist single block fallback")
        if evidence is None:
            _verify_fit_statistics(row, labels, centers, weights, data, purity, cache)
        else:
            selected = transient_selection[
                (transient_selection.requested_k == row.requested_k)
                & (transient_selection.replicate == row.replicate)
            ].iloc[0]
            candidate = verified_candidates.get((row.replicate, row.candidate_id))
            _require(candidate is not None, "finalist/transient candidate identity")
            for name in fields - {"partition_parameters", "proposal_cuts"}:
                _require(
                    getattr(row, name) == selected[name] == candidate[name],
                    "finalist/transient " + name,
                )
            _require(
                row.partition_parameters == json.loads(candidate["partition_parameters"])
                and proposal == [int(cut) for cut in candidate["chain_cuts"].split(",") if cut],
                "finalist/transient parameters",
            )
    # One finalist per attempt: comparing all attempts has the same two-stage
    # tie order as per-K replicate selection followed by global selection.
    return selection.sort_values(["bic", "num_clusters", "requested_k", "replicate"], kind="stable").iloc[0]


def verify_run(directory, *, require_complete=True, _candidate_evidence=None, _progress=None):
    """Verify saved fits; schema 4 retains finalists, not the complete proposal bank."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("output_schema_version")
    _require(
        schema in (3, 4),
        "unsupported output schema; schema 2 requires pinned CliPP1.5 19f310b",
    )
    for name, value in IDENTITIES.items():
        if name not in {"software_version", "output_schema_version"}:
            _require(manifest.get(name) == value, "contract identity " + name)
    _require(
        _candidate_evidence is None or (schema == 4 and not require_complete), "transient verification scope"
    )
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
    _require(set(manifest["inputs"]) == {"snv", "cna", "purity"}, "original input records")
    for name, record in manifest["inputs"].items():
        _require(
            record["path"] == f"inputs/{name}.txt" and artifacts.get(record["path"]) == record["sha256"],
            "original input identity",
        )
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
    purity = manifest["purity"]
    _require(isinstance(purity, (float, int)) and 0 < purity <= 1, "purity")
    _verify_input_mapping(root, data, ledger, purity)
    order = _integer_vector(raw / "chain_order.txt")
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
    validated_config = FitConfig.from_dict(config)
    resolved, actual = manifest["backend_resolved"], manifest["backend_actual"]
    _require(
        manifest["backend_requested"] == config["device"]
        and resolved in {"auto", "cpu", "cuda"}
        and actual in {"cpu", "cuda"}
        and (config["device"] == "auto" or resolved == config["device"])
        and (resolved == "auto" or actual == resolved)
        and (resolved != "auto" or n > 1000 or actual == "cpu"),
        "configuration/backend binding",
    )
    _require(manifest["capacities"] == validated_config.capacities(n), "configuration/capacities")
    subsampling = config["subsample_size"] is not None
    expected_raw = {"chain_order.txt", "pilot_cp.txt", "chain_initialization.json"}
    for rep in range(1, config["replicates"] + 1):
        suffix = f"_rep{rep}" if subsampling else ""
        if subsampling:
            expected_raw.add(f"sample_indices_rep{rep}.txt")
        expected_raw.update(
            f"K{k}_{name}{suffix}.{extension}"
            for k in manifest["capacities"]
            for name, extension in (("phi", "txt"), ("label", "txt"), ("fit", "tsv"))
        )
    _require({p.name for p in raw.iterdir()} == expected_raw, "configured raw attempt inventory")
    _require(
        {p.name for p in pre.iterdir()} == {"retained.tsv", "input_ledger.tsv"}, "canonical input inventory"
    )
    _require(
        {p.name for p in final.iterdir()}
        == (
            {"mutations.tsv", "clusters.tsv"}
            | ({"bic_selection.tsv", "chain_candidates.tsv"} if schema == 3 else set())
        ),
        "compact result inventory",
    )
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
        _require(
            record["indices_file"] == f"preliminary_result/sample_indices_rep{record['replicate']}.txt",
            "subsample indices reference",
        )
        indices = _integer_vector(root / record["indices_file"])
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
    if schema == 3:
        selection = _table(final / "bic_selection.tsv")
        candidates = final / "chain_candidates.tsv"
        verified_candidates = _verify_candidates(
            candidates,
            selection,
            manifest["capacities"],
            config["replicates"],
            parent_hashes,
            order,
            data,
            purity,
            _progress=_progress,
        )
        selected_rows = selection[selection.selected]
        _require(len(selected_rows) == 1, "exactly one selected result")
        selected = selected_rows.iloc[0]
        valid = selection[selection.selected_for_k]
        _require(valid.requested_k.tolist() == manifest["capacities"], "one result per requested capacity")
        best = valid.sort_values(["bic", "num_clusters", "requested_k", "replicate"], kind="stable").iloc[0]
        _require(
            int(best.requested_k) == int(selected.requested_k)
            and int(best.replicate) == int(selected.replicate),
            "BIC-form selection",
        )
        evidence = SimpleNamespace(**verified_candidates[selected.replicate, selected.candidate_id])
    else:
        selected = _verify_finalists(
            manifest["fits"],
            manifest["capacities"],
            config["replicates"],
            parent_hashes,
            order,
            data,
            purity,
            _candidate_evidence,
            _progress=_progress,
        )
        evidence = selected
    _require(selected.requested_k == manifest["selected_k"], "selected capacity")
    _require(
        manifest["selected_outputs"]
        == {"mutations": "final_result/mutations.tsv", "clusters": "final_result/clusters.tsv"},
        "selected output references",
    )
    # Finalist likelihoods, scores, weights and retained provenance were checked
    # above. Bind selected tables exactly to that evidence, then replay only
    # the posterior quantities unique to the selected mutation output.
    labels, centers, weights = _candidate_parameters(evidence, order, purity)
    calls, structure = _table(final / "mutations.tsv"), _table(final / "clusters.tsv")
    if schema == 4:
        _require(
            calls.columns.tolist()
            == [
                "chromosome_index",
                "position",
                "mutation_id",
                "original_row",
                "cluster_index",
                "major_cn",
                "multiplicity",
                "expected_multiplicity",
            ]
            and structure.columns.tolist()
            == [
                "cluster_index",
                "num_SNV",
                "cellular_prevalence",
                "cancer_cell_fraction",
                "purity",
                "assignment_proportion",
            ],
            "minimal selected output columns",
        )
    for column in ("mutation_id", "original_row", "chromosome_index", "position", "major_cn"):
        _require(np.array_equal(calls[column], data[column]), "selected mutation " + column)
    _require(
        np.array_equal(calls.cluster_index, labels)
        and np.array_equal(structure.cellular_prevalence, centers),
        "selected parameters match fit evidence",
    )
    if schema == 3:
        _require(np.array_equal(structure.mixture_weight, weights), "selected mixture weights")
    counts = np.bincount(labels)
    _require(
        np.array_equal(structure.cluster_index, np.arange(len(centers)))
        and np.array_equal(structure.num_SNV, counts),
        "membership counts",
    )
    _close(structure.cancer_cell_fraction, centers / purity, "CCF conversion")
    _close(structure.purity, purity, "output purity")
    _close(structure.assignment_proportion, counts / n, "assignment proportions")
    _, posterior = _kernel(data, centers[labels], purity)
    _verify_posterior_modes(data, centers[labels], purity, calls.multiplicity, posterior)
    if schema == 3:
        _close(calls.multiplicity_probability, [p.max() for p in posterior], "posterior mode probabilities")
    _close(calls.expected_multiplicity, [p @ np.arange(1, len(p) + 1) for p in posterior], "posterior means")
    k = manifest["selected_k"]
    return {
        "status": "verified",
        "selected_k": k,
        "occupied_clusters": int(selected.num_clusters),
        "num_mutations": n,
        "bic": float(selected.bic),
        "backend": manifest["backend_actual"],
        "scope": (
            "artifact/statistical/structural consistency; not global optimality or scientific accuracy"
            if schema == 3
            else "artifact/finalist/selected-result statistical consistency; discarded candidate bank and refinement ancestry "
            + (
                "verified from transient evidence before publication"
                if _candidate_evidence is not None
                else "not replayed"
            )
            + "; not global optimality or scientific accuracy"
        ),
    }
