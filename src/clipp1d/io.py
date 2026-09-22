"""Single-sample long-table validation, independent of pandas and CliPP2.

Schema/normalization contract adapted from CliPP2 77525a6; see UPSTREAM.md.
"""

import csv
import gzip
import hashlib
import io
import math
import zlib
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .policy import Policy
from .types import InputError, Mutation, TumorInput

SCHEMA_COLUMNS = (
    "mutation_id", "sample_id", "alt_count", "ref_count", "count_observed", "purity",
    "normal_cn", "segment_id", "cn_state_id", "cn_state_fraction", "allele_a_cn", "allele_b_cn",
)


def _number(value, name, integer=False, missing=False):
    if missing and value == ".":
        return None
    if integer:
        # Validate the supplied decimal before float64 conversion can round it.
        try:
            exact = Decimal(value)
        except InvalidOperation as exc:
            raise InputError(f"{name} must be numeric") from exc
        if (not exact.is_finite() or exact < 0 or exact > 2**53 or
                exact != exact.to_integral_value()):
            raise InputError(f"{name} must be an exactly representable nonnegative integer")
        return int(exact)
    try:
        v = float(value)
    except ValueError as exc:
        raise InputError(f"{name} must be numeric") from exc
    if not math.isfinite(v) or v < 0:
        raise InputError(f"{name} must be finite and nonnegative")
    return v


def _identifier(v):
    if not v or v == "." or v.startswith("#") or v.strip() != v or any(c in v for c in "\t\r\n"):
        raise InputError(f"Invalid identifier: {v!r}")
    return v


def read_tumor(path, policy=Policy()):
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    metadata, rows, header = {}, [], None
    try:
        content = gzip.decompress(payload) if path.suffix.lower() == ".gz" else payload
        decoded = content.decode("utf-8")
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
        raise InputError("Input must be valid UTF-8 TSV, optionally gzip compressed") from exc
    with io.StringIO(decoded, newline="") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("##"):
                if header is not None or "=" not in line:
                    raise InputError("Metadata must precede the header and use ##key=value")
                key, value = line[2:].split("=", 1)
                if key in metadata or not key or not value:
                    raise InputError("Duplicate or empty metadata")
                metadata[_identifier(key)] = value
                continue
            if line.startswith("#"):
                continue
            try:
                fields = next(csv.reader([line], delimiter="\t", strict=True))
            except csv.Error as exc:
                raise InputError("Invalid TSV quoting") from exc
            if header is None:
                header = fields
                if (len(set(header)) != len(header) or
                        any(not c or c.strip() != c for c in header) or
                        not set(SCHEMA_COLUMNS).issubset(header)):
                    raise InputError("Header must contain the twelve unique model columns")
                continue
            if len(fields) != len(header):
                raise InputError("Invalid row width or blank value (use '.' for missing)")
            row = dict(zip(header, fields))
            if any(row[key] == "" for key in SCHEMA_COLUMNS):
                raise InputError("Invalid row width or blank value (use '.' for missing)")
            for key in ("mutation_id", "sample_id", "segment_id", "cn_state_id"):
                row[key] = _identifier(row[key])
            if row["count_observed"] not in ("0", "1"):
                raise InputError("count_observed must be 0 or 1")
            row["count_observed"] = row["count_observed"] == "1"
            for key in ("alt_count", "ref_count", "allele_a_cn", "allele_b_cn"):
                row[key] = _number(row[key], key, integer=True,
                                   missing=key.endswith("count") and not row["count_observed"])
            for key in ("purity", "normal_cn", "cn_state_fraction"):
                row[key] = _number(row[key], key)
            if not 0 < row["purity"] <= 1 or row["cn_state_fraction"] <= 0:
                raise InputError("Require purity in (0,1] and positive CN fractions")
            if row["allele_a_cn"] < row["allele_b_cn"]:
                raise InputError("Require allele_a_cn >= allele_b_cn")
            rows.append(row)
    if not rows:
        raise InputError("Input contains no mutation rows")
    for key, expected in (("schema", "clipp2.tumor.long.v1"),
                          ("coordinate_system", "1-based-inclusive"), ("missing_value", ".")):
        if key in metadata and metadata[key] != expected:
            raise InputError(f"Unsupported {key}: {metadata[key]}")
    samples = {r["sample_id"] for r in rows}
    if len(samples) != 1:
        raise InputError("Exactly one sample_id is required; select a sample explicitly upstream")
    purities = [r["purity"] for r in rows]
    mean_purity = math.fsum(purities) / len(purities)
    if any(abs(p - mean_purity) > 1e-10 for p in purities):
        raise InputError("Purity must be constant within the sample")
    groups, definitions = {}, {}
    for r in rows:
        groups.setdefault(r["mutation_id"], []).append(r)
        key = (r["segment_id"], r["cn_state_id"])
        definitions.setdefault(key, []).append((r["cn_state_fraction"], r["allele_a_cn"], r["allele_b_cn"]))
    states = {}
    for (segment, state_id), values in definitions.items():
        fractions = [v[0] for v in values]
        mean_fraction = math.fsum(fractions) / len(fractions)
        if len({v[1:] for v in values}) != 1 or any(
                abs(f - mean_fraction) > 1e-10 for f in fractions):
            raise InputError("Inconsistent CN state definition")
        states.setdefault(segment, {})[state_id] = (min(fractions), *values[0][1:])
    for state_map in states.values():
        if abs(math.fsum(s[0] for s in state_map.values()) - 1) > 1e-8:
            raise InputError("CN fractions must sum to one per segment")
    mutations = []
    for mid, group in sorted(groups.items()):
        first = group[0]
        exact = ("alt_count", "ref_count", "count_observed", "segment_id")
        if (any(any(r[k] != first[k] for k in exact) for r in group) or
                max(r["normal_cn"] for r in group) - min(r["normal_cn"] for r in group) > 1e-10):
            raise InputError(f"Repeated observation fields disagree for {mid}")
        state_ids = [r["cn_state_id"] for r in group]
        if len(set(state_ids)) != len(state_ids):
            raise InputError(f"Duplicate CN state row for {mid}")
        expected = states[first["segment_id"]]
        if set(state_ids) != set(expected):
            raise InputError(f"Incomplete CN state set for {mid}")
        aggregated = {}
        for sid in sorted(expected):
            fraction, a, b = expected[sid]
            aggregated.setdefault((a, b), []).append(fraction)
        cn = tuple((math.fsum(fs), a, b) for (a, b), fs in sorted(aggregated.items()))
        reason = None
        if max(s[1] for s in cn) > policy.max_major_cn:
            reason = "MAJOR_CN_ABOVE_LIMIT"
        elif not first["count_observed"]:
            reason = "MISSING_COUNTS"
        elif first["alt_count"] + first["ref_count"] == 0:
            reason = "ZERO_DEPTH"
        elif max(s[1] for s in cn) == 0:
            raise InputError(f"NO_POSITIVE_MUTANT_COPY_PATH: {mid}")
        mutations.append(Mutation(mid, first["alt_count"], first["ref_count"],
                                  first["count_observed"], min(r["normal_cn"] for r in group), cn, reason))
    # Match upstream canonicalization after exclusion; absent retained data is handled by compiler.
    retained_ids = {m.mutation_id for m in mutations if m.exclusion is None}
    purity = min((r["purity"] for r in rows if r["mutation_id"] in retained_ids), default=min(purities))
    tumor_id = _identifier(metadata.get("tumor_id", path.name.removesuffix(".gz").removesuffix(".tsv")))
    return TumorInput(tumor_id,
                      next(iter(samples)), purity, tuple(mutations), digest)
