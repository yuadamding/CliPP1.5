"""One autosomal, inclusive-interval input contract, independent of fitting.

Preserves the 19f310b R input policy, including normalized coordinate spelling
used by the frozen-chain tie break. No genotype/CN state is inferred or filtered.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INT_MAX = np.iinfo(np.int32).max


def _read(path, required):
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"Every input row must have exactly the header's number of fields: {path}")
    if len(set(rows[0])) != len(rows[0]) or not set(required) <= set(rows[0]):
        raise ValueError(f"Missing or duplicated input columns: {path}")
    return pd.DataFrame(rows[1:], columns=rows[0], dtype=str)


def _numbers(values):
    numbers = np.full(len(values), np.nan)
    # Parse directly to binary64: pandas can round a fractional coordinate to
    # an integer. Match R's hex support without Python's digit separators.
    for index, token in enumerate(values):
        token = str(token)
        if "_" in token:
            continue
        try:
            numbers[index] = (
                float.fromhex(token) if token.lower().startswith(("0x", "+0x", "-0x")) else float(token)
            )
        except (ValueError, OverflowError):
            pass
    return numbers


def _chromosomes(values):
    text = values.str.replace(r"(?i)^chr", "", regex=True)
    numbers = _numbers(text)
    if not text.str.fullmatch("[0-9]+").all() or not np.isin(numbers, np.arange(1, 23)).all():
        raise ValueError("Only autosomes 1..22 or chr1..chr22 are supported")
    return numbers.astype(np.int64)


def _coordinates(values, name):
    numbers = _numbers(values)
    if not (
        np.isfinite(numbers) & (numbers == np.floor(numbers)) & (numbers >= 1) & (numbers <= INT_MAX)
    ).all():
        raise ValueError(name + " must contain 1-based positive integer coordinates within int32")
    return numbers.astype(np.int64)


def _coordinate_text(value):
    """R's shortest fixed/scientific spelling for positive int32-valued doubles."""
    fixed = str(int(value))
    mantissa, exponent = format(float(value), ".14e").split("e")
    scientific = mantissa.rstrip("0").rstrip(".") + "e" + exponent
    return scientific if len(scientific) < len(fixed) else fixed


@dataclass
class CanonicalInput:
    retained: pd.DataFrame
    ledger: pd.DataFrame
    segments: pd.DataFrame
    purity: float

    @property
    def arrays(self):
        return tuple(
            np.ascontiguousarray(self.retained[name], dtype=np.int32)
            for name in ("alt_count", "depth", "major_cn", "total_cn")
        ) + (self.purity,)

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in (("retained.tsv", self.retained), ("input_ledger.tsv", self.ledger)):
            table.to_csv(directory / name, sep="\t", index=False, na_rep="NA", float_format="%.17g")

    def summary(self):
        n, major = len(self.retained), int(self.retained.major_cn.max())
        # Persistent padded model arrays only: one bool and one float64 per state.
        padded = n * major
        return {
            "num_input_rows": len(self.ledger),
            "num_retained": n,
            "num_excluded": len(self.ledger) - n,
            "exclusion_reasons": self.ledger.loc[self.ledger.status == "excluded", "reason"]
            .value_counts()
            .to_dict(),
            "cna_covered_input_rows": int(self.ledger.matched_segment_id.notna().sum()),
            "purity": self.purity,
            "maximum_major_cn": major,
            "scope": "autosomes 1..22; 1-based inclusive intervals",
            "resource_estimate": {
                "padded_support_states": padded,
                "model_padded_arrays_bytes": 9 * padded,
                "one_float64_support_temporary_bytes": 8 * padded,
                "initializer_matrix_cache_limit_bytes": 64 * 1024**2,
                "likelihood_column_cache_limit_bytes": 32 * 1024**2,
                "scope": "allocation estimates, not peak-memory limits; full N is initialized and scored even with subsampling",
            },
            "warnings": (
                [
                    "Padded full multiplicity support exceeds 256 MiB before temporaries; profile this input before a large panel"
                ]
                if 9 * padded > 256 * 1024**2
                else []
            ),
        }


def preprocess(snv_input, cn_input, purity_input, *, output=None):
    tokens = Path(purity_input).read_text().split()
    if len(tokens) != 1:
        raise ValueError("Purity file must contain exactly one scalar")
    purity = _numbers(pd.Series(tokens))[0]
    if not np.isfinite(purity) or not 0 < purity <= 1:
        raise ValueError("Purity must be in (0, 1]")
    snv = _read(snv_input, ("chromosome_index", "position", "ref_count", "alt_count"))
    segments = _read(
        cn_input, ("chromosome_index", "start_position", "end_position", "major_cn", "minor_cn", "total_cn")
    )
    chromosome = _chromosomes(snv.chromosome_index)
    position = _coordinates(snv.position, "SNV position")
    if len(set(zip(chromosome, position))) != len(snv):
        raise ValueError("Duplicate normalized SNV coordinates; one variant per locus is supported")
    coordinate_text = np.array([_coordinate_text(p) for p in position])
    ids = (
        snv.mutation_id.to_numpy()
        if "mutation_id" in snv
        else np.array([f"{c}:{p}" for c, p in zip(chromosome, coordinate_text)])
    )
    if len(set(ids)) != len(ids) or any(value in ("", "NA", "NaN") for value in ids):
        raise ValueError("Mutation IDs must be nonempty and unique")
    segments["chromosome_index"] = _chromosomes(segments.chromosome_index)
    for column in ("start_position", "end_position"):
        segments[column] = _coordinates(segments[column], "CNA " + column)
    if (segments.start_position > segments.end_position).any():
        raise ValueError("CNA start must not exceed end (inclusive coordinates)")
    for column in ("major_cn", "minor_cn", "total_cn"):
        segments[column] = _numbers(segments[column])
    cn = segments[["major_cn", "minor_cn", "total_cn"]].to_numpy()
    if (
        not np.isfinite(cn).all()
        or not (cn == np.floor(cn)).all()
        or (cn > INT_MAX).any()
        or (cn[:, 0] < 1).any()
        or (cn[:, 1] < 0).any()
        or (cn[:, 0] < cn[:, 1]).any()
        or (cn[:, 2] != cn[:, 0] + cn[:, 1]).any()
    ):
        raise ValueError("Invalid integer major/minor/total copy numbers")
    segments[["major_cn", "minor_cn", "total_cn"]] = cn.astype(np.int64)
    segments["segment_id"] = np.arange(1, len(segments) + 1)
    matched = np.full(len(snv), -1, dtype=int)
    for chrom, block in segments.groupby("chromosome_index", sort=False):
        block = block.sort_values(["start_position", "end_position"], kind="stable")
        starts, ends = block.start_position.to_numpy(), block.end_position.to_numpy()
        if np.any(starts[1:] <= np.maximum.accumulate(ends)[:-1]):
            raise ValueError(
                f"Ambiguous overlapping CNA intervals on chromosome {chrom}; inclusive endpoints cannot overlap"
            )
        rows = np.flatnonzero(chromosome == chrom)
        index = np.searchsorted(starts, position[rows], side="right") - 1
        covered = (index >= 0) & (position[rows] <= ends[np.maximum(index, 0)])
        matched[rows[covered]] = block.index.to_numpy()[index[covered]]
    alt, ref = _numbers(snv.alt_count), _numbers(snv.ref_count)
    depth = alt + ref
    valid = (
        np.isfinite(alt)
        & np.isfinite(ref)
        & (alt >= 0)
        & (ref >= 0)
        & (depth > 0)
        & (alt == np.floor(alt))
        & (ref == np.floor(ref))
        & (depth <= INT_MAX)
    )
    reason = np.where(
        ~valid, "invalid_counts_or_zero_depth", np.where(matched < 0, "no_cna_segment", "retained")
    )
    keep = reason == "retained"
    ledger = pd.DataFrame(
        {
            "original_row": np.arange(1, len(snv) + 1),
            "mutation_id": ids,
            "original_chromosome": snv.chromosome_index,
            "original_position": snv.position,
            "chromosome_index": chromosome.astype(str),
            "position": coordinate_text,
            "matched_segment_id": pd.array(np.where(matched < 0, np.nan, matched + 1), dtype="Int64"),
            "status": np.where(keep, "retained", "excluded"),
            "reason": reason,
        }
    )
    retained = ledger.loc[keep, ["original_row", "mutation_id", "chromosome_index", "position"]].reset_index(
        drop=True
    )
    for column, values in (("ref_count", ref), ("alt_count", alt), ("depth", depth)):
        retained[column] = values[keep].astype(np.int64)
    for column in ("major_cn", "minor_cn", "total_cn"):
        retained[column] = segments[column].to_numpy()[matched[keep]].astype(np.int64)
    retained["matched_segment_id"] = matched[keep] + 1
    # Preserve the historical 15-significant-digit R-to-native purity handoff.
    result = CanonicalInput(retained, ledger, segments, float(format(purity, ".15g")))
    if output is not None:
        result.write(output)
    if not keep.any():
        raise ValueError("No retained mutations; see input_ledger.tsv")
    return result


def validate_inputs(snv_input, cn_input, purity_input):
    """Preflight using the production parser; never load the native library or fit."""
    return preprocess(snv_input, cn_input, purity_input).summary()
