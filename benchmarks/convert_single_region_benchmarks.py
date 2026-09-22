#!/usr/bin/env python3
"""Convert legacy benchmarks, preserving SimClone diploid normal-CN convention.

Based on the workspace converter; the original was retained as historical evidence.
Run with PYTHONPATH=/data/CliPP2 and ml1 to access the canonical TSV writer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from CliPP2.io.tumor_txt import TUMOR_TXT_SCHEMA, load_tumor_txt, write_tumor_txt


REQUIRED_SNV_COLUMNS = frozenset(
    {"chromosome_index", "position", "alt_count", "ref_count"}
)
REQUIRED_CNA_COLUMNS = frozenset(
    {
        "chromosome_index",
        "start_position",
        "end_position",
        "major_cn",
        "minor_cn",
        "total_cn",
    }
)
REQUIRED_SOURCE_FILES = ("snv.txt", "cna.txt", "purity.txt")
SIMCLONE_SEX_ROOT = Path("/data/CliPP_Sim/testing_SimClone1000_truth")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_array(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains nonfinite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} contains noninteger values")
    return rounded.astype(np.int64)


def _chromosome_key(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    if not text:
        raise ValueError("empty chromosome identifier")
    return text.upper() if text.upper() in {"X", "Y", "MT", "M"} else text


def _source_sex(tumor_id: str, sex_root: Path) -> str:
    metadata_root = sex_root / tumor_id
    files = sorted(metadata_root.glob("**/dataset_g.txt"))
    if len(files) != 1:
        raise ValueError(
            f"{tumor_id}: expected one SimClone dataset_g.txt, found {len(files)}"
        )
    table = pd.read_csv(files[0], sep="\t")
    if table.shape[0] != 1 or "sex" not in table.columns:
        raise ValueError(f"{tumor_id}: malformed SimClone sex metadata")
    sex = str(table.loc[0, "sex"]).strip().lower()
    if sex not in {"male", "female"}:
        raise ValueError(f"{tumor_id}: unsupported sex {sex!r}")
    return sex


def _normal_copy_number(chromosome: str, *, sex: str, cohort: str) -> float:
    if cohort == "SimClone1000":
        return 2.0  # Simulator uses diploid normal cells, including X/Y.
    if chromosome == "X":
        return 1.0 if sex == "male" else 2.0
    if chromosome == "Y":
        return 1.0 if sex == "male" else 0.0
    return 2.0


def _validated_source_tables(
    source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    missing = [name for name in REQUIRED_SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source.name}: missing source files {missing}")

    snv = pd.read_csv(source / "snv.txt", sep="\t", dtype={"chromosome_index": str})
    cna = pd.read_csv(source / "cna.txt", sep="\t", dtype={"chromosome_index": str})
    missing_snv = sorted(REQUIRED_SNV_COLUMNS.difference(snv.columns))
    missing_cna = sorted(REQUIRED_CNA_COLUMNS.difference(cna.columns))
    if missing_snv or missing_cna:
        raise ValueError(
            f"{source.name}: missing SNV columns {missing_snv}; CNA columns {missing_cna}"
        )
    if snv.empty or cna.empty:
        raise ValueError(f"{source.name}: SNV and CNA tables must be nonempty")

    snv = snv.copy()
    cna = cna.copy()
    snv["chromosome"] = snv["chromosome_index"].map(_chromosome_key)
    cna["chromosome"] = cna["chromosome_index"].map(_chromosome_key)
    for name in ("position", "alt_count", "ref_count"):
        snv[name] = _integer_array(snv[name], name=f"{source.name} SNV {name}")
    for name in (
        "start_position",
        "end_position",
        "major_cn",
        "minor_cn",
        "total_cn",
    ):
        cna[name] = _integer_array(cna[name], name=f"{source.name} CNA {name}")

    if (snv[["position", "alt_count", "ref_count"]] < 0).to_numpy().any():
        raise ValueError(f"{source.name}: negative SNV position or read count")
    if (snv["position"] == 0).any():
        raise ValueError(f"{source.name}: mutation positions must be 1-based positive")
    if (cna[["major_cn", "minor_cn", "total_cn"]] < 0).to_numpy().any():
        raise ValueError(f"{source.name}: negative copy number")
    if (cna["start_position"] <= 0).any() or (
        cna["end_position"] < cna["start_position"]
    ).any():
        raise ValueError(f"{source.name}: invalid CNA interval")
    if not np.array_equal(
        cna["total_cn"].to_numpy(),
        (cna["major_cn"] + cna["minor_cn"]).to_numpy(),
    ):
        raise ValueError(f"{source.name}: total_cn differs from major_cn + minor_cn")

    purity_tokens = (source / "purity.txt").read_text(encoding="utf-8").split()
    if len(purity_tokens) != 1:
        raise ValueError(f"{source.name}: purity.txt must contain exactly one value")
    purity = float(purity_tokens[0])
    if not np.isfinite(purity) or not 0.0 < purity <= 1.0:
        raise ValueError(f"{source.name}: invalid purity {purity_tokens[0]!r}")
    return snv, cna, purity


def _map_segments(snv: pd.DataFrame, cna: pd.DataFrame, *, tumor_id: str) -> pd.DataFrame:
    matched_parts: list[pd.DataFrame] = []
    for chromosome, snv_group in snv.groupby("chromosome", sort=False):
        segments = cna.loc[cna["chromosome"] == chromosome].sort_values(
            ["start_position", "end_position"], kind="stable"
        )
        if segments.empty:
            raise ValueError(f"{tumor_id}: no CNA segment for chromosome {chromosome}")
        starts = segments["start_position"].to_numpy(dtype=np.int64)
        ends = segments["end_position"].to_numpy(dtype=np.int64)
        if np.any(starts[1:] <= ends[:-1]):
            raise ValueError(f"{tumor_id}: overlapping CNA segments on {chromosome}")
        positions = snv_group["position"].to_numpy(dtype=np.int64)
        local = np.searchsorted(starts, positions, side="right") - 1
        clipped = np.clip(local, 0, len(ends) - 1)
        valid = (local >= 0) & (positions <= ends[clipped])
        if not np.all(valid):
            raise ValueError(
                f"{tumor_id}: SNVs outside CNA segments on {chromosome}: "
                f"{positions[~valid][:5].tolist()}"
            )
        chosen = segments.iloc[local].copy()
        chosen.index = snv_group.index
        matched_parts.append(chosen)
    return pd.concat(matched_parts).loc[snv.index]


def _canonical_table(
    source: Path,
    *,
    cohort: str,
    sex_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    tumor_id = source.name
    snv, cna, purity = _validated_source_tables(source)
    reversed_allele_segments = int((cna["major_cn"] < cna["minor_cn"]).sum())
    source_major = cna["major_cn"].to_numpy(dtype=np.int64, copy=True)
    source_minor = cna["minor_cn"].to_numpy(dtype=np.int64, copy=True)
    cna["major_cn"] = np.maximum(source_major, source_minor)
    cna["minor_cn"] = np.minimum(source_major, source_minor)
    matched = _map_segments(snv, cna, tumor_id=tumor_id)
    sex = _source_sex(tumor_id, sex_root) if cohort == "SimClone1000" else "unknown"
    chromosomes = snv["chromosome"].astype(str)
    coordinate_ids = "chr" + chromosomes + ":" + snv["position"].astype(str)
    coordinate_multiplicity = coordinate_ids.groupby(coordinate_ids, sort=False).transform(
        "size"
    )
    coordinate_occurrence = coordinate_ids.groupby(coordinate_ids, sort=False).cumcount() + 1
    mutation_ids = coordinate_ids.where(
        coordinate_multiplicity == 1,
        coordinate_ids + "#" + coordinate_occurrence.astype(str),
    )
    normal_cn = np.array(
        [_normal_copy_number(value, sex=sex, cohort=cohort) for value in chromosomes], dtype=float
    )
    if np.any(
        (1.0 - purity) * normal_cn
        + purity * matched["total_cn"].to_numpy(dtype=float)
        <= 0.0
    ):
        raise ValueError(f"{tumor_id}: nonpositive normal-plus-tumor copy denominator")

    segment_ids = (
        "chr"
        + chromosomes
        + ":"
        + matched["start_position"].astype(str)
        + "-"
        + matched["end_position"].astype(str)
    )
    table = pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "sample_id": "region1",
            "chromosome": chromosomes,
            "position": snv["position"].to_numpy(dtype=np.int64),
            "ref": ".",
            "alt": ".",
            "alt_count": snv["alt_count"].to_numpy(dtype=np.int64),
            "ref_count": snv["ref_count"].to_numpy(dtype=np.int64),
            "count_observed": 1,
            "purity": purity,
            "normal_cn": normal_cn,
            "segment_id": segment_ids,
            "segment_start": matched["start_position"].to_numpy(dtype=np.int64),
            "segment_end": matched["end_position"].to_numpy(dtype=np.int64),
            "cn_state_id": "state1",
            "cn_state_fraction": 1.0,
            "allele_a_cn": matched["major_cn"].to_numpy(dtype=np.int64),
            "allele_b_cn": matched["minor_cn"].to_numpy(dtype=np.int64),
            "allele_mode": "unphased",
        }
    )
    details = {
        "num_mutations": int(len(table)),
        "num_segments_used": int(segment_ids.nunique()),
        "purity": float(purity),
        "sex": sex,
        "x_mutations": int((chromosomes == "X").sum()),
        "duplicate_coordinate_loci": int(
            coordinate_ids.loc[coordinate_ids.duplicated(keep=False)].nunique()
        ),
        "reordered_allele_segments": reversed_allele_segments,
        "source_snv_sha256": _sha256(source / "snv.txt"),
        "source_cna_sha256": _sha256(source / "cna.txt"),
        "source_purity_sha256": _sha256(source / "purity.txt"),
    }
    return table, details


def _convert_one(
    source: Path,
    output_root: Path,
    *,
    cohort: str,
    sex_root: Path,
    force: bool,
    converter_hash: str,
) -> dict[str, object]:
    started = time.monotonic()
    destination = output_root / f"{source.name}.tsv"
    if destination.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {destination}; use --force")
    table, details = _canonical_table(source, cohort=cohort, sex_root=sex_root)
    metadata = {
        "schema": TUMOR_TXT_SCHEMA,
        "tumor_id": source.name,
        "genome_build": "GRCh37",
        "coordinate_system": "1-based-inclusive",
        "missing_value": ".",
        "source_cohort": cohort,
        "source_format": "legacy_single_region_snv_cna_purity_v1",
        "source_sex": details["sex"],
        "normal_cn_contract": ("SimClone_simulator_diploid_normal_all_chromosomes"
                               if cohort == "SimClone1000" else "legacy_unknown_sex"),
        "converter_sha256": converter_hash,
        "source_snv_sha256": details["source_snv_sha256"],
        "source_cna_sha256": details["source_cna_sha256"],
        "source_purity_sha256": details["source_purity_sha256"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{source.name}.{os.getpid()}.{time.time_ns()}.tmp.tsv"
    try:
        write_tumor_txt(temporary, table, metadata)
        loaded = load_tumor_txt(temporary)
        if loaded.tumor_id != source.name or loaded.num_mutations != len(table):
            raise AssertionError(f"{source.name}: post-write identity validation failed")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "cohort": cohort,
        "tumor_id": source.name,
        "source_dir": str(source),
        "output_file": str(destination),
        **details,
        "output_sha256": _sha256(destination),
        "elapsed_seconds": time.monotonic() - started,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows).sort_values(["cohort", "tumor_id"], kind="stable")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def _convert_cohort(
    *,
    cohort: str,
    source_root: Path,
    output_root: Path,
    sex_root: Path,
    workers: int,
    force: bool,
    converter_hash: str,
) -> list[dict[str, object]]:
    sources = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not sources:
        raise ValueError(f"{source_root}: no tumor directories")
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _convert_one,
                source,
                output_root,
                cohort=cohort,
                sex_root=sex_root,
                force=force,
                converter_hash=converter_hash,
            ): source
            for source in sources
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"conversion failed for {cohort}/{source.name}: {exc}") from exc
            if completed % 50 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {"cohort": cohort, "converted": completed, "total": len(futures)}
                    ),
                    flush=True,
                )
    _write_manifest(output_root / "conversion_manifest.tsv", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SimClone1000 and PhylogicNDT500 into CliPP2 long TSVs."
    )
    parser.add_argument("--simclone-root", type=Path, default=Path("/storage/CliPP2/SimClone1000"))
    parser.add_argument("--simclone-output", type=Path, default=Path("/storage/CliPP2/SimClone1000_TSV"))
    parser.add_argument("--phylogic-root", type=Path, default=Path("/storage/CliPP2/PhylogicNDT500"))
    parser.add_argument("--phylogic-output", type=Path, default=Path("/storage/CliPP2/PhylogicNDT500_TSV"))
    parser.add_argument("--simclone-sex-root", type=Path, default=SIMCLONE_SEX_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    converter_hash = _sha256(Path(__file__).resolve())
    all_rows: list[dict[str, object]] = []
    for cohort, source_root, output_root in (
        ("SimClone1000", args.simclone_root.resolve(), args.simclone_output.resolve()),
        ("PhylogicNDT500", args.phylogic_root.resolve(), args.phylogic_output.resolve()),
    ):
        all_rows.extend(
            _convert_cohort(
                cohort=cohort,
                source_root=source_root,
                output_root=output_root,
                sex_root=args.simclone_sex_root.resolve(),
                workers=args.workers,
                force=args.force,
                converter_hash=converter_hash,
            )
        )
    summary = {
        "schema": TUMOR_TXT_SCHEMA,
        "converter_sha256": converter_hash,
        "tumors": len(all_rows),
        "mutations": int(sum(int(row["num_mutations"]) for row in all_rows)),
        "cohorts": {
            cohort: sum(row["cohort"] == cohort for row in all_rows)
            for cohort in ("SimClone1000", "PhylogicNDT500")
        },
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
