"""File-backed and command-line integration for Zigo-based sex checking."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from snputils.snp import sex_check
from snputils.snp._sex_check import _normalize_chromosome
from snputils.snp.genobj import SNPObject
from snputils.snp.io.read import (
    BCFReader,
    BEDReader,
    BGENReader,
    PGENReader,
    SNPReader,
    VCFReader,
)


log = logging.getLogger(__name__)


def _chromosome_x_indices(chromosomes: Optional[np.ndarray]) -> np.ndarray:
    if chromosomes is None or np.asarray(chromosomes).size == 0:
        raise ValueError("The genotype input does not contain chromosome metadata.")
    values = np.asarray(chromosomes, dtype=object).ravel()
    mask = np.fromiter(
        (_normalize_chromosome(value) in {"x", "23"} for value in values),
        dtype=bool,
        count=values.shape[0],
    )
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        available = list(dict.fromkeys(str(value) for value in values))
        preview = available[:10]
        suffix = "..." if len(available) > len(preview) else ""
        raise ValueError(
            "No chromosome-X variants were found. "
            f"Observed chromosome labels: {preview}{suffix}."
        )
    return indexes


def _read_x_only_snp(path: Union[str, Path], *, assume_x: bool = False) -> SNPObject:
    reader = SNPReader(Path(path))
    if isinstance(reader, BGENReader):
        raise ValueError(
            "BGEN input stores genotype probabilities rather than the hard calls required "
            "for sex checking. Convert or hard-call the data before running this command."
        )

    genotype_mode = "dosage" if isinstance(reader, BEDReader) else "auto"
    if assume_x:
        return reader.read(genotype_mode=genotype_mode)

    if isinstance(reader, VCFReader):
        metadata = reader.read(fields=["CHROM"], samples=[])
        x_indexes = _chromosome_x_indices(metadata.variants_chrom)
        x_labels = list(
            dict.fromkeys(str(metadata.variants_chrom[index]) for index in x_indexes)
        )
        if len(x_labels) == 1:
            return reader.read(region=x_labels[0], genotype_mode=genotype_mode)

        # Mixed X naming within one VCF is unusual. Reading once and letting the
        # in-memory API select every accepted label avoids duplicate sample joins.
        return reader.read(genotype_mode=genotype_mode)

    if isinstance(reader, (BEDReader, PGENReader, BCFReader)):
        metadata = reader.read(fields=["#CHROM"])
        x_indexes = _chromosome_x_indices(metadata.variants_chrom)
        return reader.read(variant_idxs=x_indexes, genotype_mode=genotype_mode)

    raise ValueError(f"Sex checking does not support {type(reader).__name__} file input.")


def _canonical_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower().lstrip("#"))


def read_reported_sex(path: Union[str, Path]) -> dict[str, Any]:
    """Read a headered sample-sex table into a mapping keyed by sample ID."""
    table = pd.read_csv(path, sep=None, engine="python", dtype=str)
    if table.empty:
        raise ValueError("Reported-sex table is empty.")

    canonical = {_canonical_column_name(column): column for column in table.columns}
    id_column = next(
        (
            canonical[name]
            for name in ("iid", "individualid", "indid", "sample", "sampleid")
            if name in canonical
        ),
        None,
    )
    sex_column = next(
        (
            canonical[name]
            for name in ("sex", "gender", "reportedsex", "pedsex")
            if name in canonical
        ),
        None,
    )
    if id_column is None or sex_column is None:
        raise ValueError(
            "Reported-sex table must contain a sample column (IID, Individual ID, or sample) "
            "and a sex column (SEX or Gender)."
        )

    raw_sample_ids = table[id_column]
    sample_ids = raw_sample_ids.astype(str).str.strip()
    if raw_sample_ids.isna().any() or sample_ids.eq("").any() or sample_ids.duplicated().any():
        raise ValueError("Reported-sex sample IDs must be non-empty and unique.")
    return dict(zip(sample_ids, table[sex_column]))


def _write_results(results: pd.DataFrame, results_path: Union[str, Path]) -> Path:
    output = Path(results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, sep="\t", index=False, na_rep="NA", compression="infer")
    return output


def run_sex_check(
    snp_source: Union[str, Path, SNPObject],
    reported_sex: Optional[Union[Mapping[str, Any], Sequence[Any], np.ndarray]] = None,
    *,
    results_path: Optional[Union[str, Path]] = None,
    assume_x: bool = False,
    low_information_threshold: int = 500,
) -> pd.DataFrame:
    """Run sex checking from a genotype path or an in-memory SNPObject.

    File-backed input uses snputils readers and loads chromosome-X genotypes only
    for BED, PGEN, BCF, and consistently named VCF inputs. BGEN probabilities are
    not hard-called implicitly.
    """
    if isinstance(snp_source, SNPObject):
        snpobj = snp_source
    else:
        snpobj = _read_x_only_snp(snp_source, assume_x=assume_x)

    results = sex_check(
        snpobj,
        reported_sex=reported_sex,
        assume_x=assume_x,
        low_information_threshold=low_information_threshold,
    )
    if results_path is not None:
        output = _write_results(results, results_path)
        log.info("Sex-check results written to %s", output)
    return results


def run_sex_check_command(args) -> int:
    reported = read_reported_sex(args.sex_path) if args.sex_path is not None else None
    run_sex_check(
        args.snp_path,
        reported_sex=reported,
        results_path=args.results_path,
        assume_x=args.assume_x,
        low_information_threshold=args.low_information_threshold,
    )
    return 0


__all__ = ["read_reported_sex", "run_sex_check"]
