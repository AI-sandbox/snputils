from __future__ import annotations

from typing import cast, Literal, Optional, overload, Union

import numpy as np


GenotypeMode = Literal["auto", "dosage", "phased"]
ExplicitGenotypeMode = Literal["dosage", "phased"]


@overload
def normalize_genotype_mode(
    genotype_mode: str,
    *,
    allow_auto: Literal[False],
) -> ExplicitGenotypeMode: ...


@overload
def normalize_genotype_mode(
    genotype_mode: str,
    *,
    allow_auto: Literal[True] = True,
) -> GenotypeMode: ...


def normalize_genotype_mode(
    genotype_mode: str,
    *,
    allow_auto: bool = True,
) -> GenotypeMode:
    """Validate and normalize a requested genotype representation."""
    if not isinstance(genotype_mode, str):
        raise TypeError("genotype_mode must be a string.")

    normalized = genotype_mode.strip().lower()
    allowed = {"dosage", "phased"}
    if allow_auto:
        allowed.add("auto")
    if normalized not in allowed:
        choices = "'auto', 'dosage', or 'phased'" if allow_auto else "'dosage' or 'phased'"
        raise ValueError(f"genotype_mode must be {choices}.")
    return cast(GenotypeMode, normalized)


def sum_diploid_alleles(
    first: np.ndarray,
    second: np.ndarray,
    *,
    missing_value: Union[int, float] = -1,
    dtype: np.dtype = np.int8,
    missing_as_haploid: Optional[Union[bool, np.ndarray]] = None,
) -> np.ndarray:
    """Sum two allele-call arrays, preserving missing calls as one sentinel.

    If `missing_as_haploid` is provided, rows marked True treat one missing allele
    as haploid-like:
        0 + missing -> 0
        1 + missing -> 2
        missing + missing -> missing
    """
    first_arr = np.asarray(first)
    second_arr = np.asarray(second)

    summed = (first_arr + second_arr).astype(dtype, copy=False)
    missing = (first_arr < 0) | (second_arr < 0)

    if missing_as_haploid is not None and np.any(missing_as_haploid):
        haploid_mask = np.asarray(missing_as_haploid, dtype=bool)

        if haploid_mask.ndim == 0:
            haploid_mask = np.full(first_arr.shape[:1], bool(haploid_mask), dtype=bool)

        while haploid_mask.ndim < missing.ndim:
            haploid_mask = haploid_mask[..., None]

        one_missing_haploid = haploid_mask & ((first_arr < 0) ^ (second_arr < 0))
        if np.any(one_missing_haploid):
            summed = summed.copy()
            present = np.where(first_arr >= 0, first_arr, second_arr)
            summed[one_missing_haploid] = (
                2 * present[one_missing_haploid]
            ).astype(dtype, copy=False)

            missing = missing & ~one_missing_haploid

    if np.any(missing):
        if summed.base is not None or not summed.flags.writeable:
            summed = summed.copy()
        summed[missing] = missing_value

    return summed


def sum_diploid_genotypes(
    genotypes: np.ndarray,
    *,
    missing_value: Union[int, float] = -1,
    dtype: np.dtype = np.int8,
    missing_as_haploid: Optional[Union[bool, np.ndarray]] = None,
) -> np.ndarray:
    """Sum a final diploid allele axis, preserving missing calls as one sentinel."""
    gt = np.asarray(genotypes)
    if gt.ndim < 1 or gt.shape[-1] != 2:
        raise ValueError("Diploid genotype arrays must have a final axis of length 2.")

    return sum_diploid_alleles(
        gt[..., 0],
        gt[..., 1],
        missing_value=missing_value,
        dtype=dtype,
        missing_as_haploid=missing_as_haploid,
    )
