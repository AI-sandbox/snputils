from __future__ import annotations

from typing import Union

import numpy as np


def sum_diploid_alleles(
    first: np.ndarray,
    second: np.ndarray,
    *,
    missing_value: Union[int, float] = -1,
    dtype: np.dtype = np.int8,
) -> np.ndarray:
    """Sum two allele-call arrays, preserving missing calls as one sentinel."""
    first_arr = np.asarray(first)
    second_arr = np.asarray(second)
    summed = (first_arr + second_arr).astype(dtype, copy=False)
    missing = (first_arr < 0) | (second_arr < 0)
    if np.any(missing):
        summed = summed.copy()
        summed[missing] = missing_value
    return summed


def sum_diploid_genotypes(
    genotypes: np.ndarray,
    *,
    missing_value: Union[int, float] = -1,
    dtype: np.dtype = np.int8,
) -> np.ndarray:
    """Sum a final diploid allele axis, preserving missing calls as one sentinel."""
    gt = np.asarray(genotypes)
    if gt.ndim < 1 or gt.shape[-1] != 2:
        raise ValueError("Diploid genotype arrays must have a final axis of length 2.")
    summed = gt.sum(axis=-1, dtype=dtype)
    missing = np.any(gt < 0, axis=-1)
    if np.any(missing):
        summed = summed.copy()
        summed[missing] = missing_value
    return summed
