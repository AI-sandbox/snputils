from __future__ import annotations

from typing import Union

import numpy as np


def _missing_mask(values: np.ndarray, before: Union[int, float, str]) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        if isinstance(before, (float, np.floating)) and np.isnan(before):
            return np.isnan(values)
        if isinstance(before, (int, np.integer, float, np.floating)) and float(before) < 0:
            return values < 0
    return values == before


def _native_missing_code(after: Union[int, float, str]) -> np.int32:
    if isinstance(after, (int, np.integer)):
        return np.int32(after)
    if isinstance(after, (float, np.floating)) and float(after).is_integer():
        return np.int32(after)
    return np.int32(-9)


def phased_to_hardcalls(
    calldata_gt: np.ndarray,
    *,
    rename_missing_values: bool,
    before: Union[int, float, str],
    after: Union[int, float, str],
) -> np.ndarray:
    gt = np.asarray(calldata_gt)
    if gt.ndim not in (2, 3):
        raise ValueError("`calldata_gt` must be a 2D or 3D array.")

    missing_code = _native_missing_code(after)
    hardcalls = np.asarray(gt, dtype=np.int16)

    if gt.ndim == 3:
        missing = np.any(_missing_mask(hardcalls, before), axis=2) if rename_missing_values else None
        hardcalls = hardcalls.sum(axis=2, dtype=np.int16)
    else:
        missing = _missing_mask(hardcalls, before) if rename_missing_values else None

    if missing is not None and np.any(missing):
        hardcalls = hardcalls.copy()
        hardcalls[missing] = missing_code

    return hardcalls.astype(np.int8, copy=False)


def phased_to_flat_alleles(
    calldata_gt: np.ndarray,
    *,
    rename_missing_values: bool,
    before: Union[int, float, str],
    after: Union[int, float, str],
) -> np.ndarray:
    gt = np.asarray(calldata_gt)
    if gt.ndim != 3:
        raise ValueError("`calldata_gt` must be a 3D array to write phased alleles.")

    alleles = np.asarray(gt, dtype=np.int16)

    if rename_missing_values:
        missing = _missing_mask(alleles, before)
        if np.any(missing):
            alleles = alleles.copy()
            alleles[missing] = _native_missing_code(after)

    num_variants, num_samples, num_alleles = alleles.shape
    return alleles.astype(np.int32, copy=False).reshape(num_variants, num_samples * num_alleles)
