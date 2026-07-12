from __future__ import annotations

from typing import Optional, Tuple, Union

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


def validate_hardcall_values(
    genotypes: np.ndarray,
    *,
    allowed_values: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    if np.asarray(genotypes).dtype.kind == "c":
        raise ValueError("Hard-call genotypes must be real numeric values.")
    try:
        numeric = np.asarray(genotypes, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hard-call genotypes must be numeric values.") from exc
    if np.any(np.isinf(numeric)):
        raise ValueError("Hard-call genotypes cannot contain infinite values.")

    called = np.isfinite(numeric) & (numeric >= 0)
    if allowed_values is None:
        valid = numeric == np.floor(numeric)
        expected = "nonnegative integer allele indexes"
    else:
        valid = np.isin(numeric, allowed_values)
        expected = "hard calls encoded as " + ", ".join(map(str, allowed_values))
    if np.any(called & ~valid):
        index = tuple(np.argwhere(called & ~valid)[0])
        value = numeric[index]
        raise ValueError(f"Expected {expected}; found invalid value {value!r}.")
    return numeric


def phased_to_hardcalls(
    genotypes: np.ndarray,
    *,
    rename_missing_values: bool,
    before: Union[int, float, str],
    after: Union[int, float, str],
) -> np.ndarray:
    gt = np.asarray(genotypes)
    if gt.ndim not in (2, 3):
        raise ValueError("`genotypes` must be a 2D or 3D array.")

    missing_code = _native_missing_code(after)
    numeric = validate_hardcall_values(
        gt,
        allowed_values=(0, 1) if gt.ndim == 3 else (0, 1, 2),
    )

    if gt.ndim == 3:
        if rename_missing_values:
            missing_entries = _missing_mask(numeric, before) | ~np.isfinite(numeric)
            hardcalls = np.where(missing_entries, 0, numeric).astype(np.int16)
            missing = np.any(missing_entries, axis=2)
        else:
            if np.any(~np.isfinite(numeric)):
                raise ValueError("Non-finite hard calls require missing-value renaming.")
            hardcalls = numeric.astype(np.int16)
            missing = None
        hardcalls = hardcalls.sum(axis=2, dtype=np.int16)
    else:
        if rename_missing_values:
            missing = _missing_mask(numeric, before) | ~np.isfinite(numeric)
            hardcalls = np.where(missing, 0, numeric).astype(np.int16)
        else:
            if np.any(~np.isfinite(numeric)):
                raise ValueError("Non-finite hard calls require missing-value renaming.")
            hardcalls = numeric.astype(np.int16)
            missing = None

    if missing is not None and np.any(missing):
        hardcalls = hardcalls.copy()
        hardcalls[missing] = missing_code

    return hardcalls.astype(np.int8, copy=False)


def phased_to_flat_alleles(
    genotypes: np.ndarray,
    *,
    rename_missing_values: bool,
    before: Union[int, float, str],
    after: Union[int, float, str],
) -> np.ndarray:
    gt = np.asarray(genotypes)
    if gt.ndim != 3:
        raise ValueError("`genotypes` must be a 3D array to write phased alleles.")

    numeric = validate_hardcall_values(gt)

    if rename_missing_values:
        missing = _missing_mask(numeric, before) | ~np.isfinite(numeric)
        alleles = np.where(missing, _native_missing_code(after), numeric).astype(np.int32)
    else:
        if np.any(~np.isfinite(numeric)):
            raise ValueError("Non-finite hard calls require missing-value renaming.")
        alleles = numeric.astype(np.int32)

    num_variants, num_samples, num_alleles = alleles.shape
    return alleles.reshape(num_variants, num_samples * num_alleles)
