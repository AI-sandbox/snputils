from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


def _membership_matrix(pop_indices: np.ndarray, n_pops: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a sample-by-population matrix for BLAS-backed population aggregation.
    """
    membership = np.zeros((pop_indices.size, n_pops), dtype=float)
    membership[np.arange(pop_indices.size), pop_indices] = 1.0
    if weights is not None:
        membership *= np.asarray(weights, dtype=float)[:, np.newaxis]
    return membership


def _dot_by_chunks(
    values: np.ndarray,
    membership: np.ndarray,
    *,
    missing_is_negative: bool = False,
    chunk_rows: int = 100_000,
) -> np.ndarray:
    """
    Multiply a SNP-by-sample matrix by a sample-by-population matrix without
    materializing a full float copy of large integer genotype arrays.
    """
    n_snps = values.shape[0]
    out = np.empty((n_snps, membership.shape[1]), dtype=float)
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, n_snps, chunk_rows):
        end = min(start + chunk_rows, n_snps)
        chunk = values[start:end]
        if missing_is_negative:
            chunk = np.where(chunk >= 0, chunk, 0)
        out[start:end] = chunk.astype(float, copy=False) @ membership
    return out


def _aggregate_2d_integer_genotypes(
    gt: np.ndarray,
    pop_indices: np.ndarray,
    pops: np.ndarray,
    *,
    pseudohaploid: Union[bool, int] = False,
    force_diploid: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Fast path for 2D integer dosage/haploid calls.

    The generic path below converts the whole genotype matrix to float and
    builds per-sample count matrices. For large complete PLINK dosages this is
    unnecessarily expensive; a population membership matrix lets BLAS aggregate
    all populations in one pass.
    """
    n_snps, n_samples = gt.shape
    n_pops = pops.size
    min_val = gt.min(initial=0)
    max_val = gt.max(initial=0)
    has_missing = min_val < 0

    if max_val <= 1 and not force_diploid:
        ploidy_per_sample = np.ones(n_samples, dtype=float)
        alt_weights = None
    else:
        ploidy_per_sample = np.full(n_samples, 2.0, dtype=float)
        alt_weights = None
        if pseudohaploid is not False:
            n_test = 1000 if pseudohaploid is True else int(pseudohaploid)
            n_check = min(n_snps, n_test)
            g_check = gt[:n_check, :]
            is_pseudohaploid = np.sum(g_check == 1, axis=0) == 0
            if np.any(is_pseudohaploid):
                ploidy_per_sample[is_pseudohaploid] = 1.0
                alt_weights = np.ones(n_samples, dtype=float)
                alt_weights[is_pseudohaploid] = 0.5

    alt_membership = _membership_matrix(pop_indices, n_pops, alt_weights)
    alt_sum = _dot_by_chunks(gt, alt_membership, missing_is_negative=has_missing)

    if has_missing:
        count_membership = _membership_matrix(pop_indices, n_pops, ploidy_per_sample)
        called = _dot_by_chunks(gt >= 0, count_membership)
        counts = np.rint(called).astype(np.int64, copy=False)
    else:
        counts_per_pop = np.bincount(pop_indices, weights=ploidy_per_sample, minlength=n_pops)
        counts = np.broadcast_to(np.rint(counts_per_pop).astype(np.int64), (n_snps, n_pops)).copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        afs = np.divide(alt_sum, counts, out=np.full_like(alt_sum, np.nan, dtype=float), where=counts > 0)

    return afs, counts, pops.tolist()


def _normalize_lai_for_mask(
    calldata_lai: np.ndarray,
    *,
    n_snps: int,
    n_samples: int,
) -> np.ndarray:
    """
    Normalize LAI to 3D shape (n_snps, n_samples, 2) for haplotype masking.
    """
    lai = np.asarray(calldata_lai)
    if lai.ndim == 2:
        try:
            lai = lai.reshape(n_snps, n_samples, 2)
        except Exception as exc:
            raise ValueError(
                "LAI shape is incompatible with genotypes. "
                "Expected (n_snps, n_samples*2) to reshape."
            ) from exc
    if lai.ndim != 3:
        raise ValueError("LAI must be 3D (n_snps, n_samples, 2) or 2D (n_snps, n_samples*2).")
    if lai.shape != (n_snps, n_samples, 2):
        raise ValueError(
            "LAI shape is incompatible with genotypes. "
            f"Expected {(n_snps, n_samples, 2)}, got {lai.shape}."
        )
    return lai


def aggregate_pop_allele_freq(
    genotypes: np.ndarray,
    sample_labels: Sequence[Any],
    *,
    ancestry: Optional[Union[str, int]] = None,
    calldata_lai: Optional[np.ndarray] = None,
    pseudohaploid: Union[bool, int] = False,
    force_diploid_2d: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Aggregate sample-level genotypes into per-population allele frequencies.

    Genotype encoding supported:
        - 3D (n_snps, n_samples, 2): haplotype calls in {0,1}, missing as negative or NaN
        - 2D (n_snps, n_samples): diploid dosages in {0,1,2} or haploid in {0,1}; missing as negative or NaN

    Returns:
        afs: float array (n_snps, n_pops)
        counts: int array (n_snps, n_pops) with called haplotypes per SNP and population
        pops: list of population labels in column order
    """
    gt = np.asarray(genotypes)
    if gt.ndim not in (2, 3):
        raise ValueError("'genotypes' must be 2D or 3D array")

    n_snps = gt.shape[0]
    n_samples = gt.shape[1]

    sample_labels = np.asarray(sample_labels)
    if sample_labels.ndim != 1:
        sample_labels = sample_labels.ravel()
    if sample_labels.shape[0] != n_samples:
        raise ValueError("'sample_labels' must have length equal to the number of samples in `genotypes`.")

    pops, pop_indices = np.unique(sample_labels, return_inverse=True)
    n_pops = pops.size

    if ancestry is not None:
        if gt.ndim != 3:
            raise ValueError("Ancestry-specific masking requires 3D genotype array (n_snps, n_samples, 2).")
        if calldata_lai is None:
            raise ValueError("Ancestry-specific masking requires SNP-level LAI (`calldata_lai`).")
        lai = _normalize_lai_for_mask(calldata_lai, n_snps=n_snps, n_samples=n_samples)
        mask = lai.astype(str) == str(ancestry)
        gt = gt.astype(float, copy=True)
        gt[~mask] = np.nan

    if ancestry is None and gt.ndim == 2 and np.issubdtype(gt.dtype, np.integer):
        return _aggregate_2d_integer_genotypes(
            gt,
            pop_indices,
            pops,
            pseudohaploid=pseudohaploid,
            force_diploid=force_diploid_2d,
        )

    # Compute alt allele counts and haplotype counts per SNP and sample
    if gt.ndim == 3:
        # (n_snps, n_samples, 2)
        g = gt.astype(float)
        g[g < 0] = np.nan
        alt_counts_per_sample = np.nansum(g, axis=2)
        hap_count_per_sample = 2 - np.sum(np.isnan(g), axis=2)

        if pseudohaploid is not False:
            n_test = 1000 if pseudohaploid is True else int(pseudohaploid)
            n_check = min(n_snps, n_test)
            g_check = alt_counts_per_sample[:n_check, :]
            # Sample is pseudohaploid if it has no heterozygote calls (alt count == 1)
            is_pseudohaploid = np.nansum(g_check == 1.0, axis=0) == 0
            mask = is_pseudohaploid[np.newaxis, :] & (hap_count_per_sample == 2.0)
            hap_count_per_sample[mask] = 1.0
            alt_counts_per_sample[mask] /= 2.0
    else:
        # (n_snps, n_samples)
        g = gt.astype(float)
        g[g < 0] = np.nan
        all_nan = np.all(np.isnan(g))
        max_val = np.nan if all_nan else np.nanmax(g)
        if all_nan:
            hap_count_per_sample = np.zeros_like(g)
            alt_counts_per_sample = np.zeros_like(g)
        elif max_val <= 1 and not force_diploid_2d:
            # Haploid-style 2D calls
            hap_count_per_sample = np.where(np.isnan(g), 0.0, 1.0)
            alt_counts_per_sample = np.where(np.isnan(g), 0.0, g)
        else:
            # Diploid dosage-style 2D calls
            hap_count_per_sample = np.where(np.isnan(g), 0.0, 2.0)
            alt_counts_per_sample = np.where(np.isnan(g), 0.0, g)

            if pseudohaploid is not False:
                n_test = 1000 if pseudohaploid is True else int(pseudohaploid)
                n_check = min(n_snps, n_test)
                g_check = g[:n_check, :]
                is_pseudohaploid = np.nansum(g_check == 1.0, axis=0) == 0
                mask = is_pseudohaploid[np.newaxis, :] & (hap_count_per_sample == 2.0)
                hap_count_per_sample[mask] = 1.0
                alt_counts_per_sample[mask] /= 2.0

    afs = np.zeros((n_snps, n_pops), dtype=float)
    counts = np.zeros((n_snps, n_pops), dtype=float)
    for pop_idx in range(n_pops):
        cols = np.where(pop_indices == pop_idx)[0]
        if cols.size == 0:
            continue
        counts[:, pop_idx] = np.sum(hap_count_per_sample[:, cols], axis=1)
        alt_sum = np.sum(alt_counts_per_sample[:, cols], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            afs[:, pop_idx] = np.where(counts[:, pop_idx] > 0, alt_sum / counts[:, pop_idx], np.nan)

    return afs, counts.astype(np.int64), pops.tolist()
