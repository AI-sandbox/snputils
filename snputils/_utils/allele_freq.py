from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np


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
    calldata_gt: np.ndarray,
    sample_labels: Sequence[Any],
    *,
    ancestry: Optional[Union[str, int]] = None,
    calldata_lai: Optional[np.ndarray] = None,
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
    gt = np.asarray(calldata_gt)
    if gt.ndim not in (2, 3):
        raise ValueError("'calldata_gt' must be 2D or 3D array")

    n_snps = gt.shape[0]
    n_samples = gt.shape[1]

    sample_labels = np.asarray(sample_labels)
    if sample_labels.ndim != 1:
        sample_labels = sample_labels.ravel()
    if sample_labels.shape[0] != n_samples:
        raise ValueError("'sample_labels' must have length equal to the number of samples in `calldata_gt`.")

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

    # Compute alt allele counts and haplotype counts per SNP and sample
    if gt.ndim == 3:
        # (n_snps, n_samples, 2)
        g = gt.astype(float)
        g[g < 0] = np.nan
        alt_counts_per_sample = np.nansum(g, axis=2)
        hap_count_per_sample = 2 - np.sum(np.isnan(g), axis=2)
    else:
        # (n_snps, n_samples)
        g = gt.astype(float)
        g[g < 0] = np.nan
        all_nan = np.all(np.isnan(g))
        max_val = np.nan if all_nan else np.nanmax(g)
        if all_nan:
            hap_count_per_sample = np.zeros_like(g)
            alt_counts_per_sample = np.zeros_like(g)
        elif max_val <= 1:
            # Haploid-style 2D calls
            hap_count_per_sample = np.where(np.isnan(g), 0.0, 1.0)
            alt_counts_per_sample = np.where(np.isnan(g), 0.0, g)
        else:
            # Diploid dosage-style 2D calls
            hap_count_per_sample = np.where(np.isnan(g), 0.0, 2.0)
            alt_counts_per_sample = np.where(np.isnan(g), 0.0, g)

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
