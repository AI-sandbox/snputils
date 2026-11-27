import warnings

import numpy as np


def snp_indices_by_chromosome(chromosomes: np.ndarray, positions: np.ndarray) -> dict[object, np.ndarray]:
    """Group original SNP indices by chromosome, excluding rows with missing coordinates."""
    import pandas as pd

    labels = np.atleast_1d(np.asarray(chromosomes))
    if np.ma.is_masked(chromosomes):
        labels = np.where(chromosomes.mask, None, labels)
    labels = pd.Categorical(labels)
    unique_chrom = labels.unique()
    missing_positions = pd.isna(np.asarray(positions)) | np.ma.getmask(positions)

    missing_fields = []
    if unique_chrom.isna().any():
        missing_fields.append("missing chromosome labels")
    if missing_positions.any():
        missing_fields.append("missing positions")
    if missing_fields:
        warnings.warn("Skipping SNPs with " + " or ".join(missing_fields) + ".", stacklevel=3)

    valid_positions = ~missing_positions
    return {
        chrom: np.flatnonzero((labels == chrom) & valid_positions)
        for chrom in unique_chrom.dropna()
    }


def known_lai_mask(lai: np.ndarray) -> np.ndarray:
    values = np.asarray(lai)
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype(float, copy=False)
        return np.isfinite(numeric) & (numeric >= 0)

    normalized = np.char.lower(np.char.strip(values.astype(str)))
    return ~np.isin(normalized, ("", ".", "-1", "-1.0", "na", "nan", "none"))


def known_lai_values(lai: np.ndarray) -> np.ndarray:
    values = np.asarray(lai).ravel()
    if np.issubdtype(values.dtype, np.unsignedinteger):
        return values
    return values[known_lai_mask(values)]
