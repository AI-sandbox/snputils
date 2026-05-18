from __future__ import annotations

from typing import Sequence

import numpy as np

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.phenotype.genobj import CovariateObject

DEFAULT_ADMIXTURE_ANCESTRY_MAP = {"0": "AFR", "1": "EUR", "2": "NAT"}


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.mean(values))
    sd = float(np.std(centered, ddof=0))
    if sd <= 0.0 or not np.isfinite(sd):
        return centered
    return centered / sd


def build_feature_layout(
    n_features: int,
    *,
    spacing_bp: int,
    span_bp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    counts = np.full(22, n_features // 22, dtype=np.int64)
    counts[: n_features % 22] += 1

    chroms: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for chrom, count in enumerate(counts.tolist(), start=1):
        if count == 0:
            continue
        start = np.arange(count, dtype=np.int64) * spacing_bp + 1
        chroms.append(np.full(count, chrom, dtype=np.int64))
        starts.append(start)
        ends.append(start + span_bp - 1)

    return (
        np.concatenate(chroms).astype(np.int64, copy=False),
        np.concatenate(starts).astype(np.int64, copy=False),
        np.concatenate(ends).astype(np.int64, copy=False),
    )


def build_lai_object(
    sample_ids: Sequence[str],
    lai: np.ndarray,
    chromosomes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    ancestry_map: dict[str, str],
) -> LocalAncestryObject:
    haplotypes = [f"{sid}.{phase}" for sid in sample_ids for phase in (0, 1)]
    return LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=list(sample_ids),
        ancestry_map=ancestry_map,
        chromosomes=np.asarray(chromosomes, dtype=np.int64),
        physical_pos=np.column_stack([starts, ends]).astype(np.int64, copy=False),
        window_sizes=np.ones(int(lai.shape[0]), dtype=np.int64),
    )


def covariate_coefficients(n_covariates: int) -> np.ndarray:
    base = np.array([0.60, -0.35, 0.25, -0.15, 0.10], dtype=np.float64)
    if n_covariates <= len(base):
        return base[:n_covariates].copy()
    extra = np.linspace(0.08, 0.02, n_covariates - len(base), dtype=np.float64)
    return np.concatenate([base, extra])


def build_synthetic_admixture_dataset(
    n_samples: int,
    n_windows: int,
    n_covariates: int,
    seed: int,
    *,
    ancestry_map: dict[str, str] | None = None,
) -> dict[str, object]:
    """Create a local-ancestry dataset with sparse quantitative trait effects.

    The generated data are intentionally small and self-contained, making them
    suitable for tutorials, tests, and benchmarks that need realistic-looking
    admixture mapping inputs without downloading external files.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_covariates < 0:
        raise ValueError("n_covariates must be non-negative.")

    ancestry_map = dict(DEFAULT_ADMIXTURE_ANCESTRY_MAP if ancestry_map is None else ancestry_map)
    rng = np.random.default_rng(seed)
    sample_ids = [f"A{i:05d}" for i in range(n_samples)]
    chromosomes, starts, ends = build_feature_layout(n_windows, spacing_bp=25_000, span_bp=4_999)

    n_ancestries = len(ancestry_map)
    lai = np.empty((n_windows, n_samples * 2), dtype=np.uint8)
    sample_props = rng.dirichlet(np.array([5.0, 3.5, 2.5], dtype=np.float64), size=n_samples)
    switch_prob = 0.035
    for sample_idx in range(n_samples):
        props = sample_props[sample_idx]
        for phase in range(2):
            hap_idx = 2 * sample_idx + phase
            state = int(rng.choice(n_ancestries, p=props))
            prev_chrom = -1
            for w, chrom in enumerate(chromosomes.tolist()):
                if chrom != prev_chrom or rng.random() < switch_prob:
                    state = int(rng.choice(n_ancestries, p=props))
                    prev_chrom = chrom
                lai[w, hap_idx] = np.uint8(state)

    n_effects = min(3, n_windows)
    effect_windows = np.unique(
        np.linspace(max(0, n_windows // 10), max(0, n_windows - 1), num=n_effects, dtype=np.int64)
    )
    effect_ancestries = np.arange(effect_windows.size, dtype=np.int64) % n_ancestries
    effect_sizes = np.array([0.30, -0.26, 0.22], dtype=np.float64)[: effect_windows.size]
    sparse_signal = np.zeros(n_samples, dtype=np.float64)
    for window_idx, ancestry_code, effect in zip(
        effect_windows.tolist(),
        effect_ancestries.tolist(),
        effect_sizes.tolist(),
    ):
        dosage = (
            (lai[window_idx, 0::2] == ancestry_code).astype(np.float64)
            + (lai[window_idx, 1::2] == ancestry_code).astype(np.float64)
        )
        sparse_signal += effect * standardize(dosage)

    global_ancestry_counts = np.zeros((n_ancestries - 1, n_samples), dtype=np.float64)
    global_chunk_size = 4096
    for start in range(0, n_windows, global_chunk_size):
        stop = min(start + global_chunk_size, n_windows)
        chunk = lai[start:stop]
        maternal = chunk[:, 0::2]
        paternal = chunk[:, 1::2]
        for ancestry_code in range(n_ancestries - 1):
            global_ancestry_counts[ancestry_code] += (
                np.sum(maternal == ancestry_code, axis=0, dtype=np.float64)
                + np.sum(paternal == ancestry_code, axis=0, dtype=np.float64)
            )
    global_ancestry = np.column_stack(
        [
            standardize(global_ancestry_counts[ancestry_code] / (2.0 * n_windows))
            for ancestry_code in range(n_ancestries - 1)
        ]
    )

    if n_covariates > 0:
        covar_names = [f"COV{i + 1}" for i in range(n_covariates)]
        covar_matrix = np.zeros((n_samples, n_covariates), dtype=np.float64)
        n_global_covars = min(n_covariates, global_ancestry.shape[1])
        if n_global_covars:
            covar_matrix[:, :n_global_covars] = global_ancestry[:, :n_global_covars]
        for idx in range(n_global_covars, n_covariates):
            covar_matrix[:, idx] = standardize(rng.normal(size=n_samples))
    else:
        covar_names = []
        covar_matrix = np.empty((n_samples, 0), dtype=np.float64)

    phenotype = 3.0 + sparse_signal
    if n_covariates > 0:
        phenotype += covar_matrix @ covariate_coefficients(n_covariates)
    phenotype += rng.normal(scale=2.5, size=n_samples)

    return {
        "sample_ids": sample_ids,
        "phenotype": phenotype.astype(np.float64, copy=False),
        "covar_names": covar_names,
        "covar_matrix": covar_matrix,
        "covariates": (
            CovariateObject(sample_ids, covar_matrix, covariate_names=covar_names)
            if n_covariates > 0
            else None
        ),
        "laiobj": build_lai_object(sample_ids, lai, chromosomes, starts, ends, ancestry_map),
        "effect_windows": effect_windows,
        "effect_ancestries": effect_ancestries,
        "effect_sizes": effect_sizes,
    }
