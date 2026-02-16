from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def write_msp(
    path: Path,
    sample_ids: Sequence[str],
    lai: np.ndarray,
    chromosomes: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    ancestry_map: Dict[int, str],
) -> None:
    haplotypes = [f"{sid}.{phase}" for sid in sample_ids for phase in (0, 1)]
    comment = "#Subpopulation order/codes: " + "\t".join(
        f"{ancestry_map[k]}={k}" for k in sorted(ancestry_map)
    )
    header = ["#chm", "spos", "epos", "sgpos", "egpos", "n snps"] + haplotypes

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(comment + "\n")
        handle.write("\t".join(header) + "\n")
        for w in range(lai.shape[0]):
            row = [
                str(int(chromosomes[w])),
                str(int(starts[w])),
                str(int(ends[w])),
                str(int(starts[w])),
                str(int(ends[w])),
                "1",
            ]
            handle.write("\t".join(row) + "\t" + "\t".join(map(str, lai[w].tolist())) + "\n")


def make_small_dataset(
    n_samples: int,
    n_windows: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:04d}" for i in range(n_samples)]
    chromosomes = ((np.arange(n_windows) % 22) + 1).astype(np.int64)
    starts = (np.arange(n_windows) * 1000 + 1).astype(np.int64)
    ends = starts + 999
    ancestry_map = {0: "AFR", 1: "EUR", 2: "NAT"}
    lai = rng.integers(0, 3, size=(n_windows, n_samples * 2), dtype=np.uint8)
    return sample_ids, lai, chromosomes, starts, ends, ancestry_map


def make_synthetic_dataset(
    n_samples: int, n_windows: int, seed: int
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:05d}" for i in range(n_samples)]

    y = (rng.random(n_samples) < 0.45).astype(np.int8)
    if int(np.sum(y)) == 0:
        y[0] = 1
    if int(np.sum(y)) == n_samples:
        y[0] = 0

    chromosomes = ((np.arange(n_windows) % 22) + 1).astype(np.int64)
    starts = (np.arange(n_windows) * 5000 + 1).astype(np.int64)
    ends = starts + 999

    ancestry_map = {0: "AFR", 1: "EUR", 2: "NAT"}
    lai = np.empty((n_windows, n_samples * 2), dtype=np.uint8)

    for w in range(n_windows):
        base = rng.dirichlet(np.array([4.0, 3.0, 3.0], dtype=float))
        is_effect_window = (w % 17) == 0
        for s in range(n_samples):
            probs = base.copy()
            if is_effect_window and int(y[s]) == 1:
                probs[0] = min(0.90, probs[0] + 0.12)
                rem = 1.0 - probs[0]
                denom = probs[1] + probs[2]
                if denom <= 0.0:
                    probs[1] = rem
                    probs[2] = 0.0
                else:
                    probs[1] = probs[1] / denom * rem
                    probs[2] = probs[2] / denom * rem
            lai[w, 2 * s] = np.uint8(rng.choice([0, 1, 2], p=probs))
            lai[w, 2 * s + 1] = np.uint8(rng.choice([0, 1, 2], p=probs))

    return sample_ids, y, lai, chromosomes, starts, ends, ancestry_map


def make_synthetic_quantitative_dataset(
    n_samples: int, n_windows: int, seed: int
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """Generate a synthetic dataset with a continuous (quantitative) phenotype.

    The phenotype is drawn from a linear model:
      y = beta0 + beta1 * mean_dosage_at_effect_windows + noise

    Effect windows are every 17th window (same pattern as the binary dataset).
    """
    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:05d}" for i in range(n_samples)]

    chromosomes = ((np.arange(n_windows) % 22) + 1).astype(np.int64)
    starts = (np.arange(n_windows) * 5000 + 1).astype(np.int64)
    ends = starts + 999

    ancestry_map = {0: "AFR", 1: "EUR", 2: "NAT"}
    lai = rng.integers(0, 3, size=(n_windows, n_samples * 2), dtype=np.uint8)

    # Identify effect windows and compute mean dosage for ancestry 0 (AFR).
    effect_windows = [w for w in range(n_windows) if (w % 17) == 0]
    if effect_windows:
        dosage_sum = np.zeros(n_samples, dtype=np.float64)
        for w in effect_windows:
            for s in range(n_samples):
                dosage_sum[s] += int(lai[w, 2 * s] == 0) + int(lai[w, 2 * s + 1] == 0)
        mean_dosage = dosage_sum / len(effect_windows)
    else:
        mean_dosage = np.zeros(n_samples, dtype=np.float64)

    beta0 = 5.0
    beta1 = 1.5
    noise = rng.normal(0.0, 1.0, size=n_samples)
    y = beta0 + beta1 * mean_dosage + noise

    return sample_ids, y, lai, chromosomes, starts, ends, ancestry_map


def make_synthetic_dataset_with_covariates(
    n_samples: int,
    n_windows: int,
    n_covariates: int,
    seed: int,
) -> Tuple[
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, str],
    List[str],
    np.ndarray,
    List[str],
    List[str],
]:
    """Generate LAI + binary/quantitative phenotypes + confounded covariates.

    Covariates are correlated with ancestry dosage at effect windows and with
    both binary and quantitative phenotypes so covariate-adjusted and unadjusted
    mapping results differ in a detectable way.
    """
    if n_covariates < 1:
        raise ValueError("n_covariates must be >= 1")

    rng = np.random.default_rng(seed)
    sample_ids = [f"S{i:05d}" for i in range(n_samples)]

    chromosomes = ((np.arange(n_windows) % 22) + 1).astype(np.int64)
    starts = (np.arange(n_windows) * 5000 + 1).astype(np.int64)
    ends = starts + 999

    ancestry_map = {0: "AFR", 1: "EUR", 2: "NAT"}
    lai = rng.integers(0, 3, size=(n_windows, n_samples * 2), dtype=np.uint8)

    effect_windows = [w for w in range(n_windows) if (w % 11) == 0]
    if effect_windows:
        dosage_anchor = np.zeros(n_samples, dtype=np.float64)
        for w in effect_windows:
            dosage_anchor += (lai[w, 0::2] == 0).astype(np.float64) + (lai[w, 1::2] == 0).astype(np.float64)
        dosage_anchor = dosage_anchor / float(len(effect_windows))
    else:
        dosage_anchor = np.zeros(n_samples, dtype=np.float64)

    anchor_centered = dosage_anchor - np.mean(dosage_anchor)
    anchor_std = np.std(anchor_centered)
    if anchor_std > 0.0:
        anchor = anchor_centered / anchor_std
    else:
        anchor = anchor_centered

    covar_matrix = np.zeros((n_samples, n_covariates), dtype=np.float64)
    covar_matrix[:, 0] = 0.9 * anchor + rng.normal(0.0, 0.7, size=n_samples)
    for j in range(1, n_covariates):
        scale = 0.65 - 0.08 * min(j, 5)
        covar_matrix[:, j] = (
            scale * anchor + 0.45 * covar_matrix[:, 0] + rng.normal(0.0, 1.0, size=n_samples)
        )

    q_noise = rng.normal(0.0, 1.0, size=n_samples)
    y_quant = (
        3.0
        + 0.8 * anchor
        + 1.1 * covar_matrix[:, 0]
        - 0.5 * covar_matrix[:, min(1, n_covariates - 1)]
        + q_noise
    )

    logit = (
        -0.25
        + 0.7 * anchor
        + 0.9 * covar_matrix[:, 0]
        - 0.5 * covar_matrix[:, min(1, n_covariates - 1)]
        + rng.normal(0.0, 0.35, size=n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
    y_binary = (rng.random(n_samples) < prob).astype(np.int8)
    if int(np.sum(y_binary)) == 0:
        y_binary[0] = 1
    if int(np.sum(y_binary)) == n_samples:
        y_binary[0] = 0

    covar_names = [f"COV{i+1}" for i in range(n_covariates)]

    keep_size = max(4, int(round(0.75 * n_samples)))
    keep_idx = np.sort(rng.choice(n_samples, size=keep_size, replace=False))
    keep_ids = [sample_ids[int(i)] for i in keep_idx.tolist()]

    remaining = np.setdiff1d(np.arange(n_samples), keep_idx, assume_unique=False)
    if remaining.size == 0:
        remove_ids = []
    else:
        remove_size = max(1, int(round(0.1 * n_samples)))
        remove_size = min(remove_size, int(remaining.size))
        remove_pick = np.sort(rng.choice(remaining, size=remove_size, replace=False))
        remove_ids = [sample_ids[int(i)] for i in remove_pick.tolist()]

    return (
        sample_ids,
        y_binary,
        y_quant,
        lai,
        chromosomes,
        starts,
        ends,
        ancestry_map,
        covar_names,
        covar_matrix,
        keep_ids,
        remove_ids,
    )
