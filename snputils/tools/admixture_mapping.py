import argparse
import csv
import gzip
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from snputils.ancestry.io.local.read import MSPReader
from snputils.phenotype.io.read import PhenotypeReader

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

log = logging.getLogger(__name__)

HLA_START = 25_477_797
HLA_END = 36_448_354


@dataclass
class _RegressionResult:
    beta: float
    se: float
    z: float
    p: float
    test: str
    errcode: str


def parse_admixmap_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="admixture_mapping", description="Admixture Mapping.")
    parser.add_argument(
        "--plot",
        required=False,
        default=False,
        type=bool,
        help="Plot Admixture Mapping Results {True, *False*}",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        default=32768,
        type=int,
        help="Maximum number of windows processed per chunk while building ancestry dosages.",
    )
    parser.add_argument(
        "--memory",
        required=False,
        default=None,
        type=int,
        help="Peak RSS-delta memory cap in MiB for internal chunked processing.",
    )
    parser.add_argument(
        "--keep_hla",
        required=False,
        action="store_true",
        help="Keep chr6 HLA windows (default is to remove them).",
    )
    parser.add_argument(
        "--quantitative",
        required=False,
        action="store_true",
        default=None,
        help="Optional override to force quantitative (linear) mode. Default is automatic trait detection.",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Print progress (windows processed, elapsed time, rate) during admixture mapping.",
    )
    parser.add_argument(
        "--covar",
        required=False,
        type=str,
        default=None,
        help="Path to covariate file (whitespace-delimited, header with #FID IID or #IID plus covariate columns).",
    )
    parser.add_argument(
        "--covar-col-nums",
        required=False,
        type=str,
        default=None,
        help='Covariate column numbers relative to first covariate column (e.g. "1-5,7").',
    )
    parser.add_argument(
        "--covar-variance-standardize",
        required=False,
        action="store_true",
        help="Center and variance-standardize selected covariates.",
    )
    parser.add_argument(
        "--ci",
        required=False,
        type=float,
        default=None,
        help="Confidence level in (0, 1), e.g. 0.95 for L95/U95 columns.",
    )
    parser.add_argument(
        "--adjust",
        required=False,
        action="store_true",
        help="Add Bonferroni and Benjamini-Hochberg FDR adjusted p-values.",
    )
    parser.add_argument(
        "--keep",
        required=False,
        type=str,
        default=None,
        help="Path to keep file (FID IID or IID per line) for sample inclusion.",
    )
    parser.add_argument(
        "--remove",
        required=False,
        type=str,
        default=None,
        help="Path to remove file (FID IID or IID per line) for sample exclusion.",
    )
    required_argv = parser.add_argument_group("required arguments")
    required_argv.add_argument("--pheID", required=True, type=str, help="Phenotype ID.")
    required_argv.add_argument(
        "--phe_path",
        required=True,
        type=str,
        help="Path to phenotype file (headered text with IID column and one phenotype column; e.g. .txt, .phe, .pheno).",
    )
    required_argv.add_argument(
        "--msp_path", required=True, type=str, help="Path of the .msp file (include file)."
    )
    required_argv.add_argument(
        "--results_path",
        required=True,
        type=str,
        help="Path used to save resulting data in compressed .tsv file.",
    )
    return parser.parse_args(argv)


def _parse_covar_col_nums(col_nums: Optional[str], n_covariates: int) -> List[int]:
    if n_covariates <= 0:
        return []
    if col_nums is None or str(col_nums).strip() == "":
        return list(range(n_covariates))

    selected: List[int] = []
    seen: Set[int] = set()
    for token in str(col_nums).split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0:
                raise ValueError("--covar-col-nums must be 1-indexed and positive.")
            if end < start:
                raise ValueError(f"Invalid covariate range '{part}' in --covar-col-nums.")
            indices = range(start - 1, end)
        else:
            idx = int(part)
            if idx <= 0:
                raise ValueError("--covar-col-nums must be 1-indexed and positive.")
            indices = [idx - 1]

        for idx0 in indices:
            if idx0 < 0 or idx0 >= n_covariates:
                raise ValueError(
                    f"--covar-col-nums selects column {idx0 + 1}, but only "
                    f"{n_covariates} covariate columns are available."
                )
            if idx0 not in seen:
                seen.add(idx0)
                selected.append(idx0)

    if not selected:
        raise ValueError("--covar-col-nums did not select any covariate column.")
    return selected


def _read_covar(
    path: Union[str, Path],
    col_nums: Optional[str],
    variance_standardize: bool,
) -> Tuple[List[str], List[str], np.ndarray]:
    covar_df = pd.read_csv(path, sep=r"\s+", dtype=str)
    if covar_df.empty:
        raise ValueError("Empty covariate file.")

    columns = list(covar_df.columns)
    lowered = [str(col).lstrip("#").upper() for col in columns]
    if "IID" not in lowered:
        raise ValueError("Covariate file must include an IID column in the header.")
    iid_col = columns[lowered.index("IID")]
    iid_series = covar_df[iid_col].astype(str)
    if iid_series.duplicated().any():
        raise ValueError("Covariate IID values must be unique.")
    sample_ids = iid_series.tolist()

    covar_start = lowered.index("IID") + 1
    if covar_start >= len(columns):
        raise ValueError("Covariate file header does not contain covariate columns.")
    all_covar_cols = columns[covar_start:]

    selected_rel = _parse_covar_col_nums(col_nums, len(all_covar_cols))
    selected_cols = [all_covar_cols[idx] for idx in selected_rel]
    covar_names = [str(name) for name in selected_cols]

    covar_numeric = covar_df[selected_cols].copy()
    for col in selected_cols:
        coldata = covar_numeric[col].astype(str).str.strip()
        if str(col).upper() == "SEX":
            coldata = coldata.str.upper().replace(
                {"MALE": "1", "M": "1", "FEMALE": "2", "F": "2", "UNKNOWN": "", "NA": "", "NAN": ""}
            )
        covar_numeric[col] = pd.to_numeric(coldata, errors="coerce")
    complete = ~covar_numeric.isna().any(axis=1)
    n_dropped = int((~complete).sum())
    if n_dropped > 0:
        log.info(
            "Dropping %d samples with missing covariate values (listwise deletion).",
            n_dropped,
        )
    covar_numeric = covar_numeric.loc[complete].to_numpy(dtype=np.float64, copy=True)
    sample_ids = [sid for sid, keep in zip(sample_ids, complete) if keep]
    if len(sample_ids) == 0:
        raise ValueError(
            "No samples remain after dropping rows with missing covariates. "
            "Check covariate file for non-numeric or missing values."
        )

    covar_matrix = covar_numeric
    if covar_matrix.ndim != 2 or covar_matrix.shape[1] == 0:
        raise ValueError("No covariates selected from covariate file.")

    if variance_standardize:
        means = np.mean(covar_matrix, axis=0)
        stds = np.std(covar_matrix, axis=0, ddof=0)
        if np.any(stds <= 0.0):
            zero_var_names = [covar_names[i] for i in np.where(stds <= 0.0)[0]]
            raise ValueError(
                "Cannot variance-standardize covariates with zero variance: "
                f"{zero_var_names}"
            )
        covar_matrix = (covar_matrix - means[None, :]) / stds[None, :]

    return sample_ids, covar_names, covar_matrix


def _read_sample_list(path: Union[str, Path]) -> Set[str]:
    selected: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                selected.add(str(parts[1]))
            else:
                selected.add(str(parts[0]))
    return selected


def _normalize_chromosome(chromosome: object) -> Union[int, str]:
    text = str(chromosome).strip()
    text_lower = text.lower().replace("chr", "")
    return int(text_lower) if text_lower.isdigit() else text


def _chromosome_as_int(chromosome: object) -> Optional[int]:
    text_lower = str(chromosome).strip().lower().replace("chr", "")
    return int(text_lower) if text_lower.isdigit() else None


def _remove_hla_windows(msp_obj) -> int:
    if msp_obj.chromosomes is None or msp_obj.physical_pos is None:
        return 0

    indexes_to_remove: List[int] = []
    for i, chrom in enumerate(msp_obj.chromosomes):
        chrom_int = _chromosome_as_int(chrom)
        if chrom_int != 6:
            continue
        start_pos = int(msp_obj.physical_pos[i, 0])
        end_pos = int(msp_obj.physical_pos[i, 1])
        if (HLA_START <= start_pos <= HLA_END) or (HLA_START <= end_pos <= HLA_END):
            indexes_to_remove.append(i)

    if indexes_to_remove:
        msp_obj.filter_windows(indexes=np.asarray(indexes_to_remove, dtype=int), include=False, inplace=True)
    return len(indexes_to_remove)


def _resolve_ancestries(msp_obj) -> List[Tuple[int, str]]:
    if msp_obj.ancestry_map is None:
        return [(int(code), f"ANC{int(code)}") for code in sorted(np.unique(msp_obj.lai).astype(int))]
    pairs = []
    for code_str, label in msp_obj.ancestry_map.items():
        pairs.append((int(code_str), str(label)))
    return sorted(pairs, key=lambda x: x[0])


def _resolve_ancestries_from_metadata(
    msp_reader: MSPReader,
    ancestry_map: Optional[Dict[str, str]],
    chunk_size: int,
    sample_indices: Optional[np.ndarray] = None,
) -> List[Tuple[int, str]]:
    if ancestry_map is not None:
        pairs = [(int(code), str(label)) for code, label in ancestry_map.items()]
        return sorted(pairs, key=lambda x: x[0])

    unique_codes: set[int] = set()
    for chunk in msp_reader.iter_windows(chunk_size=chunk_size, sample_indices=sample_indices):
        unique_codes.update(int(code) for code in np.unique(chunk["lai"]))
    return [(code, f"ANC{code}") for code in sorted(unique_codes)]


def _align_samples(
    msp_samples: Sequence[str],
    phe_samples: Sequence[str],
    y: np.ndarray,
    quantitative: bool = False,
    keep_ids: Optional[Set[str]] = None,
    remove_ids: Optional[Set[str]] = None,
    covar_samples: Optional[Sequence[str]] = None,
    covar_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(msp_samples)}
    covar_to_idx: Optional[Dict[str, int]] = None
    if covar_samples is not None:
        if covar_matrix is None:
            raise ValueError("Internal error: covariate samples provided without covariate matrix.")
        covar_to_idx = {sample_id: idx for idx, sample_id in enumerate(covar_samples)}

    msp_indexes: List[int] = []
    y_aligned: list = []
    covar_aligned: List[np.ndarray] = []
    aligned_samples: List[str] = []

    for sid, yi in zip(phe_samples, y):
        if keep_ids is not None and sid not in keep_ids:
            continue
        if remove_ids is not None and sid in remove_ids:
            continue

        idx = sample_to_idx.get(sid)
        if idx is None:
            continue
        if covar_to_idx is not None:
            cov_idx = covar_to_idx.get(sid)
            if cov_idx is None:
                continue
        msp_indexes.append(idx)
        y_aligned.append(float(yi) if quantitative else int(yi))
        if covar_to_idx is not None and covar_matrix is not None:
            covar_aligned.append(covar_matrix[cov_idx].astype(np.float64, copy=False))
        aligned_samples.append(sid)

    if not msp_indexes:
        raise ValueError("No overlapping samples between phenotype and MSP.")

    if quantitative:
        y_arr = np.asarray(y_aligned, dtype=np.float64)
        if np.var(y_arr) <= 0.0:
            raise ValueError("Quantitative phenotype has zero variance after MSP/PHE sample intersection.")
    else:
        y_arr = np.asarray(y_aligned, dtype=np.int8)
        if int(np.sum(y_arr)) == 0:
            raise ValueError("No cases after MSP/PHE sample intersection.")
        if int(np.sum(y_arr)) == len(y_arr):
            raise ValueError("No controls after MSP/PHE sample intersection.")

    covar_out = (
        np.asarray(covar_aligned, dtype=np.float64)
        if covar_to_idx is not None
        else None
    )
    return np.asarray(msp_indexes, dtype=np.int64), y_arr, aligned_samples, covar_out


def _compute_group_counts_batch(dosage_batch: np.ndarray, y_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = dosage_batch.shape[1]
    cases_total = int(np.sum(y_binary))

    mask1 = dosage_batch == 1
    mask2 = dosage_batch == 2

    n1 = np.sum(mask1, axis=1, dtype=np.int64)
    n2 = np.sum(mask2, axis=1, dtype=np.int64)
    n0 = n_samples - n1 - n2

    y_int64 = y_binary.astype(np.int64, copy=False)
    c1 = mask1 @ y_int64
    c2 = mask2 @ y_int64
    c0 = cases_total - c1 - c2

    n_counts = np.stack([n0, n1, n2], axis=1)
    c_counts = np.stack([c0, c1, c2], axis=1)
    return n_counts.astype(np.float64), c_counts.astype(np.float64)


def _compute_group_counts_from_lai(
    maternal: np.ndarray,
    paternal: np.ndarray,
    ancestry_code: int,
    y_binary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group (dosage-bin) counts directly from haplotype arrays.

    Avoids materializing the full dosage array by computing counts directly
    from the maternal/paternal match indicators.
    """
    n_samples = maternal.shape[1]
    cases_total = int(np.sum(y_binary))

    m = maternal == ancestry_code
    p = paternal == ancestry_code

    y_int = y_binary.astype(np.int64, copy=False)

    sum_m = np.sum(m, axis=1, dtype=np.int64)
    sum_p = np.sum(p, axis=1, dtype=np.int64)

    both = m & p
    n2 = np.sum(both, axis=1, dtype=np.int64)
    c2 = both @ y_int
    del both

    n1 = sum_m + sum_p - 2 * n2
    n0 = n_samples - n1 - n2

    cm = m @ y_int
    cp = p @ y_int
    c1 = cm + cp - 2 * c2
    c0 = cases_total - c1 - c2

    n_counts = np.stack([n0, n1, n2], axis=1)
    c_counts = np.stack([c0, c1, c2], axis=1)
    return n_counts.astype(np.float64), c_counts.astype(np.float64)


def _compute_dosage_from_lai(
    maternal: np.ndarray,
    paternal: np.ndarray,
    ancestry_code: int,
) -> np.ndarray:
    """Materialize per-sample ancestry dosage from maternal/paternal LAI haplotypes."""
    return (
        (maternal == ancestry_code).astype(np.uint8)
        + (paternal == ancestry_code).astype(np.uint8)
    )


def _prepare_fwl(y: np.ndarray, covar_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare Frisch-Waugh-Lovell projection terms for linear regression with covariates."""
    if covar_matrix.ndim != 2:
        raise ValueError("Covariate matrix must be 2-dimensional.")
    n_samples = covar_matrix.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("Phenotype and covariate sample counts do not match.")

    intercept = np.ones((n_samples, 1), dtype=np.float64)
    c_aug = np.concatenate([intercept, covar_matrix.astype(np.float64, copy=False)], axis=1)
    q, _ = np.linalg.qr(c_aug, mode="reduced")
    y_f64 = y.astype(np.float64, copy=False)
    y_resid = y_f64 - q @ (q.T @ y_f64)
    return y_resid, q


def _fit_linear_batch_with_covariates(
    dosage_batch: np.ndarray,
    y_resid: np.ndarray,
    q: np.ndarray,
    n_covar: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized OLS using FWL residualization for all windows in a chunk."""
    W, n_samples = dosage_batch.shape

    beta_out = np.full(W, np.nan, dtype=np.float64)
    se_out = np.full(W, np.nan, dtype=np.float64)
    t_out = np.full(W, np.nan, dtype=np.float64)
    p_out = np.full(W, np.nan, dtype=np.float64)
    errcode_out = np.full(W, ".", dtype=object)

    df = float(n_samples - (2 + int(n_covar)))
    if df <= 0.0:
        errcode_out[:] = "NO_OBS"
        return beta_out, se_out, t_out, p_out, errcode_out

    dosage_f64 = dosage_batch.astype(np.float64, copy=False)
    proj = dosage_f64 @ q
    d_resid = dosage_f64 - proj @ q.T

    sxx = np.sum(d_resid * d_resid, axis=1)
    no_var = sxx <= 0.0
    errcode_out[no_var] = "NO_VARIATION"

    can_fit = ~no_var
    fit_idx = np.where(can_fit)[0]
    if fit_idx.size == 0:
        return beta_out, se_out, t_out, p_out, errcode_out

    d_fit = d_resid[fit_idx]
    sxx_fit = sxx[fit_idx]
    sxy_fit = d_fit @ y_resid
    beta_fit = sxy_fit / sxx_fit

    yss = float(np.sum(y_resid * y_resid))
    sse_fit = yss - beta_fit * sxy_fit
    sse_fit = np.maximum(sse_fit, 0.0)
    mse_fit = sse_fit / df
    se_fit = np.sqrt(mse_fit / sxx_fit)

    valid = np.isfinite(se_fit) & (se_fit > 0.0)
    valid_idx = fit_idx[valid]
    if valid_idx.size > 0:
        beta_out[valid_idx] = beta_fit[valid]
        se_out[valid_idx] = se_fit[valid]
        t_stat = beta_fit[valid] / se_fit[valid]
        t_out[valid_idx] = t_stat
        p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
        p_vals[~np.isfinite(p_vals)] = np.nan
        p_out[valid_idx] = p_vals

    invalid_idx = fit_idx[~valid]
    errcode_out[invalid_idx] = "DEGENERATE"
    return beta_out, se_out, t_out, p_out, errcode_out


def _compute_linear_stats_from_lai(
    maternal: np.ndarray,
    paternal: np.ndarray,
    ancestry_code: int,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sufficient statistics for OLS on dosage groups directly from haplotype arrays.

    For each window and dosage group k in {0, 1, 2}, computes:
      - n[w, k]:      count of samples with dosage k
      - sum_y[w, k]:  sum of y for those samples
      - sum_y2[w, k]: sum of y^2 for those samples

    These three arrays are sufficient to perform closed-form OLS.
    """
    n_samples = maternal.shape[1]

    m = maternal == ancestry_code
    p = paternal == ancestry_code

    both = m & p
    n2 = np.sum(both, axis=1, dtype=np.int64)

    sum_m = np.sum(m, axis=1, dtype=np.int64)
    sum_p = np.sum(p, axis=1, dtype=np.int64)
    n1 = sum_m + sum_p - 2 * n2
    n0 = n_samples - n1 - n2

    y_f64 = y.astype(np.float64, copy=False)
    y_sq = y_f64 * y_f64

    sum_y_total = float(np.sum(y_f64))
    sum_y2_total = float(np.sum(y_sq))

    # dosage=2: both maternal and paternal match
    sy2 = both.astype(np.float64) @ y_f64
    sy2_sq = both.astype(np.float64) @ y_sq
    del both

    # dosage>=1 via maternal or paternal match
    sy_m = m.astype(np.float64) @ y_f64
    sy_p = p.astype(np.float64) @ y_f64
    sy_m_sq = m.astype(np.float64) @ y_sq
    sy_p_sq = p.astype(np.float64) @ y_sq

    # dosage=1: exactly one haplotype matches
    sy1 = sy_m + sy_p - 2.0 * sy2
    sy1_sq = sy_m_sq + sy_p_sq - 2.0 * sy2_sq

    # dosage=0: the remainder
    sy0 = sum_y_total - sy1 - sy2
    sy0_sq = sum_y2_total - sy1_sq - sy2_sq

    n_counts = np.stack([n0, n1, n2], axis=1).astype(np.float64)
    sum_y = np.stack([sy0, sy1, sy2], axis=1)
    sum_y2 = np.stack([sy0_sq, sy1_sq, sy2_sq], axis=1)
    return n_counts, sum_y, sum_y2


_X_DOSAGE = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_X_DOSAGE_SQ = np.array([0.0, 1.0, 4.0], dtype=np.float64)


def _standard_logistic_batch_vectorized(
    n: np.ndarray,
    c: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized standard logistic IRLS for grouped data across W windows.

    Each window has 3 groups with dosage x=[0,1,2].  All W regressions are
    run simultaneously using NumPy broadcasting, giving orders-of-magnitude
    speedup over per-window scalar IRLS.

    Returns:
        Tuple of (beta1, se, z, p, converged) arrays, each shape (W,).
    """
    x = _X_DOSAGE
    x_sq = _X_DOSAGE_SQ
    W = n.shape[0]

    n_total = np.sum(n, axis=1)
    case_rate = np.clip(
        np.sum(c, axis=1) / np.maximum(n_total, 1.0), 1e-12, 1.0 - 1e-12
    )

    b0 = np.log(case_rate / (1.0 - case_rate))
    b1 = np.zeros(W, dtype=np.float64)
    converged = np.zeros(W, dtype=bool)

    for _ in range(max_iter):
        eta = b0[:, None] + np.outer(b1, x)
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))

        r = c - n * mu
        score0 = np.sum(r, axis=1)
        score1 = r @ x

        w_d = n * mu * (1.0 - mu)
        i00 = np.sum(w_d, axis=1)
        i01 = w_d @ x
        i11 = w_d @ x_sq
        det = i00 * i11 - i01 * i01

        good = det > 1e-12
        safe_det = np.where(good, det, 1.0)
        inv_det = np.where(good, 1.0 / safe_det, 0.0)

        d0 = (i11 * score0 - i01 * score1) * inv_det
        d1 = (i00 * score1 - i01 * score0) * inv_det

        update = ~converged & good
        b0 += np.where(update, d0, 0.0)
        b1 += np.where(update, d1, 0.0)

        just_conv = update & (np.maximum(np.abs(d0), np.abs(d1)) < tol)
        converged |= just_conv

        if np.all(converged | ~good):
            break

    # Final SE, z, p for converged windows.
    eta = b0[:, None] + np.outer(b1, x)
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
    w_d = n * mu * (1.0 - mu)
    i00 = np.sum(w_d, axis=1)
    i01 = w_d @ x
    i11 = w_d @ x_sq
    det = i00 * i11 - i01 * i01

    safe_det = np.where(det > 1e-12, det, 1.0)
    se2 = np.where(det > 1e-12, i00 / safe_det, np.nan)
    valid = converged & np.isfinite(se2) & (se2 > 0.0)

    se = np.full(W, np.nan, dtype=np.float64)
    z = np.full(W, np.nan, dtype=np.float64)
    p = np.full(W, np.nan, dtype=np.float64)

    valid_idx = np.where(valid)[0]
    if valid_idx.size > 0:
        se[valid_idx] = np.sqrt(se2[valid_idx])
        z[valid_idx] = b1[valid_idx] / se[valid_idx]
        p_vals = 2.0 * stats.norm.sf(np.abs(z[valid_idx]))
        p_vals[~np.isfinite(p_vals)] = np.nan
        p[valid_idx] = p_vals

    converged &= valid
    return b1, se, z, p, converged


def _fit_logistic_batch(
    n_batch: np.ndarray,
    c_batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch logistic regression with vectorized standard IRLS and per-window Firth fallback.

    Returns:
        Tuple of (beta, se, z, p, test, errcode) arrays, each shape (W,).
    """
    W = n_batch.shape[0]
    x = _X_DOSAGE

    beta_out = np.full(W, np.nan, dtype=np.float64)
    se_out = np.full(W, np.nan, dtype=np.float64)
    z_out = np.full(W, np.nan, dtype=np.float64)
    p_out = np.full(W, np.nan, dtype=np.float64)
    test_out = np.full(W, "ADD", dtype=object)
    errcode_out = np.full(W, ".", dtype=object)

    obs_ct = np.sum(n_batch, axis=1)
    n_cases = np.sum(c_batch, axis=1)
    n_controls = obs_ct - n_cases

    no_obs = obs_ct <= 0
    errcode_out[no_obs] = "NO_OBS"

    no_case_ctrl = ~no_obs & ((n_cases <= 0) | (n_controls <= 0))
    errcode_out[no_case_ctrl] = "NO_CASE_CTRL"

    g_mean = np.sum(n_batch * x[None, :], axis=1) / np.maximum(obs_ct, 1.0)
    g_var = np.sum(n_batch * (x[None, :] - g_mean[:, None]) ** 2, axis=1)
    no_variation = ~no_obs & ~no_case_ctrl & (g_var <= 0.0)
    errcode_out[no_variation] = "NO_VARIATION"

    can_fit = ~(no_obs | no_case_ctrl | no_variation)
    fit_idx = np.where(can_fit)[0]

    if fit_idx.size == 0:
        return beta_out, se_out, z_out, p_out, test_out, errcode_out

    n_fit = n_batch[fit_idx]
    c_fit = c_batch[fit_idx]

    b1, se, z, p, converged = _standard_logistic_batch_vectorized(n_fit, c_fit)

    conv_global = fit_idx[converged]
    beta_out[conv_global] = b1[converged]
    se_out[conv_global] = se[converged]
    z_out[conv_global] = z[converged]
    p_out[conv_global] = p[converged]

    # Firth fallback for windows that did not converge with standard IRLS.
    nonconv_idx = fit_idx[~converged]
    for idx in nonconv_idx:
        firth = _fit_firth_logistic_grouped(n_batch[idx], c_batch[idx])
        if firth is not None:
            beta_out[idx] = firth.beta
            se_out[idx] = firth.se
            z_out[idx] = firth.z
            p_out[idx] = firth.p
            test_out[idx] = "FIRTH"
            errcode_out[idx] = "."
        else:
            test_out[idx] = "FIRTH"
            errcode_out[idx] = "NONCONVERGENCE"

    return beta_out, se_out, z_out, p_out, test_out, errcode_out


def _penalized_loglik_with_covariates(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = x @ beta
    mu = np.clip(_expit(eta), 1e-12, 1.0 - 1e-12)
    ll = float(np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))

    w = mu * (1.0 - mu)
    info = x.T @ (x * w[:, None])
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0.0 or not np.isfinite(logdet):
        return float("-inf")
    return ll + 0.5 * float(logdet)


def _fit_firth_logistic_with_covariates(
    dosage: np.ndarray,
    y: np.ndarray,
    covar: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Optional[_RegressionResult]:
    n_samples = int(y.size)
    if n_samples <= 0:
        return None
    n_cases = int(np.sum(y))
    if n_cases <= 0 or n_cases >= n_samples:
        return None
    if np.all(dosage == dosage[0]):
        return None

    g = dosage.astype(np.float64, copy=False)
    c = covar.astype(np.float64, copy=False)
    x = np.concatenate(
        [
            np.ones((n_samples, 1), dtype=np.float64),
            g[:, None],
            c,
        ],
        axis=1,
    )
    p = x.shape[1]
    if n_samples <= p:
        return None

    case_rate = float(np.mean(y))
    beta = np.zeros(p, dtype=np.float64)
    beta[0] = _logit(case_rate)

    converged = False
    for _ in range(max_iter):
        eta = x @ beta
        mu = _expit(eta)
        w = mu * (1.0 - mu)
        info = x.T @ (x * w[:, None])
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            return None

        # diag(H) for weighted hat matrix: h_i = w_i * x_i^T I^{-1} x_i
        quad = np.sum((x @ info_inv) * x, axis=1)
        h = w * quad
        adjusted = y - mu + h * (0.5 - mu)
        score = x.T @ adjusted
        delta = info_inv @ score

        current_ll = _penalized_loglik_with_covariates(x, y, beta)
        step = 1.0
        accepted = False
        while step > 1e-8:
            candidate = beta + step * delta
            candidate_ll = _penalized_loglik_with_covariates(x, y, candidate)
            if candidate_ll >= current_ll:
                beta = candidate
                accepted = True
                break
            step *= 0.5

        if not accepted:
            return None
        if float(np.max(np.abs(step * delta))) < tol:
            converged = True
            break

    if not converged or not np.all(np.isfinite(beta)):
        return None

    eta = x @ beta
    mu = _expit(eta)
    w = mu * (1.0 - mu)
    info = x.T @ (x * w[:, None])
    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None
    se2 = float(info_inv[1, 1])
    if (not np.isfinite(se2)) or se2 <= 0.0:
        return None

    se = float(math.sqrt(se2))
    z = float(beta[1] / se)
    p_val = float(2.0 * stats.norm.sf(abs(z)))
    if not np.isfinite(p_val):
        p_val = float("nan")
    return _RegressionResult(
        beta=float(beta[1]),
        se=se,
        z=z,
        p=p_val,
        test="FIRTH",
        errcode=".",
    )


def _fit_logistic_batch_with_covariates(
    dosage_batch: np.ndarray,
    y_binary: np.ndarray,
    covar_matrix: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched individual-level logistic regression with covariates."""
    W, n_samples = dosage_batch.shape
    n_covar = int(covar_matrix.shape[1])
    p = 2 + n_covar

    beta_out = np.full(W, np.nan, dtype=np.float64)
    se_out = np.full(W, np.nan, dtype=np.float64)
    z_out = np.full(W, np.nan, dtype=np.float64)
    p_out = np.full(W, np.nan, dtype=np.float64)
    test_out = np.full(W, "ADD", dtype=object)
    errcode_out = np.full(W, ".", dtype=object)

    if n_samples <= p:
        errcode_out[:] = "NO_OBS"
        return beta_out, se_out, z_out, p_out, test_out, errcode_out

    y = y_binary.astype(np.float64, copy=False)
    n_cases = int(np.sum(y_binary))
    if n_cases <= 0 or n_cases >= n_samples:
        errcode_out[:] = "NO_CASE_CTRL"
        return beta_out, se_out, z_out, p_out, test_out, errcode_out

    g = dosage_batch.astype(np.float64, copy=False)
    g_centered = g - np.mean(g, axis=1, keepdims=True)
    g_var = np.sum(g_centered * g_centered, axis=1)
    no_var = g_var <= 0.0
    errcode_out[no_var] = "NO_VARIATION"
    fit_idx = np.where(~no_var)[0]
    if fit_idx.size == 0:
        return beta_out, se_out, z_out, p_out, test_out, errcode_out

    covar = covar_matrix.astype(np.float64, copy=False)
    covar_t = covar.T
    beta = np.zeros((W, p), dtype=np.float64)
    beta[:, 0] = _logit(float(np.mean(y)))
    converged = np.zeros(W, dtype=bool)

    for _ in range(max_iter):
        active = fit_idx[~converged[fit_idx]]
        if active.size == 0:
            break

        g_a = g[active]
        b_a = beta[active]

        eta = b_a[:, 0][:, None] + b_a[:, 1][:, None] * g_a
        if n_covar > 0:
            eta = eta + (b_a[:, 2:] @ covar_t)
        mu = _expit(eta)
        w = mu * (1.0 - mu)
        r = y[None, :] - mu

        score = np.zeros((active.size, p), dtype=np.float64)
        score[:, 0] = np.sum(r, axis=1)
        score[:, 1] = np.sum(r * g_a, axis=1)
        if n_covar > 0:
            score[:, 2:] = r @ covar

        info = np.zeros((active.size, p, p), dtype=np.float64)
        wg = w * g_a
        info[:, 0, 0] = np.sum(w, axis=1)
        info[:, 0, 1] = np.sum(wg, axis=1)
        info[:, 1, 0] = info[:, 0, 1]
        info[:, 1, 1] = np.sum(wg * g_a, axis=1)
        if n_covar > 0:
            i0c = w @ covar
            i1c = wg @ covar
            icc = np.einsum("wn,nk,nj->wkj", w, covar, covar, optimize=True)
            info[:, 0, 2:] = i0c
            info[:, 2:, 0] = i0c
            info[:, 1, 2:] = i1c
            info[:, 2:, 1] = i1c
            info[:, 2:, 2:] = icc

        delta = np.full((active.size, p), np.nan, dtype=np.float64)
        try:
            solved = np.linalg.solve(info, score[..., None])[..., 0]
            delta[:] = solved
        except np.linalg.LinAlgError:
            for j in range(active.size):
                try:
                    delta[j] = np.linalg.solve(info[j], score[j])
                except np.linalg.LinAlgError:
                    continue

        finite = np.all(np.isfinite(delta), axis=1)
        if not np.any(finite):
            break

        active_good = active[finite]
        delta_good = delta[finite]
        max_abs = np.max(np.abs(delta_good), axis=1)
        scale = np.minimum(1.0, 5.0 / np.maximum(max_abs, 1e-12))
        delta_good = delta_good * scale[:, None]
        beta[active_good] += delta_good

        just_converged = np.max(np.abs(delta_good), axis=1) < tol
        converged[active_good[just_converged]] = True

    conv_idx = fit_idx[converged[fit_idx]]
    fallback_idx: List[int] = fit_idx[~converged[fit_idx]].tolist()

    if conv_idx.size > 0:
        g_c = g[conv_idx]
        b_c = beta[conv_idx]
        eta = b_c[:, 0][:, None] + b_c[:, 1][:, None] * g_c
        if n_covar > 0:
            eta = eta + (b_c[:, 2:] @ covar_t)
        mu = _expit(eta)
        w = mu * (1.0 - mu)
        wg = w * g_c

        info = np.zeros((conv_idx.size, p, p), dtype=np.float64)
        info[:, 0, 0] = np.sum(w, axis=1)
        info[:, 0, 1] = np.sum(wg, axis=1)
        info[:, 1, 0] = info[:, 0, 1]
        info[:, 1, 1] = np.sum(wg * g_c, axis=1)
        if n_covar > 0:
            i0c = w @ covar
            i1c = wg @ covar
            icc = np.einsum("wn,nk,nj->wkj", w, covar, covar, optimize=True)
            info[:, 0, 2:] = i0c
            info[:, 2:, 0] = i0c
            info[:, 1, 2:] = i1c
            info[:, 2:, 1] = i1c
            info[:, 2:, 2:] = icc

        inv_info = np.full_like(info, np.nan)
        try:
            inv_solved = np.linalg.inv(info)
            inv_info[:] = inv_solved
        except np.linalg.LinAlgError:
            for j in range(conv_idx.size):
                try:
                    inv_info[j] = np.linalg.inv(info[j])
                except np.linalg.LinAlgError:
                    continue

        se2 = inv_info[:, 1, 1]
        valid = np.isfinite(se2) & (se2 > 0.0)
        if np.any(valid):
            good_idx = conv_idx[valid]
            se_vals = np.sqrt(se2[valid])
            beta_vals = beta[good_idx, 1]
            z_vals = beta_vals / se_vals
            p_vals = 2.0 * stats.norm.sf(np.abs(z_vals))
            p_vals[~np.isfinite(p_vals)] = np.nan
            beta_out[good_idx] = beta_vals
            se_out[good_idx] = se_vals
            z_out[good_idx] = z_vals
            p_out[good_idx] = p_vals

        invalid_conv = conv_idx[~valid]
        fallback_idx.extend(invalid_conv.tolist())

    seen_fallback: Set[int] = set()
    for idx in fallback_idx:
        if idx in seen_fallback:
            continue
        seen_fallback.add(idx)
        firth = _fit_firth_logistic_with_covariates(g[idx], y_binary, covar)
        if firth is not None:
            beta_out[idx] = firth.beta
            se_out[idx] = firth.se
            z_out[idx] = firth.z
            p_out[idx] = firth.p
            test_out[idx] = "FIRTH"
            errcode_out[idx] = "."
        else:
            test_out[idx] = "FIRTH"
            errcode_out[idx] = "NONCONVERGENCE"

    return beta_out, se_out, z_out, p_out, test_out, errcode_out


def _odds_ratio_batch(beta: np.ndarray) -> np.ndarray:
    """Vectorised exp(beta) for logistic regression, with clipping for numerical stability."""
    finite = np.isfinite(beta)
    safe_beta = np.where(finite, np.clip(beta, -700, 700), 0.0)
    result = np.exp(safe_beta)
    result[~finite] = np.nan
    result[finite & (beta > 700)] = np.inf
    result[finite & (beta < -700)] = 0.0
    return result


def _fit_linear_batch(
    n_batch: np.ndarray,
    sum_y_batch: np.ndarray,
    sum_y2_batch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized closed-form OLS on grouped data for all windows simultaneously.

    For each window, performs linear regression y = beta0 + beta1 * dosage using
    sufficient statistics (n, sum_y, sum_y2) for dosage groups k=0,1,2.

    Returns:
        Tuple of (beta, se, t_stat, p, errcode) arrays, each shape (W,).
    """
    x = _X_DOSAGE  # [0, 1, 2]
    W = n_batch.shape[0]

    beta_out = np.full(W, np.nan, dtype=np.float64)
    se_out = np.full(W, np.nan, dtype=np.float64)
    t_out = np.full(W, np.nan, dtype=np.float64)
    p_out = np.full(W, np.nan, dtype=np.float64)
    errcode_out = np.full(W, ".", dtype=object)

    n_total = np.sum(n_batch, axis=1)  # (W,)

    no_obs = n_total <= 2.0
    errcode_out[no_obs] = "NO_OBS"

    # Check for variation in dosage
    x_bar = np.sum(n_batch * x[None, :], axis=1) / np.maximum(n_total, 1.0)
    sxx = np.sum(n_batch * (x[None, :] - x_bar[:, None]) ** 2, axis=1)
    no_var = ~no_obs & (sxx <= 0.0)
    errcode_out[no_var] = "NO_VARIATION"

    can_fit = ~(no_obs | no_var)
    fit_idx = np.where(can_fit)[0]
    if fit_idx.size == 0:
        return beta_out, se_out, t_out, p_out, errcode_out

    n_f = n_batch[fit_idx]
    sy_f = sum_y_batch[fit_idx]
    sy2_f = sum_y2_batch[fit_idx]
    nt_f = n_total[fit_idx]
    xbar_f = x_bar[fit_idx]
    sxx_f = sxx[fit_idx]

    ybar_f = np.sum(sy_f, axis=1) / nt_f
    sxy_f = np.sum(sy_f * x[None, :], axis=1) - nt_f * xbar_f * ybar_f
    syy_f = np.sum(sy2_f, axis=1) - nt_f * ybar_f ** 2

    beta1 = sxy_f / sxx_f
    ssr = syy_f - sxy_f ** 2 / sxx_f
    ssr = np.maximum(ssr, 0.0)  # numerical guard
    df = nt_f - 2.0
    mse = ssr / np.maximum(df, 1.0)
    se = np.sqrt(mse / sxx_f)

    valid = np.isfinite(se) & (se > 0.0) & (df > 0.0)
    valid_idx = fit_idx[valid]

    beta_out[valid_idx] = beta1[valid]
    se_out[valid_idx] = se[valid]
    t_stat = beta1[valid] / se[valid]
    t_out[valid_idx] = t_stat
    p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df=df[valid])
    p_vals[~np.isfinite(p_vals)] = np.nan
    p_out[valid_idx] = p_vals

    invalid_fit = fit_idx[~valid]
    errcode_out[invalid_fit] = "DEGENERATE"

    return beta_out, se_out, t_out, p_out, errcode_out


def _compute_effective_chunk_size(
    batch_size: int,
    n_samples: int,
    memory_mib: Optional[int],
    covariates_present: bool = False,
    quantitative: bool = False,
) -> int:
    base_chunk = max(1, int(batch_size))
    if memory_mib is None:
        return base_chunk
    if memory_mib < 1:
        raise ValueError("--memory must be >= 1 MiB.")

    # Memory-per-window estimate (bytes/sample/window):
    #  - No covariates: ~16
    #  - Linear with covariates (FWL): ~32
    #  - Logistic with covariates (IRLS): ~56
    # The 0.30 usable fraction below reserves headroom for fixed-cost process
    # allocations (Python runtime, gzip buffers, file handles, etc.).
    if not covariates_present:
        bytes_per_sample = 16
    elif quantitative:
        bytes_per_sample = 32
    else:
        bytes_per_sample = 56

    bytes_per_window = max(1, int(n_samples) * bytes_per_sample)
    budget_bytes = int(memory_mib) * 1024 * 1024
    usable_bytes = max(1, int(budget_bytes * 0.30))
    memory_chunk = max(1, usable_bytes // bytes_per_window)
    return max(1, min(base_chunk, int(memory_chunk)))


def _get_process_rss_mb() -> float:
    if psutil is not None:
        try:
            return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
        except (psutil.Error, PermissionError):
            pass

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def _enforce_memory_budget(
    memory_mib: Optional[int],
    rss_baseline_mb: Optional[float],
    context: str,
) -> None:
    if memory_mib is None or rss_baseline_mb is None:
        return

    rss_now_mb = _get_process_rss_mb()
    rss_delta_mb = rss_now_mb - rss_baseline_mb
    budget_mb = float(memory_mib)
    tolerance_mb = max(30.0, budget_mb * 0.05)
    if rss_delta_mb > (budget_mb + tolerance_mb):
        raise MemoryError(
            f"Internal memory budget exceeded during {context}: "
            f"RSS delta {rss_delta_mb:.2f} MiB > --memory {int(memory_mib)} MiB "
            f"(tolerance {tolerance_mb:.2f} MiB)."
        )


def _fisher_info_inverse(n: np.ndarray, mu: np.ndarray, x: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    w = n * mu * (1.0 - mu)
    i00 = float(np.sum(w))
    i01 = float(np.sum(w * x))
    i11 = float(np.sum(w * x * x))
    det = i00 * i11 - i01 * i01
    if det <= 1e-12:
        return None
    inv = np.array([[i11, -i01], [-i01, i00]], dtype=np.float64) / det
    return inv, det


def _logit(p: float) -> float:
    eps = 1e-12
    p_clip = min(max(p, eps), 1.0 - eps)
    return math.log(p_clip / (1.0 - p_clip))


def _expit(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _fit_standard_logistic_grouped(
    n: np.ndarray, c: np.ndarray, max_iter: int = 50, tol: float = 1e-8
) -> Optional[_RegressionResult]:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    n_total = float(np.sum(n))
    case_rate = float(np.sum(c) / n_total)
    beta = np.array([_logit(case_rate), 0.0], dtype=np.float64)

    converged = False
    for _ in range(max_iter):
        eta = beta[0] + beta[1] * x
        mu = _expit(eta)
        score0 = float(np.sum(c - n * mu))
        score1 = float(np.sum((c - n * mu) * x))
        info = _fisher_info_inverse(n, mu, x)
        if info is None:
            return None
        inv_i, _ = info
        delta = inv_i @ np.array([score0, score1], dtype=np.float64)
        beta = beta + delta
        if float(np.max(np.abs(delta))) < tol:
            converged = True
            break

    if not converged or not np.all(np.isfinite(beta)):
        return None

    eta = beta[0] + beta[1] * x
    mu = _expit(eta)
    info = _fisher_info_inverse(n, mu, x)
    if info is None:
        return None
    inv_i, _ = info
    se2 = float(inv_i[1, 1])
    if (not np.isfinite(se2)) or se2 <= 0.0:
        return None

    se = math.sqrt(se2)
    z = float(beta[1] / se)
    p = float(2.0 * stats.norm.sf(abs(z)))
    if not np.isfinite(p):
        p = float("nan")

    return _RegressionResult(
        beta=float(beta[1]),
        se=se,
        z=z,
        p=p,
        test="ADD",
        errcode=".",
    )


def _penalized_loglik(beta: np.ndarray, n: np.ndarray, c: np.ndarray, x: np.ndarray) -> float:
    eta = beta[0] + beta[1] * x
    mu = np.clip(_expit(eta), 1e-12, 1.0 - 1e-12)
    ll = float(np.sum(c * np.log(mu) + (n - c) * np.log(1.0 - mu)))
    info = _fisher_info_inverse(n, mu, x)
    if info is None:
        return float("-inf")
    _, det = info
    return ll + 0.5 * math.log(det)


def _fit_firth_logistic_grouped(
    n: np.ndarray, c: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> Optional[_RegressionResult]:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    n_total = float(np.sum(n))
    case_rate = float(np.sum(c) / n_total)
    beta = np.array([_logit(case_rate), 0.0], dtype=np.float64)

    converged = False
    for _ in range(max_iter):
        eta = beta[0] + beta[1] * x
        mu = _expit(eta)
        info = _fisher_info_inverse(n, mu, x)
        if info is None:
            return None
        inv_i, _ = info

        w_individual = mu * (1.0 - mu)
        quad = inv_i[0, 0] + (2.0 * inv_i[0, 1] * x) + (inv_i[1, 1] * x * x)
        h_individual = w_individual * quad
        adjusted = c - n * mu + n * h_individual * (0.5 - mu)
        score = np.array([np.sum(adjusted), np.sum(adjusted * x)], dtype=np.float64)
        delta = inv_i @ score

        current_ll = _penalized_loglik(beta, n, c, x)
        step = 1.0
        accepted = False
        while step > 1e-8:
            candidate = beta + step * delta
            candidate_ll = _penalized_loglik(candidate, n, c, x)
            if candidate_ll >= current_ll:
                beta = candidate
                accepted = True
                break
            step *= 0.5

        if not accepted:
            return None

        if float(np.max(np.abs(step * delta))) < tol:
            converged = True
            break

    if not converged or not np.all(np.isfinite(beta)):
        return None

    eta = beta[0] + beta[1] * x
    mu = _expit(eta)
    info = _fisher_info_inverse(n, mu, x)
    if info is None:
        return None
    inv_i, _ = info

    se2 = float(inv_i[1, 1])
    if (not np.isfinite(se2)) or se2 <= 0.0:
        return None
    se = math.sqrt(se2)
    z = float(beta[1] / se)
    p = float(2.0 * stats.norm.sf(abs(z)))
    if not np.isfinite(p):
        p = float("nan")

    return _RegressionResult(
        beta=float(beta[1]),
        se=se,
        z=z,
        p=p,
        test="FIRTH",
        errcode=".",
    )


def _fit_logistic_hybrid_grouped(n: np.ndarray, c: np.ndarray) -> _RegressionResult:
    obs_ct = int(np.sum(n))
    n_cases = int(np.sum(c))
    n_controls = obs_ct - n_cases
    if obs_ct <= 0:
        return _RegressionResult(np.nan, np.nan, np.nan, np.nan, "ADD", "NO_OBS")
    if n_cases <= 0 or n_controls <= 0:
        return _RegressionResult(np.nan, np.nan, np.nan, np.nan, "ADD", "NO_CASE_CTRL")

    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    g_mean = float(np.sum(n * x) / obs_ct)
    g_var = float(np.sum(n * (x - g_mean) ** 2))
    if g_var <= 0.0:
        return _RegressionResult(np.nan, np.nan, np.nan, np.nan, "ADD", "NO_VARIATION")

    fit = _fit_standard_logistic_grouped(n, c)
    if fit is not None:
        return fit

    firth = _fit_firth_logistic_grouped(n, c)
    if firth is not None:
        return firth

    return _RegressionResult(np.nan, np.nan, np.nan, np.nan, "FIRTH", "NONCONVERGENCE")


def _confidence_interval_label(ci: float) -> str:
    pct = ci * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-8:
        return str(int(rounded))
    text = f"{pct:.2f}".rstrip("0").rstrip(".")
    return text


def _compute_logistic_ci_or(
    beta: np.ndarray,
    se: np.ndarray,
    ci: float,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - ci
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lower = _odds_ratio_batch(beta - z_crit * se)
    upper = _odds_ratio_batch(beta + z_crit * se)
    invalid = ~np.isfinite(beta) | ~np.isfinite(se) | (se < 0.0)
    lower[invalid] = np.nan
    upper[invalid] = np.nan
    return lower, upper


def _compute_linear_ci_beta(
    beta: np.ndarray,
    se: np.ndarray,
    ci: float,
    df: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if df <= 0.0:
        return (
            np.full(beta.shape[0], np.nan, dtype=np.float64),
            np.full(beta.shape[0], np.nan, dtype=np.float64),
        )
    alpha = 1.0 - ci
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    lower = beta - t_crit * se
    upper = beta + t_crit * se
    invalid = ~np.isfinite(beta) | ~np.isfinite(se) | (se < 0.0)
    lower[invalid] = np.nan
    upper[invalid] = np.nan
    return lower, upper


def _compute_multiple_testing_adjustments(p_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = p_values.astype(np.float64, copy=False)
    n = p.shape[0]
    bonf = np.full(n, np.nan, dtype=np.float64)
    fdr = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(p) & (p >= 0.0) & (p <= 1.0)
    m = int(np.sum(valid))
    if m == 0:
        return bonf, fdr

    p_valid = p[valid]
    bonf[valid] = np.minimum(p_valid * m, 1.0)

    order = np.argsort(p_valid, kind="mergesort")
    sorted_p = p_valid[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    bh_sorted = sorted_p * (m / ranks)
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    bh_sorted = np.clip(bh_sorted, 0.0, 1.0)

    fdr_valid = np.empty(m, dtype=np.float64)
    fdr_valid[order] = bh_sorted
    fdr[valid] = fdr_valid
    return bonf, fdr


def _apply_multiple_testing_adjustment(
    output_file: Union[str, Path],
    p_values: Sequence[float],
) -> None:
    p_arr = np.asarray(p_values, dtype=np.float64)
    bonf, fdr = _compute_multiple_testing_adjustments(p_arr)

    src = Path(output_file)
    tmp = src.with_suffix(src.suffix + ".tmp")
    row_count = 0
    with gzip.open(src, mode="rt", encoding="utf-8", newline="") as in_handle:
        reader = csv.reader(in_handle, delimiter="\t")
        header = next(reader)
        if "ERRCODE" in header:
            err_idx = header.index("ERRCODE")
        else:
            err_idx = len(header)
        new_header = header[:err_idx] + ["BONF", "FDR_BH"] + header[err_idx:]

        with gzip.open(tmp, mode="wt", encoding="utf-8", newline="") as out_handle:
            writer = csv.writer(out_handle, delimiter="\t")
            writer.writerow(new_header)
            for idx, row in enumerate(reader):
                if idx >= p_arr.size:
                    raise ValueError("Output row count exceeds collected p-value count.")
                row_new = row[:err_idx] + [bonf[idx], fdr[idx]] + row[err_idx:]
                writer.writerow(row_new)
                row_count += 1

    if row_count != p_arr.size:
        raise ValueError(
            f"Collected {p_arr.size} p-values, but output contains {row_count} data rows."
        )
    tmp.replace(src)


def _resolve_output_path(results_path: Union[str, Path], phe_id: str) -> Path:
    text = str(results_path)
    path = Path(text)

    if path.suffix in {".tsv", ".gz"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if text.endswith("/") or path.is_dir() or path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{phe_id}_admixmap.tsv.gz"

    out = Path(text + f"{phe_id}_admixmap.tsv.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def run_admixture_mapping(
    phe_path: Union[str, Path],
    msp_path: Union[str, Path],
    results_path: Union[str, Path],
    phe_id: str,
    batch_size: int = 256,
    keep_hla: bool = False,
    memory: Optional[int] = None,
    return_results: bool = True,
    quantitative: Optional[bool] = None,
    verbose: bool = False,
    covar_path: Optional[Union[str, Path]] = None,
    covar_col_nums: Optional[str] = None,
    covar_variance_standardize: bool = False,
    ci: Optional[float] = None,
    adjust: bool = False,
    keep_path: Optional[Union[str, Path]] = None,
    remove_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    if memory is not None and int(memory) < 2:
        raise MemoryError("--memory must be >= 2 MiB for internal admixture mapping.")
    if ci is not None and (ci <= 0.0 or ci >= 1.0):
        raise ValueError("--ci must be in the open interval (0, 1).")

    phenotype_obj = PhenotypeReader(phe_path).read(quantitative=quantitative)
    phe_samples = phenotype_obj.samples
    y = phenotype_obj.values
    trait_is_quantitative = bool(phenotype_obj.is_quantitative)

    keep_ids = _read_sample_list(keep_path) if keep_path is not None else None
    remove_ids = _read_sample_list(remove_path) if remove_path is not None else None

    covar_samples: Optional[List[str]] = None
    covar_matrix: Optional[np.ndarray] = None
    if covar_path is not None:
        covar_samples, _covar_names, covar_matrix = _read_covar(
            covar_path,
            col_nums=covar_col_nums,
            variance_standardize=covar_variance_standardize,
        )

    msp_reader = MSPReader(msp_path)
    metadata = msp_reader.read_metadata()

    msp_sample_indexes, y_aligned, aligned_samples, covar_aligned = _align_samples(
        metadata.samples,
        phe_samples,
        y,
        quantitative=trait_is_quantitative,
        keep_ids=keep_ids,
        remove_ids=remove_ids,
        covar_samples=covar_samples,
        covar_matrix=covar_matrix,
    )
    del aligned_samples

    covariates_present = covar_aligned is not None
    n_covar = int(covar_aligned.shape[1]) if covariates_present else 0

    chunk_size = _compute_effective_chunk_size(
        batch_size=batch_size,
        n_samples=int(msp_sample_indexes.size),
        memory_mib=memory,
        covariates_present=covariates_present,
        quantitative=trait_is_quantitative,
    )
    ancestries = _resolve_ancestries_from_metadata(
        msp_reader=msp_reader,
        ancestry_map=metadata.ancestry_map,
        chunk_size=chunk_size,
        sample_indices=msp_sample_indexes,
    )
    if not ancestries:
        raise ValueError("No ancestries available in MSP data.")

    obs_ct = int(y_aligned.size)
    rss_baseline_mb = _get_process_rss_mb() if memory is not None else None

    output_file = _resolve_output_path(results_path, phe_id)

    ci_cols: List[str] = []
    if ci is not None:
        ci_suffix = _confidence_interval_label(ci)
        ci_cols = [f"L{ci_suffix}", f"U{ci_suffix}"]

    if trait_is_quantitative:
        core_columns = [
            "#CHROM", "POS", "END", "ID", "REF", "ALT", "A1",
            "ANCESTRY", "TEST", "OBS_CT", "BETA", "SE", "T_STAT", "P",
        ]
    else:
        core_columns = [
            "#CHROM", "POS", "END", "ID", "REF", "ALT", "A1",
            "ANCESTRY", "TEST", "OBS_CT", "BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P",
        ]
    columns_without_adjust = core_columns + ci_cols + ["ERRCODE"]
    final_columns = core_columns + ci_cols + (["BONF", "FDR_BH"] if adjust else []) + ["ERRCODE"]

    covar_f64: Optional[np.ndarray] = None
    y_resid: Optional[np.ndarray] = None
    q_fwl: Optional[np.ndarray] = None
    if trait_is_quantitative:
        y_f64 = y_aligned.astype(np.float64, copy=False)
        if covariates_present:
            covar_f64 = covar_aligned.astype(np.float64, copy=False)
            y_resid, q_fwl = _prepare_fwl(y_f64, covar_f64)
    else:
        y_binary = y_aligned.astype(np.int64, copy=False)
        if covariates_present:
            covar_f64 = covar_aligned.astype(np.float64, copy=False)

    records: List[Dict[str, object]] = [] if return_results else []
    collected_p_values: Optional[List[float]] = [] if adjust else None
    kept_window_counter: Dict[int, int] = {int(code): 0 for code, _ in ancestries}
    removed_hla_windows_total = 0
    n_windows_processed_total = 0
    chunk_index = 0
    try:
        with gzip.open(output_file, mode="wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(columns_without_adjust)

            if verbose:
                print("Reading MSP file...", flush=True)
            for chunk in msp_reader.iter_windows(chunk_size=chunk_size, sample_indices=msp_sample_indexes):
                n_win_processed = 0
                chunk_lai = chunk["lai"]
                if chunk_lai.size == 0:
                    continue

                chunk_chrom = chunk["chromosomes"]
                chunk_phys = chunk["physical_pos"]
                if chunk_phys is not None:
                    chunk_starts = chunk_phys[:, 0].astype(np.int64, copy=False)
                    chunk_ends = chunk_phys[:, 1].astype(np.int64, copy=False)
                else:
                    chunk_starts = chunk["window_indexes"].astype(np.int64, copy=False) + 1
                    chunk_ends = chunk_starts.copy()

                if not keep_hla and chunk_phys is not None:
                    keep_mask = np.ones(chunk_lai.shape[0], dtype=bool)
                    for i, chrom in enumerate(chunk_chrom):
                        chrom_int = _chromosome_as_int(chrom)
                        if chrom_int != 6:
                            continue
                        if (HLA_START <= int(chunk_starts[i]) <= HLA_END) or (HLA_START <= int(chunk_ends[i]) <= HLA_END):
                            keep_mask[i] = False

                    removed_hla_windows_total += int((~keep_mask).sum())
                    if not np.any(keep_mask):
                        continue

                    chunk_lai = chunk_lai[keep_mask]
                    chunk_chrom = chunk_chrom[keep_mask]
                    chunk_starts = chunk_starts[keep_mask]
                    chunk_ends = chunk_ends[keep_mask]

                _enforce_memory_budget(memory, rss_baseline_mb, context="chunk loading")

                maternal = chunk_lai[:, 0::2]
                paternal = chunk_lai[:, 1::2]
                n_windows_in_chunk = chunk_lai.shape[0]
                n_win_processed = n_windows_in_chunk

                if verbose and (chunk_index % 100 == 0):
                    print(f"  Chunk {chunk_index}: {n_windows_in_chunk} windows (total so far: {n_windows_processed_total + n_win_processed:,})", flush=True)
                chunk_index += 1
                chunk_chrom_norm = [_normalize_chromosome(chunk_chrom[i]) for i in range(n_windows_in_chunk)]
                chunk_starts_int = chunk_starts.tolist()
                chunk_ends_int = chunk_ends.tolist()

                for ancestry_code, ancestry_label in ancestries:
                    ancestry_code_int = int(ancestry_code)

                    if trait_is_quantitative:
                        if covariates_present:
                            if covar_f64 is None or y_resid is None or q_fwl is None:
                                raise ValueError("Internal error: missing covariate projection state.")
                            dosage_batch = _compute_dosage_from_lai(
                                maternal,
                                paternal,
                                ancestry_code_int,
                            )
                            beta_arr, se_arr, t_arr, p_arr, errcode_arr = _fit_linear_batch_with_covariates(
                                dosage_batch,
                                y_resid,
                                q_fwl,
                                n_covar=n_covar,
                            )
                            df_linear = float(obs_ct - (2 + n_covar))
                        else:
                            n_batch, sy_batch, sy2_batch = _compute_linear_stats_from_lai(
                                maternal, paternal, ancestry_code_int, y_f64,
                            )
                            beta_arr, se_arr, t_arr, p_arr, errcode_arr = _fit_linear_batch(
                                n_batch, sy_batch, sy2_batch,
                            )
                            df_linear = float(obs_ct - 2)

                        if ci is not None:
                            ci_low_arr, ci_high_arr = _compute_linear_ci_beta(
                                beta_arr,
                                se_arr,
                                ci=ci,
                                df=df_linear,
                            )
                        else:
                            ci_low_arr = ci_high_arr = None

                        current_index = kept_window_counter[ancestry_code_int]
                        rows_buf: List[List[object]] = []
                        for i in range(n_windows_in_chunk):
                            row_values = [
                                chunk_chrom_norm[i],
                                chunk_starts_int[i],
                                chunk_ends_int[i],
                                f"w{current_index + i + 1}_{ancestry_label}",
                                "N",
                                ancestry_label,
                                ancestry_label,
                                ancestry_label,
                                "LINEAR",
                                obs_ct,
                                beta_arr[i],
                                se_arr[i],
                                t_arr[i],
                                p_arr[i],
                            ]
                            if ci is not None and ci_low_arr is not None and ci_high_arr is not None:
                                row_values.extend([ci_low_arr[i], ci_high_arr[i]])
                            row_values.append(errcode_arr[i])
                            rows_buf.append(row_values)
                            if return_results:
                                records.append(dict(zip(columns_without_adjust, row_values)))
                            if collected_p_values is not None:
                                collected_p_values.append(float(p_arr[i]))
                        writer.writerows(rows_buf)
                    else:
                        if covariates_present:
                            if covar_f64 is None:
                                raise ValueError("Internal error: missing aligned covariate matrix.")
                            dosage_batch = _compute_dosage_from_lai(
                                maternal,
                                paternal,
                                ancestry_code_int,
                            )
                            beta_arr, se_arr, z_arr, p_arr, test_arr, errcode_arr = _fit_logistic_batch_with_covariates(
                                dosage_batch,
                                y_binary,
                                covar_f64,
                            )
                        else:
                            n_counts_batch, c_counts_batch = _compute_group_counts_from_lai(
                                maternal, paternal, ancestry_code_int, y_binary,
                            )
                            beta_arr, se_arr, z_arr, p_arr, test_arr, errcode_arr = _fit_logistic_batch(
                                n_counts_batch, c_counts_batch,
                            )
                        or_arr = _odds_ratio_batch(beta_arr)
                        if ci is not None:
                            ci_low_arr, ci_high_arr = _compute_logistic_ci_or(beta_arr, se_arr, ci=ci)
                        else:
                            ci_low_arr = ci_high_arr = None

                        current_index = kept_window_counter[ancestry_code_int]
                        rows_buf = []
                        for i in range(n_windows_in_chunk):
                            row_values = [
                                chunk_chrom_norm[i],
                                chunk_starts_int[i],
                                chunk_ends_int[i],
                                f"w{current_index + i + 1}_{ancestry_label}",
                                "N",
                                ancestry_label,
                                ancestry_label,
                                ancestry_label,
                                test_arr[i],
                                obs_ct,
                                beta_arr[i],
                                or_arr[i],
                                se_arr[i],
                                z_arr[i],
                                p_arr[i],
                            ]
                            if ci is not None and ci_low_arr is not None and ci_high_arr is not None:
                                row_values.extend([ci_low_arr[i], ci_high_arr[i]])
                            row_values.append(errcode_arr[i])
                            rows_buf.append(row_values)
                            if return_results:
                                records.append(dict(zip(columns_without_adjust, row_values)))
                            if collected_p_values is not None:
                                collected_p_values.append(float(p_arr[i]))
                        writer.writerows(rows_buf)

                    kept_window_counter[ancestry_code_int] = current_index + n_windows_in_chunk
                    _enforce_memory_budget(memory, rss_baseline_mb, context=f"ancestry {ancestry_label}")

                n_windows_processed_total += n_win_processed
    except Exception:
        try:
            output_file.unlink()
        except FileNotFoundError:
            pass
        raise

    if verbose and n_windows_processed_total > 0:
        print(f"  Done. Processed {n_windows_processed_total:,} windows.", flush=True)

    if not keep_hla:
        log.info("Removed %s HLA windows.", removed_hla_windows_total)

    if adjust:
        if collected_p_values is None:
            raise ValueError("Internal error: adjusted p-values requested but collection is unavailable.")
        _apply_multiple_testing_adjustment(output_file, collected_p_values)
        if return_results and records:
            bonf_arr, fdr_arr = _compute_multiple_testing_adjustments(
                np.asarray(collected_p_values, dtype=np.float64)
            )
            for rec, bonf_val, fdr_val in zip(records, bonf_arr.tolist(), fdr_arr.tolist()):
                rec["BONF"] = bonf_val
                rec["FDR_BH"] = fdr_val

    if return_results:
        results = pd.DataFrame.from_records(records)
        results = results.reindex(columns=final_columns)
    else:
        results = pd.DataFrame(columns=final_columns)
    log.info("Admixture mapping results written to %s", output_file)
    return results


def admixmap(argv: Sequence[str]):
    args = parse_admixmap_args(argv)
    run_admixture_mapping(
        phe_path=args.phe_path,
        msp_path=args.msp_path,
        results_path=args.results_path,
        phe_id=args.pheID,
        batch_size=args.batch_size,
        keep_hla=args.keep_hla,
        memory=args.memory,
        return_results=False,
        quantitative=args.quantitative,
        verbose=args.verbose,
        covar_path=args.covar,
        covar_col_nums=args.covar_col_nums,
        covar_variance_standardize=args.covar_variance_standardize,
        ci=args.ci,
        adjust=args.adjust,
        keep_path=args.keep,
        remove_path=args.remove,
    )
    return None
