import argparse
import csv
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from snputils.ancestry.io.local.read import MSPReader
from snputils.phenotype.io.read import PhenotypeReader

from ._association import (
    _apply_multiple_testing_adjustment,
    _compute_effective_chunk_size,
    _compute_group_counts_batch,
    _compute_linear_ci_beta,
    _compute_logistic_ci_or,
    _compute_multiple_testing_adjustments,
    _confidence_interval_label,
    _enforce_memory_budget,
    _fit_linear_batch,
    _fit_linear_batch_with_covariates,
    _fit_logistic_batch,
    _fit_logistic_batch_with_covariates,
    _get_process_rss_mb,
    _normalize_chromosome,
    _odds_ratio_batch,
    _prepare_fwl,
    _read_covar,
    _read_sample_list,
    _resolve_output_path,
)

log = logging.getLogger(__name__)

HLA_START = 25_477_797
HLA_END = 36_448_354


def add_admixmap_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        required=False,
        default=32768,
        type=int,
        help="Maximum number of windows processed per chunk while building ancestry dosages.",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        required=False,
        default=None,
        type=int,
        help="Peak RSS-delta memory cap in MiB for internal chunked processing.",
    )
    parser.add_argument(
        "--keep-hla",
        dest="keep_hla",
        required=False,
        action="store_true",
        help="Keep chr6 HLA windows (default is to remove them).",
    )
    parser.add_argument(
        "--quantitative",
        dest="quantitative",
        required=False,
        action="store_true",
        default=None,
        help="Optional override to force quantitative (linear) mode. Default is automatic trait detection.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        required=False,
        action="store_true",
        help="Print progress (windows processed, elapsed time, rate) during admixture mapping.",
    )
    parser.add_argument(
        "--covar-path",
        dest="covar_path",
        required=False,
        type=str,
        default=None,
        help="Path to covariate file (whitespace-delimited, header with #FID IID or #IID plus covariate columns).",
    )
    parser.add_argument(
        "--covar-col-nums",
        dest="covar_col_nums",
        required=False,
        type=str,
        default=None,
        help='Covariate column numbers relative to first covariate column (e.g. "1-5,7").',
    )
    parser.add_argument(
        "--covar-variance-standardize",
        dest="covar_variance_standardize",
        required=False,
        action="store_true",
        help="Center and variance-standardize selected covariates.",
    )
    parser.add_argument(
        "--ci",
        dest="ci",
        required=False,
        type=float,
        default=None,
        help="Confidence level in (0, 1), e.g. 0.95 for L95/U95 columns.",
    )
    parser.add_argument(
        "--adjust",
        dest="adjust",
        required=False,
        action="store_true",
        help="Add Bonferroni and Benjamini-Hochberg FDR adjusted p-values.",
    )
    parser.add_argument(
        "--keep-path",
        dest="keep_path",
        required=False,
        type=str,
        default=None,
        help="Path to keep file (FID IID or IID per line) for sample inclusion.",
    )
    parser.add_argument(
        "--remove-path",
        dest="remove_path",
        required=False,
        type=str,
        default=None,
        help="Path to remove file (FID IID or IID per line) for sample exclusion.",
    )
    required_argv = parser.add_argument_group("required arguments")
    required_argv.add_argument("--phe-id", dest="phe_id", required=True, type=str, help="Phenotype ID.")
    required_argv.add_argument(
        "--phe-path",
        dest="phe_path",
        required=True,
        type=str,
        help="Path to phenotype file (headered text with IID column and one phenotype column; e.g. .txt, .phe, .pheno).",
    )
    required_argv.add_argument(
        "--msp-path", dest="msp_path", required=True, type=str, help="Path of the .msp file (include file)."
    )
    required_argv.add_argument(
        "--results-path",
        dest="results_path",
        required=True,
        type=str,
        help="Path used to save resulting data in compressed .tsv file.",
    )


def parse_admixmap_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="admixture-map", description="Admixture Mapping.")
    add_admixmap_arguments(parser)
    return parser.parse_args(argv)


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
    return run_admixmap_command(args)


def run_admixmap_command(args: argparse.Namespace) -> int:
    run_admixture_mapping(
        phe_path=args.phe_path,
        msp_path=args.msp_path,
        results_path=args.results_path,
        phe_id=args.phe_id,
        batch_size=args.batch_size,
        keep_hla=args.keep_hla,
        memory=args.memory,
        return_results=False,
        quantitative=args.quantitative,
        verbose=args.verbose,
        covar_path=args.covar_path,
        covar_col_nums=args.covar_col_nums,
        covar_variance_standardize=args.covar_variance_standardize,
        ci=args.ci,
        adjust=args.adjust,
        keep_path=args.keep_path,
        remove_path=args.remove_path,
    )
    return 0
