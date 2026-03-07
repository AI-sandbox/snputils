import argparse
import csv
import gzip
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from snputils.phenotype.io.read import PhenotypeReader
from snputils.snp.io.read import BEDReader, PGENReader, SNPReader, VCFReader
from snputils.snp.io.read.vcf import VCFReaderPolars
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


def add_gwas_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        required=False,
        default=32768,
        type=int,
        help="Maximum number of variants processed per chunk.",
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
        help="Print progress (variants processed, elapsed time, rate) during GWAS.",
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
    parser.add_argument(
        "--vcf-backend",
        dest="vcf_backend",
        required=False,
        choices=("polars", "scikit-allel"),
        default="polars",
        help="VCF reader backend (used only when --snp-path is VCF).",
    )
    required_argv = parser.add_argument_group("required arguments")
    required_argv.add_argument(
        "--phe-id",
        dest="phe_id",
        required=True,
        type=str,
        help="Phenotype ID / column name to analyze.",
    )
    required_argv.add_argument(
        "--phe-path",
        dest="phe_path",
        required=True,
        type=str,
        help="Path to phenotype file (headered text with IID column and one or more phenotype columns; e.g. .txt, .phe, .pheno).",
    )
    required_argv.add_argument(
        "--snp-path",
        dest="snp_path",
        required=True,
        type=str,
        help="Path to genotype input (VCF/BED/PGEN).",
    )
    required_argv.add_argument(
        "--results-path",
        dest="results_path",
        required=True,
        type=str,
        help="Path used to save resulting data in compressed .tsv file.",
    )


def parse_gwas_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="gwas", description="Genome-wide association study (GWAS).")
    add_gwas_arguments(parser)
    return parser.parse_args(argv)


def _read_vcf_sample_ids(path: Union[str, Path]) -> List[str]:
    vcf_path = Path(path)
    if vcf_path.suffixes[-2:] == [".vcf", ".gz"]:
        open_func = gzip.open
    elif vcf_path.suffix == ".vcf":
        open_func = open
    else:
        raise ValueError(f"Unsupported VCF extension for sample parsing: {vcf_path.suffixes}")

    with open_func(vcf_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("#CHROM") and not raw_line.startswith("CHROM"):
                continue
            line = raw_line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) == 1:
                parts = line.split()
            if len(parts) <= 9:
                raise ValueError("VCF header does not contain sample columns.")
            return [str(sample) for sample in parts[9:]]

    raise ValueError("VCF header line (#CHROM) not found.")


def _read_snp_samples(snp_reader: object) -> List[str]:
    if isinstance(snp_reader, (VCFReaderPolars, VCFReader)):
        return _read_vcf_sample_ids(getattr(snp_reader, "filename"))

    if isinstance(snp_reader, (BEDReader, PGENReader)):
        sample_obj = snp_reader.read(fields=["IID"])
        samples = sample_obj.samples
        if samples is None:
            raise ValueError("Failed to read sample IDs from SNP input.")
        return [str(sample) for sample in np.asarray(samples, dtype=object).tolist()]

    raise ValueError(f"Unsupported SNP reader type: {type(snp_reader).__name__}")


def _align_samples_to_snp_order(
    snp_samples: Sequence[str],
    phe_samples: Sequence[str],
    y: np.ndarray,
    quantitative: bool = False,
    keep_ids: Optional[Set[str]] = None,
    remove_ids: Optional[Set[str]] = None,
    covar_samples: Optional[Sequence[str]] = None,
    covar_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    phe_to_idx: Dict[str, int] = {sample_id: idx for idx, sample_id in enumerate(phe_samples)}
    covar_to_idx: Optional[Dict[str, int]] = None
    if covar_samples is not None:
        if covar_matrix is None:
            raise ValueError("Internal error: covariate samples provided without covariate matrix.")
        covar_to_idx = {sample_id: idx for idx, sample_id in enumerate(covar_samples)}

    snp_indexes: List[int] = []
    y_aligned: List[Union[int, float]] = []
    covar_aligned: List[np.ndarray] = []
    aligned_samples: List[str] = []

    for snp_idx, sid in enumerate(snp_samples):
        sid_str = str(sid)
        if keep_ids is not None and sid_str not in keep_ids:
            continue
        if remove_ids is not None and sid_str in remove_ids:
            continue

        phe_idx = phe_to_idx.get(sid_str)
        if phe_idx is None:
            continue

        if covar_to_idx is not None:
            cov_idx = covar_to_idx.get(sid_str)
            if cov_idx is None:
                continue

        snp_indexes.append(snp_idx)
        yi = y[phe_idx]
        y_aligned.append(float(yi) if quantitative else int(yi))
        if covar_to_idx is not None and covar_matrix is not None:
            covar_aligned.append(covar_matrix[cov_idx].astype(np.float64, copy=False))
        aligned_samples.append(sid_str)

    if not snp_indexes:
        raise ValueError("No overlapping samples between phenotype and SNP input.")

    if quantitative:
        y_arr = np.asarray(y_aligned, dtype=np.float64)
        if np.var(y_arr) <= 0.0:
            raise ValueError("Quantitative phenotype has zero variance after SNP/PHE sample intersection.")
    else:
        y_arr = np.asarray(y_aligned, dtype=np.int8)
        if int(np.sum(y_arr)) == 0:
            raise ValueError("No cases after SNP/PHE sample intersection.")
        if int(np.sum(y_arr)) == len(y_arr):
            raise ValueError("No controls after SNP/PHE sample intersection.")

    covar_out = np.asarray(covar_aligned, dtype=np.float64) if covar_to_idx is not None else None
    return np.asarray(snp_indexes, dtype=np.int64), y_arr, aligned_samples, covar_out


def _iter_snp_chunks(
    snp_reader: object,
    chunk_size: int,
    sample_indices: np.ndarray,
    aligned_samples: Sequence[str],
) -> Iterator[Dict[str, Optional[np.ndarray]]]:
    if isinstance(snp_reader, (BEDReader, PGENReader)):
        for chunk in snp_reader.iter_read(
            fields=["GT", "#CHROM", "POS", "ID", "REF", "ALT"],
            sample_idxs=np.asarray(sample_indices, dtype=np.uint32),
            sum_strands=True,
            chunk_size=chunk_size,
        ):
            yield {
                "calldata_gt": chunk.calldata_gt,
                "variants_chrom": chunk.variants_chrom,
                "variants_pos": chunk.variants_pos,
                "variants_id": chunk.variants_id,
                "variants_ref": chunk.variants_ref,
                "variants_alt": chunk.variants_alt,
            }
        return

    if isinstance(snp_reader, VCFReaderPolars):
        for chunk in snp_reader.iter_read(
            fields=["#CHROM", "CHROM", "POS", "ID", "REF", "ALT"],
            samples=list(aligned_samples),
            sum_strands=True,
            chunk_size=chunk_size,
        ):
            yield {
                "calldata_gt": chunk.calldata_gt,
                "variants_chrom": chunk.variants_chrom,
                "variants_pos": chunk.variants_pos,
                "variants_id": chunk.variants_id,
                "variants_ref": chunk.variants_ref,
                "variants_alt": chunk.variants_alt,
            }
        return

    if isinstance(snp_reader, VCFReader):
        full = snp_reader.read(
            fields=[
                "variants/CHROM",
                "variants/POS",
                "variants/ID",
                "variants/REF",
                "variants/ALT",
                "calldata/GT",
            ],
            samples=list(aligned_samples),
            sum_strands=True,
        )
        if full.calldata_gt is None:
            return
        n_variants = int(full.calldata_gt.shape[0])
        for start in range(0, n_variants, int(chunk_size)):
            stop = min(start + int(chunk_size), n_variants)
            yield {
                "calldata_gt": full.calldata_gt[start:stop],
                "variants_chrom": None if full.variants_chrom is None else full.variants_chrom[start:stop],
                "variants_pos": None if full.variants_pos is None else full.variants_pos[start:stop],
                "variants_id": None if full.variants_id is None else full.variants_id[start:stop],
                "variants_ref": None if full.variants_ref is None else full.variants_ref[start:stop],
                "variants_alt": None if full.variants_alt is None else full.variants_alt[start:stop],
            }
        return

    raise ValueError(f"Unsupported SNP reader type for chunking: {type(snp_reader).__name__}")


def _coerce_variant_text_array(
    values: Optional[np.ndarray],
    length: int,
    default: str,
) -> np.ndarray:
    out = np.full(length, default, dtype=object)
    if values is None:
        return out

    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0:
        return out
    if arr.size != length:
        raise ValueError("Variant metadata length mismatch for GWAS chunk.")

    for idx, raw in enumerate(arr.tolist()):
        text = str(raw).strip()
        if text and text != "." and text.upper() != "NAN":
            out[idx] = text
    return out


def _coerce_variant_chrom_array(
    values: Optional[np.ndarray],
    length: int,
) -> np.ndarray:
    out = np.full(length, ".", dtype=object)
    if values is None:
        return out

    arr = np.asarray(values, dtype=object).reshape(-1)
    if arr.size == 0:
        return out
    if arr.size != length:
        raise ValueError("Variant chromosome length mismatch for GWAS chunk.")

    for idx, raw in enumerate(arr.tolist()):
        text = str(raw).strip()
        if text and text.upper() != "NAN":
            out[idx] = _normalize_chromosome(text)
    return out


def _coerce_variant_pos_array(
    values: Optional[np.ndarray],
    length: int,
    offset: int,
) -> np.ndarray:
    out = np.arange(offset + 1, offset + length + 1, dtype=np.int64)
    if values is None:
        return out

    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return out
    if arr.size != length:
        raise ValueError("Variant position length mismatch for GWAS chunk.")

    numeric = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric)
    out[valid] = numeric[valid].astype(np.int64)
    return out


def _build_variant_id_array(
    values: Optional[np.ndarray],
    chrom: np.ndarray,
    pos: np.ndarray,
    offset: int,
) -> np.ndarray:
    length = int(pos.shape[0])
    out = np.empty(length, dtype=object)

    raw = None
    if values is not None:
        arr = np.asarray(values, dtype=object).reshape(-1)
        if arr.size not in (0, length):
            raise ValueError("Variant ID length mismatch for GWAS chunk.")
        raw = arr if arr.size == length else None

    for idx in range(length):
        if raw is not None:
            text = str(raw[idx]).strip()
            if text and text != "." and text.upper() != "NAN":
                out[idx] = text
                continue
        chrom_text = str(chrom[idx])
        pos_val = int(pos[idx])
        out[idx] = f"{chrom_text}:{pos_val}" if chrom_text != "." else f"v{offset + idx + 1}"

    return out


def _extract_chunk_arrays(
    chunk: Dict[str, Optional[np.ndarray]],
    variant_offset: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt = chunk.get("calldata_gt")
    if gt is None:
        raise ValueError("Missing genotype calls in GWAS chunk.")

    dosage = np.asarray(gt)
    if dosage.ndim == 3:
        dosage = dosage.sum(axis=2, dtype=np.int16)
    if dosage.ndim != 2:
        raise ValueError("GWAS expects genotype chunks with shape (variants, samples).")
    if dosage.shape[1] == 0:
        raise ValueError("No samples available in GWAS genotype chunk.")

    invalid = (dosage < 0) | (dosage > 2)
    if np.any(invalid):
        raise ValueError(
            "GWAS currently requires diploid dosages encoded as 0/1/2 with no missing values."
        )

    dosage_uint8 = dosage.astype(np.uint8, copy=False)
    n_variants = int(dosage_uint8.shape[0])
    chrom = _coerce_variant_chrom_array(chunk.get("variants_chrom"), n_variants)
    pos = _coerce_variant_pos_array(chunk.get("variants_pos"), n_variants, offset=variant_offset)
    variant_id = _build_variant_id_array(chunk.get("variants_id"), chrom, pos, offset=variant_offset)
    ref = _coerce_variant_text_array(chunk.get("variants_ref"), n_variants, default="N")
    alt = _coerce_variant_text_array(chunk.get("variants_alt"), n_variants, default=".")
    return dosage_uint8, chrom, pos, variant_id, ref, alt


def _compute_linear_stats_from_dosage_batch(
    dosage_batch: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_f64 = y.astype(np.float64, copy=False)
    y_sq = y_f64 * y_f64
    n_variants = int(dosage_batch.shape[0])

    n_batch = np.empty((n_variants, 3), dtype=np.float64)
    sum_y_batch = np.empty((n_variants, 3), dtype=np.float64)
    sum_y2_batch = np.empty((n_variants, 3), dtype=np.float64)

    for dosage_code in (0, 1, 2):
        mask = dosage_batch == dosage_code
        n_batch[:, dosage_code] = np.sum(mask, axis=1, dtype=np.int64)
        sum_y_batch[:, dosage_code] = mask @ y_f64
        sum_y2_batch[:, dosage_code] = mask @ y_sq

    return n_batch, sum_y_batch, sum_y2_batch


def run_gwas(
    phe_path: Union[str, Path],
    snp_path: Union[str, Path],
    results_path: Union[str, Path],
    phe_id: str,
    batch_size: int = 256,
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
    vcf_backend: str = "polars",
) -> pd.DataFrame:
    if memory is not None and int(memory) < 2:
        raise MemoryError("--memory must be >= 2 MiB for internal GWAS processing.")
    if ci is not None and (ci <= 0.0 or ci >= 1.0):
        raise ValueError("--ci must be in the open interval (0, 1).")

    phenotype_obj = PhenotypeReader(phe_path).read(
        phenotype_col=phe_id,
        quantitative=quantitative,
    )
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

    snp_reader = SNPReader(snp_path, vcf_backend=vcf_backend)
    snp_samples = _read_snp_samples(snp_reader)

    sample_indexes, y_aligned, aligned_samples, covar_aligned = _align_samples_to_snp_order(
        snp_samples=snp_samples,
        phe_samples=phe_samples,
        y=y,
        quantitative=trait_is_quantitative,
        keep_ids=keep_ids,
        remove_ids=remove_ids,
        covar_samples=covar_samples,
        covar_matrix=covar_matrix,
    )

    covariates_present = covar_aligned is not None
    n_covar = int(covar_aligned.shape[1]) if covariates_present else 0
    chunk_size = _compute_effective_chunk_size(
        batch_size=batch_size,
        n_samples=int(sample_indexes.size),
        memory_mib=memory,
        covariates_present=covariates_present,
        quantitative=trait_is_quantitative,
    )

    obs_ct = int(y_aligned.size)
    rss_baseline_mb = _get_process_rss_mb() if memory is not None else None
    output_file = _resolve_output_path(results_path, phe_id, default_suffix="_gwas.tsv.gz")

    ci_cols: List[str] = []
    if ci is not None:
        ci_suffix = _confidence_interval_label(ci)
        ci_cols = [f"L{ci_suffix}", f"U{ci_suffix}"]

    if trait_is_quantitative:
        core_columns = [
            "#CHROM", "POS", "END", "ID", "REF", "ALT", "A1",
            "TEST", "OBS_CT", "BETA", "SE", "T_STAT", "P",
        ]
    else:
        core_columns = [
            "#CHROM", "POS", "END", "ID", "REF", "ALT", "A1",
            "TEST", "OBS_CT", "BETA", "OR", "LOG(OR)_SE", "Z_STAT", "P",
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

    variants_processed_total = 0
    chunk_index = 0
    try:
        with gzip.open(output_file, mode="wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(columns_without_adjust)

            if verbose:
                print("Reading SNP input...", flush=True)

            for chunk in _iter_snp_chunks(
                snp_reader=snp_reader,
                chunk_size=chunk_size,
                sample_indices=sample_indexes,
                aligned_samples=aligned_samples,
            ):
                dosage_batch, chrom_arr, pos_arr, id_arr, ref_arr, alt_arr = _extract_chunk_arrays(
                    chunk,
                    variant_offset=variants_processed_total,
                )
                n_variants = int(dosage_batch.shape[0])
                if n_variants == 0:
                    continue

                if verbose and (chunk_index % 100 == 0):
                    print(
                        f"  Chunk {chunk_index}: {n_variants} variants "
                        f"(total so far: {variants_processed_total + n_variants:,})",
                        flush=True,
                    )
                chunk_index += 1

                if trait_is_quantitative:
                    if covariates_present:
                        if covar_f64 is None or y_resid is None or q_fwl is None:
                            raise ValueError("Internal error: missing covariate projection state.")
                        beta_arr, se_arr, t_arr, p_arr, errcode_arr = _fit_linear_batch_with_covariates(
                            dosage_batch,
                            y_resid,
                            q_fwl,
                            n_covar=n_covar,
                        )
                        df_linear = float(obs_ct - (2 + n_covar))
                    else:
                        n_batch, sy_batch, sy2_batch = _compute_linear_stats_from_dosage_batch(dosage_batch, y_f64)
                        beta_arr, se_arr, t_arr, p_arr, errcode_arr = _fit_linear_batch(
                            n_batch, sy_batch, sy2_batch
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

                    rows_buf: List[List[object]] = []
                    for i in range(n_variants):
                        row_values: List[object] = [
                            chrom_arr[i],
                            int(pos_arr[i]),
                            int(pos_arr[i]),
                            id_arr[i],
                            ref_arr[i],
                            alt_arr[i],
                            alt_arr[i],
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
                        beta_arr, se_arr, z_arr, p_arr, test_arr, errcode_arr = _fit_logistic_batch_with_covariates(
                            dosage_batch,
                            y_binary,
                            covar_f64,
                        )
                    else:
                        n_counts_batch, c_counts_batch = _compute_group_counts_batch(
                            dosage_batch,
                            y_binary,
                        )
                        beta_arr, se_arr, z_arr, p_arr, test_arr, errcode_arr = _fit_logistic_batch(
                            n_counts_batch,
                            c_counts_batch,
                        )
                    or_arr = _odds_ratio_batch(beta_arr)
                    if ci is not None:
                        ci_low_arr, ci_high_arr = _compute_logistic_ci_or(beta_arr, se_arr, ci=ci)
                    else:
                        ci_low_arr = ci_high_arr = None

                    rows_buf = []
                    for i in range(n_variants):
                        row_values = [
                            chrom_arr[i],
                            int(pos_arr[i]),
                            int(pos_arr[i]),
                            id_arr[i],
                            ref_arr[i],
                            alt_arr[i],
                            alt_arr[i],
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

                variants_processed_total += n_variants
                _enforce_memory_budget(memory, rss_baseline_mb, context="GWAS chunk processing")
    except Exception:
        try:
            output_file.unlink()
        except FileNotFoundError:
            pass
        raise

    if verbose and variants_processed_total > 0:
        print(f"  Done. Processed {variants_processed_total:,} variants.", flush=True)

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

    log.info("GWAS results written to %s", output_file)
    return results


def gwas(argv: Sequence[str]):
    args = parse_gwas_args(argv)
    return run_gwas_command(args)


def run_gwas_command(args: argparse.Namespace) -> int:
    run_gwas(
        phe_path=args.phe_path,
        snp_path=args.snp_path,
        results_path=args.results_path,
        phe_id=args.phe_id,
        batch_size=args.batch_size,
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
        vcf_backend=args.vcf_backend,
    )
    return 0
