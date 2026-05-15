#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.ancestry.io.local.write import AdmixtureMappingPGENWriter, AdmixtureMappingVCFWriter
from snputils.tools import run_admixture_mapping

DEFAULT_N_SAMPLES = 128
DEFAULT_ADMIX_WINDOWS = 2_048
DEFAULT_N_COVARIATES = 3
DEFAULT_BATCH_SIZE = 32_768
DEFAULT_SEED = 20260512
DEFAULT_PHE_ID = "TRAIT"
DEFAULT_ANCESTRY_MAP = {"0": "AFR", "1": "EUR", "2": "NAT"}
DEFAULT_N_REPS = 5


def result_paths(work_dir: Path) -> dict[str, Path]:
    result_dir = work_dir / "results"
    return {
        "result_dir": result_dir,
        "summary": result_dir / "concordance_summary.csv",
        "timing": result_dir / "timing_results.csv",
        "concordance_admixture_mapping": result_dir / "concordance_admixture_mapping.csv",
        "snputils_admixture_mapping": result_dir / "snputils_admixture_mapping.tsv",
        "plink_admixture_mapping": result_dir / "plink_admixture_mapping.csv",
    }


def cached_results_exist(paths: dict[str, Path]) -> bool:
    required = [
        paths["summary"],
        paths["timing"],
        paths["concordance_admixture_mapping"],
        paths["plink_admixture_mapping"],
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def require_plink(name_or_path: str) -> str:
    candidate = Path(name_or_path)
    if candidate.exists():
        return str(candidate.resolve())
    resolved = shutil.which(name_or_path)
    if resolved is None:
        raise RuntimeError(
            f"PLINK binary {name_or_path!r} is not on PATH. "
            "Install plink2 or pass --plink-bin /path/to/plink2."
        )
    return resolved


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def remove_pgen_fileset(pfile_prefix: Path) -> None:
    """Remove a PLINK 2 PGEN fileset (.pgen / .psam / .pvar, and optional .pvar.zst)."""
    remove_if_exists(pfile_prefix.with_suffix(".pgen"))
    remove_if_exists(pfile_prefix.with_suffix(".psam"))
    remove_if_exists(pfile_prefix.with_suffix(".pvar"))
    remove_if_exists(pfile_prefix.parent / f"{pfile_prefix.name}.pvar.zst")


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


def write_quantitative_phe(path: Path, sample_ids: Sequence[str], phe_name: str, y: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"FID IID {phe_name}\n")
        for sid, yi in zip(sample_ids, y):
            handle.write(f"{sid} {sid} {float(yi):.12g}\n")


def write_covar(path: Path, sample_ids: Sequence[str], covar_names: Sequence[str], covar: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        header = " ".join(["FID", "IID", *covar_names])
        handle.write(header + "\n")
        for sid, row in zip(sample_ids, covar):
            values = " ".join(f"{float(v):.12g}" for v in row.tolist())
            handle.write(f"{sid} {sid} {values}\n")


def covariate_coefficients(n_covariates: int) -> np.ndarray:
    base = np.array([0.60, -0.35, 0.25, -0.15, 0.10], dtype=np.float64)
    if n_covariates <= len(base):
        return base[:n_covariates].copy()
    extra = np.linspace(0.08, 0.02, n_covariates - len(base), dtype=np.float64)
    return np.concatenate([base, extra])


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


def build_synthetic_admixture_dataset(
    n_samples: int,
    n_windows: int,
    n_covariates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sample_ids = [f"A{i:05d}" for i in range(n_samples)]
    chromosomes, starts, ends = build_feature_layout(n_windows, spacing_bp=25_000, span_bp=4_999)

    n_ancestries = len(DEFAULT_ANCESTRY_MAP)
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
    # Keep the phenotype sparse: only a few local ancestry windows are causal.
    # Nearby windows may tag the same ancestry block, but most windows should be null.
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
    # Use genome-wide ancestry proportions as covariates, as in practical
    # admixture mapping, without materializing an ancestry x window x sample cube.
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

    y = 3.0 + sparse_signal
    if n_covariates > 0:
        y += covar_matrix @ covariate_coefficients(n_covariates)
    y += rng.normal(scale=2.5, size=n_samples)

    laiobj = build_lai_object(
        sample_ids,
        lai,
        chromosomes,
        starts,
        ends,
        DEFAULT_ANCESTRY_MAP,
    )
    return {
        "sample_ids": sample_ids,
        "phenotype": y.astype(np.float64, copy=False),
        "covar_names": covar_names,
        "covar_matrix": covar_matrix,
        "laiobj": laiobj,
        "effect_windows": effect_windows,
        "effect_ancestries": effect_ancestries,
        "effect_sizes": effect_sizes,
    }


def run_command(cmd: list[str], log_path: Path) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    log_path.write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def load_plink_linear_results(out_dir: Path, out_prefix_name: str) -> pd.DataFrame:
    glm_files = sorted(out_dir.glob(f"{out_prefix_name}*.glm.linear"))
    if not glm_files:
        any_glm = sorted(out_dir.glob(f"{out_prefix_name}*.glm.*"))
        raise RuntimeError(
            f"No PLINK .glm.linear output found for prefix {out_prefix_name!r}. "
            f"Available outputs: {[p.name for p in any_glm[:10]]}"
        )

    glm = pd.concat((pd.read_csv(path, sep=r"\s+") for path in glm_files), ignore_index=True)
    if "TEST" in glm.columns:
        glm = glm[glm["TEST"] == "ADD"]
    if "ERRCODE" in glm.columns:
        glm = glm[(glm["ERRCODE"] == ".") | (glm["ERRCODE"].isna())]
    return glm.reset_index(drop=True)


def run_plink_admixture_mapping(
    plink_bin: str,
    ancestry_vcfs: dict[str, Path],
    phe_path: Path,
    covar_path: Path | None,
    covar_names: Sequence[str],
    work_dir: Path,
) -> tuple[pd.DataFrame, float]:
    results: list[pd.DataFrame] = []
    t0 = time.perf_counter()

    for ancestry_label, vcf_path in ancestry_vcfs.items():
        out_prefix = work_dir / f"plink_admixture_mapping_{ancestry_label}"
        cmd = [
            plink_bin,
            "--vcf",
            str(vcf_path),
            "--double-id",
            "--allow-no-sex",
            "--pheno",
            str(phe_path),
        ]
        if covar_path is not None and covar_names:
            cmd.extend(["--covar", str(covar_path), "--covar-name", ",".join(covar_names)])

        glm_args = ["hide-covar"] if covar_path is not None and covar_names else ["allow-no-covars"]
        cmd.extend(["--glm", *glm_args, "--out", str(out_prefix)])
        run_command(cmd, work_dir / f"{out_prefix.name}.log")

        glm = load_plink_linear_results(work_dir, out_prefix.name)
        out = pd.DataFrame(
            {
                "#CHROM": pd.to_numeric(glm["#CHROM"], errors="coerce"),
                "POS": pd.to_numeric(glm["POS"], errors="coerce"),
                "ANCESTRY": ancestry_label,
                "P_plink": pd.to_numeric(glm.get("P"), errors="coerce"),
            }
        )
        out = out.dropna(subset=["#CHROM", "POS", "P_plink"]).copy()
        out["#CHROM"] = out["#CHROM"].astype(int)
        out["POS"] = out["POS"].astype(int)
        results.append(out)

    wall = time.perf_counter() - t0
    if not results:
        raise RuntimeError("No PLINK admixture-mapping results were produced.")
    return pd.concat(results, ignore_index=True), float(wall)


def run_plink_admixture_mapping_pgen(
    plink_bin: str,
    ancestry_pgen_prefixes: dict[str, Path],
    phe_path: Path,
    covar_path: Path | None,
    covar_names: Sequence[str],
    work_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """Run PLINK admixture mapping using PGEN filesets (``--pfile``).

    PGEN is plink2's native binary format — the most favourable choice for
    plink, avoiding the text-parsing overhead of VCF.
    """
    results: list[pd.DataFrame] = []
    t0 = time.perf_counter()

    for ancestry_label, pfile_prefix in ancestry_pgen_prefixes.items():
        out_prefix = work_dir / f"plink_admixture_mapping_{ancestry_label}"
        cmd = [
            plink_bin,
            "--pfile",
            str(pfile_prefix),
            "--allow-no-sex",
            "--pheno",
            str(phe_path),
        ]
        if covar_path is not None and covar_names:
            cmd.extend(["--covar", str(covar_path), "--covar-name", ",".join(covar_names)])

        glm_args = ["hide-covar"] if covar_path is not None and covar_names else ["allow-no-covars"]
        cmd.extend(["--glm", *glm_args, "--out", str(out_prefix)])
        run_command(cmd, work_dir / f"{out_prefix.name}.log")

        glm = load_plink_linear_results(work_dir, out_prefix.name)
        out = pd.DataFrame(
            {
                "#CHROM": pd.to_numeric(glm["#CHROM"], errors="coerce"),
                "POS": pd.to_numeric(glm["POS"], errors="coerce"),
                "ANCESTRY": ancestry_label,
                "P_plink": pd.to_numeric(glm.get("P"), errors="coerce"),
            }
        )
        out = out.dropna(subset=["#CHROM", "POS", "P_plink"]).copy()
        out["#CHROM"] = out["#CHROM"].astype(int)
        out["POS"] = out["POS"].astype(int)
        results.append(out)

    wall = time.perf_counter() - t0
    if not results:
        raise RuntimeError("No PLINK admixture-mapping results were produced.")
    return pd.concat(results, ignore_index=True), float(wall)


def run_plink_admixture_mapping_from_lai(
    plink_bin: str,
    laiobj: LocalAncestryObject,
    ancestry_file_prefix: Path,
    phe_path: Path,
    covar_path: Path | None,
    covar_names: Sequence[str],
    work_dir: Path,
    *,
    admix_format: str = "vcf",
) -> tuple[pd.DataFrame, float]:
    """Write LAI data to the chosen format and run PLINK admixture mapping.

    Timing starts from the same in-memory ``laiobj`` that snputils also uses —
    a symmetric starting point, because plink cannot read LAI data natively.

    Args:
        admix_format: ``"vcf"`` (default) writes one VCF per ancestry and uses
            ``--vcf``; ``"pgen"`` writes one PGEN fileset per ancestry and uses
            ``--pfile``.
    """
    if admix_format == "pgen":
        for ancestry_label in DEFAULT_ANCESTRY_MAP.values():
            pgen_prefix = ancestry_file_prefix.with_name(
                f"{ancestry_file_prefix.stem}_{ancestry_label}"
            )
            remove_pgen_fileset(pgen_prefix)

        # Timer covers LAI→PGEN conversion + all per-ancestry plink2 --glm runs.
        # Both output writing (plink .glm.linear) and input I/O are symmetric with
        # the snputils side, which also writes its results to disk.
        t0 = time.perf_counter()
        AdmixtureMappingPGENWriter(laiobj, ancestry_file_prefix).write()
        ancestry_pgen_prefixes = {
            label: ancestry_file_prefix.with_name(
                f"{ancestry_file_prefix.stem}_{label}"
            )
            for label in DEFAULT_ANCESTRY_MAP.values()
        }
        results, _wall_glm = run_plink_admixture_mapping_pgen(
            plink_bin,
            ancestry_pgen_prefixes,
            phe_path,
            covar_path,
            covar_names,
            work_dir,
        )
    else:
        for ancestry_label in DEFAULT_ANCESTRY_MAP.values():
            remove_if_exists(
                ancestry_file_prefix.with_name(
                    f"{ancestry_file_prefix.stem}_{ancestry_label}{ancestry_file_prefix.suffix}"
                )
            )

        t0 = time.perf_counter()
        AdmixtureMappingVCFWriter(laiobj, ancestry_file_prefix).write()
        ancestry_vcfs = {
            label: ancestry_file_prefix.with_name(
                f"{ancestry_file_prefix.stem}_{label}{ancestry_file_prefix.suffix}"
            )
            for label in DEFAULT_ANCESTRY_MAP.values()
        }
        results, _wall_glm = run_plink_admixture_mapping(
            plink_bin,
            ancestry_vcfs,
            phe_path,
            covar_path,
            covar_names,
            work_dir,
        )

    wall = time.perf_counter() - t0
    return results, float(wall)


def snputils_linear_results_filtered(results: pd.DataFrame) -> pd.DataFrame:
    out = results
    if "TEST" in out.columns:
        out = out[out["TEST"] == "LINEAR"]
    if "ERRCODE" in out.columns:
        out = out[(out["ERRCODE"] == ".") | (out["ERRCODE"].isna())]
    return out


def extract_snputils_admixture_pvalues(results: pd.DataFrame) -> pd.DataFrame:
    out = snputils_linear_results_filtered(results)
    out["#CHROM"] = pd.to_numeric(out["#CHROM"], errors="coerce")
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    out["P_snputils"] = pd.to_numeric(out["P"], errors="coerce")
    out = out.dropna(subset=["#CHROM", "POS", "P_snputils"]).copy()
    out["#CHROM"] = out["#CHROM"].astype(int)
    out["POS"] = out["POS"].astype(int)
    out["ANCESTRY"] = out["ANCESTRY"].astype(str)
    return out[["#CHROM", "POS", "ANCESTRY", "P_snputils"]].drop_duplicates(
        subset=["#CHROM", "POS", "ANCESTRY"]
    ).reset_index(drop=True)


def read_snputils_admixture_mapping_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["#CHROM", "POS", "ANCESTRY", "P", "TEST", "ERRCODE"],
    )


def summarize_pvalue_concordance(
    analysis: str,
    snputils_df: pd.DataFrame,
    plink_df: pd.DataFrame,
    key_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = snputils_df.merge(plink_df, on=list(key_cols), how="inner")
    if merged.empty:
        raise RuntimeError(f"No overlapping {analysis} rows remained after merging snputils and PLINK results.")

    eps = np.finfo(np.float64).tiny
    p_snputils = np.clip(merged["P_snputils"].to_numpy(dtype=np.float64), eps, 1.0)
    p_plink = np.clip(merged["P_plink"].to_numpy(dtype=np.float64), eps, 1.0)
    neglog_snputils = -np.log10(p_snputils)
    neglog_plink = -np.log10(p_plink)

    merged = merged.copy()
    merged["abs_p_diff"] = np.abs(p_snputils - p_plink)
    merged["neglog10_p_snputils"] = neglog_snputils
    merged["neglog10_p_plink"] = neglog_plink
    merged["abs_neglog10_p_diff"] = np.abs(neglog_snputils - neglog_plink)

    pearson_r = float(np.corrcoef(neglog_snputils, neglog_plink)[0, 1]) if len(merged) > 1 else float("nan")
    rmse_neglog10 = float(np.sqrt(np.mean(np.square(neglog_snputils - neglog_plink))))

    summary = {
        "analysis": analysis,
        "n": int(len(merged)),
        "pearson_r_neglog10_p": pearson_r,
        "rmse_neglog10_p": rmse_neglog10,
        "max_abs_neglog10_p_diff": float(np.max(merged["abs_neglog10_p_diff"].to_numpy(dtype=np.float64))),
        "median_abs_p_diff": float(np.median(merged["abs_p_diff"].to_numpy(dtype=np.float64))),
        "max_abs_p_diff": float(np.max(merged["abs_p_diff"].to_numpy(dtype=np.float64))),
    }
    return merged, summary


def print_stdout_summary(summary: pd.DataFrame, timing: pd.DataFrame) -> None:
    print(summary.to_string(index=False))
    print(timing.to_string(index=False))
    rows = timing[timing["analysis"] == "admixture_mapping"].set_index("method")
    if {"snputils", "plink"}.issubset(rows.index):
        mean_col = "seconds_mean" if "seconds_mean" in rows.columns else "seconds"
        std_col = "seconds_std" if "seconds_std" in rows.columns else None
        snp_s = float(rows.loc["snputils", mean_col])
        plink_s = float(rows.loc["plink", mean_col])

        def _fmt(m: str) -> str:
            s = float(rows.loc[m, mean_col])
            if std_col is not None:
                sd = float(rows.loc[m, std_col])
                return f"{s:.3f}s ±{sd:.3f}s"
            return f"{s:.3f}s"

        if snp_s > 0:
            print(
                f"admixture_mapping wall time: snputils {_fmt('snputils')}, "
                f"MSP→PGEN→PLINK {_fmt('plink')} "
                f"(ratio plink/snputils {plink_s / snp_s:.3f}x)"
            )
        else:
            print(f"admixture_mapping wall time: snputils {_fmt('snputils')}, MSP→PGEN→PLINK {_fmt('plink')}")


def draw_neglog10_p_panel(ax, merged: pd.DataFrame, *, title: str, color: str) -> None:
    x = merged["neglog10_p_plink"].to_numpy(dtype=float)
    y = merged["neglog10_p_snputils"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        ax.set_title(title)
        ax.text(0.5, 0.5, r"No finite paired $-\log_{10}(P)$", transform=ax.transAxes, ha="center", va="center")
        return

    ax.scatter(
        x[finite],
        y[finite],
        s=18,
        alpha=0.72,
        color=color,
        edgecolor="white",
        linewidth=0.25,
        rasterized=True,
    )
    lo = float(min(x[finite].min(), y[finite].min()))
    hi = float(max(x[finite].max(), y[finite].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1e-6
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("PLINK $-\\log_{10}(P)$")
    ax.set_ylabel("snputils $-\\log_{10}(P)$")
    ax.set_title(title)

    diff = y[finite] - x[finite]
    if finite.sum() > 1 and np.std(x[finite]) > 0 and np.std(y[finite]) > 0:
        r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    else:
        r = float("nan")
    rmse = float(np.sqrt(np.mean(diff**2))) if finite.any() else float("nan")
    max_abs = float(np.max(np.abs(diff))) if finite.any() else float("nan")

    ax.text(
        0.04,
        0.96,
        f"n = {int(finite.sum())}\nr = {r:.6g}\nRMSE = {rmse:.2e}\nmax abs Δ = {max_abs:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )
    ax.grid(color="0.9", linewidth=0.6)


def _draw_timing_bar_panel(ax, timing_df: pd.DataFrame, analysis: str, title: str) -> None:
    subset = (
        timing_df.loc[timing_df["analysis"] == analysis]
        .drop_duplicates("method")
        .set_index("method")
    )
    mean_col = "seconds_mean" if "seconds_mean" in subset.columns else "seconds"
    std_col = "seconds_std" if "seconds_std" in subset.columns else None

    order = ("snputils", "plink")
    labels = ("snputils", "MSP→PGEN→PLINK")
    times = [float(subset.loc[m, mean_col]) if m in subset.index else float("nan") for m in order]
    errs = (
        [float(subset.loc[m, std_col]) if m in subset.index else float("nan") for m in order]
        if std_col is not None
        else None
    )

    colors = ("#0072B2", "#CC79A7")
    bars = ax.bar(
        labels,
        times,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        yerr=errs,
        capsize=5,
        error_kw={"linewidth": 1.4, "ecolor": "0.25", "capthick": 1.4},
    )
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    finite_times = [t for t in times if np.isfinite(t)]
    ymax = max(finite_times) if finite_times else 1.0
    # Extra headroom so error-bar caps don't clip at the top of the axes.
    err_max = max((e for e in (errs or []) if np.isfinite(e)), default=0.0)
    ax.set_ylim(0, (ymax + err_max) * 1.18 if ymax > 0 else 1.0)
    for bar, sec, err in zip(bars, times, errs if errs else [None] * len(times)):
        if np.isfinite(sec):
            label = f"{sec:.2f}s"
            if err is not None and np.isfinite(err):
                label += f"±{err:.2f}s"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (err if err is not None and np.isfinite(err) else 0.0),
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )


def save_association_plink_pdfs(
    merged_admixture: pd.DataFrame,
    pdf_path: Path,
    *,
    timing_df: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    pdf_path = pdf_path.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig_admix, ax_admix = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
        draw_neglog10_p_panel(
            ax_admix, merged_admixture, title="Admixture mapping", color="#009E73"
        )
        pdf.savefig(fig_admix, bbox_inches="tight")
        plt.close(fig_admix)

        fig_admix_t, ax_admix_t = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
        _draw_timing_bar_panel(
            ax_admix_t,
            timing_df,
            "admixture_mapping",
            "Admixture mapping wall time",
        )
        pdf.savefig(fig_admix_t, bbox_inches="tight")
        plt.close(fig_admix_t)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare snputils admixture-mapping p-values against PLINK on synthetic data. "
            "Writes CSV results, timings, and a summary PDF."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/association_plink_concordance"))
    parser.add_argument(
        "--plink-bin",
        default="plink2",
        help="PLINK binary name or absolute path. Default: %(default)s.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Samples per synthetic benchmark dataset. Default: %(default)s.",
    )
    parser.add_argument(
        "--admix-windows",
        type=int,
        default=DEFAULT_ADMIX_WINDOWS,
        help="Synthetic admixture-mapping window count. Default: %(default)s.",
    )
    parser.add_argument(
        "--n-covariates",
        type=int,
        default=DEFAULT_N_COVARIATES,
        help="Covariate count shared by both analyses. Default: %(default)s.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "snputils admixture-mapping windows per in-memory chunk; increase if RAM allows "
            "(fewer chunks, less overhead on very large variant counts). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed for both synthetic datasets. Default: %(default)s.",
    )
    parser.add_argument(
        "--admix-format",
        choices=["vcf", "pgen"],
        default="vcf",
        help=(
            "File format used to pass LAI data to PLINK for admixture mapping. "
            "'vcf' (default) writes one VCF per ancestry and uses --vcf; "
            "'pgen' writes one PGEN fileset per ancestry and uses --pfile."
        ),
    )
    parser.add_argument(
        "--snputils-admix-input",
        choices=["msp", "laiobj"],
        default="msp",
        help=(
            "Input source used for snputils admixture mapping. 'msp' reads the "
            "written text MSP; 'laiobj' uses the in-memory LocalAncestryObject, "
            "matching the source used to write PLINK ancestry files."
        ),
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=DEFAULT_N_REPS,
        help=(
            "Number of independent timing repetitions for each method. "
            "Rep 1 is the canonical run (concordance is verified); subsequent reps "
            "are timing-only. Mean ± std are reported and shown in the bar plot. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the synthetic inputs and recompute concordance even if cached CSVs already exist.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="PDF path for concordance and timing plots. Default: WORK_DIR/concordance_grid.pdf",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.n_samples <= 3:
        raise ValueError("--n-samples must be at least 4.")
    if args.admix_windows <= 0:
        raise ValueError("--admix-windows must be positive.")
    if args.n_covariates < 0:
        raise ValueError("--n-covariates must be >= 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.n_reps < 1:
        raise ValueError("--n-reps must be at least 1.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate_args(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    paths = result_paths(args.work_dir)
    pdf_path = args.output_pdf if args.output_pdf is not None else args.work_dir / "concordance_grid.pdf"
    if cached_results_exist(paths) and not args.force:
        summary = pd.read_csv(paths["summary"])
        timing = pd.read_csv(paths["timing"])
        save_association_plink_pdfs(
            pd.read_csv(paths["concordance_admixture_mapping"]),
            pdf_path,
            timing_df=timing,
        )
        print(f"Using cached concordance and timing results under {paths['result_dir']} (use --force to recompute).")
        print_stdout_summary(summary, timing)
        print(f"Wrote {pdf_path}")
        return 0

    plink_bin = require_plink(args.plink_bin)

    data_dir = args.work_dir / "data"
    plink_dir = args.work_dir / "plink"
    data_dir.mkdir(parents=True, exist_ok=True)
    plink_dir.mkdir(parents=True, exist_ok=True)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)

    admix_data = build_synthetic_admixture_dataset(
        n_samples=args.n_samples,
        n_windows=args.admix_windows,
        n_covariates=args.n_covariates,
        seed=args.seed + 1,
    )
    admix_msp = data_dir / "admixture_mapping.msp"
    admix_phe = data_dir / "admixture_mapping.phe"
    admix_covar = data_dir / "admixture_mapping.covar"
    # Prefix used when writing ancestry-specific files for PLINK.
    # VCF path keeps the .vcf suffix so AdmixtureMappingVCFWriter can detect the
    # extension; PGEN path is a bare stem (PGENWriter appends .pgen/.psam/.pvar).
    if args.admix_format == "pgen":
        admix_file_prefix = data_dir / "admixture_mapping"
    else:
        admix_file_prefix = data_dir / "admixture_mapping.vcf"
    remove_if_exists(admix_msp)
    remove_if_exists(admix_phe)
    remove_if_exists(admix_covar)
    # Remove stale per-ancestry files from both formats to avoid leftover artifacts.
    for ancestry_label in DEFAULT_ANCESTRY_MAP.values():
        remove_if_exists(
            (data_dir / "admixture_mapping.vcf").with_name(
                f"admixture_mapping_{ancestry_label}.vcf"
            )
        )
        remove_pgen_fileset(data_dir / f"admixture_mapping_{ancestry_label}")
    if args.snputils_admix_input == "msp":
        admix_data["laiobj"].save_msp(admix_msp)
    write_quantitative_phe(admix_phe, admix_data["sample_ids"], DEFAULT_PHE_ID, admix_data["phenotype"])
    if args.n_covariates > 0:
        write_covar(admix_covar, admix_data["sample_ids"], admix_data["covar_names"], admix_data["covar_matrix"])

    covar_col_nums = f"1-{args.n_covariates}" if args.n_covariates > 0 else None
    _snputils_admix_source = (
        "in-memory LocalAncestryObject"
        if args.snputils_admix_input == "laiobj"
        else "MSP (uncompressed TSV)"
    )
    _admix_plink_scope = (
        f"snputils run_admixture_mapping on {_snputils_admix_source} (return_results=False) vs "
        f"plink ancestry-PGEN (--pfile --glm)"
        if args.admix_format == "pgen"
        else
        f"snputils run_admixture_mapping on {_snputils_admix_source} (return_results=False) vs "
        "plink ancestry-VCF plus plink --vcf --glm"
    )
    metadata = {
        "dataset_strategy": "synthetic",
        "trait_model": (
            "sparse_local_ancestry_quantitative_with_global_ancestry_covariates"
            if args.n_covariates > 0
            else "sparse_local_ancestry_quantitative"
        ),
        "plink_bin": plink_bin,
        "n_samples": int(args.n_samples),
        "admix_windows": int(args.admix_windows),
        "n_covariates": int(args.n_covariates),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "admix_format": args.admix_format,
        "snputils_admix_input": args.snputils_admix_input,
        "admixture_mapping_causal_windows_0_based": [int(x) for x in admix_data["effect_windows"]],
        "admixture_mapping_causal_ancestries": [
            DEFAULT_ANCESTRY_MAP[str(int(x))] for x in admix_data["effect_ancestries"]
        ],
        "admixture_mapping_causal_effects": [float(x) for x in admix_data["effect_sizes"]],
        "admixture_mapping_input": (
            "in_memory_laiobj"
            if args.snputils_admix_input == "laiobj"
            else str(admix_msp.resolve())
        ),
        "admixture_mapping_plink_file_prefix": str(admix_file_prefix.resolve()),
        "runtime_scope": {
            "admixture_mapping": _admix_plink_scope,
        },
    }
    with (args.work_dir / "benchmark_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    remove_if_exists(paths["result_dir"] / "snputils_admixture_mapping.tsv.gz")

    snputils_admix_input = (
        admix_data["laiobj"] if args.snputils_admix_input == "laiobj" else admix_msp
    )

    snputils_times: list[float] = []
    plink_times: list[float] = []

    t0 = time.perf_counter()
    run_admixture_mapping(
        phe_path=admix_phe,
        lai_source=snputils_admix_input,
        results_path=paths["snputils_admixture_mapping"],
        phe_id=DEFAULT_PHE_ID,
        batch_size=args.batch_size,
        keep_hla=True,
        return_results=False,
        quantitative=True,
        covar_path=admix_covar if args.n_covariates > 0 else None,
        covar_col_nums=covar_col_nums,
    )
    snputils_times.append(time.perf_counter() - t0)

    snputils_admix_p = extract_snputils_admixture_pvalues(
        read_snputils_admixture_mapping_results(paths["snputils_admixture_mapping"])
    )

    plink_admix_p, wall_plink = run_plink_admixture_mapping_from_lai(
        plink_bin,
        admix_data["laiobj"],
        admix_file_prefix,
        admix_phe,
        admix_covar if args.n_covariates > 0 else None,
        admix_data["covar_names"],
        plink_dir,
        admix_format=args.admix_format,
    )
    plink_times.append(wall_plink)
    plink_admix_p.to_csv(paths["plink_admixture_mapping"], index=False)

    _timing_snputils_path = paths["result_dir"] / "_snputils_admixture_mapping_timing.tsv"
    for rep in range(1, args.n_reps):
        print(f"  timing rep {rep + 1}/{args.n_reps} ...", flush=True)
        t0 = time.perf_counter()
        run_admixture_mapping(
            phe_path=admix_phe,
            lai_source=snputils_admix_input,
            results_path=_timing_snputils_path,
            phe_id=DEFAULT_PHE_ID,
            batch_size=args.batch_size,
            keep_hla=True,
            return_results=False,
            quantitative=True,
            covar_path=admix_covar if args.n_covariates > 0 else None,
            covar_col_nums=covar_col_nums,
        )
        snputils_times.append(time.perf_counter() - t0)

        _, t_plink = run_plink_admixture_mapping_from_lai(
            plink_bin,
            admix_data["laiobj"],
            admix_file_prefix,
            admix_phe,
            admix_covar if args.n_covariates > 0 else None,
            admix_data["covar_names"],
            plink_dir,
            admix_format=args.admix_format,
        )
        plink_times.append(t_plink)
    remove_if_exists(_timing_snputils_path)

    admix_merged, admix_summary = summarize_pvalue_concordance(
        "admixture_mapping",
        snputils_admix_p,
        plink_admix_p,
        ["#CHROM", "POS", "ANCESTRY"],
    )

    admix_merged.to_csv(paths["concordance_admixture_mapping"], index=False)

    summary = pd.DataFrame([admix_summary])
    summary.to_csv(paths["summary"], index=False)

    ddof = 1 if len(snputils_times) > 1 else 0
    timing = pd.DataFrame(
        [
            {
                "analysis": "admixture_mapping",
                "method": "snputils",
                "seconds_mean": float(np.mean(snputils_times)),
                "seconds_std": float(np.std(snputils_times, ddof=ddof)),
                "n_reps": len(snputils_times),
            },
            {
                "analysis": "admixture_mapping",
                "method": "plink",
                "seconds_mean": float(np.mean(plink_times)),
                "seconds_std": float(np.std(plink_times, ddof=ddof)),
                "n_reps": len(plink_times),
            },
        ]
    )
    timing.to_csv(paths["timing"], index=False)

    save_association_plink_pdfs(admix_merged, pdf_path, timing_df=timing)
    print_stdout_summary(summary, timing)
    print(f"Wrote {paths['result_dir']} and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
