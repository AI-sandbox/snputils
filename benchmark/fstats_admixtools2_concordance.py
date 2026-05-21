#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from snputils.datasets import load_dataset
from snputils.datasets._registry import get_dataset_spec
from snputils.snp.genobj import SNPObject
from snputils.snp.io.read.auto import SNPReader
from snputils.stats import d_stat, f2, f3, f4, f4_ratio, fst, genomic_block_labels
from snputils.stats.fstats import _prepare_inputs


DEFAULT_BENCHMARK_POPULATIONS = (
    "YRI",
    "LWK",
    "MSL",
    "GWD",
    "CEU",
    "FIN",
    "GBR",
    "IBS",
    "CHB",
    "JPT",
    "GIH",
    "PEL",
)

BENCHMARK_1KGP_RESOURCE = "phase3"


def local_registry_autosome_vcfs(genotype_dir: Path, *, resource: str = BENCHMARK_1KGP_RESOURCE) -> list[Path]:
    chrom_resource = get_dataset_spec("1kgp").genotype_resource(resource)
    root = Path(genotype_dir).expanduser().resolve()
    paths = [root / chrom_resource.filename(c) for c in chrom_resource.chromosomes]
    missing_or_empty = [p for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing_or_empty:
        sample = ", ".join(p.name for p in missing_or_empty[:5])
        extra = "" if len(missing_or_empty) <= 5 else f" (and {len(missing_or_empty) - 5} more)"
        raise FileNotFoundError(
            f"Missing or empty VCF files under {root} for registry resource {resource!r}. "
            f"Expected filenames like {chrom_resource.filename(chrom_resource.chromosomes[0])!r}; "
            f"not found or empty: {sample}{extra}."
        )
    return paths


STAT_ORDER = ("f2", "f3", "f4", "d_stat", "f4_ratio", "fst")
STAT_COLORS = {
    "f2": "#009E73",
    "f3": "#0072B2",
    "f4": "#D55E00",
    "d_stat": "#CC79A7",
    "f4_ratio": "#56B4E9",
    "fst": "#E69F00",
}

STAT_LABELS: dict[str, str] = {
    "f2": r"$f_2$",
    "f3": r"$f_3$",
    "f4": r"$f_4$",
    "d_stat": r"$D$-statistic",
    "f4_ratio": r"$f_4$-ratio",
    "fst": r"$F_{\mathrm{ST}}$",
}
DEFAULT_MAX_D_STAT_COMBINATIONS = 1_000
DEFAULT_MAX_F4_RATIO_COMBINATIONS = 1_000

F4_RATIO_KEY_COLS = ["pop1", "pop2", "pop3", "pop4", "pop5"]

DEFAULT_F4_RATIO_MIN_ABS_DEN = 1e-4
DEFAULT_F4_RATIO_MIN_ABS_DEN_Z = 3.0
DEFAULT_F4_RATIO_EST_MIN = 0.0
DEFAULT_F4_RATIO_EST_MAX = 1.0
DEFAULT_F4_RATIO_ALPHA_BINS = 20
DEFAULT_N_REPS = 5

CONCORDANCE_POP_KEYS: dict[str, tuple[str, ...]] = {
    "f2": ("pop1", "pop2"),
    "f3": ("pop1", "pop2", "pop3"),
    "f4": ("pop1", "pop2", "pop3", "pop4"),
    "d_stat": ("pop1", "pop2", "pop3", "pop4"),
    "f4_ratio": ("pop1", "pop2", "pop3", "pop4", "pop5"),
    "fst": ("pop1", "pop2"),
}

TIMING_BAR_COLORS = {"snputils": "#0072B2", "admixtools2": "#CC79A7"}

TIMING_N_METHOD_SLOTS = 2
TIMING_GROUP_PITCH = 0.44
TIMING_BAR_WIDTH = min(0.26, TIMING_GROUP_PITCH * 0.92)

GRID_CONCORDANCE_SUP_LABEL_FONTSIZE = 18
TIMING_TITLE_FONTSIZE = 13
TIMING_YLABEL_FONTSIZE = 15
TIMING_XTICK_LABEL_FONTSIZE = 12
TIMING_YTICK_LABEL_FONTSIZE = 11
TIMING_BAR_LABEL_FONTSIZE = 11


def evenly_spaced_subset(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    indices = np.linspace(0, len(df) - 1, max_rows, dtype=int)
    return df.iloc[indices].reset_index(drop=True)


def statistic_combinations(
    populations: Sequence[str],
    *,
    max_d_stat_combinations: int | None = DEFAULT_MAX_D_STAT_COMBINATIONS,
    max_f4_ratio_combinations: int | None = DEFAULT_MAX_F4_RATIO_COMBINATIONS,
) -> dict[str, pd.DataFrame]:
    pairs = pd.DataFrame(list(itertools.combinations(populations, 2)), columns=["pop1", "pop2"])
    triples = pd.DataFrame(
        [
            (target, ref1, ref2)
            for target in populations
            for ref1, ref2 in itertools.combinations([p for p in populations if p != target], 2)
        ],
        columns=["pop1", "pop2", "pop3"],
    )
    quads = pd.DataFrame(
        list(itertools.permutations(populations, 4)),
        columns=["pop1", "pop2", "pop3", "pop4"],
    )
    f4_ratios = pd.DataFrame(
        list(itertools.permutations(populations, 5)),
        columns=["pop1", "pop2", "pop3", "pop4", "pop5"],
    )
    d_stat_combos = evenly_spaced_subset(quads, max_d_stat_combinations)

    return {
        "f2": pairs,
        "f3": triples,
        "f4": quads,
        "d_stat": d_stat_combos,
        "f4_ratio": f4_ratios,
        "fst": pairs,
    }


def stable_f4_ratio_combinations(
    fstat_data,
    block_labels: np.ndarray,
    combos: pd.DataFrame,
    *,
    max_rows: int | None,
    min_abs_den: float = DEFAULT_F4_RATIO_MIN_ABS_DEN,
    min_abs_den_z: float = DEFAULT_F4_RATIO_MIN_ABS_DEN_Z,
    est_min: float = DEFAULT_F4_RATIO_EST_MIN,
    est_max: float = DEFAULT_F4_RATIO_EST_MAX,
) -> pd.DataFrame:
    """
    Select qpf4ratio combinations whose denominator is not near zero and whose
    alpha estimate is in the usual interpretable range.

    For a five-pop tuple p1,p2,p3,p4,p5, ADMIXTOOLS qpf4ratio uses:
      numerator   f4(p1, p2; p3, p4)
      denominator f4(p1, p2; p5, p4)
    """
    combos = combos.reset_index(drop=True)
    if combos.empty:
        return combos

    num_f4 = f4(
        fstat_data,
        a=combos["pop1"].tolist(),
        b=combos["pop2"].tolist(),
        c=combos["pop3"].tolist(),
        d=combos["pop4"].tolist(),
        blocks=block_labels,
    )

    den_f4 = f4(
        fstat_data,
        a=combos["pop1"].tolist(),
        b=combos["pop2"].tolist(),
        c=combos["pop5"].tolist(),
        d=combos["pop4"].tolist(),
        blocks=block_labels,
    )

    scored = combos.copy()
    scored["_num_est"] = num_f4["est"].to_numpy(dtype=float)
    scored["_den_est"] = den_f4["est"].to_numpy(dtype=float)
    scored["_den_z"] = den_f4["z"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        scored["_alpha_probe"] = scored["_num_est"] / scored["_den_est"]

    stable = (
        np.isfinite(scored["_alpha_probe"])
        & np.isfinite(scored["_den_est"])
        & np.isfinite(scored["_den_z"])
        & (scored["_den_est"].abs() >= min_abs_den)
        & (scored["_den_z"].abs() >= min_abs_den_z)
        & (scored["_alpha_probe"] >= est_min)
        & (scored["_alpha_probe"] <= est_max)
    )

    scored = scored.loc[stable].copy()
    if scored.empty:
        raise RuntimeError(
            "No stable f4-ratio combinations passed the filter. "
            "Try lowering DEFAULT_F4_RATIO_MIN_ABS_DEN or DEFAULT_F4_RATIO_MIN_ABS_DEN_Z."
        )

    scored["_den_abs"] = scored["_den_est"].abs()
    scored["_den_z_abs"] = scored["_den_z"].abs()

    if max_rows is not None and max_rows > 0:
        bin_edges = np.linspace(est_min, est_max, DEFAULT_F4_RATIO_ALPHA_BINS + 1)
        scored["_alpha_bin"] = pd.cut(scored["_alpha_probe"], bins=bin_edges, include_lowest=True, labels=False)
        per_bin = max(1, math.ceil(max_rows / DEFAULT_F4_RATIO_ALPHA_BINS))
        selected = (
            scored.sort_values(["_alpha_bin", "_den_z_abs", "_den_abs"], ascending=[True, False, False])
            .groupby("_alpha_bin", dropna=True, group_keys=False)
            .head(per_bin)
        )
        selected = selected.sort_values("_alpha_probe")
        if len(selected) > max_rows:
            selected = evenly_spaced_subset(selected, max_rows)
        elif len(selected) < max_rows:
            remaining = scored.drop(index=selected.index, errors="ignore")
            remaining = remaining.sort_values("_alpha_probe")
            selected = pd.concat([selected, evenly_spaced_subset(remaining, max_rows - len(selected))], axis=0)
        scored = selected.sort_values("_alpha_probe")

    return scored[F4_RATIO_KEY_COLS].reset_index(drop=True)


def result_paths(work_dir: Path) -> dict[str, Path]:
    result_dir = work_dir / "results"
    paths = {
        "result_dir": result_dir,
        "combo_dir": work_dir / "combinations",
        "summary": result_dir / "concordance_summary.csv",
        "timing": result_dir / "timing_results.csv",
    }
    for stat in STAT_ORDER:
        paths[f"snputils_{stat}"] = result_dir / f"snputils_{stat}.csv"
        paths[f"admixtools2_{stat}"] = result_dir / f"admixtools2_{stat}.csv"
        paths[f"concordance_{stat}"] = result_dir / f"concordance_{stat}.csv"
    return paths


def cached_results_exist(paths: dict[str, Path]) -> bool:
    required = [paths["summary"], paths["timing"]]
    required.extend(paths[f"concordance_{stat}"] for stat in STAT_ORDER)
    return all(path.exists() for path in required)


def run_snputils(
    plink_prefix: Path,
    subset: SNPObject,
    block_size_bp: int,
    combo_csv_dir: Path,
    *,
    max_d_stat_combinations: int | None,
    max_f4_ratio_combinations: int | None,
) -> dict[str, pd.DataFrame]:
    if subset.sample_fid is None:
        raise RuntimeError("Subset is missing population labels in SNPObject.sample_fid.")
    populations = list(dict.fromkeys(str(pop) for pop in subset.sample_fid))
    combos = statistic_combinations(
        populations,
        max_d_stat_combinations=max_d_stat_combinations,
        max_f4_ratio_combinations=max_f4_ratio_combinations,
    )

    reader = SNPReader(plink_prefix.with_suffix(".bed"))
    snpobj = reader.read(sum_strands=True)

    if snpobj.samples is None or snpobj.genotypes is None:
        raise RuntimeError("BEDReader did not return genotypes or sample IDs.")
    if list(snpobj.samples) != list(subset.samples):
        raise RuntimeError("Sample order from PLINK does not match the VCF subset (IID mismatch).")

    block_labels = genomic_block_labels(snpobj.variants_chrom, snpobj.variants_pos, block_size_bp)
    fstat_data = _prepare_inputs(snpobj)

    combos["f4_ratio"] = stable_f4_ratio_combinations(
        fstat_data,
        block_labels,
        combos["f4_ratio"],
        max_rows=max_f4_ratio_combinations,
    )

    f2_df = f2(
        fstat_data,
        pop1=combos["f2"]["pop1"].tolist(),
        pop2=combos["f2"]["pop2"].tolist(),
        blocks=block_labels,
        apply_correction=True,
    )

    f3_df = (
        f3(
            fstat_data,
            target=combos["f3"]["pop1"].tolist(),
            ref1=combos["f3"]["pop2"].tolist(),
            ref2=combos["f3"]["pop3"].tolist(),
            blocks=block_labels,
            apply_correction=True,
        )
        .rename(columns={"target": "pop1", "ref1": "pop2", "ref2": "pop3"})
    )

    f4_df = (
        f4(
            fstat_data,
            a=combos["f4"]["pop1"].tolist(),
            b=combos["f4"]["pop2"].tolist(),
            c=combos["f4"]["pop3"].tolist(),
            d=combos["f4"]["pop4"].tolist(),
            blocks=block_labels,
        )
        .rename(columns={"a": "pop1", "b": "pop2", "c": "pop3", "d": "pop4"})
    )

    d_stat_df = (
        d_stat(
            fstat_data,
            a=combos["d_stat"]["pop1"].tolist(),
            b=combos["d_stat"]["pop2"].tolist(),
            c=combos["d_stat"]["pop3"].tolist(),
            d=combos["d_stat"]["pop4"].tolist(),
            blocks=block_labels,
        )
        .rename(columns={"a": "pop1", "b": "pop2", "c": "pop3", "d": "pop4"})
    )

    f4_ratio_combos = combos["f4_ratio"]
    f4_ratio_num = list(
        zip(
            f4_ratio_combos["pop1"],
            f4_ratio_combos["pop2"],
            f4_ratio_combos["pop3"],
            f4_ratio_combos["pop4"],
        )
    )
    f4_ratio_den = list(
        zip(
            f4_ratio_combos["pop1"],
            f4_ratio_combos["pop2"],
            f4_ratio_combos["pop5"],
            f4_ratio_combos["pop4"],
        )
    )
    f4_ratio_df = pd.concat(
        [
            f4_ratio_combos.reset_index(drop=True),
            f4_ratio(
                fstat_data,
                num=f4_ratio_num,
                den=f4_ratio_den,
                blocks=block_labels,
            ).drop(columns=["num", "den"], errors="ignore"),
        ],
        axis=1,
    )

    fst_df = fst(
        fstat_data,
        pop1=combos["fst"]["pop1"].tolist(),
        pop2=combos["fst"]["pop2"].tolist(),
        blocks=block_labels,
        method="hudson",
    )

    combo_csv_dir.mkdir(parents=True, exist_ok=True)
    for stat, stat_combos in combos.items():
        stat_combos.to_csv(combo_csv_dir / f"{stat}_combinations.csv", index=False)

    return {
        "f2": f2_df,
        "f3": f3_df,
        "f4": f4_df,
        "d_stat": d_stat_df,
        "f4_ratio": f4_ratio_df,
        "fst": fst_df,
    }


def write_admixtools_runner(
    prefix: Path,
    populations: Sequence[str],
    block_size_bp: int,
    combo_dir: Path,
    outdir: Path,
) -> Path:
    script = outdir / "run_admixtools2_concordance.R"
    script.write_text(
        textwrap.dedent(
            f"""
            suppressPackageStartupMessages(library(admixtools))

            if (!exists("read_table2", envir = asNamespace("admixtools"), inherits = TRUE)) {{
              stop("ADMIXTOOLS2 currently expects readr::read_table2; install readr 2.1.5 if your readr release removed it.")
            }}

            prefix <- {json.dumps(str(prefix.resolve()))}
            combo_dir <- {json.dumps(str(combo_dir.resolve()))}
            outdir <- {json.dumps(str(outdir.resolve()))}
            pops <- {"c(" + ", ".join(json.dumps(pop) for pop in populations) + ")"}
            block_size_bp <- {int(block_size_bp)}

            f2_combos <- as.matrix(read.csv(file.path(combo_dir, "f2_combinations.csv"), stringsAsFactors = FALSE))
            f3_combos <- as.matrix(read.csv(file.path(combo_dir, "f3_combinations.csv"), stringsAsFactors = FALSE))
            f4_combos <- as.matrix(read.csv(file.path(combo_dir, "f4_combinations.csv"), stringsAsFactors = FALSE))
            d_stat_combos <- as.matrix(read.csv(file.path(combo_dir, "d_stat_combinations.csv"), stringsAsFactors = FALSE))
            f4_ratio_combos <- as.matrix(read.csv(file.path(combo_dir, "f4_ratio_combinations.csv"), stringsAsFactors = FALSE))

            f2_blocks <- f2_from_geno(
              prefix,
              pops = pops,
              blgsize = block_size_bp,
              maxmiss = 0,
              minmaf = 0,
              maxmaf = 0.5,
              poly_only = FALSE,
              format = "plink",
              adjust_pseudohaploid = FALSE,
              remove_na = FALSE,
              apply_corr = TRUE,
              verbose = FALSE
            )
            f2_out <- as.data.frame(f2(f2_blocks, f2_combos, sure = TRUE, verbose = FALSE))
            f3_out <- as.data.frame(f3(f2_blocks, f3_combos, sure = TRUE, verbose = FALSE))
            f4_out <- as.data.frame(f4(f2_blocks, f4_combos, sure = TRUE, verbose = FALSE))
            d_stat_out <- as.data.frame(qpdstat(
              prefix,
              d_stat_combos,
              blgsize = block_size_bp,
              f4mode = FALSE,
              allsnps = FALSE,
              poly_only = FALSE,
              verbose = FALSE
            ))
            f4_ratio_out <- as.data.frame(qpf4ratio(
              f2_blocks,
              f4_ratio_combos,
              poly_only = FALSE,
              verbose = FALSE
            ))
            fst_out <- as.data.frame(fst(
              prefix,
              pops,
              blgsize = block_size_bp,
              maxmiss = 0,
              minmaf = 0,
              maxmaf = 0.5,
              poly_only = FALSE,
              apply_corr = TRUE,
              verbose = FALSE
            ))

            write.csv(f2_out, file.path(outdir, "admixtools2_f2.csv"), row.names = FALSE)
            write.csv(f3_out, file.path(outdir, "admixtools2_f3.csv"), row.names = FALSE)
            write.csv(f4_out, file.path(outdir, "admixtools2_f4.csv"), row.names = FALSE)
            names(d_stat_out)[names(d_stat_out) == "f4"] <- "est"
            write.csv(d_stat_out, file.path(outdir, "admixtools2_d_stat.csv"), row.names = FALSE)
            names(f4_ratio_out)[names(f4_ratio_out) == "alpha"] <- "est"
            write.csv(f4_ratio_out, file.path(outdir, "admixtools2_f4_ratio.csv"), row.names = FALSE)
            write.csv(fst_out, file.path(outdir, "admixtools2_fst.csv"), row.names = FALSE)
            """
        ).lstrip()
    )
    return script


def run_admixtools2(
    prefix: Path,
    populations: Sequence[str],
    block_size_bp: int,
    combo_dir: Path,
    outdir: Path,
) -> None:
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript is not on PATH; install R before running ADMIXTOOLS2 concordance.")
    check = subprocess.run(
        ["Rscript", "-e", "quit(status = !requireNamespace('admixtools', quietly = TRUE))"],
        cwd=outdir,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "The ADMIXTOOLS2 R package is not installed. A reproducible install is:\n"
            "Rscript -e 'install.packages(\"https://cran.r-project.org/src/contrib/Archive/readr/readr_2.1.5.tar.gz\", "
            "repos = NULL, type = \"source\")'\n"
            "MAKEFLAGS='CXX11STD=-std=gnu++14' Rscript -e 'install.packages(\"admixtools\", "
            "repos = c(\"https://evolecolgroup.r-universe.dev\", \"https://cloud.r-project.org\"))'"
        )

    script = write_admixtools_runner(
        prefix,
        populations,
        block_size_bp,
        combo_dir,
        outdir,
    )
    subprocess.run(["Rscript", script.name], cwd=outdir, check=True)


def summarize_concordance(
    stat: str,
    snputils_df: pd.DataFrame,
    admixtools_df: pd.DataFrame,
    key_cols: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    merged = snputils_df.merge(
        admixtools_df,
        on=list(key_cols),
        suffixes=("_snputils", "_admixtools2"),
        validate="one_to_one",
    )
    merged["delta_est"] = merged["est_snputils"] - merged["est_admixtools2"]
    merged["abs_delta_est"] = merged["delta_est"].abs()
    x = merged["est_snputils"].to_numpy(dtype=float)
    y = merged["est_admixtools2"].to_numpy(dtype=float)
    diff = x - y
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() > 1 and np.std(x[finite]) > 0 and np.std(y[finite]) > 0:
        r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    else:
        r = float("nan")
    summary = {
        "stat": stat,
        "n": int(finite.sum()),
        "pearson_r": r,
        "max_abs_delta": float(np.max(np.abs(diff[finite]))) if finite.any() else float("nan"),
        "mean_abs_delta": float(np.mean(np.abs(diff[finite]))) if finite.any() else float("nan"),
        "rmse": float(math.sqrt(np.mean(diff[finite] ** 2))) if finite.any() else float("nan"),
    }
    return merged, summary


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

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


def draw_concordance_panel(ax, merged: pd.DataFrame, stat: str) -> None:
    title = STAT_LABELS.get(stat, stat)
    x = merged["est_admixtools2"].to_numpy(dtype=float)
    y = merged["est_snputils"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        ax.set_title(title)
        ax.text(0.5, 0.5, "No finite paired estimates", transform=ax.transAxes, ha="center", va="center")
        return
    ax.scatter(
        x[finite],
        y[finite],
        s=18,
        alpha=0.72,
        color=STAT_COLORS.get(stat, "#4C78A8"),
        edgecolor="white",
        linewidth=0.25,
    )
    lo = float(min(x[finite].min(), y[finite].min()))
    hi = float(max(x[finite].max(), y[finite].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1e-6
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    if finite.sum() > 1 and np.std(x[finite]) > 0 and np.std(y[finite]) > 0:
        r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    else:
        r = float("nan")
    max_abs_delta = float(np.max(np.abs(y[finite] - x[finite]))) if finite.any() else float("nan")
    ax.text(
        0.04,
        0.96,
        f"n = {int(finite.sum())}\nr = {r:.6g}\nmax abs diff = {max_abs_delta:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )
    ax.grid(color="0.9", linewidth=0.6)


def normalize_fstats_timing_csv(timing_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure seconds_mean / seconds_std exist (legacy CSVs used ``seconds`` only)."""
    out = timing_df.copy()
    if "seconds_mean" not in out.columns and "seconds" in out.columns:
        out["seconds_mean"] = out["seconds"]
    if "seconds_mean" not in out.columns:
        raise ValueError("timing CSV must contain seconds_mean or legacy seconds column.")
    if "seconds_std" not in out.columns:
        out["seconds_std"] = np.nan
    return out


def _draw_fstats_timing_bar_panel(ax, timing_df: pd.DataFrame) -> None:
    timing_df = normalize_fstats_timing_csv(timing_df)
    overall = timing_df.loc[timing_df["component"] == "overall"].drop_duplicates("method").set_index("method")
    mean_col = "seconds_mean"
    std_col = "seconds_std"
    order = ["snputils", "admixtools2"]
    labels = ["snputils", "ADMIXTOOLS2"]
    times = [float(overall.loc[m, mean_col]) if m in overall.index else float("nan") for m in order]
    errs_raw = [float(overall.loc[m, std_col]) if m in overall.index else float("nan") for m in order]
    errs = errs_raw if any(math.isfinite(e) for e in errs_raw) else None
    colors = [TIMING_BAR_COLORS[k] for k in order]
    group_pitch = TIMING_GROUP_PITCH
    bar_width = TIMING_BAR_WIDTH
    x_centers = np.arange(TIMING_N_METHOD_SLOTS, dtype=float) * group_pitch

    bars = ax.bar(
        x_centers,
        times,
        width=bar_width,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        yerr=errs,
        capsize=3,
        error_kw={"linewidth": 1.4, "ecolor": "0.25", "capthick": 1.4},
    )
    ax.set_xticks(x_centers)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=TIMING_XTICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TIMING_YTICK_LABEL_FONTSIZE)
    edge_pad = max(0.08, bar_width * 0.35)
    ax.set_xlim(x_centers[0] - bar_width / 2 - edge_pad, x_centers[-1] + bar_width / 2 + edge_pad)
    ax.set_ylabel("Wall-clock time (s)", fontsize=TIMING_YLABEL_FONTSIZE)
    ax.set_title("Overall computation (same PLINK subset)", fontsize=TIMING_TITLE_FONTSIZE)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    finite_times = [t for t in times if np.isfinite(t)]
    ymax = max(finite_times) if finite_times else 1.0
    err_max = max((e for e in (errs or []) if math.isfinite(e)), default=0.0)
    ax.set_ylim(0, (ymax + err_max) * 1.18 if ymax > 0 else 1.0)
    err_list = errs if errs else [None] * len(times)
    for bar, sec, err in zip(bars, times, err_list):
        if np.isfinite(sec):
            label = f"{sec:.2f}s"
            if err is not None and math.isfinite(err):
                label += f"±{err:.2f}s"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (err if err is not None and math.isfinite(err) else 0.0),
                label,
                ha="center",
                va="bottom",
                fontsize=TIMING_BAR_LABEL_FONTSIZE,
            )


def save_concordance_and_timing_pdfs(
    merged_by_stat: dict[str, pd.DataFrame],
    pdf_path: Path,
    *,
    timing_df: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    configure_matplotlib()
    pdf_path = pdf_path.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        stats = [stat for stat in STAT_ORDER if stat in merged_by_stat]
        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 7.0), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, stat in zip(axes_arr, stats):
            draw_concordance_panel(ax, merged_by_stat[stat], stat)
        for ax in axes_arr[len(stats) :]:
            ax.axis("off")
        for idx, ax in enumerate(axes_arr[: len(stats)]):
            row, col = divmod(idx, ncols)
            ax.tick_params(labelbottom=(row == nrows - 1), labelleft=(col == 0))
        fig.supxlabel("ADMIXTOOLS2 estimate", fontsize=GRID_CONCORDANCE_SUP_LABEL_FONTSIZE)
        fig.supylabel("snputils estimate", fontsize=GRID_CONCORDANCE_SUP_LABEL_FONTSIZE)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        timing_df = normalize_fstats_timing_csv(timing_df)
        fig_t, ax_t = plt.subplots(figsize=(4.25, 3.6), constrained_layout=True)
        _draw_fstats_timing_bar_panel(ax_t, timing_df)
        pdf.savefig(fig_t, bbox_inches="tight")
        plt.close(fig_t)


def concordance_summary_table(
    snputils_dfs: dict[str, pd.DataFrame],
    result_dir: Path,
    *,
    pdf_path: Path,
    timing_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    specs = {stat: list(keys) for stat, keys in CONCORDANCE_POP_KEYS.items()}
    summaries = []
    merged_by_stat: dict[str, pd.DataFrame] = {}
    for stat, keys in specs.items():
        admix = pd.read_csv(result_dir / f"admixtools2_{stat}.csv")
        merged, summary = summarize_concordance(stat, snputils_dfs[stat], admix, keys)
        summaries.append(summary)
        merged_by_stat[stat] = merged
    save_concordance_and_timing_pdfs(merged_by_stat, pdf_path, timing_df=timing_df)
    return pd.DataFrame(summaries), merged_by_stat


def load_cached_results(paths: dict[str, Path], pdf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_by_stat = {stat: pd.read_csv(paths[f"concordance_{stat}"]) for stat in STAT_ORDER}
    summary = pd.read_csv(paths["summary"])
    timing = normalize_fstats_timing_csv(pd.read_csv(paths["timing"]))
    save_concordance_and_timing_pdfs(merged_by_stat, pdf_path, timing_df=timing)
    return summary, timing


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare snputils f2/f3/f4/D/f4-ratio/Fst estimates with ADMIXTOOLS2 on a 1000 Genomes subset "
            "(PLINK bed/bim/fam), and save raw concordance CSVs, overall wall-clock timing, and a PDF summary."
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/fstats_admixtools2_concordance"))
    parser.add_argument(
        "--panel-url",
        default=None,
        help="Population panel URL. Default: the 1000 Genomes panel registered in snputils.datasets.",
    )
    parser.add_argument(
        "--genotype-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory that already contains the 22 phase3 autosome VCF.gz files expected by "
            "snputils.datasets (same names as a full ``load_dataset(..., resource=phase3, output_dir=DIR)`` run). "
            "Use this to reuse downloads from another work directory without fetching again."
        ),
    )
    parser.add_argument(
        "--vcf",
        default=None,
        metavar="PATH_OR_URL",
        help=(
            "Single VCF path or URL. If omitted, uses 1000 Genomes phase3 autosomes chr1-chr22 "
            "(max-variants split evenly across chromosomes), from --genotype-dir if set."
        ),
    )
    parser.add_argument(
        "--no-download-vcf",
        dest="download_vcf",
        action="store_false",
        help=(
            "Disable automatic URL downloads. Remote VCF streaming is not supported; use this only with local VCF paths."
        ),
    )
    parser.set_defaults(download_vcf=True)
    parser.add_argument("--populations", nargs="+", default=list(DEFAULT_BENCHMARK_POPULATIONS))
    parser.add_argument(
        "--samples-per-pop",
        type=int,
        default=50,
        help="Number of individuals taken per population from the panel (panel order). Default 50.",
    )
    parser.add_argument("--max-variants", type=int, default=1_000_000)
    parser.add_argument(
        "--max-d-stat-combinations",
        type=int,
        default=DEFAULT_MAX_D_STAT_COMBINATIONS,
        help=(
            "Maximum D-statistic population quadruples to benchmark, sampled deterministically across all permutations. "
            "Use 0 or a negative value to run all combinations. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--max-f4-ratio-combinations",
        type=int,
        default=DEFAULT_MAX_F4_RATIO_COMBINATIONS,
        help=(
            "Maximum f4-ratio population quintuples to benchmark, sampled deterministically across all permutations. "
            "Use 0 or a negative value to run all combinations. Default: %(default)s."
        ),
    )
    parser.add_argument("--block-size-bp", type=int, default=50_000)
    parser.add_argument(
        "--include-monomorphic",
        action="store_true",
        help="Keep sites that are monomorphic in the selected samples",
    )
    parser.add_argument(
        "--skip-admixtools2",
        action="store_true",
        help=(
            "Only write the PLINK subset and subset_metadata.json (skip snputils, R, "
            "and PDF generation)."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help=(
            "Path for benchmark figures (PDF): concordance grid plus overall timing. "
            "Default: WORK_DIR/concordance_grid.pdf"
        ),
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=DEFAULT_N_REPS,
        help=(
            "Number of independent timing repetitions per method. Rep 1 verifies concordance and writes outputs; "
            "later reps are timing-only. Mean ± std are saved and shown on the timing plot. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute raw concordance and timing results even if cached result files already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    paths = result_paths(args.work_dir)
    pdf_path = args.output_pdf if args.output_pdf is not None else args.work_dir / "concordance_grid.pdf"

    if args.n_reps < 1:
        raise ValueError("--n-reps must be at least 1.")

    if cached_results_exist(paths) and not args.force:
        summary, timing = load_cached_results(paths, pdf_path)
        print(f"Using cached raw concordance and timing results under {paths['result_dir']} (use --force to recompute).")
        print(summary.to_string(index=False))
        overall = normalize_fstats_timing_csv(timing)
        overall = overall.loc[overall["component"] == "overall"].drop_duplicates("method").set_index("method")
        mean_col = "seconds_mean"
        std_col = "seconds_std"

        def _fmt_wall(method: str) -> str:
            if method not in overall.index:
                return "n/a"
            m = float(overall.loc[method, mean_col])
            if std_col in overall.columns and math.isfinite(float(overall.loc[method, std_col])):
                sd = float(overall.loc[method, std_col])
                return f"{m:.3f}s ±{sd:.3f}s"
            return f"{m:.3f}s"

        if "snputils" in overall.index and "admixtools2" in overall.index:
            snp_m = float(overall.loc["snputils", mean_col])
            adm_m = float(overall.loc["admixtools2", mean_col])
            ratio_txt = (
                f" (ratio admix/snputils {adm_m / snp_m:.3f}x)"
                if snp_m > 0 and math.isfinite(snp_m) and math.isfinite(adm_m)
                else ""
            )
            print(
                f"Wall time: snputils {_fmt_wall('snputils')}, ADMIXTOOLS2 {_fmt_wall('admixtools2')}{ratio_txt}"
            )
        print(f"Wrote {pdf_path} from cached results")
        return 0

    if args.genotype_dir is not None and args.vcf is not None:
        raise ValueError("Use either --vcf or --genotype-dir, not both.")

    if args.genotype_dir is None:
        genotype_sources = None if args.vcf is None else [args.vcf]
    else:
        genotype_sources = local_registry_autosome_vcfs(args.genotype_dir)

    subset = load_dataset(
        "1kgp",
        resource=BENCHMARK_1KGP_RESOURCE,
        output_dir=args.work_dir,
        genotype_sources=genotype_sources,
        download_genotypes=args.download_vcf,
        populations=args.populations,
        samples_per_population=args.samples_per_pop,
        max_variants=args.max_variants,
        require_biallelic=True,
        require_complete=True,
        require_polymorphic=not args.include_monomorphic,
        snv_only=True,
        panel_url=args.panel_url,
        sum_strands=False,
    )
    if subset.sample_fid is None or subset.sample_sex is None:
        raise RuntimeError("load_dataset did not return population and sex labels on SNPObject.")

    if args.vcf is None:
        print(
            f"Built subset: {subset.n_samples} samples, {subset.n_snps} biallelic SNPs across "
            "1000 Genomes Phase 3 autosomes, "
            f"populations={','.join(args.populations)}"
        )
        prefix = args.work_dir / "kg_autosomes_subset"
    else:
        print(
            f"Built subset: {subset.n_samples} samples, {subset.n_snps} biallelic SNPs from {args.vcf}, "
            f"populations={','.join(args.populations)}"
        )
        prefix = args.work_dir / "kg_single_vcf_subset"

    subset_sample_pops = [str(pop) for pop in subset.sample_fid]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    subset.save_bed(
        prefix.with_suffix(".bed"),
        rename_missing_values=False,
        sample_phenotype="-9",
    )
    n_blocks = int(len(np.unique(genomic_block_labels(subset.variants_chrom, subset.variants_pos, args.block_size_bp))))
    with (args.work_dir / "subset_metadata.json").open("w") as handle:
        json.dump(
            {
                "genotype_format": "plink",
                "plink_prefix": str(prefix.resolve()),
                "single_vcf_mode": args.vcf is not None,
                "autosomes_chr1_chr22": args.vcf is None,
                "genotype_dir": str(args.genotype_dir.resolve()) if args.genotype_dir is not None else None,
                "download_vcf_before_subset": bool(args.download_vcf),
                "samples": subset.samples.tolist(),
                "sample_populations": subset_sample_pops,
                "n_variants": int(subset.n_snps),
                "block_size_bp": int(args.block_size_bp),
                "n_blocks": n_blocks,
                "max_d_stat_combinations": int(args.max_d_stat_combinations),
                "max_f4_ratio_combinations": int(args.max_f4_ratio_combinations),
            },
            handle,
            indent=2,
        )

    if args.skip_admixtools2:
        print(f"Wrote PLINK .bed/.bim/.fam and subset metadata under {args.work_dir}")
        return 0

    populations = list(dict.fromkeys(subset_sample_pops))
    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    paths["combo_dir"].mkdir(parents=True, exist_ok=True)

    snputils_times: list[float] = []
    admix_times: list[float] = []

    t0_snputils = time.perf_counter()
    snputils_dfs = run_snputils(
        prefix,
        subset,
        args.block_size_bp,
        paths["combo_dir"],
        max_d_stat_combinations=args.max_d_stat_combinations,
        max_f4_ratio_combinations=args.max_f4_ratio_combinations,
    )
    snputils_times.append(time.perf_counter() - t0_snputils)
    for stat, df in snputils_dfs.items():
        df.to_csv(paths[f"snputils_{stat}"], index=False)

    t0_admix = time.perf_counter()
    run_admixtools2(prefix, populations, args.block_size_bp, paths["combo_dir"], paths["result_dir"])
    admix_times.append(time.perf_counter() - t0_admix)

    for rep in range(1, args.n_reps):
        print(f"  timing rep {rep + 1}/{args.n_reps} ...", flush=True)
        t0_s = time.perf_counter()
        _ = run_snputils(
            prefix,
            subset,
            args.block_size_bp,
            paths["combo_dir"],
            max_d_stat_combinations=args.max_d_stat_combinations,
            max_f4_ratio_combinations=args.max_f4_ratio_combinations,
        )
        snputils_times.append(time.perf_counter() - t0_s)
        t0_a = time.perf_counter()
        run_admixtools2(prefix, populations, args.block_size_bp, paths["combo_dir"], paths["result_dir"])
        admix_times.append(time.perf_counter() - t0_a)

    ddof = 1 if len(snputils_times) > 1 else 0
    timing_df = pd.DataFrame(
        [
            {
                "method": "snputils",
                "component": "overall",
                "seconds_mean": float(np.mean(snputils_times)),
                "seconds_std": float(np.std(snputils_times, ddof=ddof)),
                "n_reps": len(snputils_times),
            },
            {
                "method": "admixtools2",
                "component": "overall",
                "seconds_mean": float(np.mean(admix_times)),
                "seconds_std": float(np.std(admix_times, ddof=ddof)),
                "n_reps": len(admix_times),
            },
        ]
    )
    timing_df.to_csv(paths["timing"], index=False)

    summary, merged_by_stat = concordance_summary_table(
        snputils_dfs,
        paths["result_dir"],
        pdf_path=pdf_path,
        timing_df=timing_df,
    )
    summary.to_csv(paths["summary"], index=False)
    for stat, merged in merged_by_stat.items():
        merged.to_csv(paths[f"concordance_{stat}"], index=False)

    print(summary.to_string(index=False))
    snp_m = float(np.mean(snputils_times))
    adm_m = float(np.mean(admix_times))
    ratio_txt = f" (ratio admix/snputils {adm_m / snp_m:.3f}x)" if snp_m > 0 else ""
    ddof_pr = 1 if len(snputils_times) > 1 else 0
    snp_sd = float(np.std(snputils_times, ddof=ddof_pr))
    adm_sd = float(np.std(admix_times, ddof=ddof_pr))
    print(
        f"Wall time: snputils {snp_m:.3f}s ±{snp_sd:.3f}s, ADMIXTOOLS2 {adm_m:.3f}s ±{adm_sd:.3f}s{ratio_txt}"
    )
    print(f"Wrote raw results under {paths['result_dir']}")
    print(f"Wrote {pdf_path} (concordance + timing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
