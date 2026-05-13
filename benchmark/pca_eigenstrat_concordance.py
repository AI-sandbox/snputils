#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from benchmark.fstats_admixtools2_concordance import (
        BENCHMARK_1KGP_RESOURCE,
        DEFAULT_BENCHMARK_POPULATIONS,
        local_registry_autosome_vcfs,
    )
except ModuleNotFoundError:
    from fstats_admixtools2_concordance import (
        BENCHMARK_1KGP_RESOURCE,
        DEFAULT_BENCHMARK_POPULATIONS,
        local_registry_autosome_vcfs,
    )
from snputils.datasets import load_dataset
from snputils.processing import PCA
from snputils.snp.genobj import SNPObject
from snputils.snp.io.read.auto import SNPReader
from snputils.snp.io.write._plink import coerce_sex_codes


@dataclass(frozen=True)
class SmartPcaOutputs:
    evec: Path
    eval: Path
    log: Path
    par: Path


def write_plink_for_eigensoft(subset: SNPObject, prefix: Path) -> None:
    """
    Write a PACKEDBED fileset with phenotype set to population (for convertf/smartpca).
    Mirrors ``SNPObject.save_bed`` usage in ``fstats_admixtools2_concordance``.
    """
    if subset.samples is None or subset.sample_fid is None:
        raise RuntimeError("Subset needs samples and population labels (SNPObject.sample_fid).")
    subset.save_bed(
        prefix.with_suffix(".bed"),
        rename_missing_values=False,
        sample_phenotype=list(subset.sample_fid),
    )


def write_convertf_param(plink_prefix: Path, eigen_prefix: Path, param_path: Path) -> Path:
    param_path.write_text(
        textwrap.dedent(
            f"""
            genotypename: {plink_prefix.with_suffix(".bed").resolve()}
            snpname: {plink_prefix.with_suffix(".bim").resolve()}
            indivname: {plink_prefix.with_suffix(".fam").resolve()}
            outputformat: EIGENSTRAT
            genooutfilename: {eigen_prefix.with_suffix(".geno").resolve()}
            snpoutfilename: {eigen_prefix.with_suffix(".snp").resolve()}
            indoutfilename: {eigen_prefix.with_suffix(".ind").resolve()}
            familynames: NO
            noxdata: YES
            """
        ).strip()
        + "\n"
    )
    return param_path


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name!r} is not on PATH; install EIGENSOFT and ensure its bin directory is on PATH.")
    return path


def run_convertf(plink_prefix: Path, eigen_prefix: Path, outdir: Path) -> Path:
    convertf = require_executable("convertf")
    par = write_convertf_param(plink_prefix, eigen_prefix, outdir / "convertf.PACKEDPED.EIGENSTRAT.par")
    log_path = outdir / "convertf.log"
    with log_path.open("w") as log:
        subprocess.run([convertf, "-p", str(par)], cwd=outdir, stdout=log, stderr=subprocess.STDOUT, check=True)
    for suffix in (".geno", ".snp", ".ind"):
        output = eigen_prefix.with_suffix(suffix)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"convertf did not create {output}")
    return log_path


def normalize_eigenstrat_indiv(subset: SNPObject, eigen_prefix: Path) -> None:
    """
    Make convertf's .ind labels explicit and verify sample order before smartpca.

    convertf should preserve PACKEDPED individual order when familynames is NO.
    This check catches unexpected family-name concatenation or sample reordering
    before concordance is computed against the wrong rows.
    """
    ind_path = eigen_prefix.with_suffix(".ind")
    ind = pd.read_csv(ind_path, sep=r"\s+", header=None, names=["sample", "sex", "population"], dtype=str)
    if ind["sample"].tolist() != list(subset.samples):
        raise RuntimeError(
            "convertf output .ind sample order/name mismatch; inspect convertf.log before comparing PCA coordinates."
        )
    n = len(subset.samples)
    plink_codes = coerce_sex_codes(subset.sample_sex, n, missing_code="0")
    ind_sex = ["M" if str(c) == "1" else "F" if str(c) == "2" else "U" for c in np.asarray(plink_codes)]
    ind["sex"] = ind_sex
    ind["population"] = subset.sample_fid
    ind.to_csv(ind_path, sep="\t", header=False, index=False)


def write_smartpca_param(
    eigen_prefix: Path,
    outputs: SmartPcaOutputs,
    *,
    n_components: int,
    maxpops: int,
    numthreads: int | None,
) -> Path:
    lines = [
        f"genotypename: {eigen_prefix.with_suffix('.geno').resolve()}",
        f"snpname: {eigen_prefix.with_suffix('.snp').resolve()}",
        f"indivname: {eigen_prefix.with_suffix('.ind').resolve()}",
        f"evecoutname: {outputs.evec.resolve()}",
        f"evaloutname: {outputs.eval.resolve()}",
        "altnormstyle: NO",
        "usenorm: NO",
        f"numoutevec: {n_components}",
        "numoutlieriter: 0",
        "familynames: NO",
        f"maxpops: {maxpops}",
        f"grmoutname: {(outputs.evec.parent / 'grmjunk').resolve()}",
    ]
    if numthreads is not None:
        lines.append(f"numthreads: {numthreads}")

    outputs.par.write_text("\n".join(lines) + "\n")
    return outputs.par


def run_smartpca(
    eigen_prefix: Path,
    outdir: Path,
    *,
    n_components: int,
    maxpops: int,
    numthreads: int | None,
) -> tuple[SmartPcaOutputs, float]:
    smartpca = require_executable("smartpca")
    outputs = SmartPcaOutputs(
        evec=outdir / "eigenstrat_pca.evec",
        eval=outdir / "eigenstrat_pca.eval",
        log=outdir / "smartpca.log",
        par=outdir / "smartpca.par",
    )
    write_smartpca_param(
        eigen_prefix,
        outputs,
        n_components=n_components,
        maxpops=maxpops,
        numthreads=numthreads,
    )
    with outputs.log.open("w") as log:
        t0 = time.perf_counter()
        subprocess.run([smartpca, "-p", str(outputs.par)], cwd=outdir, stdout=log, stderr=subprocess.STDOUT, check=True)
        wall_smartpca_s = time.perf_counter() - t0
    if not outputs.evec.exists() or outputs.evec.stat().st_size == 0:
        raise RuntimeError(f"smartpca did not create {outputs.evec}")
    if not outputs.eval.exists() or outputs.eval.stat().st_size == 0:
        raise RuntimeError(f"smartpca did not create {outputs.eval}")
    return outputs, float(wall_smartpca_s)


def run_snputils_pca(
    plink_prefix: Path,
    n_components: int,
    backend: str,
    *,
    fitting: str = "exact",
    torch_device: str = "cpu",
) -> tuple[pd.DataFrame, float]:
    """
    Load PLINK BED and run PCA. Returns coords and wall seconds for ``fit_transform`` only
    (excludes SNPReader I/O — see benchmark timing plot footnote).
    """
    reader = SNPReader(plink_prefix.with_suffix(".bed"))
    snpobj = reader.read(sum_strands=True)
    if snpobj.samples is None or snpobj.calldata_gt is None:
        raise RuntimeError("BEDReader did not return genotypes or sample IDs.")
    if backend == "pytorch":
        pca = PCA(
            backend=backend,
            n_components=n_components,
            fitting=fitting,
            device=torch_device,
        )
    else:
        pca = PCA(backend=backend, n_components=n_components, fitting=fitting)
    t0 = time.perf_counter()
    coords = pca.fit_transform(snpobj)
    wall_fit_s = time.perf_counter() - t0
    if backend == "pytorch":
        dev = pca.device
        using_gpu = getattr(dev, "type", str(dev)) == "cuda"
        print(
            f"snputils PCA: TorchPCA on device {dev} (using_gpu={using_gpu}), fitting={fitting!r}"
        )
    else:
        print(
            f"snputils PCA: scikit-learn backend (not TorchPCA; no GPU), fitting={fitting!r}"
        )
    if hasattr(coords, "cpu"):
        coords = coords.cpu().numpy()
    coords = np.asarray(coords, dtype=float)
    columns = [f"PC{i}" for i in range(1, coords.shape[1] + 1)]
    df = pd.DataFrame(coords, columns=columns).assign(sample=list(snpobj.samples))[["sample", *columns]]
    return df, float(wall_fit_s)


def parse_smartpca_evec(evec_path: Path, n_components: int) -> pd.DataFrame:
    rows: list[list[str]] = []
    with evec_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    if not rows:
        raise RuntimeError(f"No sample rows found in {evec_path}")

    expected = n_components + 2
    bad = [row for row in rows if len(row) < expected]
    if bad:
        raise RuntimeError(f"Malformed smartpca .evec row in {evec_path}: {bad[0]}")

    columns = ["sample", *[f"PC{i}" for i in range(1, n_components + 1)], "population"]
    df = pd.DataFrame([row[:expected] for row in rows], columns=columns)
    for col in columns[1:-1]:
        df[col] = df[col].astype(float)
    return df


def standardize(values: pd.Series) -> pd.Series:
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if sd == 0 or not math.isfinite(sd):
        raise ValueError("Cannot standardize a constant or non-finite PCA component.")
    return (values - mean) / sd


def summarize_concordance(snputils_df: pd.DataFrame, eigen_df: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = snputils_df.merge(
        eigen_df,
        on="sample",
        suffixes=("_snputils", "_eigenstrat"),
        validate="one_to_one",
    )
    summaries: list[dict[str, float | int | str]] = []
    for i in range(1, n_components + 1):
        pc = f"PC{i}"
        snp_col = f"{pc}_snputils"
        eig_col = f"{pc}_eigenstrat"
        merged[f"{snp_col}_z"] = standardize(merged[snp_col])
        merged[f"{eig_col}_z"] = standardize(merged[eig_col])

        x = merged[f"{eig_col}_z"].to_numpy(dtype=float)
        y = merged[f"{snp_col}_z"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[finite], y[finite])[0, 1]) if finite.sum() > 1 else float("nan")
        sign = -1.0 if r < 0 else 1.0
        merged[f"{eig_col}_z_aligned"] = sign * merged[f"{eig_col}_z"]
        aligned_x = sign * x
        diff = y[finite] - aligned_x[finite]

        if finite.sum() > 1:
            slope, intercept = np.polyfit(aligned_x[finite], y[finite], deg=1)
        else:
            slope, intercept = float("nan"), float("nan")
        summaries.append(
            {
                "component": pc,
                "n": int(finite.sum()),
                "pearson_r_abs": abs(r),
                "sign_applied_to_eigenstrat": sign,
                "rmse_z": float(math.sqrt(np.mean(diff**2))) if finite.any() else float("nan"),
                "max_abs_delta_z": float(np.max(np.abs(diff))) if finite.any() else float("nan"),
                "slope_after_z_alignment": float(slope),
                "intercept_after_z_alignment": float(intercept),
            }
        )
    return merged, pd.DataFrame(summaries)


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


PCA_TIMING_COLORS = {"snputils_pca_fit": "#0072B2", "smartpca": "#CC79A7"}


def pca_benchmark_result_paths(work_dir: Path) -> dict[str, Path]:
    """CSV outputs that together suffice to regenerate the benchmark PDF."""
    return {
        "concordance_by_sample": work_dir / "pca_concordance_by_sample.csv",
        "concordance_summary": work_dir / "pca_concordance_summary.csv",
        "computation_timing": work_dir / "pca_computation_timing.csv",
    }


def cached_pca_benchmark_ready(paths: dict[str, Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def load_cached_pca_benchmark_tables(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    merged = pd.read_csv(paths["concordance_by_sample"])
    summary = pd.read_csv(paths["concordance_summary"])
    timing_df = pd.read_csv(paths["computation_timing"])
    timing: dict[str, float] = {
        str(m): float(s) for m, s in zip(timing_df["method"].tolist(), timing_df["seconds"].tolist())
    }
    return merged, summary, timing


def print_pca_stdout_summary(summary: pd.DataFrame, timing: Mapping[str, float]) -> None:
    print(summary.to_string(index=False))
    wall_snputils_fit = float(timing.get("snputils_pca_fit", float("nan")))
    wall_smartpca = float(timing.get("smartpca", float("nan")))
    ratio = ""
    if wall_snputils_fit > 0 and math.isfinite(wall_snputils_fit) and math.isfinite(wall_smartpca):
        ratio = f" (ratio smartpca/snputils {wall_smartpca / wall_snputils_fit:.3f}x)"
    print(
        f"PCA wall time (computation only): snputils fit_transform {wall_snputils_fit:.3f}s, "
        f"smartpca {wall_smartpca:.3f}s{ratio}"
    )


def save_concordance_pdf(
    merged: pd.DataFrame,
    summary: pd.DataFrame,
    pdf_path: Path,
    n_components: int,
    *,
    pca_wall_seconds: dict[str, float] | None = None,
    device: str | None = None,
    fitting: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    configure_matplotlib()
    pdf_path = pdf_path.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        ncols = min(3, n_components)
        nrows = math.ceil(n_components / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.35 * nrows), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)

        for i in range(1, n_components + 1):
            ax = axes_arr[i - 1]
            pc = f"PC{i}"
            x = merged[f"{pc}_eigenstrat_z_aligned"].to_numpy(dtype=float)
            y = merged[f"{pc}_snputils_z"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[finite], y[finite], s=18, alpha=0.7, color="#0072B2", edgecolor="white", linewidth=0.25)
            lo = float(min(x[finite].min(), y[finite].min()))
            hi = float(max(x[finite].max(), y[finite].max()))
            pad = (hi - lo) * 0.05 if hi > lo else 1e-6
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=0.9)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("EIGENSTRAT z-score")
            ylabel = "snputils z-score"
            if device == "cuda" or device == "gpu" or device == "cuda:0":
                mode = " (GPU"
            else:
                mode = " (CPU"
            if fitting == "lowrank":
                mode += ", lowrank)"
            else:
                mode += ", exact)"
            ax.set_ylabel(ylabel + mode)
            ax.set_title(pc)
            row = summary.loc[summary["component"] == pc].iloc[0]
            ax.text(
                0.04,
                0.96,
                f"n = {int(row['n'])}\n|r| = {row['pearson_r_abs']:.6g}\nRMSE z = {row['rmse_z']:.2e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
            )
            ax.grid(color="0.9", linewidth=0.6)

        for ax in axes_arr[n_components:]:
            ax.axis("off")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if pca_wall_seconds:
            labels_map = {f"snputils_pca_fit": f"snputils PCA\n{mode}", "smartpca": "EIGENSOFT\nsmartpca"}
            order = ("snputils_pca_fit", "smartpca")
            labels = [labels_map[k] for k in order]
            times = [float(pca_wall_seconds.get(k, float("nan"))) for k in order]
            colors = [PCA_TIMING_COLORS[k] for k in order]

            fig_t, ax_t = plt.subplots(figsize=(4.2, 3.5), constrained_layout=True)
            bars = ax_t.bar(labels, times, color=colors, edgecolor="white", linewidth=0.6)
            ax_t.set_ylabel("Wall-clock time (s)")
            ax_t.set_title("PCA computation")
            ax_t.grid(axis="y", color="0.9", linewidth=0.6)
            finite_times = [t for t in times if np.isfinite(t)]
            ymax = max(finite_times) if finite_times else 1.0
            ax_t.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
            for bar, sec in zip(bars, times):
                if np.isfinite(sec):
                    ax_t.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{sec:.2f}s",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                    )
            pdf.savefig(fig_t, bbox_inches="tight")
            plt.close(fig_t)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare snputils PCA with EIGENSOFT smartpca on a 1000 Genomes subset. "
            "The script writes a PLINK .bed/.bim/.fam subset via snputils, converts it to EIGENSTRAT with convertf, "
            "runs smartpca, and reports sign-invariant per-component concordance."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/pca_eigenstrat_concordance"))
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
            "snputils.datasets (same filenames as storing downloads under ``--work-dir``). "
            "Reads genotypes from here while ``--work-dir`` holds outputs."
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
    parser.add_argument("--max-variants", type=int, default=100_000)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument(
        "--snputils-backend",
        choices=("sklearn", "pytorch"),
        default="sklearn",
        help="PCA implementation: sklearn or pytorch (snputils TorchPCA). Default sklearn.",
    )
    parser.add_argument(
        "--snputils-torch-pca",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force snputils TorchPCA (``--snputils-torch-pca``) or sklearn PCA (``--no-snputils-torch-pca``). "
            "When omitted, ``--snputils-backend`` decides (pytorch = TorchPCA)."
        ),
    )
    parser.add_argument(
        "--snputils-torch-device",
        default="cpu",
        help=(
            "PyTorch device for TorchPCA (cpu, cuda, gpu, cuda:0, ...). Ignored when not using TorchPCA. "
            "If CUDA is unavailable, snputils falls back to CPU."
        ),
    )
    parser.add_argument(
        "--snputils-pca-fitting",
        dest="snputils_pca_fitting",
        choices=("exact", "lowrank"),
        default="exact",
        help=(
            "snputils PCA SVD mode: exact (default; sklearn svd_solver=full, torch economy SVD) or "
            "lowrank (approximate; sklearn randomized, torch.svd_lowrank)."
        ),
    )
    parser.add_argument(
        "--include-monomorphic",
        action="store_true",
        help="Keep sites that are monomorphic in the selected samples.",
    )
    parser.add_argument(
        "--smartpca-numthreads",
        type=int,
        default=None,
        metavar="N",
        help="If set, write numthreads N into the smartpca par file; default omits it (smartpca/EIGENSOFT default).",
    )
    parser.add_argument(
        "--skip-eigenstrat",
        action="store_true",
        help="Only write the PLINK .bed/.bim/.fam subset and subset metadata; skip convertf, smartpca, and PDF generation.",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help=(
            "Path for the benchmark PDF: page 1 concordance grids, page 2 PCA-only wall times "
            "(snputils fit_transform vs smartpca). Default: WORK_DIR/concordance_grid.pdf"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run load_dataset, snputils PCA, convertf/smartpca, and concordance even when "
            "cached CSV results already exist under --work-dir."
        ),
    )
    return parser.parse_args(argv)


def resolve_snputils_pca_backend(args: argparse.Namespace) -> str:
    """``--snputils-torch-pca`` / ``--no-snputils-torch-pca`` override ``--snputils-backend`` when set."""
    if args.snputils_torch_pca is True:
        return "pytorch"
    if args.snputils_torch_pca is False:
        return "sklearn"
    return args.snputils_backend


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_pop <= 0:
        raise ValueError("--samples-per-pop must be positive.")
    if args.max_variants <= 0:
        raise ValueError("--max-variants must be positive.")
    if args.n_components <= 0:
        raise ValueError("--n-components must be positive.")
    if args.vcf is None and args.max_variants < 22:
        raise ValueError("--max-variants must be at least 22 when using all autosomes.")
    n_samples = len(args.populations) * args.samples_per_pop
    if args.n_components >= n_samples:
        raise ValueError(f"--n-components must be less than the selected sample count ({n_samples}).")
    if args.genotype_dir is not None and args.vcf is not None:
        raise ValueError("Use either --vcf or --genotype-dir, not both.")
    if args.snputils_torch_pca is False and args.snputils_backend == "pytorch":
        raise ValueError("Conflicting options: --no-snputils-torch-pca with --snputils-backend pytorch.")
    if args.smartpca_numthreads is not None and args.smartpca_numthreads < 1:
        raise ValueError("--smartpca-numthreads must be a positive integer when set.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate_args(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = args.output_pdf if args.output_pdf is not None else args.work_dir / "concordance_grid.pdf"
    bench_paths = pca_benchmark_result_paths(args.work_dir)

    if (
        not args.skip_eigenstrat
        and cached_pca_benchmark_ready(bench_paths)
        and not args.force
    ):
        merged, summary, timing_for_pdf = load_cached_pca_benchmark_tables(bench_paths)
        n_components = len(summary)
        save_concordance_pdf(
            merged,
            summary,
            pdf_path,
            n_components,
            pca_wall_seconds=timing_for_pdf,
            device=args.snputils_torch_device,
            fitting=args.snputils_pca_fitting,
        )
        print(
            f"PCA benchmark CSVs already exist under {args.work_dir}; regenerated PDF only. "
            "Use --force to recompute genotype load, PCA, and concordance tables."
        )
        print_pca_stdout_summary(summary, timing_for_pdf)
        print(f"Wrote {pdf_path} (concordance + PCA timing)")
        return 0

    snputils_pca_backend = resolve_snputils_pca_backend(args)

    if not args.skip_eigenstrat:
        require_executable("convertf")
        require_executable("smartpca")

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
            f"Built PCA subset: {subset.n_samples} samples, {subset.n_snps} biallelic SNPs across "
            "1000 Genomes Phase 3 autosomes, "
            f"populations={','.join(args.populations)}"
        )
        plink_prefix = args.work_dir / "kg_autosomes_subset"
    else:
        print(
            f"Built PCA subset: {subset.n_samples} samples, {subset.n_snps} biallelic SNPs from {args.vcf}, "
            f"populations={','.join(args.populations)}"
        )
        plink_prefix = args.work_dir / "kg_single_vcf_subset"

    plink_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_plink_for_eigensoft(subset, plink_prefix)
    subset_sample_pops = [str(pop) for pop in subset.sample_fid]

    metadata = {
        "genotype_format": "plink",
        "plink_prefix": str(plink_prefix.resolve()),
        "single_vcf_mode": args.vcf is not None,
        "genotype_dir": str(args.genotype_dir.resolve()) if args.genotype_dir is not None else None,
        "autosomes_chr1_chr22": args.vcf is None,
        "panel_url": args.panel_url,
        "download_vcf_before_subset": bool(args.download_vcf),
        "samples": subset.samples.tolist(),
        "sample_populations": subset_sample_pops,
        "n_variants": int(subset.n_snps),
        "n_components": int(args.n_components),
        "snputils_pca_backend": snputils_pca_backend,
        "snputils_backend": snputils_pca_backend,
        "snputils_backend_cli": args.snputils_backend,
        "snputils_torch_pca_explicit": args.snputils_torch_pca,
        "snputils_torch_device": args.snputils_torch_device if snputils_pca_backend == "pytorch" else None,
        "snputils_pca_fitting": args.snputils_pca_fitting,
        "smartpca_numthreads": args.smartpca_numthreads,
    }
    with (args.work_dir / "subset_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    if args.skip_eigenstrat:
        print(f"Wrote PLINK .bed/.bim/.fam and subset metadata under {args.work_dir}")
        return 0

    snputils_df, wall_snputils_fit = run_snputils_pca(
        plink_prefix,
        args.n_components,
        snputils_pca_backend,
        fitting=args.snputils_pca_fitting,
        torch_device=args.snputils_torch_device,
    )
    snputils_df.to_csv(args.work_dir / "snputils_pca.csv", index=False)

    with tempfile.TemporaryDirectory(prefix="pca_eigenstrat_", dir=str(args.work_dir)) as rundir_raw:
        rundir = Path(rundir_raw)
        eigen_prefix = rundir / "kg_subset_eigenstrat"
        run_convertf(plink_prefix, eigen_prefix, rundir)
        normalize_eigenstrat_indiv(subset, eigen_prefix)

        smartpca_outputs, wall_smartpca = run_smartpca(
            eigen_prefix,
            rundir,
            n_components=args.n_components,
            maxpops=max(100, len(set(subset_sample_pops)) + 10),
            numthreads=args.smartpca_numthreads,
        )
        eigen_df = parse_smartpca_evec(smartpca_outputs.evec, args.n_components)
        eigen_df.to_csv(args.work_dir / "eigenstrat_pca.csv", index=False)
        shutil.copy2(smartpca_outputs.evec, args.work_dir / smartpca_outputs.evec.name)
        shutil.copy2(smartpca_outputs.eval, args.work_dir / smartpca_outputs.eval.name)
        shutil.copy2(smartpca_outputs.log, args.work_dir / smartpca_outputs.log.name)
        shutil.copy2(smartpca_outputs.par, args.work_dir / smartpca_outputs.par.name)

    merged, summary = summarize_concordance(snputils_df, eigen_df, args.n_components)
    merged.to_csv(bench_paths["concordance_by_sample"], index=False)
    summary.to_csv(bench_paths["concordance_summary"], index=False)

    pca_timing = pd.DataFrame(
        [
            {"method": "snputils_pca_fit", "seconds": wall_snputils_fit},
            {"method": "smartpca", "seconds": wall_smartpca},
        ]
    )
    pca_timing.to_csv(bench_paths["computation_timing"], index=False)

    timing_for_pdf = dict(zip(pca_timing["method"], pca_timing["seconds"]))
    save_concordance_pdf(
        merged,
        summary,
        pdf_path,
        args.n_components,
        pca_wall_seconds=timing_for_pdf,
        device=args.snputils_torch_device if snputils_pca_backend == "pytorch" else None,
        fitting=args.snputils_pca_fitting,
    )

    print_pca_stdout_summary(summary, timing_for_pdf)
    print(f"Wrote {pdf_path} (concordance + PCA timing)")
    print(f"Wrote {bench_paths['computation_timing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
