#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from snputils.snp.genobj import SNPObject
from snputils.snp.io.read.auto import SNPReader
from snputils.stats import f2, f3, f4, genomic_block_labels


DEFAULT_PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)
PHASE3_1000G_RELEASE_DIR = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
)
AUTOSOME_CHROMOSOMES: tuple[int, ...] = tuple(range(1, 23))
DEFAULT_POPULATIONS = (
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


def default_phase3_autosome_vcf_url(chromosome: int) -> str:
    if chromosome not in AUTOSOME_CHROMOSOMES:
        raise ValueError(f"chromosome must be in {AUTOSOME_CHROMOSOMES[0]}..{AUTOSOME_CHROMOSOMES[-1]}, got {chromosome}")
    return (
        f"{PHASE3_1000G_RELEASE_DIR}"
        f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    )


def download_if_missing(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading {url} -> {path}")
    with urllib.request.urlopen(url) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)
    return path


def read_1000g_panel(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, sep="\t", dtype=str)
    rename = {
        "pop": "population",
        "super_pop": "super_population",
        "gender": "sex",
    }
    panel = panel.rename(columns={k: v for k, v in rename.items() if k in panel.columns})
    required = {"sample", "population"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"1000 Genomes panel is missing columns: {sorted(missing)}")
    if "sex" not in panel.columns:
        panel["sex"] = "U"
    panel["sex"] = panel["sex"].map({"male": "M", "female": "F", "M": "M", "F": "F"}).fillna("U")
    return panel


def select_samples(panel: pd.DataFrame, populations: Sequence[str], samples_per_pop: int) -> pd.DataFrame:
    selected = []
    for pop in populations:
        rows = panel.loc[panel["population"] == pop].head(samples_per_pop)
        if len(rows) < samples_per_pop:
            raise ValueError(f"Population {pop!r} has only {len(rows)} samples in the panel")
        selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def sample_populations(selected_samples: pd.DataFrame, samples: Sequence[str]) -> list[str]:
    sample_to_pop = dict(zip(selected_samples["sample"], selected_samples["population"]))
    return [sample_to_pop[sample] for sample in samples]


def sample_sexes(selected_samples: pd.DataFrame, samples: Sequence[str]) -> list[str]:
    sample_to_sex = dict(zip(selected_samples["sample"], selected_samples["sex"]))
    return [sample_to_sex.get(sample, "U") for sample in samples]


def _fill_missing_variant_ids(snpobj: SNPObject) -> SNPObject:
    if snpobj.variants_id is None:
        return snpobj
    ids = np.asarray(snpobj.variants_id, dtype=object).copy()
    missing = (ids == ".") | (ids == "")
    if np.any(missing):
        ids[missing] = np.asarray(
            [
                f"{chrom}:{pos}:{ref}:{alt}"
                for chrom, pos, ref, alt in zip(
                    snpobj.variants_chrom[missing],
                    snpobj.variants_pos[missing],
                    snpobj.variants_ref[missing],
                    snpobj.variants_alt[missing],
                )
            ],
            dtype=object,
        )
        snpobj = snpobj.copy()
        snpobj.variants_id = ids
    return snpobj


def read_snp_subset(
    source: str | Path,
    selected_samples: pd.DataFrame,
    *,
    max_variants: int,
    require_complete: bool = True,
    require_polymorphic: bool = True,
) -> SNPObject:
    if str(source).startswith(("http://", "https://", "ftp://")):
        raise ValueError("Remote VCF streaming is not supported here; download the VCF before reading it.")

    selected = selected_samples["sample"].tolist()
    chunks: list[SNPObject] = []
    n_variants = 0
    reader = SNPReader(source)
    for chunk in reader.iter_read(samples=selected, sum_strands=False, chunk_size=50_000):
        chunk = chunk.filter_biallelic_variants(snv_only=True)
        if require_complete:
            chunk = chunk.filter_complete_genotypes()
        if require_polymorphic:
            chunk = chunk.filter_polymorphic_variants()
        if chunk.n_snps == 0:
            continue
        remaining = max_variants - n_variants
        if chunk.n_snps > remaining:
            chunk = chunk.filter_variants(indexes=np.arange(remaining), include=True)
        chunks.append(_fill_missing_variant_ids(chunk))
        n_variants += chunk.n_snps
        if n_variants >= max_variants:
            break

    if n_variants < max_variants:
        raise RuntimeError(f"Only found {n_variants} eligible variants; requested {max_variants}")

    subset = SNPObject.concat_variants(chunks)
    subset.sample_fid = np.asarray(sample_populations(selected_samples, subset.samples), dtype=object)
    return subset


def split_int_evenly(total: int, n_parts: int) -> list[int]:
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def read_snp_subset_autosomes(
    selected_samples: pd.DataFrame,
    vcf_sources: Sequence[str | Path],
    *,
    max_variants_total: int,
    require_complete: bool = True,
    require_polymorphic: bool = True,
) -> SNPObject:
    if len(vcf_sources) == 0:
        raise ValueError("vcf_sources is empty")
    if max_variants_total < len(vcf_sources):
        raise ValueError(
            f"max_variants_total ({max_variants_total}) must be >= len(vcf_sources) ({len(vcf_sources)}) "
            "so every source contributes at least one variant."
        )
    quotas = split_int_evenly(max_variants_total, len(vcf_sources))
    parts = [
        read_snp_subset(
            src,
            selected_samples,
            max_variants=q,
            require_complete=require_complete,
            require_polymorphic=require_polymorphic,
        )
        for src, q in zip(vcf_sources, quotas)
    ]
    return SNPObject.concat_variants(parts)


def write_plink(snpobj: SNPObject, prefix: Path, sample_sex: Sequence[str]) -> None:
    """
    Write PLINK 1.0 binary (.bed/.bim/.fam) with native snputils I/O.

    Population labels use PLINK FID (field 1), matching ADMIXTOOLS2 `pops` and the
    [data formats](https://uqrmaie1.github.io/admixtools/articles/io.html) convention for binary PLINK.
    """
    prefix.parent.mkdir(parents=True, exist_ok=True)
    snpobj.save_bed(
        prefix.with_suffix(".bed"),
        rename_missing_values=False,
        sample_fid=snpobj.sample_fid,
        sample_sex=sample_sex,
        sample_phenotype="-9",
    )


def statistic_combinations(populations: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return pairs, triples, quads


def run_snputils(
    plink_prefix: Path,
    subset: SNPObject,
    block_size_bp: int,
    combo_csv_dir: Path,
) -> dict[str, pd.DataFrame]:
    if subset.sample_fid is None:
        raise RuntimeError("Subset is missing population labels in SNPObject.sample_fid.")
    populations = list(dict.fromkeys(str(pop) for pop in subset.sample_fid))
    pairs, triples, quads = statistic_combinations(populations)

    reader = SNPReader(plink_prefix.with_suffix(".bed"))
    snpobj = reader.read(sum_strands=True)

    if snpobj.samples is None or snpobj.calldata_gt is None:
        raise RuntimeError("BEDReader did not return genotypes or sample IDs.")
    if list(snpobj.samples) != list(subset.samples):
        raise RuntimeError("Sample order from PLINK does not match the VCF subset (IID mismatch).")

    block_labels = genomic_block_labels(snpobj.variants_chrom, snpobj.variants_pos, block_size_bp)
    f2_df = f2(
        snpobj,
        pop1=pairs["pop1"].tolist(),
        pop2=pairs["pop2"].tolist(),
        blocks=block_labels,
        apply_correction=True,
    )
    f3_df = (
        f3(
            snpobj,
            target=triples["pop1"].tolist(),
            ref1=triples["pop2"].tolist(),
            ref2=triples["pop3"].tolist(),
            blocks=block_labels,
            apply_correction=True,
        )
        .rename(columns={"target": "pop1", "ref1": "pop2", "ref2": "pop3"})
    )
    f4_df = (
        f4(
            snpobj,
            a=quads["pop1"].tolist(),
            b=quads["pop2"].tolist(),
            c=quads["pop3"].tolist(),
            d=quads["pop4"].tolist(),
            blocks=block_labels,
        )
        .rename(columns={"a": "pop1", "b": "pop2", "c": "pop3", "d": "pop4"})
    )
    combo_csv_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(combo_csv_dir / "f2_combinations.csv", index=False)
    triples.to_csv(combo_csv_dir / "f3_combinations.csv", index=False)
    quads.to_csv(combo_csv_dir / "f4_combinations.csv", index=False)
    return {"f2": f2_df, "f3": f3_df, "f4": f4_df}


def r_vector(values: Sequence[str]) -> str:
    return "c(" + ", ".join(json.dumps(v) for v in values) + ")"


def write_admixtools_runner(
    prefix: Path,
    populations: Sequence[str],
    block_size_bp: int,
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
            outdir <- {json.dumps(str(outdir.resolve()))}
            pops <- {r_vector(populations)}
            block_size_bp <- {int(block_size_bp)}

            f2_combos <- as.matrix(read.csv(file.path(outdir, "f2_combinations.csv"), stringsAsFactors = FALSE))
            f3_combos <- as.matrix(read.csv(file.path(outdir, "f3_combinations.csv"), stringsAsFactors = FALSE))
            f4_combos <- as.matrix(read.csv(file.path(outdir, "f4_combinations.csv"), stringsAsFactors = FALSE))

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

            write.csv(f2_out, file.path(outdir, "admixtools2_f2.csv"), row.names = FALSE)
            write.csv(f3_out, file.path(outdir, "admixtools2_f3.csv"), row.names = FALSE)
            write.csv(f4_out, file.path(outdir, "admixtools2_f4.csv"), row.names = FALSE)
            """
        ).lstrip()
    )
    return script


def run_admixtools2(prefix: Path, populations: Sequence[str], block_size_bp: int, outdir: Path) -> None:
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


STAT_COLORS = {"f2": "#009E73", "f3": "#0072B2", "f4": "#D55E00"}
# Population key columns for merging snputils ↔ ADMIXTOOLS2 (order matches ADMIXTOOLS2 keys).
CONCORDANCE_POP_KEYS: dict[str, tuple[str, ...]] = {
    "f2": ("pop1", "pop2"),
    "f3": ("pop1", "pop2", "pop3"),
    "f4": ("pop1", "pop2", "pop3", "pop4"),
}


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_concordance_panel(ax, merged: pd.DataFrame, stat: str) -> None:
    x = merged["est_admixtools2"].to_numpy(dtype=float)
    y = merged["est_snputils"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
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
    ax.set_xlabel("ADMIXTOOLS2 estimate")
    ax.set_ylabel("snputils estimate")
    ax.set_title(stat)
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
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )
    ax.grid(color="0.9", linewidth=0.6)


def save_concordance_grid_pdf(merged_by_stat: dict[str, pd.DataFrame], pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    stats = [stat for stat in ("f2", "f3", "f4") if stat in merged_by_stat]
    fig, axes = plt.subplots(1, len(stats), figsize=(3.25 * len(stats), 3.25), constrained_layout=True)
    if len(stats) == 1:
        axes = [axes]
    for ax, stat in zip(axes, stats):
        draw_concordance_panel(ax, merged_by_stat[stat], stat)
    pdf_path = pdf_path.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def concordance_summary_table(
    snputils_dfs: dict[str, pd.DataFrame],
    rundir: Path,
    *,
    pdf_path: Path,
) -> pd.DataFrame:
    specs = {stat: list(keys) for stat, keys in CONCORDANCE_POP_KEYS.items()}
    summaries = []
    merged_by_stat: dict[str, pd.DataFrame] = {}
    for stat, keys in specs.items():
        admix = pd.read_csv(rundir / f"admixtools2_{stat}.csv")
        merged, summary = summarize_concordance(stat, snputils_dfs[stat], admix, keys)
        summaries.append(summary)
        merged_by_stat[stat] = merged
    save_concordance_grid_pdf(merged_by_stat, pdf_path)
    return pd.DataFrame(summaries)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare snputils f2/f3/f4 estimates with ADMIXTOOLS2 on a 1000 Genomes subset "
            "(PLINK bed/bim/fam), and save one concordance grid PDF plus a stdout summary."
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/fstats_admixtools2_concordance"))
    parser.add_argument("--panel-url", default=DEFAULT_PANEL_URL)
    parser.add_argument(
        "--vcf",
        default=None,
        metavar="PATH_OR_URL",
        help=(
            "Single VCF path or URL. If omitted, uses 1000 Genomes phase3 autosomes chr1-chr22 "
            "(max-variants split evenly across chromosomes)."
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
    parser.add_argument("--populations", nargs="+", default=list(DEFAULT_POPULATIONS))
    parser.add_argument(
        "--samples-per-pop",
        type=int,
        default=50,
        help="Number of individuals taken per population from the panel (panel order). Default 50.",
    )
    parser.add_argument("--max-variants", type=int, default=1_000_000)
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
            "Path for the multi-panel concordance figure only (PDF). "
            "Default: WORK_DIR/concordance_grid.pdf"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    panel_path = download_if_missing(args.panel_url, args.work_dir / "integrated_call_samples_v3.20130502.ALL.panel")
    panel = read_1000g_panel(panel_path)
    selected = select_samples(panel, args.populations, args.samples_per_pop)

    if args.vcf is None:
        autosome_urls = [default_phase3_autosome_vcf_url(c) for c in AUTOSOME_CHROMOSOMES]
        if args.download_vcf:
            vcf_sources = [
                download_if_missing(url, args.work_dir / Path(url).name) for url in autosome_urls
            ]
        else:
            raise ValueError("Remote VCF streaming is not supported by this benchmark; keep VCF downloading enabled.")
        print(
            f"Building subset: {len(selected)} samples, {args.max_variants} biallelic SNPs across "
            f"autosomes chr{AUTOSOME_CHROMOSOMES[0]}-chr{AUTOSOME_CHROMOSOMES[-1]} "
            f"(~{args.max_variants // len(AUTOSOME_CHROMOSOMES)} per chromosome), "
            f"populations={','.join(args.populations)}"
        )
        subset = read_snp_subset_autosomes(
            selected,
            vcf_sources,
            max_variants_total=args.max_variants,
            require_polymorphic=not args.include_monomorphic,
        )
        prefix = args.work_dir / "kg_autosomes_subset"
        subset_meta_sources: list[str] = [str(u) for u in vcf_sources]
    else:
        vcf_source: str | Path = args.vcf
        if args.download_vcf and str(vcf_source).startswith(("http://", "https://", "ftp://")):
            vcf_source = download_if_missing(str(vcf_source), args.work_dir / Path(str(vcf_source)).name)
        elif str(vcf_source).startswith(("http://", "https://", "ftp://")):
            raise ValueError("Remote VCF streaming is not supported by this benchmark; keep VCF downloading enabled.")
        print(
            f"Building subset: {len(selected)} samples, {args.max_variants} biallelic SNPs from {vcf_source}, "
            f"populations={','.join(args.populations)}"
        )
        subset = read_snp_subset(
            vcf_source,
            selected,
            max_variants=args.max_variants,
            require_polymorphic=not args.include_monomorphic,
        )
        prefix = args.work_dir / "kg_single_vcf_subset"
        subset_meta_sources = [str(vcf_source)]

    subset_sample_sex = sample_sexes(selected, subset.samples)
    subset_sample_pops = [str(pop) for pop in subset.sample_fid]
    write_plink(subset, prefix, sample_sex=subset_sample_sex)
    n_blocks = int(len(np.unique(genomic_block_labels(subset.variants_chrom, subset.variants_pos, args.block_size_bp))))
    with (args.work_dir / "subset_metadata.json").open("w") as handle:
        json.dump(
            {
                "genotype_format": "plink",
                "plink_prefix": str(prefix.resolve()),
                "source_vcfs": subset_meta_sources,
                "single_vcf_mode": args.vcf is not None,
                "autosomes_chr1_chr22": args.vcf is None,
                "panel_url": args.panel_url,
                "download_vcf_before_subset": bool(args.download_vcf),
                "samples": subset.samples.tolist(),
                "sample_populations": subset_sample_pops,
                "n_variants": int(subset.n_snps),
                "block_size_bp": int(args.block_size_bp),
                "n_blocks": n_blocks,
            },
            handle,
            indent=2,
        )

    if args.skip_admixtools2:
        print(f"Wrote PLINK .bed/.bim/.fam and subset metadata under {args.work_dir}")
        return 0

    populations = list(dict.fromkeys(subset_sample_pops))
    pdf_path = args.output_pdf if args.output_pdf is not None else args.work_dir / "concordance_grid.pdf"

    with tempfile.TemporaryDirectory(prefix="fstats_admixtools2_", dir=str(args.work_dir)) as rundir_raw:
        rundir_path = Path(rundir_raw)
        snputils_dfs = run_snputils(prefix, subset, args.block_size_bp, rundir_path)
        run_admixtools2(prefix, populations, args.block_size_bp, rundir_path)
        summary = concordance_summary_table(snputils_dfs, rundir_path, pdf_path=pdf_path)

    print(summary.to_string(index=False))
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
