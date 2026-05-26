import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

from snputils.snp.io.read import SNPReader
from snputils.visualization._figure_export import (
    default_savefig_kwargs,
    scatter_rasterized_for_path,
)
from snputils.visualization.constants import get_palette_color

log = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def add_pca_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--snp-path",
        dest="snp_path",
        required=True,
        type=str,
        help="Path to genotype input (VCF, BED, or PGEN fileset).",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        required=True,
        type=str,
        help="Path for the PCA scatter plot (.pdf / .svg for vector output, .png, ...; see visualization._figure_export).",
    )
    parser.add_argument(
        "--coords",
        dest="coords",
        required=False,
        default=None,
        type=str,
        help="Optional path for a TSV/CSV of sample IDs and PC coordinates (see dimred_tabular).",
    )
    parser.add_argument(
        "--components",
        dest="components",
        required=False,
        default=None,
        type=str,
        help="Optional path to save PCA components as a .npy file.",
    )
    parser.add_argument(
        "--backend",
        required=False,
        choices=("sklearn", "pytorch"),
        default="sklearn",
        help="Backend used to perform PCA.",
    )
    parser.add_argument(
        "--n-components",
        dest="n_components",
        required=False,
        type=_positive_int,
        default=2,
        help="Number of principal components to compute.",
    )
    parser.add_argument(
        "--fitting",
        dest="fitting",
        choices=("exact", "lowrank"),
        default="exact",
        help=(
            "SVD mode: exact (standard PCA; sklearn uses svd_solver='full') or "
            "lowrank approximate (sklearn randomized / torch svd_lowrank)."
        ),
    )
    parser.add_argument(
        "--sum-strands",
        dest="sum_strands",
        required=False,
        action="store_true",
        help="Read diploid genotypes as per-individual summed strand counts.",
    )
    parser.add_argument(
        "--vcf-backend",
        dest="vcf_backend",
        required=False,
        choices=("default", "polars"),
        default="default",
        help="VCF reader backend (used only when input is VCF).",
    )


def parse_pca_args(argv):
    parser = argparse.ArgumentParser(
        prog="pca",
        description="Principal Component Analysis plot and save components.",
    )
    add_pca_arguments(parser)
    return parser.parse_args(argv)


def run_pca_command(args: argparse.Namespace) -> int:
    if int(args.n_components) <= 0:
        raise ValueError("--n-components must be a positive integer.")

    reader = SNPReader(Path(args.snp_path), vcf_backend=args.vcf_backend)
    snpobj = reader.read(sum_strands=args.sum_strands)

    if args.backend == "pytorch":
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PCA backend 'pytorch' requires optional dependency 'torch'. "
                "Install it with `pip install snputils[torch]` or `pip install torch`."
            ) from exc

        from ..processing import PCA

        pca = PCA(
            backend=args.backend,
            n_components=args.n_components,
            fitting=args.fitting,
            embedding_table_path=args.coords,
        )
        components = pca.fit_transform(snpobj)
        components = components.cpu().numpy()
    elif args.backend == "sklearn":
        from ..processing import PCA

        pca = PCA(
            backend="sklearn",
            n_components=args.n_components,
            fitting=args.fitting,
            embedding_table_path=args.coords,
        )
        components = np.asarray(pca.fit_transform(snpobj), dtype=float)
    else:
        raise ValueError("Unknown backend for PCA. Use 'sklearn' or 'pytorch'.")

    import matplotlib.pyplot as plt

    if components.ndim != 2 or components.shape[1] < 1:
        raise ValueError("PCA produced an invalid component matrix.")

    x = components[:, 0]
    if components.shape[1] >= 2:
        y = components[:, 1]
        y_label = "Principal Component 2"
    else:
        y = np.zeros_like(x)
        y_label = "Constant (0)"

    plt.figure(figsize=(10, 8))
    _scatter_kw: dict = {"linewidth": 0, "alpha": 0.5, "color": get_palette_color(0)}
    if scatter_rasterized_for_path(str(args.plot)):
        _scatter_kw["rasterized"] = True
    plt.scatter(x, y, **_scatter_kw)
    plt.xlabel("Principal Component 1", fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.tight_layout()

    _save_kw = default_savefig_kwargs(str(args.plot))
    plt.savefig(args.plot, **_save_kw)
    if args.components is not None:
        np.save(args.components, components)
    return 0


def plot_and_save_pca(argv: List[str]):
    args = parse_pca_args(argv)
    return run_pca_command(args)
