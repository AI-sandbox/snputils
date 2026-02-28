import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

from snputils.snp.io.read import SNPReader

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
        "--fig-path",
        dest="fig_path",
        required=True,
        type=str,
        help="Path used to save PCA plot.",
    )
    parser.add_argument(
        "--npy-path",
        dest="npy_path",
        required=True,
        type=str,
        help="Path used to save principal components in .npy format.",
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
        choices=("polars", "scikit-allel"),
        default="polars",
        help="VCF reader backend (used only when input is VCF).",
    )


def parse_pca_args(argv):
    parser = argparse.ArgumentParser(
        prog="pca",
        description="Principal Component Analysis plot and save components.",
    )
    add_pca_arguments(parser)
    return parser.parse_args(argv)


def _compute_sklearn_components(snpobj, n_components: int) -> np.ndarray:
    from sklearn.decomposition import PCA as SklearnPCA

    if snpobj.calldata_gt.ndim == 2:
        X = np.transpose(snpobj.calldata_gt.astype(float), (1, 0))
    elif snpobj.calldata_gt.ndim == 3:
        X = np.transpose(snpobj.calldata_gt.astype(float), (1, 0, 2))
        X = np.mean(X, axis=2)
    else:
        raise ValueError(
            f"Invalid shape for calldata_gt: expected 2D or 3D, got {snpobj.calldata_gt.ndim}D."
        )
    return SklearnPCA(n_components=n_components).fit_transform(X)


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
                "Install it with `pip install snputils[gpu]` or `pip install torch`."
            ) from exc

        from ..processing import PCA

        pca = PCA(backend=args.backend, n_components=args.n_components)
        components = pca.fit_transform(snpobj)
        components = components.cpu().numpy()
    elif args.backend == "sklearn":
        components = _compute_sklearn_components(snpobj, n_components=args.n_components)
    else:
        raise ValueError("Unknown backend for PCA. Use 'sklearn' or 'pytorch'.")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.scatter(components[:, 0], components[:, 1], linewidth=0, alpha=0.5)
    plt.xlabel("Principal Component 1", fontsize=20)
    plt.ylabel("Principal Component 2", fontsize=20)
    plt.tight_layout()

    plt.savefig(args.fig_path)
    np.save(args.npy_path, components)
    return 0


def plot_and_save_pca(argv: List[str]):
    args = parse_pca_args(argv)
    return run_pca_command(args)
