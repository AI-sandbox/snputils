import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class _Command:
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _run_pca(args: argparse.Namespace) -> int:
    from . import pca as pca_module

    return int(pca_module.run_pca_command(args))


def _run_admixture_map(args: argparse.Namespace) -> int:
    from . import admixture_mapping as admix_module

    return int(admix_module.run_admixmap_command(args))


def _add_pca_arguments(parser: argparse.ArgumentParser) -> None:
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
        help="Path to save the PCA scatter plot image.",
    )
    parser.add_argument(
        "--npy-path",
        dest="npy_path",
        required=True,
        type=str,
        help="Path to save PCA components as a .npy file.",
    )
    parser.add_argument(
        "--backend",
        choices=("sklearn", "pytorch"),
        default="sklearn",
        help="PCA backend to use.",
    )
    parser.add_argument(
        "--n-components",
        dest="n_components",
        default=2,
        type=_positive_int,
        help="Number of principal components to compute.",
    )
    parser.add_argument(
        "--sum-strands",
        dest="sum_strands",
        action="store_true",
        help="Read diploid genotypes as per-individual summed strand counts.",
    )
    parser.add_argument(
        "--vcf-backend",
        dest="vcf_backend",
        choices=("polars", "scikit-allel"),
        default="polars",
        help="VCF reader backend (used only when input is VCF).",
    )


def _add_admixture_map_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phe-id",
        dest="phe_id",
        required=True,
        type=str,
        help="Phenotype ID.",
    )
    parser.add_argument(
        "--phe-path",
        dest="phe_path",
        required=True,
        type=str,
        help="Path to phenotype file.",
    )
    parser.add_argument(
        "--msp-path",
        dest="msp_path",
        required=True,
        type=str,
        help="Path to MSP file.",
    )
    parser.add_argument(
        "--results-path",
        dest="results_path",
        required=True,
        type=str,
        help="Output directory or output .tsv/.tsv.gz path.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=32768,
        type=int,
        help="Max windows processed per chunk.",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        default=None,
        type=int,
        help="Peak RSS-delta memory cap in MiB.",
    )
    parser.add_argument(
        "--keep-hla",
        dest="keep_hla",
        action="store_true",
        help="Keep chr6 HLA windows (default removes them).",
    )
    parser.add_argument(
        "--quantitative",
        dest="quantitative",
        action="store_true",
        default=None,
        help="Force quantitative (linear) mode.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print progress updates.",
    )
    parser.add_argument(
        "--covar-path",
        dest="covar_path",
        default=None,
        type=str,
        help="Path to covariate file.",
    )
    parser.add_argument(
        "--covar-col-nums",
        dest="covar_col_nums",
        default=None,
        type=str,
        help='Covariate columns relative to first covariate column (e.g. "1-5,7").',
    )
    parser.add_argument(
        "--covar-variance-standardize",
        dest="covar_variance_standardize",
        action="store_true",
        help="Center and variance-standardize selected covariates.",
    )
    parser.add_argument(
        "--ci",
        dest="ci",
        default=None,
        type=float,
        help="Confidence level in (0, 1), e.g. 0.95.",
    )
    parser.add_argument(
        "--adjust",
        dest="adjust",
        action="store_true",
        help="Add Bonferroni and Benjamini-Hochberg FDR p-values.",
    )
    parser.add_argument(
        "--keep-path",
        dest="keep_path",
        default=None,
        type=str,
        help="Path to keep file (FID IID or IID per line).",
    )
    parser.add_argument(
        "--remove-path",
        dest="remove_path",
        default=None,
        type=str,
        help="Path to remove file (FID IID or IID per line).",
    )


_COMMANDS: Dict[str, _Command] = {
    "pca": _Command(
        help="Run PCA and save plot/components.",
        add_arguments=_add_pca_arguments,
        run=_run_pca,
    ),
    "admixture-map": _Command(
        help="Run admixture mapping.",
        add_arguments=_add_admixture_map_arguments,
        run=_run_admixture_map,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snputils",
        description="snputils command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command")

    for name, command in _COMMANDS.items():
        subparser = subparsers.add_parser(name, help=command.help, description=command.help)
        command.add_arguments(subparser)
        subparser.set_defaults(_handler=command.run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args_list = sys.argv[1:] if argv is None else argv
    if not args_list:
        parser.print_help(sys.stderr)
        return 1

    args = parser.parse_args(args_list)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 1
    return int(handler(args))
