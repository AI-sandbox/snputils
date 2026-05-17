import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from snputils import __version__

DOCS_URL = "https://docs.snputils.org"
SOURCE_URL = "https://github.com/AI-sandbox/snputils"


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


def _run_gwas(args: argparse.Namespace) -> int:
    from . import gwas as gwas_module

    return int(gwas_module.run_gwas_command(args))


def _run_simulate(args: argparse.Namespace) -> int:
    try:
        from snputils.simulation.simulator_cli import run_simulator_command
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print(
                "snputils simulate requires PyTorch. Install it with `pip install 'snputils[torch]'`.",
                file=sys.stderr,
            )
            return 2
        raise

    return int(run_simulator_command(args))


def _run_mdpca(args: argparse.Namespace) -> int:
    from snputils.ancestry.io.local.read import read_lai
    from snputils.processing.mdpca import mdPCA
    from snputils.snp.io.read import read_snp

    snpobj = read_snp(args.snp_path, sum_strands=False)
    laiobj = read_lai(args.lai_path)
    mdPCA(
        snpobj=snpobj,
        laiobj=laiobj,
        labels_file=args.labels_file,
        ancestry=args.ancestry,
        method=args.method,
        is_masked=not args.unmasked,
        average_strands=args.average_strands,
        force_nan_incomplete_strands=args.force_nan_incomplete_strands,
        is_weighted=args.weighted,
        groups_to_remove=args.groups_to_remove,
        min_percent_snps=args.min_percent_snps,
        group_snp_frequencies_only=not args.include_individual_frequencies,
        save_masks=args.save_masks,
        load_masks=args.load_masks,
        masks_file=args.masks_file,
        embedding_table_path=args.coords,
        covariance_matrix_file=args.covariance_matrix_file,
        n_components=args.n_components,
        rsid_or_chrompos=args.rsid_or_chrompos,
        percent_vals_masked=args.percent_vals_masked,
    )
    return 0


def _run_maasmds(args: argparse.Namespace) -> int:
    from snputils.ancestry.io.local.read import read_lai
    from snputils.processing.maasmds import maasMDS
    from snputils.snp.io.read import read_snp

    snp_paths = _split_csv(args.snp_path)
    lai_paths = _split_csv(args.lai_path)
    if len(snp_paths) != len(lai_paths):
        raise ValueError("--snp-path and --lai-path must contain the same number of comma-separated paths.")

    snpobj = [read_snp(path, sum_strands=False) for path in snp_paths]
    laiobj = [read_lai(path) for path in lai_paths]
    if len(snpobj) == 1:
        snp_arg = snpobj[0]
        lai_arg = laiobj[0]
    else:
        snp_arg = snpobj
        lai_arg = laiobj

    maasMDS(
        snpobj=snp_arg,
        laiobj=lai_arg,
        labels_file=args.labels_file,
        ancestry=args.ancestry,
        is_masked=not args.unmasked,
        average_strands=args.average_strands,
        force_nan_incomplete_strands=args.force_nan_incomplete_strands,
        is_weighted=args.weighted,
        groups_to_remove=args.groups_to_remove,
        min_percent_snps=args.min_percent_snps,
        group_snp_frequencies_only=not args.include_individual_frequencies,
        save_masks=args.save_masks,
        load_masks=args.load_masks,
        masks_file=args.masks_file,
        distance_type=args.distance_type,
        n_components=args.n_components,
        rsid_or_chrompos=args.rsid_or_chrompos,
        embedding_table_path=args.coords,
    )
    return 0


def _run_plot_manhattan(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from snputils.visualization.manhattan_plot import manhattan_plot

    manhattan_plot(
        args.results_path,
        significance_threshold=args.significance_threshold,
        point_size=args.point_size,
        line_width=args.line_width,
        line_color=args.line_color,
        title=args.title,
        save=True,
        output_filename=args.output_path,
    )
    return 0


def _run_plot_qq(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from snputils.visualization.qq_plot import qq_plot

    qq_plot(
        args.results_path,
        color=args.color,
        significance_threshold=args.significance_threshold,
        point_size=args.point_size,
        line_width=args.line_width,
        expected_line_color=args.expected_line_color,
        threshold_line_color=args.threshold_line_color,
        title=args.title,
        save=True,
        output_filename=args.output_path,
    )
    return 0


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _add_pca_arguments(parser: argparse.ArgumentParser) -> None:
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
        help="Path to save the PCA scatter plot (.pdf / .svg / .png, ...; vector formats use rasterized points at 300 dpi by default).",
    )
    parser.add_argument(
        "--coords",
        dest="coords",
        default=None,
        type=str,
        help="Optional path to write PC coordinates as TSV/CSV (see snputils.processing.dimred_tabular).",
    )
    parser.add_argument(
        "--components",
        dest="components",
        default=None,
        type=str,
        help="Optional path to save PCA components as a .npy file.",
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
        action="store_true",
        help="Read diploid genotypes as per-individual summed strand counts.",
    )
    parser.add_argument(
        "--vcf-backend",
        dest="vcf_backend",
        choices=("default", "polars"),
        default="default",
        help="VCF reader backend (used only when input is VCF).",
    )


def _add_admixture_map_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phe-id",
        dest="phe_id",
        required=True,
        type=str,
        help="Phenotype ID / column name to analyze.",
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
        "--sample-remove",
        dest="remove_path",
        default=None,
        type=str,
        help="Path to remove file (FID IID or IID per line).",
    )


def _add_gwas_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phe-id",
        dest="phe_id",
        required=True,
        type=str,
        help="Phenotype ID / column name to analyze.",
    )
    parser.add_argument(
        "--phe-path",
        dest="phe_path",
        required=True,
        type=str,
        help="Path to phenotype file with IID and one or more phenotype columns.",
    )
    parser.add_argument(
        "--snp-path",
        dest="snp_path",
        required=True,
        type=str,
        help="Path to genotype input (VCF/BED/PGEN).",
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
        help="Max variants processed per chunk.",
    )
    parser.add_argument(
        "--memory",
        dest="memory",
        default=None,
        type=int,
        help="Peak RSS-delta memory cap in MiB.",
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
        "--variant-exclude",
        dest="exclude_path",
        default=None,
        type=str,
        help="Path to variant exclusion file (one or more variant selectors per line).",
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
        "--sample-remove",
        dest="remove_path",
        default=None,
        type=str,
        help="Path to remove file (FID IID or IID per line).",
    )
    parser.add_argument(
        "--vcf-backend",
        dest="vcf_backend",
        choices=("polars", "scikit-allel"),
        default="polars",
        help="VCF reader backend (used only when input is VCF).",
    )


def _add_dimred_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snp-path", required=True, help="Path to SNP input. For maasmds, pass comma-separated paths for multiple arrays.")
    parser.add_argument("--lai-path", required=True, help="Path to MSP/local ancestry input. For maasmds, pass comma-separated paths matching --snp-path.")
    parser.add_argument("--labels-file", required=True, help="TSV labels file with indID and label columns.")
    parser.add_argument("--ancestry", required=True, help="Ancestry index or ancestry-map label to analyze.")
    parser.add_argument("--coords", required=True, help="Output TSV/CSV path for coordinates and row metadata.")
    parser.add_argument("--n-components", type=_positive_int, default=2, help="Number of dimensions/components to compute.")
    parser.add_argument("--unmasked", action="store_true", help="Use unmasked genotypes instead of ancestry-specific masking.")
    parser.add_argument("--average-strands", action="store_true", help="Average each individual's two haplotypes.")
    parser.add_argument("--force-nan-incomplete-strands", action="store_true", help="Set averaged strand pairs to NaN if either haplotype is missing.")
    parser.add_argument("--weighted", action="store_true", help="Read individual weights from the labels file.")
    parser.add_argument("--groups-to-remove", nargs="+", default=None, help="Population labels to remove before analysis.")
    parser.add_argument("--min-percent-snps", type=float, default=4, help="Minimum percent of non-missing SNPs required per row.")
    parser.add_argument("--include-individual-frequencies", action="store_true", help="Keep individual-level data when weighted group combinations are present.")
    parser.add_argument("--save-masks", action="store_true", help="Save masked genotype data to --masks-file.")
    parser.add_argument("--load-masks", action="store_true", help="Load masked genotype data from --masks-file.")
    parser.add_argument("--masks-file", default="masks.npz", help="Path for saving/loading masked genotype data.")
    parser.add_argument("--rsid-or-chrompos", type=int, choices=(1, 2), default=2, help="Variant ID mode: 1=rsID, 2=chromosome_position.")


def _add_mdpca_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dimred_common_arguments(parser)
    parser.add_argument(
        "--method",
        default="weighted_cov_pca",
        choices=(
            "weighted_cov_pca",
            "regularized_optimization_ils",
            "cov_matrix_imputation",
            "cov_matrix_imputation_ils",
            "nonmissing_pca_ils",
        ),
        help="mdPCA method.",
    )
    parser.add_argument("--covariance-matrix-file", default=None, help="Optional .npy path for the covariance matrix.")
    parser.add_argument("--percent-vals-masked", type=float, default=0, help="Percent of covariance values to mask for imputation methods.")


def _add_maasmds_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dimred_common_arguments(parser)
    parser.add_argument(
        "--distance-type",
        choices=("Manhattan", "RMS", "AP"),
        default="AP",
        help="Pairwise distance used before MDS.",
    )


def _add_simulate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vcf", required=True, help="Path to the phased VCF/VCF-gz file.")
    parser.add_argument("--metadata", required=True, help="TSV/CSV file with at least Sample / Population / Latitude / Longitude.")
    parser.add_argument("--output-dir", required=True, help="Directory in which to save the simulated batches.")
    parser.add_argument("--genetic-map", default=None, help="Genetic map table with columns: chrom, pos, cM.")
    parser.add_argument("--chromosome", type=int, default=None, help="If provided, restrict genetic map rows to this chromosome id.")
    parser.add_argument("--window-size", type=int, default=1000, help="#SNPs per window.")
    parser.add_argument("--store-latlon-as-nvec", action="store_true", help="Convert lat/lon to unit n-vectors (x,y,z).")
    parser.add_argument("--make-haploid", action="store_true", help="Flatten diploid genotypes into haplotypes.")
    parser.add_argument("--device", default="cpu", help="torch device string, e.g. 'cuda:0'.")
    parser.add_argument("--batch-size", type=int, default=256, help="#simulated haplotypes per batch.")
    parser.add_argument("--num-generations", type=int, default=10, help="Upper bound on random generations since admixture.")
    parser.add_argument("--n-batches", type=int, default=1, help="#separate batches to generate & save.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print additional debugging info.")


def _add_plot_manhattan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-path", required=True, help="Association result TSV with #CHROM, POS, and P columns.")
    parser.add_argument("--output-path", required=True, help="Output figure path (.pdf, .svg, .png, ...).")
    parser.add_argument("--significance-threshold", type=float, default=0.05, help="Nominal alpha used for the Bonferroni line.")
    parser.add_argument("--point-size", type=float, default=7.0, help="Scatter point size.")
    parser.add_argument("--line-width", type=float, default=1.0, help="Reference line width.")
    parser.add_argument("--line-color", default="r", help="Bonferroni reference line color.")
    parser.add_argument("--title", default=None, help="Optional plot title.")


def _add_plot_qq_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-path", required=True, help="Association result TSV with a P column.")
    parser.add_argument("--output-path", required=True, help="Output figure path (.pdf, .svg, .png, ...).")
    parser.add_argument("--significance-threshold", type=float, default=0.05, help="Nominal alpha used for the Bonferroni line.")
    parser.add_argument("--point-size", type=float, default=7.0, help="Scatter point size.")
    parser.add_argument("--line-width", type=float, default=1.0, help="Reference line width.")
    parser.add_argument("--color", default="black", help="Scatter point color.")
    parser.add_argument("--expected-line-color", default="red", help="Expected-null reference line color.")
    parser.add_argument("--threshold-line-color", default="orange", help="Bonferroni threshold line color.")
    parser.add_argument("--title", default=None, help="Optional plot title.")


_COMMANDS: Dict[str, _Command] = {
    "pca": _Command(
        help="Run PCA and save plot/components.",
        add_arguments=_add_pca_arguments,
        run=_run_pca,
    ),
    "mdpca": _Command(
        help="Run missing-data PCA and save an embedding table.",
        add_arguments=_add_mdpca_arguments,
        run=_run_mdpca,
    ),
    "maasmds": _Command(
        help="Run multi-array ancestry-specific MDS and save an embedding table.",
        add_arguments=_add_maasmds_arguments,
        run=_run_maasmds,
    ),
    "admixture-map": _Command(
        help="Run admixture mapping.",
        add_arguments=_add_admixture_map_arguments,
        run=_run_admixture_map,
    ),
    "gwas": _Command(
        help="Run GWAS.",
        add_arguments=_add_gwas_arguments,
        run=_run_gwas,
    ),
    "simulate": _Command(
        help="Simulate admixed haplotype batches.",
        add_arguments=_add_simulate_arguments,
        run=_run_simulate,
    ),
    "plot-manhattan": _Command(
        help="Create a Manhattan plot from association results.",
        add_arguments=_add_plot_manhattan_arguments,
        run=_run_plot_manhattan,
    ),
    "plot-qq": _Command(
        help="Create a QQ plot from association results.",
        add_arguments=_add_plot_qq_arguments,
        run=_run_plot_qq,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snputils",
        description=(
            "snputils command-line interface for common file-backed workflows. "
            f"Version: {__version__}. Docs: {DOCS_URL}. Source: {SOURCE_URL}."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"snputils {__version__}",
        help="Show the installed snputils version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for name, command in _COMMANDS.items():
        subparser = subparsers.add_parser(name, help=command.help, description=command.help)
        command.add_arguments(subparser)
        subparser.set_defaults(_handler=command.run)

    version_parser = subparsers.add_parser("version", help="Show the installed snputils version.")
    version_parser.set_defaults(_handler=lambda args: _print_version())

    return parser


def _print_version() -> int:
    print(f"snputils {__version__}")
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
