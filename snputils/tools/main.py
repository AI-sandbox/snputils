import sys
import logging

log = logging.getLogger(__name__)


def _help_text() -> str:
    return (
        "snputils CLI utilities\n"
        "\n"
        "Usage:\n"
        "  snputils <command> [options]\n"
        "\n"
        "Available commands:\n"
        "  pca                Run PCA and save plot/components.\n"
        "  dummy_tool         Print a message (example command).\n"
        "  admixture_mapping  Run admixture mapping.\n"
        "\n"
        "Examples:\n"
        "  snputils pca --vcf_file data.vcf --fig_path pca.png --npy_path pca.npy --backend sklearn\n"
        "  snputils admixture_mapping --pheID trait --phe_path trait.phe --msp_path ancestry.msp --results_path out/\n"
    )


def _print_help(error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    print(_help_text(), file=stream)


def main():
    arg_list = tuple(sys.argv)
    if len(arg_list) <= 1:
        _print_help(error=True)
        sys.exit(1)

    command = sys.argv[1]
    if command in {"-h", "--help", "help"}:
        _print_help(error=False)
        sys.exit(0)

    if command == "pca":
        from . import pca

        sys.exit(pca.plot_and_save_pca(arg_list[2:]))
    if command == "dummy_tool":
        from . import dummy_tool

        sys.exit(dummy_tool.dummy_tool(arg_list[2:]))
    if command == "admixture_mapping":
        from . import admixture_mapping

        sys.exit(admixture_mapping.admixmap(arg_list[2:]))

    log.error(
        'Invalid command "%s". Use "snputils --help" to see available commands.',
        command,
    )
    sys.exit(1)
