import logging
import multiprocessing
import subprocess
from contextlib import ExitStack
from os.path import abspath, splitext
from pathlib import Path
from typing import Any, Optional, Union

log = logging.getLogger(__name__)


def vcf_to_igd(
    vcf_file: Union[str, Path],
    igd_file: Optional[Union[str, Path]] = None,
    logfile_out: Optional[Union[str, Path]] = None,
    logfile_err: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Convert a VCF file to IGD via `grg convert`.

    Args:
        vcf_file: Input VCF path.
        igd_file: Output IGD file path. Defaults to `<vcf_stem>.igd`.
        logfile_out: File to append standard output to. If None, output is discarded.
        logfile_err: File to append standard error to. If None, output is discarded.

    Returns:
        Path to the IGD file.
    """
    vcf_path = Path(vcf_file)
    if not vcf_path.exists():
        raise FileNotFoundError(f"File {vcf_path} does not exist")

    igd_path = Path(igd_file) if igd_file is not None else _default_converted_path(vcf_path, ".igd")

    with ExitStack() as stack:
        lf_o = _log_handle(logfile_out, stack)
        lf_e = _log_handle(logfile_err, stack)
        subprocess.run(
            ["grg", "convert", abspath(str(vcf_path)), abspath(str(igd_path))],
            stdout=lf_o,
            stderr=lf_e,
            check=True,
        )

    return igd_path


def vcf_to_grg(
    vcf_file: Union[str, Path],
    range: Optional[str] = None,
    parts: Optional[int] = None,
    jobs: Optional[int] = None,
    trees: Optional[int] = None,
    binmuts: Optional[bool] = None,
    no_file_cleanup: Optional[bool] = None,
    maf_flip: Optional[bool] = None,
    population_ids: Optional[Union[str, Path]] = None,
    mutation_batch_size: Optional[int] = None,
    igd_file: Optional[Union[str, Path]] = None,
    out_file: Optional[Union[str, Path]] = None,
    verbose: Optional[bool] = None,
    no_merge: Optional[bool] = None,
    force: Optional[bool] = None,
    logfile_out: Optional[Union[str, Path]] = None,
    logfile_err: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Convert a VCF file to a GRG file via `grg construct`.

    If `igd_file` exists, it is used as construct input. If it does not exist,
    it is first created via `grg convert` and then used for construction.

    Returns:
        Path to the GRG file.
    """
    input_file = Path(vcf_file).resolve()
    if igd_file is not None:
        candidate_igd = Path(igd_file)
        if candidate_igd.exists():
            input_file = candidate_igd.resolve()
        else:
            input_file = vcf_to_igd(
                vcf_file,
                igd_file=igd_file,
                logfile_out=logfile_out,
                logfile_err=logfile_err,
            ).resolve()

    grg_path = Path(out_file) if out_file is not None else _default_converted_path(input_file, ".grg")

    args = ["grg", "construct"]
    args += _setarg(range, "-r", None)
    args += _setarg(parts, "-p", 50)
    args += _setarg(jobs, "-j", multiprocessing.cpu_count())
    args += _setarg(trees, "-t", 16)
    args += _setarg(binmuts, "--binary-muts", None)
    args += _setarg(no_file_cleanup, "--no-file-cleanup", None)
    args += _setarg(maf_flip, "--maf-flip", None)
    args += _setarg(population_ids, "--population-ids", None)
    args += _setarg(mutation_batch_size, "--mutation-batch-size", None)
    args += _setarg(str(grg_path), "--out-file", None)
    args += _setarg(verbose, "--verbose", None)
    args += _setarg(no_merge, "--no-merge", None)
    args += _setarg(force, "--force", None)
    args += [str(input_file)]
    log.debug("Running grg construct command: %s", args)

    with ExitStack() as stack:
        lf_o = _log_handle(logfile_out, stack)
        lf_e = _log_handle(logfile_err, stack)
        subprocess.run(args, stdout=lf_o, stderr=lf_e, check=True)

    return grg_path


def _default_converted_path(input_file: Union[str, Path], suffix: str) -> Path:
    default_stem = splitext(str(input_file))[0]
    if default_stem.endswith(".vcf"):
        default_stem = splitext(default_stem)[0]
    return Path(default_stem + suffix)


def _log_handle(logfile: Optional[Union[str, Path]], stack: ExitStack):
    if logfile is None:
        return subprocess.DEVNULL
    return stack.enter_context(open(logfile, "a"))


def _setarg(x: Optional[Any], flag: str, default_arg: Optional[Any] = None) -> list[str]:
    if isinstance(x, bool):
        return [flag] if x else []
    if x is None and default_arg is not None:
        return [flag, f"{default_arg}"]
    if x is not None:
        return [flag, f"{x}"]
    return []
