from pathlib import Path
from typing import Optional, Union

from snputils.ancestry.genobj.local import LocalAncestryObject


def read_lai(file: Union[str, Path], **kwargs) -> LocalAncestryObject:
    """
    Automatically detect the local ancestry data file format from the file's extension and
    read it into a `snputils.ancestry.genobj.LocalAncestryObject`.

    **Supported formats:**

    - MSP (`.msp`, `.msp.tsv`)
    - FLARE (`.anc.vcf`)
    - admix-kit LANC (`.lanc`)

    Text inputs may be plain, gzip-compressed, or Zstandard-compressed.

    Args:
        file (str or pathlib.Path):
            Path to the file to be read.
        **kwargs: Additional arguments passed to the reader method.
    """
    from snputils.ancestry.io.local.read.auto import LAIReader

    return LAIReader(file).read(**kwargs)


def read_msp(file: Union[str, Path]) -> 'LocalAncestryObject':
    """Read an MSP file into a `LocalAncestryObject`.

    Args:
        file (str or pathlib.Path):
            Path to the file to be read.

    Returns:
        LocalAncestryObject: A LocalAncestryObject instance.
    """
    from snputils.ancestry.io.local.read.msp import MSPReader

    return MSPReader(file).read()


def read_flare(file: Union[str, Path]) -> 'LocalAncestryObject':
    """Read a FLARE ancestry VCF into a `LocalAncestryObject`."""
    from snputils.ancestry.io.local.read.flare import FLAREReader

    return FLAREReader(file).read()


def read_lanc(
    file: Union[str, Path],
    *,
    pvar_file: Optional[Union[str, Path]] = None,
    psam_file: Optional[Union[str, Path]] = None,
) -> 'LocalAncestryObject':
    """Read an admix-kit LANC file into a `LocalAncestryObject`.

    By default this looks for sibling `.pvar` and `.psam` files with the same
    prefix as `file` to recover SNP coordinates and sample IDs.
    Pass `pvar_file=` and/or `psam_file=` to point at those sidecars elsewhere.
    If either sidecar is unavailable, the reader falls back to loading the LAI
    calls alone and warns that the missing metadata could not be reconstructed.
    """
    from snputils.ancestry.io.local.read.lanc import LANCReader

    return LANCReader(file, pvar_file=pvar_file, psam_file=psam_file).read()
