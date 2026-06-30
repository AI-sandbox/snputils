from pathlib import Path
from typing import Optional, Union

from snputils.ancestry.genobj.local import LocalAncestryObject


def read_lai(file: Union[str, Path], **kwargs) -> LocalAncestryObject:
    """
    Automatically detect the local ancestry data file format from the file's extension and 
    read it into a `snputils.ancestry.genobj.LocalAncestryObject`.

    **Supported formats:**

    - `.msp`: Text-based MSP format.
    - `.msp.tsv`: Text-based MSP format with TSV extension.
    - `.anc.vcf` / `.anc.vcf.gz`: FLARE local ancestry VCF output.
    - `.lanc`: admix-kit local ancestry change-point format.
    
    Args:
        file (str or pathlib.Path): 
            Path to the file to be read.
        **kwargs: Additional arguments passed to the reader method.
    """
    from snputils.ancestry.io.local.read.auto import LAIReader

    return LAIReader(file).read(**kwargs)


def read_msp(file: Union[str, Path]) -> 'LocalAncestryObject':
    """
    Read data from an `.msp` or `.msp.tsv` file and construct a `snputils.ancestry.genobj.LocalAncestryObject`.

    Args:
        file (str or pathlib.Path): 
            Path to the file to be read. It should end with `.msp` or `.msp.tsv`.

    Returns:
        LocalAncestryObject: A LocalAncestryObject instance.
    """
    from snputils.ancestry.io.local.read.msp import MSPReader

    return MSPReader(file).read()


def read_flare(file: Union[str, Path]) -> 'LocalAncestryObject':
    """
    Read data from a FLARE `.anc.vcf` or `.anc.vcf.gz` output file and construct a
    `snputils.ancestry.genobj.LocalAncestryObject`.
    """
    from snputils.ancestry.io.local.read.flare import FLAREReader

    return FLAREReader(file).read()


def read_lanc(
    file: Union[str, Path],
    *,
    pvar_file: Optional[Union[str, Path]] = None,
    psam_file: Optional[Union[str, Path]] = None,
) -> 'LocalAncestryObject':
    """
    Read data from an admix-kit `.lanc` file and construct a
    `snputils.ancestry.genobj.LocalAncestryObject`.

    By default this looks for sibling `.pvar`/`.pvar.zst` and `.psam` files
    with the same prefix as `file` to recover SNP coordinates and sample IDs.
    Pass `pvar_file=` and/or `psam_file=` to point at those sidecars elsewhere.
    If either sidecar is unavailable, the reader falls back to loading the LAI
    calls alone and warns that the missing metadata could not be reconstructed.
    """
    from snputils.ancestry.io.local.read.lanc import LANCReader

    return LANCReader(file, pvar_file=pvar_file, psam_file=psam_file).read()
