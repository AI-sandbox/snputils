from pathlib import Path
from typing import Union

from snputils.ancestry.genobj.local import LocalAncestryObject


def read_lai(file: Union[str, Path], **kwargs) -> LocalAncestryObject:
    """
    Automatically detect the local ancestry data file format from the file's extension and 
    read it into a `snputils.ancestry.genobj.LocalAncestryObject`.

    **Supported formats:**

    - `.msp`: Text-based MSP format.
    - `.msp.tsv`: Text-based MSP format with TSV extension.
    - `.anc.vcf` / `.anc.vcf.gz`: FLARE local ancestry VCF output.
    
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
