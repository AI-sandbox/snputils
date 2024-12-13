from pathlib import Path
from typing import Union

from snputils.ancestry.genobj.window import WindowLevelAncestryObject


def read_lai(file: Union[str, Path], **kwargs) -> WindowLevelAncestryObject:
    """
    Automatically detect the local ancestry data file format from the file's extension and 
    read it into a `snputils.ancestry.genobj.WindowLevelAncestryObject`.

    **Supported formats:**

    - `.msp`: Text-based MSP format.
    - `.msp.tsv`: Text-based MSP format with TSV extension.
    
    Args:
        file (str or pathlib.Path): 
            Path to the file to be read. It should end with `.msp` or `.msp.tsv`.
        **kwargs: Additional arguments passed to the reader method.
    """
    from snputils.ancestry.io.local.read.auto import LAIReader

    return LAIReader(file).read(**kwargs)


def read_msp(file: Union[str, Path]) -> 'WindowLevelAncestryObject':
    """
    Read data from an `.msp` or `.msp.tsv` file and construct a `snputils.ancestry.genobj.WindowLevelAncestryObject`.

    Args:
        file (str or pathlib.Path): 
            Path to the file to be read. It should end with `.msp` or `.msp.tsv`.

    Returns:
        **WindowLevelAncestryObject:**
            A WindowLevelAncestryObject instance.
    """
    from snputils.ancestry.io.local.read.msp import MSPReader

    return MSPReader(file).read()
