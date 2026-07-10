from pathlib import Path
from typing import Union

from snputils.ibd.genobj.ibdobj import IBDObject


def read_ibd(file: Union[str, Path], **kwargs) -> IBDObject:
    """
    Automatically detect the IBD data file format from the file's extension and read it into an `IBDObject`.

    Supported formats are hap-IBD (``.ibd``) and ancIBD (``.tsv``). Text
    inputs may be plain, gzip-compressed, or Zstandard-compressed.

    Args:
        file (str or pathlib.Path): Path to the file to be read.
        **kwargs: Additional arguments passed to the reader method.
    """
    from snputils.ibd.io.read.auto import IBDReader

    return IBDReader(file).read(**kwargs)
