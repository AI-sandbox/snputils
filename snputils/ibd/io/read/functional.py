from pathlib import Path
from typing import Union

from snputils.ibd.genobj.ibdobj import IBDObject


def read_ibd(file: Union[str, Path], **kwargs) -> IBDObject:
    """
    Automatically detect the IBD data file format from the file's extension and read it into an `IBDObject`.

    Supported formats:
    - Hap-IBD (no standard extension; defaults to tab-delimited columns without header).
    - ancIBD (template only).

    Args:
        file (str or pathlib.Path): Path to the file to be read.
        **kwargs: Additional arguments passed to the reader method.
    """
    from snputils.ibd.io.read.auto import IBDReader

    return IBDReader(file).read(**kwargs)


