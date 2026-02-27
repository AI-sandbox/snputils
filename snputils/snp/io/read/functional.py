import pathlib
from typing import TYPE_CHECKING, Union

from snputils.snp.genobj import SNPObject

if TYPE_CHECKING:
    from snputils.snp.genobj.grgobj import GRGObject


def read_snp(filename: Union[str, pathlib.Path], **kwargs) -> SNPObject:
    """
    Automatically detect the file format and read it into a SNPObject.

    Args:
        filename: Filename of the file to read.
        **kwargs: Additional arguments passed to the reader method.

    Raises:
        ValueError: If the filename does not have an extension or the extension is not supported.
    """
    from snputils.snp.io.read.auto import SNPReader

    return SNPReader(filename).read(**kwargs)


def read_bed(filename: Union[str, pathlib.Path], **kwargs) -> SNPObject:
    """
    Read a BED fileset into a SNPObject.

    Args:
        filename: Filename of the BED fileset to read.
        **kwargs: Additional arguments passed to the reader method. See :class:`snputils.snp.io.read.bed.BEDReader` for possible parameters.
    """
    from snputils.snp.io.read.bed import BEDReader

    return BEDReader(filename).read(**kwargs)


def read_pgen(filename: Union[str, pathlib.Path], **kwargs) -> SNPObject:
    """
    Read a PGEN fileset into a SNPObject.

    Args:
        filename: Filename of the PGEN fileset to read.
        **kwargs: Additional arguments passed to the reader method. See :class:`snputils.snp.io.read.pgen.PGENReader` for possible parameters.
    """
    from snputils.snp.io.read.pgen import PGENReader

    return PGENReader(filename).read(**kwargs)


def read_vcf(filename: Union[str, pathlib.Path], 
             backend: str = 'polars',
             **kwargs) -> SNPObject:
    """
    Read a VCF fileset into a SNPObject.

    Args:
        filename: Filename of the VCF fileset to read.
        backend: Backend to use for reading the VCF file. Options are 'polars' or 'scikit-allel'.
        **kwargs: Additional arguments passed to the reader method. See :class:`snputils.snp.io.read.vcf.VCFReader` for possible parameters.
    """
    from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars
    if backend == 'polars':
        print(f"Reading {filename} with polars backend")
        return VCFReaderPolars(filename).read(**kwargs)
    else:
        print(f"Reading {filename} with scikit-allel backend")
        return VCFReader(filename).read(**kwargs)


def read_grg(filename: Union[str, pathlib.Path], **kwargs) -> "GRGObject":
    """
    Read a GRG file into a GRGObject.

    Args:
        filename: Filename of the GRG file to read.
        **kwargs: Additional arguments passed to the reader method.
    """
    try:
        from snputils.snp.io.read.grg import GRGReader
    except ModuleNotFoundError as exc:
        if exc.name == "pygrgl":
            raise ImportError(
                "GRG support requires the optional dependency 'pygrgl'. "
                "Install it with: pip install pygrgl"
            ) from exc
        raise

    return GRGReader(filename).read(**kwargs)
