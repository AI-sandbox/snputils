from .read import SNPReader, BEDReader, PGENReader, VCFReader, read_snp, read_bed, read_pgen, read_vcf, read_grg
from .write import BEDWriter, PGENWriter, VCFWriter

__all__ = ['read_snp', 'read_bed', 'read_pgen', 'read_vcf', 'read_grg',
           'SNPReader', 'BEDReader', 'PGENReader', 'VCFReader',
           'BEDWriter', 'PGENWriter', 'VCFWriter', 'GRGWriter', 'GRGReader']


def __getattr__(name):
    if name == "GRGReader":
        from .read import GRGReader

        return GRGReader
    if name == "GRGWriter":
        from .write import GRGWriter

        return GRGWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
