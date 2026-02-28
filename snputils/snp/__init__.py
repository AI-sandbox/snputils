from .genobj import SNPObject
from .io import SNPReader, BEDReader, PGENReader, VCFReader, BEDWriter, PGENWriter, VCFWriter, read_snp, read_bed, read_pgen, read_vcf, read_grg

__all__ = [
    "SNPObject",
    "GRGObject",
    "SNPReader",
    "BEDReader",
    "GRGReader",
    "GRGWriter",
    "PGENReader",
    "VCFReader",
    "BEDWriter",
    "PGENWriter",
    "VCFWriter",
    "read_snp",
    "read_bed",
    "read_pgen",
    "read_vcf",
    "read_grg",
]


def __getattr__(name):
    if name == "GRGObject":
        from .genobj import GRGObject

        return GRGObject
    if name == "GRGReader":
        from .io import GRGReader

        return GRGReader
    if name == "GRGWriter":
        from .io import GRGWriter

        return GRGWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
