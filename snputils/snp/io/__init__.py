from .read import SNPReader, BEDReader, BGENReader, PGENReader, VCFReader, read_snp, read_bed, read_bgen, read_pgen, read_vcf, read_grg
from .write import BEDWriter, BGENWriter, PGENWriter, VCFWriter, vcf_to_grg, vcf_to_igd

__all__ = ['read_snp', 'read_bed', 'read_bgen', 'read_pgen', 'read_vcf', 'read_grg',
           'SNPReader', 'BEDReader', 'BGENReader', 'PGENReader', 'VCFReader',
           'BEDWriter', 'BGENWriter', 'PGENWriter', 'VCFWriter', 'GRGWriter', 'GRGReader',
           'vcf_to_grg', 'vcf_to_igd']


def __getattr__(name):
    if name == "GRGReader":
        from .read import GRGReader

        return GRGReader
    if name == "GRGWriter":
        from .write import GRGWriter

        return GRGWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
