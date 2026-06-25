from .auto import SNPReader
from .bed import BEDReader
from .bcf import BCFReader
from .bgen import BGENReader
from .pgen import PGENReader
from .vcf import VCFReader
from .functional import read_bcf, read_bed, read_bgen, read_grg, read_pgen, read_snp, read_vcf

__all__ = [
    "SNPReader",
    "BEDReader",
    "BCFReader",
    "BGENReader",
    "PGENReader",
    "VCFReader",
    "GRGReader",
    "read_snp",
    "read_bcf",
    "read_bed",
    "read_bgen",
    "read_pgen",
    "read_vcf",
    "read_grg",
]


def __getattr__(name):
    if name == "GRGReader":
        try:
            from .grg import GRGReader
        except ModuleNotFoundError as exc:
            if exc.name == "pygrgl":
                raise ImportError(
                    "GRG support requires the optional dependency 'pygrgl'. "
                    "Install it with: pip install pygrgl"
                ) from exc
            raise
        return GRGReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
