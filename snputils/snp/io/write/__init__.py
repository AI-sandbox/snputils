from .vcf import VCFWriter
from .bed import BEDWriter
from .bgen import BGENWriter
from .pgen import PGENWriter
from .grg_from_vcf import vcf_to_grg, vcf_to_igd

__all__ = ["VCFWriter", "BEDWriter", "BGENWriter", "PGENWriter", "GRGWriter", "vcf_to_grg", "vcf_to_igd"]


def __getattr__(name):
    if name == "GRGWriter":
        try:
            from .grg import GRGWriter
        except ModuleNotFoundError as exc:
            if exc.name == "pygrgl":
                raise ImportError(
                    "GRG support requires the optional dependency 'pygrgl'. "
                    "Install it with: pip install pygrgl"
                ) from exc
            raise
        return GRGWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
