from .vcf import VCFWriter
from .bed import BEDWriter
from .pgen import PGENWriter

__all__ = ["VCFWriter", "BEDWriter", "PGENWriter", "GRGWriter"]


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
