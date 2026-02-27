from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("snputils")
except PackageNotFoundError:
    __version__ = "unknown"

from .snp import SNPObject, SNPReader, BEDReader, PGENReader, VCFReader, BEDWriter, PGENWriter, VCFWriter, read_snp, read_bed, read_pgen, read_vcf, read_grg
from .ancestry import LocalAncestryObject, GlobalAncestryObject, MSPReader, MSPWriter, AdmixtureMappingVCFWriter, AdmixtureReader, AdmixtureWriter, read_lai, read_msp, read_adm, read_admixture
from .ibd import IBDObject, read_ibd, HapIBDReader, AncIBDReader, IBDReader
from .phenotype import (
    MultiPhenotypeObject,
    PhenotypeObject,
    MultiPhenReader,
    PhenotypeReader,
)
from . import visualization as viz
from .datasets import load_dataset

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
    "LocalAncestryObject",
    "GlobalAncestryObject",
    "MSPReader",
    "MSPWriter",
    "AdmixtureMappingVCFWriter",
    "AdmixtureReader",
    "AdmixtureWriter",
    "read_lai",
    "read_msp",
    "read_adm",
    "read_admixture",
    "IBDObject",
    "read_ibd",
    "HapIBDReader",
    "AncIBDReader",
    "IBDReader",
    "MultiPhenotypeObject",
    "PhenotypeObject",
    "MultiPhenReader",
    "PhenotypeReader",
    "load_dataset",
    "viz",
]


def __getattr__(name):
    if name in {"GRGObject", "GRGReader", "GRGWriter"}:
        from . import snp as _snp

        return getattr(_snp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
