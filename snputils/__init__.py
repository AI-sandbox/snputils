from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("snputils")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .snp import SNPObject, GRGObject, SNPReader, BEDReader, GRGReader, PGENReader, VCFReader, BEDWriter, PGENWriter, VCFWriter, read_snp, read_bed, read_pgen, read_vcf
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
