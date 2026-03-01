from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Tuple

try:
    __version__ = version("snputils")
except PackageNotFoundError:
    __version__ = "unknown"

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "SNPObject": (".snp", "SNPObject"),
    "GRGObject": (".snp", "GRGObject"),
    "SNPReader": (".snp", "SNPReader"),
    "BEDReader": (".snp", "BEDReader"),
    "GRGReader": (".snp", "GRGReader"),
    "GRGWriter": (".snp", "GRGWriter"),
    "PGENReader": (".snp", "PGENReader"),
    "VCFReader": (".snp", "VCFReader"),
    "BEDWriter": (".snp", "BEDWriter"),
    "PGENWriter": (".snp", "PGENWriter"),
    "VCFWriter": (".snp", "VCFWriter"),
    "read_snp": (".snp", "read_snp"),
    "read_bed": (".snp", "read_bed"),
    "read_pgen": (".snp", "read_pgen"),
    "read_vcf": (".snp", "read_vcf"),
    "read_grg": (".snp", "read_grg"),
    "LocalAncestryObject": (".ancestry", "LocalAncestryObject"),
    "GlobalAncestryObject": (".ancestry", "GlobalAncestryObject"),
    "MSPReader": (".ancestry", "MSPReader"),
    "MSPWriter": (".ancestry", "MSPWriter"),
    "AdmixtureMappingVCFWriter": (".ancestry", "AdmixtureMappingVCFWriter"),
    "AdmixtureReader": (".ancestry", "AdmixtureReader"),
    "AdmixtureWriter": (".ancestry", "AdmixtureWriter"),
    "read_lai": (".ancestry", "read_lai"),
    "read_msp": (".ancestry", "read_msp"),
    "read_adm": (".ancestry", "read_adm"),
    "read_admixture": (".ancestry", "read_admixture"),
    "IBDObject": (".ibd", "IBDObject"),
    "read_ibd": (".ibd", "read_ibd"),
    "HapIBDReader": (".ibd", "HapIBDReader"),
    "AncIBDReader": (".ibd", "AncIBDReader"),
    "IBDReader": (".ibd", "IBDReader"),
    "MultiPhenotypeObject": (".phenotype", "MultiPhenotypeObject"),
    "PhenotypeObject": (".phenotype", "PhenotypeObject"),
    "MultiPhenReader": (".phenotype", "MultiPhenReader"),
    "PhenotypeReader": (".phenotype", "PhenotypeReader"),
    "load_dataset": (".datasets", "load_dataset"),
    "viz": (".visualization", ""),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, package=__name__)
    value = module if attr_name == "" else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
