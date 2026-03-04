from importlib import import_module
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "SNPObject": (".genobj", "SNPObject"),
    "GRGObject": (".genobj", "GRGObject"),
    "SNPReader": (".io", "SNPReader"),
    "BEDReader": (".io", "BEDReader"),
    "GRGReader": (".io", "GRGReader"),
    "GRGWriter": (".io", "GRGWriter"),
    "PGENReader": (".io", "PGENReader"),
    "VCFReader": (".io", "VCFReader"),
    "BEDWriter": (".io", "BEDWriter"),
    "PGENWriter": (".io", "PGENWriter"),
    "VCFWriter": (".io", "VCFWriter"),
    "read_snp": (".io", "read_snp"),
    "read_bed": (".io", "read_bed"),
    "read_pgen": (".io", "read_pgen"),
    "read_vcf": (".io", "read_vcf"),
    "read_grg": (".io", "read_grg"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
