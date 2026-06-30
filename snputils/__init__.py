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
    "BCFReader": (".snp", "BCFReader"),
    "BGENReader": (".snp", "BGENReader"),
    "GRGReader": (".snp", "GRGReader"),
    "GRGWriter": (".snp", "GRGWriter"),
    "PGENReader": (".snp", "PGENReader"),
    "VCFReader": (".snp", "VCFReader"),
    "BEDWriter": (".snp", "BEDWriter"),
    "BCFWriter": (".snp", "BCFWriter"),
    "BGENWriter": (".snp", "BGENWriter"),
    "PGENWriter": (".snp", "PGENWriter"),
    "VCFWriter": (".snp", "VCFWriter"),
    "vcf_to_grg": (".snp", "vcf_to_grg"),
    "vcf_to_igd": (".snp", "vcf_to_igd"),
    "read_snp": (".snp", "read_snp"),
    "read_bcf": (".snp", "read_bcf"),
    "read_bed": (".snp", "read_bed"),
    "read_bgen": (".snp", "read_bgen"),
    "read_pgen": (".snp", "read_pgen"),
    "read_vcf": (".snp", "read_vcf"),
    "read_grg": (".snp", "read_grg"),
    "LocalAncestryObject": (".ancestry", "LocalAncestryObject"),
    "GlobalAncestryObject": (".ancestry", "GlobalAncestryObject"),
    "MSPReader": (".ancestry", "MSPReader"),
    "MSPWriter": (".ancestry", "MSPWriter"),
    "FLAREReader": (".ancestry", "FLAREReader"),
    "FLAREWriter": (".ancestry", "FLAREWriter"),
    "LANCReader": (".ancestry", "LANCReader"),
    "LANCWriter": (".ancestry", "LANCWriter"),
    "AdmixtureMappingVCFWriter": (".ancestry", "AdmixtureMappingVCFWriter"),
    "AdmixtureReader": (".ancestry", "AdmixtureReader"),
    "AdmixtureWriter": (".ancestry", "AdmixtureWriter"),
    "read_lai": (".ancestry", "read_lai"),
    "read_msp": (".ancestry", "read_msp"),
    "read_flare": (".ancestry", "read_flare"),
    "read_lanc": (".ancestry", "read_lanc"),
    "read_adm": (".ancestry", "read_adm"),
    "read_admixture": (".ancestry", "read_admixture"),
    "IBDObject": (".ibd", "IBDObject"),
    "read_ibd": (".ibd", "read_ibd"),
    "HapIBDReader": (".ibd", "HapIBDReader"),
    "AncIBDReader": (".ibd", "AncIBDReader"),
    "IBDReader": (".ibd", "IBDReader"),
    "PCA": (".processing", "PCA"),
    "mdPCA": (".processing", "mdPCA"),
    "maasMDS": (".processing", "maasMDS"),
    "allele_freq_stream": (".stats", "allele_freq_stream"),
    "MultiPhenotypeObject": (".phenotype", "MultiPhenotypeObject"),
    "PhenotypeObject": (".phenotype", "PhenotypeObject"),
    "CovariateObject": (".phenotype", "CovariateObject"),
    "MultiPhenReader": (".phenotype", "MultiPhenReader"),
    "PhenotypeReader": (".phenotype", "PhenotypeReader"),
    "read_pheno": (".phenotype", "read_pheno"),
    "available_datasets_list": (".datasets", "available_datasets_list"),
    "load_dataset": (".datasets", "load_dataset"),
    "build_synthetic_admixture_dataset": (".datasets", "build_synthetic_admixture_dataset"),
    "build_synthetic_chromosome_painting_dataset": (".datasets", "build_synthetic_chromosome_painting_dataset"),
    "build_synthetic_grg": (".datasets", "build_synthetic_grg"),
    "build_synthetic_maasmds_dataset": (".datasets", "build_synthetic_maasmds_dataset"),
    "build_synthetic_mdpca_dataset": (".datasets", "build_synthetic_mdpca_dataset"),
    "build_synthetic_phenotype_dataset": (".datasets", "build_synthetic_phenotype_dataset"),
    "build_synthetic_snp_dataset": (".datasets", "build_synthetic_snp_dataset"),
    "run_admixture_mapping": (".tools", "run_admixture_mapping"),
    "run_gwas": (".tools", "run_gwas"),
    "read_labels": ("._utils.labels", "read_labels"),
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
