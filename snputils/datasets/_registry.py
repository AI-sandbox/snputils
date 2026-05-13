from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union


ONE_KGP_PHASE3_PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)
ONE_KGP_PHASE3_RELEASE_DIR = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
ONE_KGP_PHASE3_AUTOSOME_CHROMOSOMES: tuple[int, ...] = tuple(range(1, 23))

ONE_KGP_GRCH38_BIALLELIC_RELEASE_DIR = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000_genomes_project/release/20181203_biallelic_SNV"
)
ONE_KGP_GRCH38_BIALLELIC_CHROMOSOMES: tuple[str, ...] = tuple(
    [str(i) for i in range(1, 23)] + ["X", "Y"]
)


@dataclass(frozen=True)
class ChromosomeResource:
    """
    Downloadable chromosome-indexed files for a dataset.
    """

    name: str
    base_url: str
    filename_template: str
    chromosomes: Sequence[str]

    def normalize_chromosome(self, chromosome: Union[str, int]) -> str:
        chrom = str(chromosome).lower().replace("chr", "")
        allowed = {str(value).lower(): str(value) for value in self.chromosomes}
        if chrom not in allowed:
            raise ValueError(
                f"Chromosome {chromosome!r} is not available for resource {self.name!r}. "
                f"Available chromosomes: {', '.join(map(str, self.chromosomes))}."
            )
        return allowed[chrom]

    def filename(self, chromosome: Union[str, int]) -> str:
        chrom = self.normalize_chromosome(chromosome)
        return self.filename_template.format(chrom=chrom)

    def url(self, chromosome: Union[str, int]) -> str:
        return f"{self.base_url.rstrip('/')}/{self.filename(chromosome)}"


@dataclass(frozen=True)
class PopulationMetadataSpec:
    """
    Tabular sample metadata for a dataset.
    """

    url: str
    filename: Optional[str] = None
    sep: str = "\t"
    column_renames: Mapping[str, str] = field(default_factory=dict)
    required_columns: Sequence[str] = ("sample", "population")
    sex_column: str = "sex"

    @property
    def default_filename(self) -> str:
        return self.filename or Path(self.url).name


@dataclass(frozen=True)
class DatasetSpec:
    """
    Registry entry for a dataset.
    """

    name: str
    aliases: Sequence[str] = ()
    genotype_resources: Mapping[str, ChromosomeResource] = field(default_factory=dict)
    default_genotype_resource: Optional[str] = None
    population_metadata: Optional[PopulationMetadataSpec] = None

    def genotype_resource(self, resource: Optional[str] = None) -> ChromosomeResource:
        resource_name = resource or self.default_genotype_resource
        if resource_name is None:
            raise ValueError(f"Dataset {self.name!r} has no default genotype resource.")
        try:
            return self.genotype_resources[resource_name]
        except KeyError as exc:
            available = ", ".join(self.genotype_resources)
            raise ValueError(
                f"Dataset {self.name!r} has no genotype resource {resource_name!r}. "
                f"Available resources: {available}."
            ) from exc


DATASETS: dict[str, DatasetSpec] = {
    "1kgp": DatasetSpec(
        name="1kgp",
        aliases=("1000g", "1000_genomes"),
        default_genotype_resource="grch38_biallelic",
        genotype_resources={
            "grch38_biallelic": ChromosomeResource(
                name="grch38_biallelic",
                base_url=ONE_KGP_GRCH38_BIALLELIC_RELEASE_DIR,
                filename_template="ALL.chr{chrom}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
                chromosomes=ONE_KGP_GRCH38_BIALLELIC_CHROMOSOMES,
            ),
            "phase3": ChromosomeResource(
                name="phase3",
                base_url=ONE_KGP_PHASE3_RELEASE_DIR,
                filename_template="ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
                chromosomes=tuple(str(i) for i in ONE_KGP_PHASE3_AUTOSOME_CHROMOSOMES),
            ),
        },
        population_metadata=PopulationMetadataSpec(
            url=ONE_KGP_PHASE3_PANEL_URL,
            column_renames={
                "pop": "population",
                "super_pop": "super_population",
                "gender": "sex",
            },
        ),
    ),
}


def register_dataset(spec: DatasetSpec) -> None:
    """
    Add or replace a dataset registry entry.
    """
    DATASETS[spec.name] = spec


def normalize_dataset_name(name: str) -> str:
    """
    Normalize a dataset name or alias to its registry key.
    """
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    for key, spec in DATASETS.items():
        if normalized == key or normalized in spec.aliases:
            return key
    raise NotImplementedError(f"Dataset {name!r} is not implemented.")


def get_dataset_spec(name: str) -> DatasetSpec:
    """
    Return a dataset registry entry by name or alias.
    """
    return DATASETS[normalize_dataset_name(name)]


def available_datasets_list() -> list[str]:
    """
    Get the list of available dataset registry names.
    """
    return list(DATASETS)
