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

ONE_KGP_HIGH_COVERAGE_2022_RELEASE_DIR = (
    "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
)
ONE_KGP_HIGH_COVERAGE_2022_METADATA_URL = (
    "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt"
)
ONE_KGP_HIGH_COVERAGE_2022_CHROMOSOMES: tuple[str, ...] = tuple(
    [str(i) for i in range(1, 23)] + ["X"]
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
    default_chromosomes: Optional[Sequence[str]] = None
    filename_overrides: Mapping[str, str] = field(default_factory=dict)
    index_filename_template: Optional[str] = None
    population_metadata: Optional[PopulationMetadataSpec] = None

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
        if chrom in self.filename_overrides:
            return self.filename_overrides[chrom]
        return self.filename_template.format(chrom=chrom)

    def url(self, chromosome: Union[str, int]) -> str:
        return f"{self.base_url.rstrip('/')}/{self.filename(chromosome)}"

    def index_filename(self, chromosome: Union[str, int]) -> Optional[str]:
        if self.index_filename_template is None:
            return None
        chrom = self.normalize_chromosome(chromosome)
        return self.index_filename_template.format(chrom=chrom, filename=self.filename(chrom))

    def index_url(self, chromosome: Union[str, int]) -> Optional[str]:
        filename = self.index_filename(chromosome)
        if filename is None:
            return None
        return f"{self.base_url.rstrip('/')}/{filename}"

    @property
    def default_chromosome_list(self) -> Sequence[str]:
        return self.default_chromosomes or self.chromosomes


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
            "high_coverage_2022": ChromosomeResource(
                name="high_coverage_2022",
                base_url=ONE_KGP_HIGH_COVERAGE_2022_RELEASE_DIR,
                filename_template="1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz",
                chromosomes=ONE_KGP_HIGH_COVERAGE_2022_CHROMOSOMES,
                default_chromosomes=("1",),
                filename_overrides={
                    "X": "1kGP_high_coverage_Illumina.chrX.filtered.SNV_INDEL_SV_phased_panel.v2.vcf.gz",
                },
                index_filename_template="{filename}.tbi",
                population_metadata=PopulationMetadataSpec(
                    url=ONE_KGP_HIGH_COVERAGE_2022_METADATA_URL,
                    sep=r"\s+",
                    column_renames={
                        "SampleID": "sample",
                        "Sample_ID": "sample",
                        "Sample": "sample",
                        "sample_id": "sample",
                        "FamilyID": "family_id",
                        "Family_ID": "family_id",
                        "FatherID": "father_id",
                        "Father_ID": "father_id",
                        "MotherID": "mother_id",
                        "Mother_ID": "mother_id",
                        "Sex": "sex",
                        "Gender": "sex",
                        "Population": "population",
                        "Superpopulation": "super_population",
                        "SuperPopulation": "super_population",
                    },
                ),
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
