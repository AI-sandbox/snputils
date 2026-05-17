import logging
from typing import Mapping, Optional, Sequence, Union, List
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.auto import SNPReader
from snputils.snp.io.read.pgen import PGENReader
from snputils._utils.data_home import get_data_home
from snputils._utils.download import download_url
from snputils._utils.plink import execute_plink_cmd
from ._registry import (
    ChromosomeResource,
    PopulationMetadataSpec,
    available_datasets_list,
    get_dataset_spec,
)

log = logging.getLogger(__name__)


def _dataset_data_dir(name: str, data_home: Optional[Union[Path, str]] = None) -> Path:
    spec = get_dataset_spec(name)
    return get_data_home(data_home) / spec.name


def _normalize_chromosomes(
    chromosomes: Union[Sequence[Union[str, int]], str, int],
    *,
    resource: Optional[ChromosomeResource] = None,
) -> list[str]:
    if isinstance(chromosomes, (str, int)):
        chromosomes = [chromosomes]
    if resource is None:
        return [str(chrom).lower().replace("chr", "") for chrom in chromosomes]
    return [resource.normalize_chromosome(chrom) for chrom in chromosomes]


def _download_dataset_file(url: str, output_path: Union[Path, str], verbose: bool = True) -> Path:
    """
    Download a dataset file unless it already exists and is non-empty.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    download_url(url, output_path, show_progress=verbose)
    return output_path


def _download_chromosome_files(
    name: str,
    chromosomes: Union[Sequence[Union[str, int]], str, int],
    *,
    resource: Optional[str] = None,
    output_dir: Optional[Union[Path, str]] = None,
    data_home: Optional[Union[Path, str]] = None,
    download: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """
    Download chromosome-indexed dataset files unless already cached.
    """
    chrom_resource = get_dataset_spec(name).genotype_resource(resource)
    normalized = _normalize_chromosomes(chromosomes, resource=chrom_resource)
    if output_dir is None:
        output_dir = _dataset_data_dir(name, data_home)
    output_dir = Path(output_dir)
    paths = []
    for chrom in normalized:
        path = output_dir / chrom_resource.filename(chrom)
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)
            continue
        if not download:
            raise ValueError(
                f"Dataset file {path} is missing. Enable downloading or provide local genotype_sources."
            )
        paths.append(_download_dataset_file(chrom_resource.url(chrom), path, verbose=verbose))
    return paths


def _is_remote_source(source: Union[Path, str]) -> bool:
    return str(source).startswith(("http://", "https://", "ftp://"))


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _prepare_genotype_sources(
    genotype_sources: Sequence[Union[Path, str]],
    *,
    output_dir: Union[Path, str],
    download: bool,
    verbose: bool,
) -> list[Path]:
    output_dir = Path(output_dir)
    paths = []
    for source in genotype_sources:
        if _is_remote_source(source):
            if not download:
                raise ValueError(
                    "Remote genotype sources require download_genotypes=True; remote VCF streaming is not supported."
                )
            paths.append(_download_dataset_file(str(source), output_dir / Path(str(source)).name, verbose=verbose))
        else:
            paths.append(Path(source))
    return paths


def load_dataset(
        name: str,
        chromosomes: Optional[Union[List[str], List[int], str, int]] = None,
        variants_ids: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        resource: Optional[str] = None,
        data_home: Optional[Union[Path, str]] = None,
        output_dir: Optional[Union[Path, str]] = None,
        genotype_sources: Optional[Union[Sequence[Union[Path, str]], Path, str]] = None,
        download_genotypes: bool = True,
        populations: Optional[Sequence[str]] = None,
        samples_per_population: Optional[int] = None,
        max_variants: Optional[int] = None,
        require_biallelic: bool = False,
        require_complete: bool = False,
        require_polymorphic: bool = False,
        snv_only: bool = False,
        metadata_path: Optional[Union[Path, str]] = None,
        metadata_url: Optional[str] = None,
        panel_path: Optional[Union[Path, str]] = None,
        panel_url: Optional[str] = None,
        verbose: bool = True,
        **read_kwargs
) -> SNPObject:
    """
    Load a genome dataset.

    Args:
        name (str): Name of the dataset to load. Call `available_datasets_list()` to get the list of available datasets.
        chromosomes (List[str] | List[int] | str | int): Chromosomes to load.
        variants_ids (List[str]): List of variant IDs to load.
        sample_ids (List[str]): List of sample IDs to load.
        resource (str): Optional dataset genotype resource name. If omitted, the dataset default is used.
        data_home (Path | str): Optional dataset cache directory root.
        output_dir (Path | str): Optional directory for downloaded source files and intermediate files.
        genotype_sources: Optional local paths or URLs to use instead of a registry chromosome resource.
        download_genotypes: Whether remote genotype sources should be downloaded.
        populations: Optional population labels to select from dataset metadata.
        samples_per_population: Optional number of samples to take from each selected population.
        max_variants: Optional maximum number of variants to read by streaming source files directly.
        require_biallelic: When ``max_variants`` is set and source files are streamed, keep only variants with
            exactly one REF allele and one ALT allele.
        require_complete: When ``max_variants`` is set and source files are streamed, keep only variants with no
            missing genotype calls across the selected samples.
        require_polymorphic: When ``max_variants`` is set and source files are streamed, keep only variants that
            are polymorphic among the selected samples after any sample filtering.
        snv_only: When ``max_variants`` is set and source files are streamed, keep only biallelic single-nucleotide
            variants. This implies the same biallelic filter as ``require_biallelic=True`` and additionally
            removes multi-base substitutions, indels, and symbolic alleles.
        metadata_path: Optional local population metadata path.
        metadata_url: Optional population metadata URL.
        panel_path: Backward-compatible alias for metadata_path.
        panel_url: Backward-compatible alias for metadata_url.
        verbose (bool): Whether to show progress.
        **read_kwargs: Keyword arguments to pass to `PGENReader.read()`.

    Returns:
        SNPObject: SNPObject containing the loaded dataset. If population metadata is used, population labels
        are stored in ``sample_fid`` and sex labels are stored in ``sample_sex``.
    """
    dataset = get_dataset_spec(name)
    chrom_resource = dataset.genotype_resource(resource)
    if chromosomes is None and genotype_sources is None:
        chromosomes = list(chrom_resource.chromosomes)
    chromosomes = [] if chromosomes is None else _normalize_chromosomes(chromosomes, resource=chrom_resource)

    data_path = Path(output_dir) if output_dir is not None else _dataset_data_dir(dataset.name, data_home)
    data_path.mkdir(parents=True, exist_ok=True)

    population_metadata = None
    selected_samples = None
    selected_metadata_url = metadata_url or panel_url
    if populations is not None or metadata_path is not None or metadata_url is not None or panel_path is not None or panel_url is not None:
        metadata_spec = dataset.population_metadata
        if selected_metadata_url is None and metadata_spec is not None:
            selected_metadata_url = metadata_spec.url
        selected_metadata_path = metadata_path or panel_path
        if selected_metadata_path is None and selected_metadata_url is not None:
            selected_metadata_path = data_path / Path(selected_metadata_url).name
        population_metadata = _load_population_metadata(
            dataset.name,
            metadata_path=selected_metadata_path,
            metadata_url=selected_metadata_url,
            verbose=verbose,
        )
        if populations is not None:
            selected_samples = _select_population_samples(
                population_metadata,
                populations,
                samples_per_population,
            )
            if sample_ids is None:
                sample_ids = selected_samples["sample"].astype(str).tolist()

    if genotype_sources is not None:
        chromosome_paths = _prepare_genotype_sources(
            _as_list(genotype_sources),
            output_dir=data_path,
            download=download_genotypes,
            verbose=verbose,
        )
    else:
        chromosome_paths = _download_chromosome_files(
            dataset.name,
            chromosomes,
            resource=resource,
            output_dir=data_path,
            download=download_genotypes,
            verbose=verbose,
        )

    sample_metadata = selected_samples if selected_samples is not None else population_metadata

    if max_variants is not None:
        snpobj = _read_snp_subset_sources(
            chromosome_paths,
            sample_ids=sample_ids,
            variant_ids=variants_ids,
            max_variants_total=max_variants,
            require_biallelic=require_biallelic,
            require_complete=require_complete,
            require_polymorphic=require_polymorphic,
            snv_only=snv_only,
        )
        if sample_metadata is not None and snpobj.samples is not None:
            snpobj.sample_fid = _population_labels_for_samples(sample_metadata, snpobj.samples)
            snpobj.sample_sex = _sex_labels_for_samples(sample_metadata, snpobj.samples)
        return snpobj

    if variants_ids is not None:
        variants_ids_txt = tempfile.NamedTemporaryFile(mode='w')
        variants_ids_txt.write("\n".join(variants_ids))
        variants_ids_txt.flush()

    if sample_ids is not None:
        sample_ids_txt = tempfile.NamedTemporaryFile(mode='w')
        sample_ids_txt.write("\n".join(sample_ids))
        sample_ids_txt.flush()

    merge_list_txt = tempfile.NamedTemporaryFile(mode='w')

    chromosome_labels = chromosomes if chromosomes else [str(i + 1) for i in range(len(chromosome_paths))]
    for chr, chr_path in zip(chromosome_labels, chromosome_paths):
        log.info(f"Processing chromosome {chr}...")
        out_file = _plink_output_prefix(chr_path)
        execute_plink_cmd(
            ["--vcf", str(chr_path.resolve())]
            + (["--keep", sample_ids_txt.name] if sample_ids is not None else [])
            + (["--extract", variants_ids_txt.name] if variants_ids is not None else [])
            + [
                "--set-missing-var-ids", "@:#",
                "--make-pgen",
                "--out", out_file,
            ], cwd=data_path)
        merge_list_txt.write(f"{out_file}\n")

    if len(chromosome_paths) > 1:
        # Merge the PGEN files into single PGEN fileset
        log.info("Merging PGEN files...")
        merge_list_txt.flush()
        execute_plink_cmd(["--pmerge-list", merge_list_txt.name, "--make-pgen", "--out", dataset.name],
                          cwd=data_path)
    else:
        # Rename the single PGEN file
        for ext in ["pgen", "psam", "pvar"]:
            Path(data_path / f"{out_file}.{ext}").rename(data_path / f"{dataset.name}.{ext}")

    # Read PGEN fileset with PGENReader into SNPObject
    log.info("Reading PGEN fileset...")
    snpobj = PGENReader(data_path / dataset.name).read(**read_kwargs)
    if sample_metadata is not None and snpobj.samples is not None:
        snpobj.sample_fid = _population_labels_for_samples(sample_metadata, snpobj.samples)
        snpobj.sample_sex = _sex_labels_for_samples(sample_metadata, snpobj.samples)

    if variants_ids is not None:
        variants_ids_txt.close()
    if sample_ids is not None:
        sample_ids_txt.close()
    merge_list_txt.close()

    return snpobj


def _split_int_evenly(total: int, n_parts: int) -> list[int]:
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _fill_missing_variant_ids(snpobj: SNPObject) -> SNPObject:
    if snpobj.variants_id is None:
        return snpobj
    ids = np.asarray(snpobj.variants_id, dtype=object).copy()
    missing = (ids == ".") | (ids == "")
    if np.any(missing):
        ids[missing] = np.asarray(
            [
                f"{chrom}:{pos}:{ref}:{alt}"
                for chrom, pos, ref, alt in zip(
                    snpobj.variants_chrom[missing],
                    snpobj.variants_pos[missing],
                    snpobj.variants_ref[missing],
                    snpobj.variants_alt[missing],
                )
            ],
            dtype=object,
        )
        snpobj = snpobj.copy()
        snpobj.variants_id = ids
    return snpobj


def _read_snp_subset(
    source: Union[Path, str],
    *,
    sample_ids: Optional[Sequence[str]],
    variant_ids: Optional[Sequence[str]],
    max_variants: int,
    require_biallelic: bool,
    require_complete: bool,
    require_polymorphic: bool,
    snv_only: bool,
    allow_fewer: bool = False,
) -> Optional[SNPObject]:
    selected = None if sample_ids is None else list(sample_ids)
    selected_variants = None if variant_ids is None else list(variant_ids)
    chunks: list[SNPObject] = []
    n_variants = 0
    reader = SNPReader(source)
    iter_kwargs = {"sum_strands": False, "chunk_size": 50_000}
    if selected is not None:
        iter_kwargs["sample_ids"] = np.asarray(selected, dtype=object)
    if selected_variants is not None:
        iter_kwargs["variant_ids"] = np.asarray(selected_variants, dtype=object)

    for chunk in reader.iter_read(**iter_kwargs):
        chunk = _fill_missing_variant_ids(chunk)
        if require_biallelic or snv_only:
            chunk = chunk.filter_biallelic_variants(snv_only=snv_only)
        if require_complete:
            chunk = chunk.filter_complete_genotypes()
        if require_polymorphic:
            chunk = chunk.filter_polymorphic_variants()
        if chunk.n_snps == 0:
            continue
        remaining = max_variants - n_variants
        if chunk.n_snps > remaining:
            chunk = chunk.filter_variants(indexes=np.arange(remaining), include=True)
        chunks.append(chunk)
        n_variants += chunk.n_snps
        if n_variants >= max_variants:
            break

    if n_variants == 0 and allow_fewer:
        return None
    if n_variants < max_variants:
        if allow_fewer:
            return SNPObject.concat_variants(chunks)
        raise RuntimeError(f"Only found {n_variants} eligible variants; requested {max_variants}")

    return SNPObject.concat_variants(chunks)


def _read_snp_subset_sources(
    sources: Sequence[Path],
    *,
    sample_ids: Optional[Sequence[str]],
    variant_ids: Optional[Sequence[str]],
    max_variants_total: int,
    require_biallelic: bool,
    require_complete: bool,
    require_polymorphic: bool,
    snv_only: bool,
) -> SNPObject:
    if len(sources) == 0:
        raise ValueError("No genotype sources were provided.")
    selected_variants = None if variant_ids is None else list(variant_ids)
    if selected_variants is not None:
        parts = []
        n_variants = 0
        for source in sources:
            remaining = max_variants_total - n_variants
            if remaining <= 0:
                break
            part = _read_snp_subset(
                source,
                sample_ids=sample_ids,
                variant_ids=selected_variants,
                max_variants=remaining,
                require_biallelic=require_biallelic,
                require_complete=require_complete,
                require_polymorphic=require_polymorphic,
                snv_only=snv_only,
                allow_fewer=True,
            )
            if part is None:
                continue
            parts.append(part)
            n_variants += part.n_snps
        if not parts:
            raise RuntimeError("No eligible variants matched variants_ids.")
        return SNPObject.concat_variants(parts)

    if max_variants_total < len(sources):
        raise ValueError(
            f"max_variants ({max_variants_total}) must be >= number of genotype sources ({len(sources)}) "
            "so every source contributes at least one variant."
        )
    quotas = _split_int_evenly(max_variants_total, len(sources))
    parts = [
        _read_snp_subset(
            source,
            sample_ids=sample_ids,
            variant_ids=None,
            max_variants=quota,
            require_biallelic=require_biallelic,
            require_complete=require_complete,
            require_polymorphic=require_polymorphic,
            snv_only=snv_only,
        )
        for source, quota in zip(sources, quotas)
    ]
    return SNPObject.concat_variants(parts)


def _plink_output_prefix(path: Path) -> str:
    name = path.name
    for suffix in (".vcf.gz", ".vcf.bgz", ".vcf"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _read_population_metadata_file(
    metadata_path: Union[Path, str],
    *,
    spec: Optional[PopulationMetadataSpec] = None,
    sep: Optional[str] = None,
    column_renames: Optional[Mapping[str, str]] = None,
    required_columns: Sequence[str] = ("sample", "population"),
    sex_column: str = "sex",
) -> pd.DataFrame:
    if spec is not None:
        sep = spec.sep if sep is None else sep
        column_renames = dict(spec.column_renames if column_renames is None else column_renames)
        required_columns = spec.required_columns
        sex_column = spec.sex_column
    else:
        sep = "\t" if sep is None else sep
        column_renames = {} if column_renames is None else column_renames

    metadata = pd.read_csv(metadata_path, sep=sep, dtype=str)
    metadata = metadata.rename(columns={k: v for k, v in column_renames.items() if k in metadata.columns})

    missing = set(required_columns).difference(metadata.columns)
    if missing:
        raise ValueError(f"Population metadata is missing columns: {sorted(missing)}")

    if sex_column not in metadata.columns:
        metadata[sex_column] = "U"
    sex = metadata[sex_column].fillna("U").astype(str).str.strip().str.lower()
    metadata[sex_column] = sex.map({
        "1": "M",
        "m": "M",
        "male": "M",
        "2": "F",
        "f": "F",
        "female": "F",
        "0": "U",
        "u": "U",
        "unknown": "U",
        "": "U",
        ".": "U",
        "nan": "U",
        "none": "U",
    }).fillna("U")
    return metadata


def _load_population_metadata(
    name: str = "1kgp",
    *,
    data_home: Optional[Union[Path, str]] = None,
    metadata_path: Optional[Union[Path, str]] = None,
    metadata_url: Optional[str] = None,
    panel_path: Optional[Union[Path, str]] = None,
    panel_url: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load sample-level population metadata for a supported dataset.

    ``panel_path`` and ``panel_url`` are accepted as aliases for compatibility
    with older 1000 Genomes-specific call sites.
    """
    spec = get_dataset_spec(name)
    metadata_spec = spec.population_metadata
    path = metadata_path or panel_path
    if metadata_spec is None and metadata_url is None and panel_url is None and path is None:
        raise NotImplementedError(f"Population metadata for dataset {name!r} is not implemented.")

    url = metadata_url or panel_url or (metadata_spec.url if metadata_spec is not None else None)
    if path is None:
        if url is None:
            raise ValueError("A population metadata path or URL is required.")
        filename = metadata_spec.default_filename if metadata_spec is not None else Path(url).name
        path = _dataset_data_dir(spec.name, data_home) / filename
    path = Path(path)

    if url is not None:
        _download_dataset_file(url, path, verbose=verbose)
    elif not path.exists():
        raise FileNotFoundError(f"Population metadata file not found: {path}")
    return _read_population_metadata_file(path, spec=metadata_spec)


def _select_population_samples(
    metadata: pd.DataFrame,
    populations: Sequence[str],
    samples_per_population: Optional[int] = None,
    *,
    sample_col: str = "sample",
    population_col: str = "population",
) -> pd.DataFrame:
    """
    Select samples from a population metadata table in population order.
    """
    required = {sample_col, population_col}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Population metadata is missing columns: {sorted(missing)}")

    selected = []
    for pop in populations:
        rows = metadata.loc[metadata[population_col] == pop]
        if samples_per_population is not None:
            rows = rows.head(samples_per_population)
            if len(rows) < samples_per_population:
                raise ValueError(
                    f"Population {pop!r} has only {len(rows)} samples; "
                    f"requested {samples_per_population}."
                )
        if len(rows) == 0:
            raise ValueError(f"Population {pop!r} has no samples in the metadata.")
        selected.append(rows)

    return pd.concat(selected, ignore_index=True)


def _metadata_for_samples(
    metadata: pd.DataFrame,
    samples: Sequence[str],
    *,
    sample_col: str = "sample",
) -> pd.DataFrame:
    if sample_col not in metadata.columns:
        raise ValueError(f"Population metadata is missing sample column {sample_col!r}.")

    indexed = metadata.set_index(sample_col, drop=False)
    missing = [sample for sample in samples if sample not in indexed.index]
    if missing:
        preview = ", ".join(map(str, missing[:5]))
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"Metadata is missing samples: {preview}{suffix}")

    return indexed.loc[list(samples)].reset_index(drop=True)


def _population_labels_for_samples(
    metadata: pd.DataFrame,
    samples: Sequence[str],
    *,
    sample_col: str = "sample",
    population_col: str = "population",
) -> list[str]:
    """
    Return population labels aligned to ``samples``.
    """
    if population_col not in metadata.columns:
        raise ValueError(f"Population metadata is missing population column {population_col!r}.")
    aligned = _metadata_for_samples(metadata, samples, sample_col=sample_col)
    return aligned[population_col].astype(str).tolist()


def _sex_labels_for_samples(
    metadata: pd.DataFrame,
    samples: Sequence[str],
    *,
    sample_col: str = "sample",
    sex_col: str = "sex",
    default: str = "U",
) -> list[str]:
    """
    Return PLINK-compatible sex labels aligned to ``samples``.
    """
    if sex_col not in metadata.columns:
        return [default] * len(samples)
    aligned = _metadata_for_samples(metadata, samples, sample_col=sample_col)
    return aligned[sex_col].fillna(default).astype(str).tolist()


# Keep the generated module-style path usable even when package-level
# ``snputils.datasets.load_dataset`` resolves to this convenience function.
load_dataset.available_datasets_list = available_datasets_list  # type: ignore[attr-defined]
