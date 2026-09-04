import logging
import operator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union
import csv

import numpy as np
import polars as pl
import pgenlib as pg

from snputils._utils.genotypes import (
    ExplicitGenotypeMode,
    GenotypeMode,
    normalize_genotype_mode,
    sum_diploid_alleles,
)
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from snputils.snp.io.read._pgenlib import (
    estimate_phased_alleles_peak_bytes,
    read_phased_alleles,
)

log = logging.getLogger(__name__)


def _normalized_non_diploid_chromosome(chrom: Any) -> Optional[str]:
    value = str(chrom).strip().lower()
    if value.startswith("chrom"):
        value = value[5:]
    elif value.startswith("chr"):
        value = value[3:]
    elif value.startswith("chm"):
        value = value[3:]
    if value in {"x", "y", "m", "mt", "mitochondria", "mitochondrial", "mitochondrion"}:
        return value
    return None


def _non_diploid_chromosome_mask_or_none(chromosomes: np.ndarray) -> Optional[np.ndarray]:
    mask = np.fromiter(
        (_normalized_non_diploid_chromosome(chrom) is not None for chrom in chromosomes),
        dtype=bool,
        count=len(chromosomes),
    )
    return mask if np.any(mask) else None


def _normalize_chromosome_ploidy(value: Optional[str]) -> str:
    if value is None:
        return "auto"

    value = str(value).strip().lower()
    if value not in {"auto", "autosomal", "mixed"}:
        raise ValueError(
            "chromosome_ploidy must be one of None, 'auto', 'autosomal', or 'mixed'."
        )
    return value


def _resolve_compressed_path(base_path: str, ext: str) -> str:
    import os
    candidate = base_path + ext
    if os.path.exists(candidate):
        return candidate
    for comp_ext in (".zst", ".gz"):
        candidate_compressed = base_path + ext + comp_ext
        if os.path.exists(candidate_compressed):
            return candidate_compressed
    return candidate


def _open_textfile(filename: str, mode: str = "rt"):
    if filename.endswith(".zst"):
        import zstandard as zstd
        return zstd.open(filename, mode, encoding="utf-8") if "t" in mode else zstd.open(filename, mode)
    elif filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode, encoding="utf-8") if "t" in mode else gzip.open(filename, mode)
    return open(filename, mode)


def _strip_bed_fileset_suffix(filename: Union[str, Path]) -> str:
    filename_str = str(filename)
    lower_filename = filename_str.lower()
    for suffix in (
        ".bim.zst",
        ".bim.gz",
        ".fam.zst",
        ".fam.gz",
        ".bed",
        ".bim",
        ".fam",
    ):
        if lower_filename.endswith(suffix):
            return filename_str[:-len(suffix)]
    return filename_str


def _read_dosage_parallel(
    filename_noext: str,
    *,
    raw_sample_ct: int,
    variant_ct: Optional[int],
    sample_subset: Optional[np.ndarray],
    variant_idxs: np.ndarray,
    num_variants: int,
    num_samples: int,
    threads: int,
) -> np.ndarray:
    """Decode BED hardcalls with independent pgenlib readers."""
    genotypes = np.empty((num_variants, num_samples), dtype=np.int8)
    if num_variants == 0 or num_samples == 0:
        return genotypes

    worker_count = min(threads, num_variants)
    output_boundaries = np.linspace(
        0,
        num_variants,
        worker_count + 1,
        dtype=np.int64,
    )

    def read_partition(worker_idx: int) -> None:
        output_start = int(output_boundaries[worker_idx])
        output_stop = int(output_boundaries[worker_idx + 1])
        partition_variant_idxs = variant_idxs[output_start:output_stop]
        partition_output = genotypes[output_start:output_stop]
        reader = pg.PgenReader(
            str.encode(filename_noext + ".bed"),
            raw_sample_ct=raw_sample_ct,
            variant_ct=variant_ct,
            sample_subset=sample_subset,
        )
        try:
            if (
                int(partition_variant_idxs[-1])
                - int(partition_variant_idxs[0])
                + 1
                == partition_variant_idxs.size
                and np.all(np.diff(partition_variant_idxs) == 1)
            ):
                reader.read_range(
                    int(partition_variant_idxs[0]),
                    int(partition_variant_idxs[-1]) + 1,
                    partition_output,
                )
            else:
                reader.read_list(partition_variant_idxs, partition_output)
        finally:
            reader.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(read_partition, range(worker_count)))
    return genotypes


@SNPBaseReader.register
class BEDReader(SNPBaseReader):
    def read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        sample_ids: Optional[np.ndarray] = None,
        sample_idxs: Optional[np.ndarray] = None,
        variant_ids: Optional[np.ndarray] = None,
        variant_idxs: Optional[np.ndarray] = None,
        genotype_mode: GenotypeMode = "dosage",
        chromosome_ploidy: Optional[str] = None,
        separator: Optional[str] = None,
        threads: int = 1,
    ) -> SNPObject:
        """
        Read a bed fileset (bed, bim, fam) into a SNPObject.

        Args:
            fields (str, None, or list of str, optional): Fields to extract data for that should be included in the returned SNPObject.
                Available fields are 'GT', 'IID', 'REF', 'ALT', '#CHROM', 'CM', 'ID', 'POS'.
                To extract all fields, set fields to None. Defaults to None.
            exclude_fields (str, None, or list of str, optional): Fields to exclude from the returned SNPObject.
                Available fields are 'GT', 'IID', 'REF', 'ALT', '#CHROM', 'CM', 'ID', 'POS'.
                To exclude no fields, set exclude_fields to None. Defaults to None.
            sample_ids: List of sample IDs to read. If None and sample_idxs is None, all samples are read.
            sample_idxs: List of sample indices to read. If None and sample_ids is None, all samples are read.
            variant_ids: List of variant IDs to read. If None and variant_idxs is None, all variants are read.
            variant_idxs: List of variant indices to read. If None and variant_ids is None, all variants are read.
            genotype_mode: ``"dosage"`` (default) returns genotype dosages in
                a single ``int8`` array with values ``{0, 1, 2}``. ``"auto"``
                is equivalent to ``"dosage"``. PLINK BED/BIM/FAM does not
                store phase, so ``"phased"`` is not supported.
            chromosome_ploidy:
                Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
                all selected variants should be treated as ordinary diploid/autosomal; this
                skips non-diploid chromosome checks and can be faster. The default None/"auto"
                preserves existing behavior.
            separator: Separator used in the pvar file. If None, the separator is automatically detected.
                If the automatic detection fails, please specify the separator manually.
            threads: Number of independent native pgenlib decoder threads.

        Returns:
            **SNPObject**:
                A SNPObject instance.
        """
        assert (
            sample_idxs is None or sample_ids is None
        ), "Only one of sample_idxs and sample_ids can be specified"
        assert (
            variant_idxs is None or variant_ids is None
        ), "Only one of variant_idxs and variant_ids can be specified"
        genotype_mode = normalize_genotype_mode(genotype_mode)
        return_dosage = genotype_mode in {"auto", "dosage"}
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        detect_non_diploid = bool(return_dosage) and chromosome_ploidy_mode != "autosomal"
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("BEDReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("BEDReader threads must be at least 1.")

        if isinstance(fields, str):
            fields = [fields]
        if isinstance(exclude_fields, str):
            exclude_fields = [exclude_fields]

        fields = fields or ["GT", "IID", "REF", "ALT", "#CHROM", "CM", "ID", "POS"]
        exclude_fields = exclude_fields or []
        fields = [field for field in fields if field not in exclude_fields]
        if "GT" in fields and not return_dosage:
            raise ValueError(
                "PLINK BED/BIM/FAM does not store phase, so genotype_mode='phased' is not supported. "
                "Use genotype_mode='dosage' to load 0/1/2 genotype dosages."
            )
        only_read_bed = fields == ["GT"] and variant_idxs is None and sample_idxs is None

        filename_noext = _strip_bed_fileset_suffix(self.filename)

        fam_filename = _resolve_compressed_path(filename_noext, ".fam")
        bim_filename = _resolve_compressed_path(filename_noext, ".bim")

        if only_read_bed:
            with _open_textfile(fam_filename, 'rt') as f:
                file_num_samples = sum(1 for _ in f)  # Get sample count from fam file
            file_num_variants = None  # Not needed
        else:
            log.info(f"Reading {bim_filename}")

            if separator is None:
                with _open_textfile(bim_filename, "rt") as file:
                    separator = csv.Sniffer().sniff(file.readline()).delimiter

            bim = pl.read_csv(
                bim_filename,
                separator=separator,
                has_header=False,
                new_columns=["#CHROM", "ID", "CM", "POS", "ALT", "REF"],
                schema_overrides={
                    "#CHROM": pl.String,
                    "ID": pl.String,
                    "CM": pl.Float64,
                    "POS": pl.Int64,
                    "ALT": pl.String,
                    "REF": pl.String
                },
                null_values=["NA"]
            ).with_row_index()
            file_num_variants = bim.height

            if variant_ids is not None:
                variant_id_values = [str(v) for v in np.atleast_1d(variant_ids)]
                variant_id_or_pos = (
                    pl.col("ID").is_in(variant_id_values)
                    | pl.concat_str(
                        [pl.col("#CHROM"), pl.lit(":"), pl.col("POS").cast(pl.String)]
                    ).is_in(variant_id_values)
                )
                variant_idxs = (
                    bim.filter(variant_id_or_pos)
                    .select("index")
                    .to_series()
                    .to_numpy()
                )

            if variant_idxs is None:
                num_variants = file_num_variants
                variant_idxs = np.arange(num_variants, dtype=np.uint32)
            else:
                requested_variant_idxs = np.asarray(variant_idxs, dtype=np.uint32).ravel()
                if np.any(requested_variant_idxs >= file_num_variants):
                    raise ValueError("One or more variant indexes are out of bounds.")
                selector = pl.DataFrame({"index": requested_variant_idxs}).with_row_index("_selector_order")
                bim = (
                    selector
                    .join(bim, on="index", how="left")
                    .sort("_selector_order")
                    .drop("_selector_order")
                )
                variant_idxs = requested_variant_idxs
                num_variants = np.size(variant_idxs)

            log.info(f"Reading {fam_filename}")

            fam = pl.read_csv(
                fam_filename,
                separator=separator,
                has_header=False,
                new_columns=["Family ID", "IID", "Father ID",
                             "Mother ID", "Sex code", "Phenotype value"],
                schema_overrides={
                    "Family ID": pl.String,
                    "IID": pl.String,
                    "Father ID": pl.String,
                    "Mother ID": pl.String,
                    "Sex code": pl.String,
                },
                null_values=["NA"]
            ).with_row_index()
            file_num_samples = fam.height

            if sample_ids is not None:
                sample_idxs = fam.filter(pl.col("IID").is_in(sample_ids)).select("index").to_series().to_numpy()

            if sample_idxs is None:
                num_samples = file_num_samples
            else:
                num_samples = np.size(sample_idxs)
                sample_idxs = np.array(sample_idxs, dtype=np.uint32)
                fam = fam.filter(pl.col("index").is_in(sample_idxs))

        if "GT" in fields:
            log.info(f"Reading {filename_noext}.bed")
            pgen_reader = pg.PgenReader(
                str.encode(filename_noext + ".bed"),
                raw_sample_ct=file_num_samples,
                variant_ct=file_num_variants,
                sample_subset=sample_idxs,
            )

            if only_read_bed:
                num_samples = pgen_reader.get_raw_sample_ct()
                num_variants = pgen_reader.get_variant_ct()
                variant_idxs = np.arange(num_variants, dtype=np.uint32)

            non_diploid_mask = None
            if detect_non_diploid and "bim" in locals() and "#CHROM" in bim.columns:
                non_diploid_mask = _non_diploid_chromosome_mask_or_none(
                    bim.get_column("#CHROM").to_numpy()
                )

            # required arrays: variant_idxs + sample_idxs + genotypes
            if not return_dosage:
                required_ram = (
                    (num_samples + num_variants) * 4
                    + estimate_phased_alleles_peak_bytes(num_variants, num_samples)
                )
            else:
                required_ram = (num_samples + num_variants) * 4 + num_variants * num_samples
                num_non_diploid = int(np.sum(non_diploid_mask)) if non_diploid_mask is not None else 0
                if num_non_diploid:
                    required_ram += estimate_phased_alleles_peak_bytes(num_non_diploid, num_samples)
            log.info(f">{required_ram / 1024**3:.2f} GiB of RAM are required to process {num_samples} samples with {num_variants} variants each")

            if not return_dosage:
                genotypes = read_phased_alleles(
                    pgen_reader,
                    variant_idxs,
                    num_variants,
                    num_samples,
                )
            else:
                if threads == 1:
                    genotypes = np.empty((num_variants, num_samples), dtype=np.int8)
                    pgen_reader.read_list(variant_idxs, genotypes)
                else:
                    genotypes = _read_dosage_parallel(
                        filename_noext,
                        raw_sample_ct=file_num_samples,
                        variant_ct=file_num_variants,
                        sample_subset=sample_idxs,
                        variant_idxs=variant_idxs,
                        num_variants=num_variants,
                        num_samples=num_samples,
                        threads=threads,
                    )
                if detect_non_diploid and only_read_bed:
                    log.debug(
                        "Skipping non-diploid BED dosage correction because BIM chromosome metadata was not loaded."
                    )
                elif detect_non_diploid and non_diploid_mask is not None:
                    non_diploid_output_rows = np.flatnonzero(non_diploid_mask)
                    non_diploid_variant_idxs = np.asarray(
                        variant_idxs[non_diploid_output_rows],
                        dtype=np.uint32,
                    )

                    separate = read_phased_alleles(
                        pgen_reader,
                        non_diploid_variant_idxs,
                        non_diploid_variant_idxs.size,
                        num_samples,
                    )

                    genotypes[non_diploid_output_rows] = sum_diploid_alleles(
                        separate[:, :, 0],
                        separate[:, :, 1],
                        missing_as_haploid=True,
                    )
            pgen_reader.close()
        else:
            genotypes = None

        log.info("Constructing SNPObject")

        fid_col = None
        if "IID" in fields and "Family ID" in fam.columns:
            fid_col = fam.get_column("Family ID").to_numpy()
        sex_col = None
        if "IID" in fields and "Sex code" in fam.columns:
            sex_col = fam.get_column("Sex code").to_numpy()

        snpobj = SNPObject(
            genotypes=genotypes if "GT" in fields else None,
            samples=fam.get_column("IID").to_numpy() if "IID" in fields and "IID" in fam.columns else None,
            sample_fid=fid_col,
            sample_sex=sex_col,
            **{f'variants_{k.lower()}': bim.get_column(v).to_numpy() if v in fields and v in bim.columns else None
               for k, v in {'ref': 'REF', 'alt': 'ALT', 'chrom': '#CHROM', 'cm': 'CM', 'id': 'ID', 'pos': 'POS'}.items()}
        )

        log.info("Finished constructing SNPObject")
        return snpobj

    def _resolve_variant_idxs_for_iter(
        self,
        *,
        variant_ids: Optional[np.ndarray],
        variant_idxs: Optional[np.ndarray],
        separator: Optional[str],
    ) -> np.ndarray:
        """
        Resolve variant selectors to canonical file-order row indices.
        """
        filename_noext = _strip_bed_fileset_suffix(self.filename)

        bim_filename = _resolve_compressed_path(filename_noext, ".bim")

        local_separator = separator
        if local_separator is None:
            with _open_textfile(bim_filename, "rt") as file:
                local_separator = csv.Sniffer().sniff(file.readline()).delimiter

        bim = pl.read_csv(
            bim_filename,
            separator=local_separator,
            has_header=False,
            new_columns=["#CHROM", "ID", "CM", "POS", "ALT", "REF"],
            schema_overrides={
                "#CHROM": pl.String,
                "ID": pl.String,
                "CM": pl.Float64,
                "POS": pl.Int64,
                "ALT": pl.String,
                "REF": pl.String,
            },
            null_values=["NA"],
        ).with_row_index()

        if variant_ids is not None:
            variant_id_values = [str(v) for v in np.atleast_1d(variant_ids)]
            variant_id_or_pos = (
                pl.col("ID").is_in(variant_id_values)
                | pl.concat_str([pl.col("#CHROM"), pl.lit(":"), pl.col("POS").cast(pl.String)]).is_in(
                    variant_id_values
                )
            )
            resolved = (
                bim.filter(variant_id_or_pos)
                .select("index")
                .to_series()
                .to_numpy()
            )
            return np.asarray(resolved, dtype=np.uint32)

        if variant_idxs is not None:
            requested = np.asarray(variant_idxs, dtype=np.uint32).ravel()
            if np.any(requested >= bim.height):
                raise ValueError("One or more variant indexes are out of bounds.")
            return requested

        return np.arange(bim.height, dtype=np.uint32)

    def iter_read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        sample_ids: Optional[np.ndarray] = None,
        sample_idxs: Optional[np.ndarray] = None,
        variant_ids: Optional[np.ndarray] = None,
        variant_idxs: Optional[np.ndarray] = None,
        genotype_mode: ExplicitGenotypeMode = "dosage",
        chromosome_ploidy: Optional[str] = None,
        separator: Optional[str] = None,
        threads: int = 1,
        chunk_size: int = 10_000,
    ) -> Iterator[SNPObject]:
        """
        Stream the BED fileset in variant chunks.

        This yields a sequence of SNPObject chunks along the SNP axis.

        chromosome_ploidy:
            Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
            all selected variants should be treated as ordinary diploid/autosomal; this
            skips non-diploid chromosome checks and can be faster. The default None/"auto"
            preserves existing behavior.
        threads: Number of pgenlib decoder threads used for each chunk.
        """
        genotype_mode = normalize_genotype_mode(genotype_mode, allow_auto=False)
        _normalize_chromosome_ploidy(chromosome_ploidy)
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("BEDReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("BEDReader threads must be at least 1.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")
        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        selectors = self._resolve_variant_idxs_for_iter(
            variant_ids=variant_ids,
            variant_idxs=variant_idxs,
            separator=separator,
        )

        n_selectors = int(selectors.size)
        for start in range(0, n_selectors, int(chunk_size)):
            stop = min(start + int(chunk_size), n_selectors)
            selector_chunk = np.asarray(selectors[start:stop], dtype=np.uint32)
            yield self.read(
                fields=fields,
                exclude_fields=exclude_fields,
                sample_ids=sample_ids,
                sample_idxs=sample_idxs,
                variant_idxs=selector_chunk,
                genotype_mode=genotype_mode,
                chromosome_ploidy=chromosome_ploidy,
                separator=separator,
                threads=threads,
            )
