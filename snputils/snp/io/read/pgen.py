import logging
import operator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List, Optional
import os

import numpy as np
import polars as pl
import pgenlib as pg

from snputils._utils.genotypes import (
    ExplicitGenotypeMode,
    GenotypeMode,
    _MULTIALLELIC_DOSAGE_ERROR,
    normalize_genotype_mode,
    sum_diploid_alleles,
)
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from snputils.snp.io.read._pgenlib import (
    estimate_phased_alleles_peak_bytes,
    read_phased_alleles,
    read_phased_alleles_into,
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


def _open_textfile(filename):
    if filename.endswith(".zst"):
        import zstandard as zstd
        return zstd.open(filename, "rt", encoding="utf-8")
    elif filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, "rt", encoding="utf-8")
    return open(filename, "rt", encoding="utf-8")


def _detect_pvar_separator(line: str) -> str:
    if "\t" in line:
        return "\t"
    return " "


def _find_pvar_path(filename_noext: str) -> Optional[str]:
    return next(
        (
            filename_noext + extension
            for extension in (".pvar", ".pvar.zst", ".pvar.gz")
            if os.path.exists(filename_noext + extension)
        ),
        None,
    )


def _open_pgen_reader(
    filename_noext: str,
    *,
    raw_sample_ct: Optional[int],
    variant_ct: Optional[int],
    sample_subset: Optional[np.ndarray],
    genotype_mode: GenotypeMode,
) -> tuple[Any, bool]:
    """Open the biallelic fast path first; consult PVAR only after PGEN reports multiallelic data."""
    reader_kwargs = {
        "raw_sample_ct": raw_sample_ct,
        "variant_ct": variant_ct,
        "sample_subset": sample_subset,
    }
    try:
        return pg.PgenReader(str.encode(filename_noext + ".pgen"), **reader_kwargs), False
    except RuntimeError as exc:
        if "multiallelic variants present" not in str(exc):
            raise
        if genotype_mode == "dosage":
            raise ValueError(_MULTIALLELIC_DOSAGE_ERROR) from exc

    pvar_filename = _find_pvar_path(filename_noext)
    if pvar_filename is None:
        raise FileNotFoundError(f"No .pvar, .pvar.zst, or .pvar.gz file found for {filename_noext}")
    with pg.PvarReader(str.encode(pvar_filename)) as pvar_reader:
        allele_idx_offsets = pvar_reader.get_allele_idx_offsets()
    return (
        pg.PgenReader(
            str.encode(filename_noext + ".pgen"),
            allele_idx_offsets=allele_idx_offsets,
            **reader_kwargs,
        ),
        True,
    )


def _read_dosage_parallel(
    filename_noext: str,
    *,
    raw_sample_ct: Optional[int],
    variant_ct: Optional[int],
    sample_subset: Optional[np.ndarray],
    variant_idxs: np.ndarray,
    num_variants: int,
    num_samples: int,
    threads: int,
) -> np.ndarray:
    """Decode ordered biallelic hardcalls with independent pgenlib readers."""
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
        reader, contains_multiallelic = _open_pgen_reader(
            filename_noext,
            raw_sample_ct=raw_sample_ct,
            variant_ct=variant_ct,
            sample_subset=sample_subset,
            genotype_mode="dosage",
        )
        try:
            if contains_multiallelic:
                raise ValueError(_MULTIALLELIC_DOSAGE_ERROR)
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


def _read_phased_parallel(
    filename_noext: str,
    *,
    raw_sample_ct: Optional[int],
    variant_ct: Optional[int],
    sample_subset: Optional[np.ndarray],
    variant_idxs: np.ndarray,
    num_variants: int,
    num_samples: int,
    threads: int,
) -> np.ndarray:
    """Decode ordered phased alleles with independent pgenlib readers."""
    genotypes = np.empty((num_variants, num_samples, 2), dtype=np.int8)
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
        reader, _ = _open_pgen_reader(
            filename_noext,
            raw_sample_ct=raw_sample_ct,
            variant_ct=variant_ct,
            sample_subset=sample_subset,
            genotype_mode="phased",
        )
        try:
            read_phased_alleles_into(
                reader,
                partition_variant_idxs,
                partition_output,
            )
        finally:
            reader.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(read_partition, range(worker_count)))
    return genotypes


@SNPBaseReader.register
class PGENReader(SNPBaseReader):
    def read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        sample_ids: Optional[np.ndarray] = None,
        sample_idxs: Optional[np.ndarray] = None,
        variant_ids: Optional[np.ndarray] = None,
        variant_idxs: Optional[np.ndarray] = None,
        genotype_mode: GenotypeMode = "auto",
        chromosome_ploidy: Optional[str] = None,
        separator: str = None,
        threads: int = 1,
    ) -> SNPObject:
        """
        Read a pgen fileset (pgen, psam, pvar) into a SNPObject.

        Args:
            fields (str, None, or list of str, optional): Fields to extract data for that should be included in the returned SNPObject.
                Available fields are 'GT', 'IID', 'REF', 'ALT', '#CHROM', 'CM', 'ID', 'POS', 'FILTER', 'QUAL', 'INFO'.
                To extract all fields, set fields to None. Defaults to None.
            exclude_fields (str, None, or list of str, optional): Fields to exclude from the returned SNPObject.
                Available fields are 'GT', 'IID', 'REF', 'ALT', '#CHROM', 'CM', 'ID', 'POS', 'FILTER', 'QUAL', 'INFO'.
                To exclude no fields, set exclude_fields to None. Defaults to None.
            sample_ids: List of sample IDs to read. If None and sample_idxs is None, all samples are read.
            sample_idxs: List of sample indices to read. If None and sample_ids is None, all samples are read.
            variant_ids: List of variant IDs to read. If None and variant_idxs is None, all variants are read.
            variant_idxs: List of variant indices to read. If None and variant_ids is None, all variants are read.
            genotype_mode: ``"dosage"`` returns one biallelic ALT-copy count per
                sample as an ``int8`` value in ``{0, 1, 2}`` and rejects multiallelic variants. ``"phased"``
                returns phased allele calls and requires PGEN hardcall phase
                information. ``"auto"`` (default) preserves phased hardcalls
                when possible and falls back to dosage for unphased hardcalls.
            chromosome_ploidy:
                Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
                all selected variants should be treated as ordinary diploid/autosomal; this
                skips non-diploid chromosome checks and can be faster. The default None/"auto"
                preserves existing behavior.
            separator: Separator used in the pvar file. If None, the separator is automatically detected.
                If the automatic detection fails, please specify the separator manually.
            threads: Number of independent native pgenlib decoder threads. Values
                above one are supported for explicit ``genotype_mode="dosage"``
                and ``genotype_mode="phased"`` reads.

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
        chromosome_ploidy_mode = _normalize_chromosome_ploidy(chromosome_ploidy)
        try:
            threads = operator.index(threads)
        except TypeError as exc:
            raise TypeError("PGENReader threads must be an integer.") from exc
        if threads < 1:
            raise ValueError("PGENReader threads must be at least 1.")
        if threads != 1 and genotype_mode == "auto":
            raise ValueError(
                "PGENReader threads require an explicit genotype_mode of "
                "'dosage' or 'phased'."
            )

        if isinstance(fields, str):
            fields = [fields]
        if isinstance(exclude_fields, str):
            exclude_fields = [exclude_fields]

        fields = fields or ["GT", "IID", "REF", "ALT", "#CHROM", "CM", "ID", "POS", "FILTER", "QUAL", "INFO"]
        exclude_fields = exclude_fields or []
        fields = [field for field in fields if field not in exclude_fields]
        only_read_pgen = fields == ["GT"] and variant_idxs is None and sample_idxs is None

        filename_noext = str(self.filename)
        for ext in [".pgen", ".pvar", ".pvar.zst", ".pvar.gz", ".psam", ".psam.zst", ".psam.gz"]:
            if filename_noext.endswith(ext):
                filename_noext = filename_noext[:-len(ext)]
                break

        if only_read_pgen:
            file_num_samples = None  # Not needed for pgen
            file_num_variants = None  # Not needed
        else:
            pvar_extensions = [".pvar", ".pvar.zst", ".pvar.gz"]
            pvar_filename = None
            for ext in pvar_extensions:
                possible_pvar = filename_noext + ext
                if os.path.exists(possible_pvar):
                    pvar_filename = possible_pvar
                    break
            if pvar_filename is None:
                raise FileNotFoundError(f"No .pvar, .pvar.zst, or .pvar.gz file found for {filename_noext}")

            log.info(f"Reading {pvar_filename}")

            pvar_has_header = True
            pvar_header_line_num = 0
            with _open_textfile(pvar_filename) as file:
                for line_num, line in enumerate(file):
                    if line.startswith("##"):  # Metadata
                        continue
                    else:
                        if separator is None:
                            separator = _detect_pvar_separator(line)
                        if line.startswith("#CHROM"):  # Header
                            pvar_header_line_num = line_num
                            header = line.strip().split()
                            break
                        elif not line.startswith("#"):  # If no header, look at line 1
                            pvar_has_header = False
                            cols_in_pvar = len(line.strip().split(separator))
                            if cols_in_pvar == 5:
                                header = ["#CHROM", "ID", "POS", "ALT", "REF"]
                            elif cols_in_pvar == 6:
                                header = ["#CHROM", "ID", "CM", "POS", "ALT", "REF"]
                            else:
                                raise ValueError(
                                    f"{pvar_filename} is not a valid pvar file."
                                )
                            break

            pvar_reading_args = {
                'separator': separator,
                'skip_rows': pvar_header_line_num,
                'has_header': pvar_has_header,
                'new_columns': None if pvar_has_header else header,
                'schema_overrides': {
                    "#CHROM": pl.String,
                    "CM": pl.Float64,
                    "POS": pl.UInt32,
                    "ID": pl.String,
                    "REF": pl.String,
                    "ALT": pl.String,
                    "QUAL": pl.String,
                    "FILTER": pl.String,
                    "INFO": pl.String,
                },
                'null_values': ["NA"],
            }
            if pvar_filename.endswith(('.zst', '.gz')):
                pvar = pl.read_csv(pvar_filename, **pvar_reading_args).lazy()
            else:
                pvar = pl.scan_csv(pvar_filename, **pvar_reading_args)

            # We need to map requested IDs to row positions before reading genotypes.
            variant_meta = pvar.select(["ID", "#CHROM", "POS"]).with_row_index().collect()
            file_num_variants = variant_meta.height

            if variant_ids is not None:
                variant_id_values = [str(v) for v in np.atleast_1d(variant_ids)]
                variant_id_or_pos = (
                    pl.col("ID").is_in(variant_id_values)
                    | pl.concat_str(
                        [pl.col("#CHROM"), pl.lit(":"), pl.col("POS").cast(pl.String)]
                    ).is_in(variant_id_values)
                )
                variant_idxs = (
                    variant_meta.filter(variant_id_or_pos)
                    .select("index")
                    .to_series()
                    .to_numpy()
                )

            if variant_idxs is None:
                num_variants = file_num_variants
                variant_idxs = np.arange(num_variants, dtype=np.uint32)
                pvar = pvar.collect()
            else:
                requested_variant_idxs = np.asarray(variant_idxs, dtype=np.uint32).ravel()
                if np.any(requested_variant_idxs >= file_num_variants):
                    raise ValueError("One or more variant indexes are out of bounds.")
                selector = pl.DataFrame({"index": requested_variant_idxs}).with_row_index("_selector_order")
                pvar = (
                    selector.lazy()
                    .join(pvar.with_row_index(), on="index", how="left")
                    .sort("_selector_order")
                    .collect()
                )
                variant_idxs = requested_variant_idxs
                num_variants = np.size(variant_idxs)
                pvar = pvar.drop(["_selector_order", "index"])

            psam_extensions = [".psam", ".psam.zst", ".psam.gz"]
            psam_filename = None
            for ext in psam_extensions:
                possible_psam = filename_noext + ext
                if os.path.exists(possible_psam):
                    psam_filename = possible_psam
                    break
            if psam_filename is None:
                raise FileNotFoundError(f"No .psam, .psam.zst, or .psam.gz file found for {filename_noext}")

            log.info(f"Reading {psam_filename}")

            with _open_textfile(psam_filename) as file:
                first_line = file.readline().strip()
                psam_has_header = first_line.startswith(("#FID", "FID", "#IID", "IID"))

            psam = pl.read_csv(
                psam_filename,
                separator=separator,
                has_header=psam_has_header,
                new_columns=None if psam_has_header else ["FID", "IID", "PAT", "MAT", "SEX", "PHENO1"],
                schema_overrides={
                    "#FID": pl.String,
                    "FID": pl.String,
                    "#IID": pl.String,
                    "IID": pl.String,
                    "PAT": pl.String,
                    "MAT": pl.String,
                    "SEX": pl.String,
                    "PHENO1": pl.String,
                },
                null_values=["NA"],
            ).with_row_index()
            if "#IID" in psam.columns:
                psam = psam.rename({"#IID": "IID"})
            if "#FID" in psam.columns:
                psam = psam.rename({"#FID": "FID"})

            file_num_samples = psam.height

            if sample_ids is not None:
                psam = psam.filter(pl.col("IID").is_in(sample_ids))
                sample_idxs = psam.select("index").to_series().to_numpy()
                num_samples = np.size(sample_idxs)
            elif sample_idxs is not None:
                num_samples = np.size(sample_idxs)
                sample_idxs = np.array(sample_idxs, dtype=np.uint32)
                psam = psam.filter(pl.col("index").is_in(sample_idxs))
            else:
                num_samples = file_num_samples

        if "GT" in fields:
            log.info(f"Reading {filename_noext}.pgen")
            pgen_reader, contains_multiallelic = _open_pgen_reader(
                filename_noext,
                raw_sample_ct=file_num_samples,
                variant_ct=file_num_variants,
                sample_subset=sample_idxs,
                genotype_mode=genotype_mode,
            )

            try:
                if only_read_pgen:
                    num_samples = pgen_reader.get_raw_sample_ct()
                    num_variants = pgen_reader.get_variant_ct()
                    variant_idxs = np.arange(num_variants, dtype=np.uint32)

                hardcall_phase_present = pgen_reader.hardcall_phase_present()
                auto_mode = genotype_mode == "auto"
                effective_return_dosage = genotype_mode == "dosage"
                if auto_mode:
                    effective_return_dosage = not hardcall_phase_present
                elif not effective_return_dosage and not hardcall_phase_present:
                    raise ValueError(
                        "This PGEN file does not contain hardcall phase information, so "
                        "genotype_mode='phased' is not supported. Use genotype_mode='dosage' "
                        "to load 0/1/2 genotype dosages."
                    )
                if effective_return_dosage and contains_multiallelic:
                    raise ValueError(_MULTIALLELIC_DOSAGE_ERROR)
                detect_non_diploid = effective_return_dosage and chromosome_ploidy_mode != "autosomal"

                non_diploid_mask = None
                if detect_non_diploid:
                    if only_read_pgen:
                        log.debug(
                            "Skipping chromosome-specific non-diploid correction because "
                            ".pvar metadata is unavailable in GT-only fast path."
                        )
                    elif "#CHROM" in pvar.columns:
                        non_diploid_mask = _non_diploid_chromosome_mask_or_none(
                            pvar.get_column("#CHROM").to_numpy()
                        )

                # required arrays: variant_idxs + sample_idxs + genotypes
                if not effective_return_dosage:
                    required_ram = (
                        (num_samples + num_variants) * 4
                        + estimate_phased_alleles_peak_bytes(
                            num_variants,
                            num_samples,
                            threads=threads,
                        )
                    )
                else:
                    required_ram = (num_samples + num_variants) * 4 + num_variants * num_samples
                    if non_diploid_mask is not None and hardcall_phase_present:
                        num_non_diploid = int(np.sum(non_diploid_mask))
                        required_ram += estimate_phased_alleles_peak_bytes(
                            num_non_diploid,
                            num_samples,
                        )
                log.info(f">{required_ram / 1024**3:.2f} GiB of RAM are required to process {num_samples} samples with {num_variants} variants each")

                if not effective_return_dosage:
                    if threads == 1:
                        genotypes = read_phased_alleles(
                            pgen_reader,
                            variant_idxs,
                            num_variants,
                            num_samples,
                        )
                    else:
                        genotypes = _read_phased_parallel(
                            filename_noext,
                            raw_sample_ct=file_num_samples,
                            variant_ct=file_num_variants,
                            sample_subset=sample_idxs,
                            variant_idxs=variant_idxs,
                            num_variants=num_variants,
                            num_samples=num_samples,
                            threads=threads,
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
                    if detect_non_diploid and non_diploid_mask is not None:
                        if hardcall_phase_present:
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
                        else:
                            log.debug(
                                "Skipping non-diploid haploid-missing correction because "
                                "separate allele calls are unavailable."
                            )
            finally:
                pgen_reader.close()
        else:
            genotypes = None

        log.info("Constructing SNPObject")

        fid_col = None
        if "IID" in fields and "FID" in psam.columns:
            fid_col = psam.get_column("FID").fill_null("NA").cast(pl.String).to_numpy()
        sex_col = None
        if "IID" in fields and "SEX" in psam.columns:
            sex_col = psam.get_column("SEX").fill_null("NA").cast(pl.String).to_numpy()

        snpobj = SNPObject(
            genotypes=genotypes if "GT" in fields else None,
            samples=psam.get_column("IID").to_numpy() if "IID" in fields and "IID" in psam.columns else None,
            sample_fid=fid_col,
            sample_sex=sex_col,
            **{f'variants_{k.lower()}': pvar.get_column(v).to_numpy() if v in fields and v in pvar.columns else None
               for k, v in {'ref': 'REF', 'alt': 'ALT', 'chrom': '#CHROM', 'cm': 'CM', 'id': 'ID', 'pos': 'POS', 'filter_pass': 'FILTER', 'qual': 'QUAL', 'info': 'INFO'}.items()}
        )

        log.info("Finished constructing SNPObject")
        return snpobj

    def _resolve_variant_idxs_for_iter(
        self,
        *,
        variant_ids: Optional[np.ndarray],
        variant_idxs: Optional[np.ndarray],
        separator: str = None,
    ) -> np.ndarray:
        """
        Resolve variant selectors to canonical file-order row indices.
        """
        filename_noext = str(self.filename)
        for ext in [".pgen", ".pvar", ".pvar.zst", ".pvar.gz", ".psam", ".psam.zst", ".psam.gz"]:
            if filename_noext.endswith(ext):
                filename_noext = filename_noext[:-len(ext)]
                break

        pvar_filename = None
        for ext in [".pvar", ".pvar.zst", ".pvar.gz"]:
            candidate = filename_noext + ext
            if os.path.exists(candidate):
                pvar_filename = candidate
                break
        if pvar_filename is None:
            raise FileNotFoundError(f"No .pvar, .pvar.zst, or .pvar.gz file found for {filename_noext}")

        local_separator = separator

        pvar_has_header = True
        pvar_header_line_num = 0
        with _open_textfile(pvar_filename) as file:
            for line_num, line in enumerate(file):
                if line.startswith("##"):
                    continue
                if local_separator is None:
                    local_separator = _detect_pvar_separator(line)
                if line.startswith("#CHROM"):
                    pvar_header_line_num = line_num
                    header = line.strip().split()
                    break
                if not line.startswith("#"):
                    pvar_has_header = False
                    cols_in_pvar = len(line.strip().split(local_separator))
                    if cols_in_pvar == 5:
                        header = ["#CHROM", "ID", "POS", "ALT", "REF"]
                    elif cols_in_pvar == 6:
                        header = ["#CHROM", "ID", "CM", "POS", "ALT", "REF"]
                    else:
                        raise ValueError(f"{pvar_filename} is not a valid pvar file.")
                    break

        pvar_reading_args = {
            "separator": local_separator,
            "skip_rows": pvar_header_line_num,
            "has_header": pvar_has_header,
            "new_columns": None if pvar_has_header else header,
            "schema_overrides": {
                "#CHROM": pl.String,
                "CM": pl.Float64,
                "POS": pl.UInt32,
                "ID": pl.String,
                "REF": pl.String,
                "ALT": pl.String,
                "QUAL": pl.String,
                "FILTER": pl.String,
                "INFO": pl.String,
            },
            "null_values": ["NA"],
        }
        if pvar_filename.endswith(((".zst", ".gz"))):
            pvar = pl.read_csv(pvar_filename, **pvar_reading_args)
        else:
            pvar = pl.scan_csv(pvar_filename, **pvar_reading_args).collect()

        variant_meta = pvar.select(["ID", "#CHROM", "POS"]).with_row_index()

        if variant_ids is not None:
            variant_id_values = [str(v) for v in np.atleast_1d(variant_ids)]
            variant_id_or_pos = (
                pl.col("ID").is_in(variant_id_values)
                | pl.concat_str([pl.col("#CHROM"), pl.lit(":"), pl.col("POS").cast(pl.String)]).is_in(
                    variant_id_values
                )
            )
            resolved = (
                variant_meta.filter(variant_id_or_pos)
                .select("index")
                .to_series()
                .to_numpy()
            )
            return np.asarray(resolved, dtype=np.uint32)

        if variant_idxs is not None:
            requested = np.asarray(variant_idxs, dtype=np.uint32).ravel()
            if np.any(requested >= variant_meta.height):
                raise ValueError("One or more variant indexes are out of bounds.")
            return requested

        return np.arange(variant_meta.height, dtype=np.uint32)

    def iter_read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        sample_ids: Optional[np.ndarray] = None,
        sample_idxs: Optional[np.ndarray] = None,
        variant_ids: Optional[np.ndarray] = None,
        variant_idxs: Optional[np.ndarray] = None,
        genotype_mode: ExplicitGenotypeMode = "phased",
        chromosome_ploidy: Optional[str] = None,
        separator: str = None,
        chunk_size: int = 10_000,
    ) -> Iterator[SNPObject]:
        """
        Stream the PGEN fileset in variant chunks.

        This yields a sequence of SNPObject chunks along the SNP axis.

        chromosome_ploidy:
            Optional hint for chromosome-specific dosage conversion. Use "autosomal" when
            all selected variants should be treated as ordinary diploid/autosomal; this
            skips non-diploid chromosome checks and can be faster. The default None/"auto"
            preserves existing behavior.
        """
        genotype_mode = normalize_genotype_mode(genotype_mode, allow_auto=False)
        _normalize_chromosome_ploidy(chromosome_ploidy)
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
            )
