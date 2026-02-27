import logging
from typing import List, Optional
import os
import csv

import numpy as np
import polars as pl
import pgenlib as pg

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader

log = logging.getLogger(__name__)


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
        sum_strands: bool = False,
        separator: str = None,
    ) -> SNPObject:
        """
        Read a pgen fileset (pgen, psam, pvar) into a SNPObject.

        Args:
            fields (str, None, or list of str, optional): Fields to extract data for that should be included in the returned SNPObject.
                Available fields are 'GT', 'IID', 'REF', 'ALT', '#CHROM', 'ID', 'POS', 'FILTER', 'QUAL'.
                To extract all fields, set fields to None. Defaults to None.
            exclude_fields (str, None, or list of str, optional): Fields to exclude from the returned SNPObject.
                Available fields are 'GT', 'IID', 'REF', 'ALT', '#CHROM', 'ID', 'POS', 'FILTER', 'QUAL'.
                To exclude no fields, set exclude_fields to None. Defaults to None.
            sample_ids: List of sample IDs to read. If None and sample_idxs is None, all samples are read.
            sample_idxs: List of sample indices to read. If None and sample_ids is None, all samples are read.
            variant_ids: List of variant IDs to read. If None and variant_idxs is None, all variants are read.
            variant_idxs: List of variant indices to read. If None and variant_ids is None, all variants are read.
            sum_strands: If True, maternal and paternal strands are combined into a single `int8` array with values `{0, 1, 2`}. 
                If False, strands are stored separately as an `int8` array with values `{0, 1}` for each strand. 
                Note: With the pgenlib backend, `False` uses `~8×` more RAM, though `calldata_gt` is only `2×` larger.
            separator: Separator used in the pvar file. If None, the separator is automatically detected.
                If the automatic detection fails, please specify the separator manually.

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

        if isinstance(fields, str):
            fields = [fields]
        if isinstance(exclude_fields, str):
            exclude_fields = [exclude_fields]

        fields = fields or ["GT", "IID", "REF", "ALT", "#CHROM", "ID", "POS", "FILTER", "QUAL"]
        exclude_fields = exclude_fields or []
        fields = [field for field in fields if field not in exclude_fields]
        only_read_pgen = fields == ["GT"] and variant_idxs is None and sample_idxs is None

        filename_noext = str(self.filename)
        for ext in [".pgen", ".pvar", ".pvar.zst", ".psam"]:
            if filename_noext.endswith(ext):
                filename_noext = filename_noext[:-len(ext)]
                break

        if only_read_pgen:
            file_num_samples = None  # Not needed for pgen
            file_num_variants = None  # Not needed
        else:
            pvar_extensions = [".pvar", ".pvar.zst"]
            pvar_filename = None
            for ext in pvar_extensions:
                possible_pvar = filename_noext + ext
                if os.path.exists(possible_pvar):
                    pvar_filename = possible_pvar
                    break
            if pvar_filename is None:
                raise FileNotFoundError(f"No .pvar or .pvar.zst file found for {filename_noext}")

            log.info(f"Reading {pvar_filename}")

            def open_textfile(filename):
                if filename.endswith('.zst'):
                    import zstandard as zstd
                    return zstd.open(filename, 'rt')
                else:
                    return open(filename, 'rt')

            pvar_has_header = True
            pvar_header_line_num = 0
            with open_textfile(pvar_filename) as file:
                for line_num, line in enumerate(file):
                    if line.startswith("##"):  # Metadata
                        continue
                    else:
                        if separator is None:
                            separator = csv.Sniffer().sniff(file.readline()).delimiter
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
                    "POS": pl.UInt32,
                    "ID": pl.String,
                    "REF": pl.String,
                    "ALT": pl.String,
                },
                'null_values': ["NA"],
            }
            if pvar_filename.endswith('.zst'):
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
                variant_idxs = np.array(variant_idxs, dtype=np.uint32)
                num_variants = np.size(variant_idxs)
                pvar = (
                    pvar.with_row_index()
                    .filter(pl.col("index").is_in(variant_idxs))
                    .drop("index")
                    .collect()
                )

            log.info(f"Reading {filename_noext}.psam")

            with open(filename_noext + ".psam") as file:
                first_line = file.readline().strip()
                psam_has_header = first_line.startswith(("#FID", "FID", "#IID", "IID"))

            psam = pl.read_csv(
                filename_noext + ".psam",
                separator=separator,
                has_header=psam_has_header,
                new_columns=None if psam_has_header else ["FID", "IID", "PAT", "MAT", "SEX", "PHENO1"],
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
            pgen_reader = pg.PgenReader(
                str.encode(filename_noext + ".pgen"),
                raw_sample_ct=file_num_samples,
                variant_ct=file_num_variants,
                sample_subset=sample_idxs,
            )

            if only_read_pgen:
                num_samples = pgen_reader.get_raw_sample_ct()
                num_variants = pgen_reader.get_variant_ct()
                variant_idxs = np.arange(num_variants, dtype=np.uint32)

            # required arrays: variant_idxs + sample_idxs + genotypes
            if not sum_strands:
                required_ram = (num_samples + num_variants + num_variants * 2 * num_samples) * 4
            else:
                required_ram = (num_samples + num_variants) * 4 + num_variants * num_samples
            log.info(f">{required_ram / 1024**3:.2f} GiB of RAM are required to process {num_samples} samples with {num_variants} variants each")

            if not sum_strands:
                genotypes = np.empty((num_variants, 2 * num_samples), dtype=np.int32)  # cannot use int8 because of pgenlib
                pgen_reader.read_alleles_list(variant_idxs, genotypes)
                genotypes = genotypes.astype(np.int8).reshape((num_variants, num_samples, 2))
            else:
                genotypes = np.empty((num_variants, num_samples), dtype=np.int8)
                pgen_reader.read_list(variant_idxs, genotypes)
            pgen_reader.close()
        else:
            genotypes = None

        log.info("Constructing SNPObject")

        snpobj = SNPObject(
            calldata_gt=genotypes if "GT" in fields else None,
            samples=psam.get_column("IID").to_numpy() if "IID" in fields and "IID" in psam.columns else None,
            **{f'variants_{k.lower()}': pvar.get_column(v).to_numpy() if v in fields and v in pvar.columns else None
               for k, v in {'ref': 'REF', 'alt': 'ALT', 'chrom': '#CHROM', 'id': 'ID', 'pos': 'POS', 'filter_pass': 'FILTER', 'qual': 'QUAL'}.items()}
        )

        log.info("Finished constructing SNPObject")
        return snpobj
