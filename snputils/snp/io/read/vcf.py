import logging
from typing import Optional, List, Any, Union
from pathlib import Path
import gzip
import csv

import allel
import numpy as np
import polars as pl
import multiprocessing
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from os.path import splitext, exists, abspath
import subprocess
from io import TextIOWrapper
import pathlib 
log = logging.getLogger(__name__)


@SNPBaseReader.register
class VCFReader(SNPBaseReader):
    def __init__(self, filename: Union[str, pathlib.Path]):
        super().__init__(filename)
        self._igd_path: Optional[pathlib.Path] = None
        self._grg_path: Optional[pathlib.Path] = None
        self.debug : bool = False
    def read(
        self,
        fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        rename_fields: Optional[dict] = None,
        fills: Optional[dict] = None,
        region: Optional[str] = None,
        samples: Optional[List[str]] = None,
        sum_strands: bool = False,
    ) -> SNPObject:
        """
        Read a vcf file into a SNPObject.

        Args:
            fields: Fields to extract data for. e.g., ['variants/CHROM', 'variants/POS',
                'calldata/GT']. If you are feeling lazy, you can drop the 'variants/'
                and 'calldata/' prefixes, in which case the fields will be matched
                against fields declared in the VCF header, with variants taking priority
                over calldata if a field with the same ID exists both in INFO and FORMAT
                headers. I.e., ['CHROM', 'POS', 'DP', 'GT'] will work, although watch out
                for fields like 'DP' which can be both INFO and FORMAT. To extract all
                fields, provide just the string '*'. To extract all variants fields
                (including all INFO fields) provide 'variants/*'. To extract all
                calldata fields (i.e., defined in FORMAT headers) provide 'calldata/*'.
            exclude_fields: Fields to exclude. E.g., for use in combination with fields='*'.
            rename_fields: Fields to be renamed. Should be a dictionary mapping old to new names.
            fills: Override the fill value used for empty values. Should be a dictionary
                mapping field names to fill values.
            region: Genomic region to extract variants for. If provided, should be a
                tabix-style region string, which can be either just a chromosome name
                (e.g., '2L'), or a chromosome name followed by 1-based beginning and
                end coordinates (e.g., '2L:100000-200000'). Note that only variants
                whose start position (POS) is within the requested range will be included.
                This is slightly different from the default tabix behaviour, where a
                variant (e.g., deletion) may be included if its position (POS) occurs
                before the requested region but its reference allele overlaps the
                region - such a variant will not be included in the data returned
                by this function.
            samples: Selection of samples to extract calldata for. If provided, should be
                a list of strings giving sample identifiers. May also be a list of
                integers giving indices of selected samples.
            sum_strands: If True, maternal and paternal strands are combined into a single `int8` array with values `{0, 1, 2`}. 
                If False, strands are stored separately as an `int8` array with values `{0, 1}` for each strand.

        Returns:
            **SNPObject**: 
                A SNPObject instance.
        """
        log.info(f"Reading {self.filename}")

        vcf_dict = allel.read_vcf(
            str(self.filename),
            fields=fields,
            exclude_fields=exclude_fields,
            rename_fields=rename_fields,
            fills=fills,
            region=region,
            samples=samples,
            alt_number=1,
        )
        assert vcf_dict is not None  # suppress Flake8 warning

        genotypes = vcf_dict["calldata/GT"].astype(np.int8)
        if sum_strands:
            genotypes = genotypes.sum(axis=2, dtype=np.int8)

        snpobj = SNPObject(
            calldata_gt=genotypes,
            samples=vcf_dict["samples"],
            variants_ref=vcf_dict["variants/REF"],
            variants_alt=vcf_dict["variants/ALT"],
            variants_chrom=vcf_dict["variants/CHROM"],
            variants_filter_pass=vcf_dict["variants/FILTER_PASS"],
            variants_id=vcf_dict["variants/ID"],
            variants_pos=vcf_dict["variants/POS"],
            variants_qual=vcf_dict["variants/QUAL"],
        )

        log.info(f"Finished reading {self.filename}")
        return snpobj
    def to_igd(self,
                igd_file : Optional[str] = None,
                logfile_out : Optional[str] = None,
                logfile_err : Optional[str] = None) -> None:
        """
        Convert the current VCF input file to IGD via `grg convert`.

        Args:
            igd_file: Output IGD file path. Defaults to `<vcf_stem>.igd`.
            logfile_out: The file to log standard output to. If None (default), no output will be logged (i.e., piped to dev null).
            logfile_err: The file to log standard error to. If None (default), no error will be logged (i.e., piped to dev null).

        """

        if not exists(self.filename):
            raise FileNotFoundError(f"File {self.filename} does not exist")

        lf_o  : Union[int, TextIOWrapper] = subprocess.DEVNULL if logfile_out is None else open(logfile_out, "a")
        lf_e  : Union[int, TextIOWrapper] = subprocess.DEVNULL if logfile_err is None else open(logfile_err, "a")
        name, _ext1 = splitext(str(self.filename))
        name, _ext2 = splitext(name)
        if igd_file is None:
            self._igd_path = pathlib.Path(name + ".igd")
        else:
            self._igd_path = pathlib.Path(igd_file)

        try:
            subprocess.run(
                ["grg", "convert", abspath(str(self.filename)), abspath(str(self._igd_path))],
                stdout=lf_o,
                stderr=lf_e,
                check=True,
            )
        finally:
            if not isinstance(lf_o, int):
                lf_o.close()
            if not isinstance(lf_e, int):
                lf_e.close()
            
    def to_grg(self,
               range: Optional[str] = None,
               parts: Optional[int] = None,
               jobs: Optional[int] = None,
               trees: Optional[int] = None,
               binmuts: Optional[bool] = None,
               no_file_cleanup: Optional[bool] = None,
               maf_flip: Optional[bool] = None,
               population_ids: Optional[str] = None,
               mutation_batch_size: Optional[int] = None,
               igd_file: Optional[str] = None,
               out_file: Optional[str] = None,
               verbose: Optional[bool] = None,
               no_merge: Optional[bool] = None,
               force: Optional[bool] = None,
               logfile_out: Optional[str] = None,
               logfile_err: Optional[str] = None
               ) -> None:
        """
        Convert VCF input to a GRG file via `grg construct`.

        If `igd_file` exists, it is used as construct input. If it does not
        exist, it is first created via `to_igd` and then used for construction.
        """
        input_file = pathlib.Path(self.filename).resolve()
        if igd_file is not None:
            candidate_igd = pathlib.Path(igd_file)
            if candidate_igd.exists():
                self._igd_path = candidate_igd.resolve()
            else:
                self.to_igd(igd_file, logfile_out, logfile_err)
            input_file = pathlib.Path(self._igd_path).resolve()

        if out_file is not None:
            self._grg_path = pathlib.Path(out_file)
        else:
            default_stem = splitext(str(input_file))[0]
            if default_stem.endswith(".vcf"):
                default_stem = splitext(default_stem)[0]
            self._grg_path = pathlib.Path(default_stem + ".grg")

        lf_o: Union[int, TextIOWrapper] = subprocess.DEVNULL if logfile_out is None else open(logfile_out, "a")
        lf_e: Union[int, TextIOWrapper] = subprocess.DEVNULL if logfile_err is None else open(logfile_err, "a")
        args = ["grg", "construct"]
        args += self._setarg(range, "-r", None)
        args += self._setarg(parts, "-p", 50)
        args += self._setarg(jobs, "-j", multiprocessing.cpu_count())
        args += self._setarg(trees, "-t", 16)
        args += self._setarg(binmuts, "--binary-muts", None)
        args += self._setarg(no_file_cleanup, "--no-file-cleanup", None)
        args += self._setarg(maf_flip, "--maf-flip", None)
        args += self._setarg(population_ids, "--population-ids", None)
        args += self._setarg(mutation_batch_size, "--mutation-batch-size", None)
        args += self._setarg(str(self._grg_path), "--out-file", None)
        args += self._setarg(verbose, "--verbose", None)
        args += self._setarg(no_merge, "--no-merge", None)
        args += self._setarg(force, "--force", None)
        args += [str(input_file)]
        log.debug("Running grg construct command: %s", args)
        try:
            subprocess.run(args, stdout=lf_o, stderr=lf_e, check=True)
        finally:
            if not isinstance(lf_o, int):
                lf_o.close()
            if not isinstance(lf_e, int):
                lf_e.close()

    def _setarg(self, x: Optional[Any], flag: str, default_arg: Optional[Any] = None) -> List[str]:
        if isinstance(x, bool):
            return [flag] if x else []
        if x is None and default_arg is not None:
            return [flag, f"{default_arg}"] 
        elif x is not None:
            return [flag, f"{x}"]
        else:
            return []
        



def _get_vcf_col_names_and_sep(vcf_path: str, separator: Optional[str] = None):
    """
    Get the column names and separator used in the VCF file.

    Args:
        vcf_path: The path to the VCF file.
        separator: Separator character. If None, the separator is automatically detected.

    Returns:
        col_names: List of column names.
        separator: Separator character.
    """
    vcf_path = Path(vcf_path)
    if vcf_path.suffixes[-2:] == ['.vcf', '.gz']:
        open_func = gzip.open
        mode = 'rt'
    elif vcf_path.suffix == '.vcf':
        open_func = open
        mode = 'r'
    else:
        raise ValueError(f"Unsupported file extension: {vcf_path.suffixes}")

    col_names = None
    with open_func(vcf_path, mode) as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("##"):
                continue
            if stripped_line.startswith("#CHROM") or stripped_line.startswith("CHROM"):
                if separator is None:
                    if "\t" in stripped_line:
                        separator = "\t"
                    else:
                        try:
                            separator = csv.Sniffer().sniff(stripped_line).delimiter
                        except csv.Error:
                            separator = "\t"
                col_names = [x.strip() for x in stripped_line.split(separator)]
                break

    if col_names is None:
        raise ValueError(
            "Could not find VCF header line. Expected a line starting with 'CHROM' or '#CHROM'."
        )

    return col_names, separator


def _infer_col_data_types(names: List):
    """
    Infer data types for VCF columns.

    Args:
        names: List of column names.

    Returns:
        col_dtypes: Dictionary mapping column names to data types.
    """
    col_dtypes = {name: pl.Utf8 for name in names}
    if 'POS' in col_dtypes:
        col_dtypes['POS'] = pl.Int32
    if '#CHROM' in col_dtypes:
        col_dtypes['#CHROM'] = pl.String
    if 'CHROM' in col_dtypes:
        col_dtypes['CHROM'] = pl.String

    return col_dtypes


def _extract_columns(names: List[str], fields: List[str], exclude_fields: List[str],
                     samples: List[str]) -> List[str]:
    """
    Extracts columns based on specified `fields`, `exclude_fields` and `samples`.

    Args:
        names: List of column names.
        fields: Fields to extract data for. This parameter specifies which data fields
            from the VCF file should be included in the result. To extract all fields,
            provide just the string '*'.
        exclude_fields: Fields to exclude. E.g., for use in combination with fields='*'.
        samples: Selection of samples to extract calldata for. If provided, should be
            a list of strings giving sample identifiers. May also be a list of
            integers giving indices of selected samples.

    Returns:
        field_columns: List of field columns.
        sample_columns: List of sample columns.
        selected_column_idxs: List of selected column indices.
    """

    # Define standard field names in a VCF file
    field_names = ['#CHROM', 'CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']

    # Find the index of the first column that is not a standard field name
    first_sample_idx = next((i for i, col in enumerate(names) if col not in field_names), len(names))

    # Identify field columns as all columns before the first sample column
    field_columns = names[:first_sample_idx]

    if fields != '*' and fields is not None:
        # Filter field columns to contain those in `fields`
        selected_fields = set(fields)
        if 'CHROM' in selected_fields and '#CHROM' in names:
            selected_fields.add('#CHROM')
        if '#CHROM' in selected_fields and 'CHROM' in names:
            selected_fields.add('CHROM')
        field_columns = [col for col in field_columns if col in selected_fields]
    elif fields == '*' and exclude_fields is not None:
        excluded_fields = set(exclude_fields)
        if 'CHROM' in excluded_fields and '#CHROM' in names:
            excluded_fields.add('#CHROM')
        if '#CHROM' in excluded_fields and 'CHROM' in names:
            excluded_fields.add('CHROM')
        field_columns = [col for col in field_columns if col not in excluded_fields]

    # Sample columns are all columns starting from the first sample column
    sample_columns = names[first_sample_idx:]

    if samples is not None:
        if len(samples) == 0:
            sample_columns = []
        elif type(samples[0]) is int:
            sample_columns = list(np.array(sample_columns)[samples])
        else:
            selected_samples = set(samples)
            sample_columns = [col for col in sample_columns if col in selected_samples]

    # Create a dictionary mapping column names to their indices
    column_idx_map = {name: index for index, name in enumerate(names)}

    # Create a list of selected column indices
    selected_column_idxs = [column_idx_map[col] for col in field_columns + sample_columns]

    # Sort the selected column indices
    selected_column_idxs.sort()

    return field_columns, sample_columns, selected_column_idxs


@SNPBaseReader.register
class VCFReaderPolars(SNPBaseReader):
    """Reads a VCF file and processes it into a SNPObject."""

    def __init__(self, filename: str):
        self._filename = filename

    def read(self,
             fields: Optional[List[str]] = None,
             exclude_fields: Optional[List[str]] = None,
             region: Optional[str] = None,
             samples: Optional[List[str]] = None,
             sum_strands: Optional[bool] = False,
             separator: Optional[str] = None
             ) -> SNPObject:
        """
        Read a vcf file into a SNPObject.

        Args:
            fields: Fields to extract data for. This parameter specifies which data fields
                from the VCF file should be included in the result. Available options include
                'CHROM'/'#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', and
                'FORMAT'.
                To extract all fields, provide just the string '*' or the default None.
            exclude_fields: Fields to exclude for use in combination with fields='*'.
                Available options include 'CHROM'/'#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL',
                'FILTER', 'INFO', and 'FORMAT'.
            region: Genomic region to extract variants for. If provided, it should be a
                tabix-style region string, specifying a chromosome name and optionally
                beginning and end coordinates (e.g., '2L:100000-200000'). TODO
            samples: Selection of samples to extract calldata for. If provided, should be
                a list of strings giving sample identifiers. May also be a list of
                integers giving indices of selected samples.  If an empty list is provided,
                no samples are extracted.
            sum_strands: True if the maternal and paternal strands are to be summed together,
                False if the strands are to be stored separately.
            separator: Separator used in the pvar file. If None, the separator is automatically detected.
                If the automatic detection fails, please specify the separator manually.

        Returns:
            snpobj: SNPObject containing the data from the VCF file. The format and content
                of this object depend on the specified parameters and the content of the VCF file.
        """
        # TODO: add support for excluding GT

        log.info(f"Reading {self._filename}")

        try:
            # Get the names and data types for the VCF file
            col_names, separator = _get_vcf_col_names_and_sep(self._filename)
            col_dtypes = _infer_col_data_types(col_names)

            # Extract columns to read based on specified 'fields', 'exclude_fields' and 'samples'
            # `selected_column_idxs` contains the indexes of the selected columns
            fields, samples, selected_column_idxs = _extract_columns(col_names, fields, exclude_fields, samples)

            # Read the VCF file into a Polars DataFrame
            vcf = pl.read_csv(
                self._filename,
                comment_prefix='##',
                has_header=True,
                separator=separator,
                columns=selected_column_idxs,
                schema_overrides=col_dtypes
            )

            log.debug("vcf polars read")

            if not samples:
                genotypes = np.array([])
            else:  # note that if samples is None, all samples are read
                # Process maternal strand:
                # Extract the first position from genotype, e.g., 0|1 -> 0
                # Replace missing values codified as ".", "-", or "" with -1, necessary for integer casting
                genotype_maternal = vcf[samples].select(pl.all().str.slice(0, length=1)) \
                                                .select(pl.all().replace({
                                                    ".": -1,
                                                    "-": -1,
                                                    "": -1
                                                })).cast(pl.Int8)

                # Process paternal strand:
                # Extract the third position from genotype, e.g., 0|1 -> 1
                # Convert ":" to ".." such that if only maternal strand is present, paternal strand is set to missing,
                # e.g., 0:0.982 -> 0..0.982 -> . -> -1
                # Replace missing values with -1
                genotype_paternal = vcf[samples].select(pl.all().str.replace_all('-1', '.')) \
                                                .select(pl.all().str.replace(':', '..')) \
                                                .select(pl.all().str.slice(2, length=1)) \
                                                .select(pl.all().replace({
                                                    ".": -1,
                                                    "": -1
                                                })).cast(pl.Int8)

                # Combine maternal and paternal genotypes
                genotypes = np.dstack((genotype_maternal, genotype_paternal))

                if sum_strands:
                    genotypes = genotypes.sum(axis=2, dtype=np.int8)

            if '#CHROM' in vcf.columns:
                chrom_column = '#CHROM'
            elif 'CHROM' in vcf.columns:
                chrom_column = 'CHROM'
            else:
                chrom_column = None

            # Create a SNPObject with the processed data
            snpobj = SNPObject(
                calldata_gt=genotypes,
                samples=np.array(samples),
                variants_ref=vcf['REF'].to_numpy() if 'REF' in fields else np.array([]),
                variants_alt=vcf['ALT'].to_numpy() if 'ALT' in fields else np.array([]),
                variants_chrom=(
                    vcf[chrom_column].to_numpy()
                    if chrom_column is not None and chrom_column in fields
                    else np.array([])
                ),
                variants_filter_pass=vcf['FILTER'].to_numpy() if 'FILTER' in fields else np.array([]),
                variants_id=vcf['ID'].to_numpy() if 'ID' in fields else np.array([]),
                variants_pos=vcf['POS'].to_numpy() if 'POS' in fields else np.array([]),
                variants_qual=vcf['QUAL'].to_numpy() if 'QUAL' in fields else np.array([])
            )

            log.info(f"Finished reading {self.filename}")

            return snpobj

        except Exception as e:
            log.warning(
                "Polars VCF parsing failed (%s). Falling back to scikit-allel reader.",
                e,
            )
            from snputils.snp.io.read import VCFReader

            # Instantiate a VCFReader object and read SNP data
            reader = VCFReader(self._filename)
            snpobj = reader.read()

            return snpobj
