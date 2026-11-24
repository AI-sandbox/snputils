import logging
import gzip
from copy import copy
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from snputils.snp.genobj import SNPObject

log = logging.getLogger(__name__)


def _vcf_allele_index(value: object, field: str = "GT") -> int:
    if isinstance(value, (float, np.floating)) and value.is_integer():
        value = int(value)
    token = str(value)
    if token == ".":
        return -1
    if not (token.isascii() and token.isdecimal()):
        raise ValueError(f"VCF {field} requires '.' or nonnegative integer allele indexes, got {value!r}.")
    return int(token)


def _encode_vcf_genotypes(
    genotypes: np.ndarray, before: Union[int, float, str], replacement: Optional[int],
) -> np.ndarray:
    if genotypes.dtype.kind == "S":
        genotypes = np.char.decode(genotypes, "ascii")
    if replacement is None:
        missing = False
    elif isinstance(before, (float, np.floating)) and np.isnan(before):
        missing = np.not_equal(genotypes, genotypes)
    else:
        # Phased PGEN reads use -9 for the same missing allele that VCF reads encode as -1.
        if before == -1 and genotypes.dtype.kind in "ifO" and np.any(genotypes == -9):
            genotypes = np.where(genotypes == -9, -1, genotypes)
        missing = genotypes == before

    native_integers = genotypes.dtype.kind in "iu"
    if native_integers:
        if np.any((genotypes < 0) & np.logical_not(missing)):
            raise ValueError("VCF GT requires '.' or nonnegative integer allele indexes.")
        # The default -1 missing marker already has the internal encoding for '.'.
        if before == replacement or not np.any(missing):
            return genotypes

    indices = genotypes.astype(object)
    if not native_integers:
        np.frompyfunc(_vcf_allele_index, 1, 1)(indices, where=np.logical_not(missing), out=indices)
    indices[missing] = replacement
    return indices


def _vcf_alt_text(alt: object) -> str:
    if isinstance(alt, bytes):
        alt = alt.decode("ascii")
    return "." if alt is None or alt == "" else str(alt)


def _validate_vcf_allele_bounds(data: SNPObject) -> None:
    # An allele index is local to its record. Zero is REF and ALT entries start at one.
    max_indices = data.genotypes.max(axis=(1, 2), initial=0)
    for i, max_index in enumerate(max_indices):
        alt = _vcf_alt_text(data.variants_alt[i])
        n_alt = 0 if alt == "." else alt.count(",") + 1
        if max_index > n_alt:
            raise ValueError(
                f"VCF allele index {max_index} at {data.variants_chrom[i]}:{data.variants_pos[i]} "
                f"is out of range for ALT={alt} (expected 0 to {n_alt} or '.')."
            )


def _normalize_info_values(
    data: SNPObject,
    variants_info: Optional[Sequence[str]],
) -> np.ndarray:
    if variants_info is not None:
        values = np.asarray(variants_info, dtype=object)
    elif data.variants_info is not None and len(np.asarray(data.variants_info)) == data.n_snps:
        values = np.asarray(data.variants_info, dtype=object)
    else:
        values = np.full(data.n_snps, ".", dtype=object)

    if values.shape[0] != data.n_snps:
        raise ValueError("variants_info length must match the number of variants.")

    return np.asarray(
        ["." if value is None or str(value) == "" else str(value) for value in values],
        dtype=object,
    )


def _normalize_qual_values(data: SNPObject) -> np.ndarray:
    values = data.variants_qual
    if values is None or len(np.asarray(values)) != data.n_snps:
        return np.full(data.n_snps, ".", dtype=object)
    out = []
    for value in np.asarray(values, dtype=object):
        if value is None:
            out.append(".")
        elif isinstance(value, (float, np.floating)) and np.isnan(value):
            out.append(".")
        else:
            text = str(value)
            out.append("." if text == "" else text)
    return np.asarray(out, dtype=object)


def _normalize_filter_values(data: SNPObject) -> np.ndarray:
    values = data.variants_filter_pass
    if values is None or len(np.asarray(values)) != data.n_snps:
        return np.full(data.n_snps, "PASS", dtype=object)

    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return np.where(arr, "PASS", ".").astype(object)

    out = []
    for value in np.asarray(values, dtype=object):
        if value is None:
            out.append(".")
        else:
            text = str(value)
            out.append("." if text == "" else text)
    return np.asarray(out, dtype=object)


class VCFWriter:
    """
    A writer class for exporting SNP data from a `snputils.snp.genobj.SNPObject` 
    into an `.vcf` file. Supports sampleless VCF writes (annotation-only) when 
    the input `SNPObject` has no samples.
    """
    def __init__(self, snpobj: SNPObject, filename: str, n_jobs: int = -1, phased: bool = False):
        """
        Args:
            snpobj (SNPObject):
                A SNPObject instance.
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. It should end with `.vcf`. 
                If the provided path does not have this extension, the `.vcf` extension will be appended.
            n_jobs: 
                Number of jobs to run in parallel. 
                - `None`: use 1 job unless within a `joblib.parallel_backend` context.  
                - `-1`: use all available processors.  
                - Any other integer: use the specified number of jobs.
            phased: 
                If True, genotype data is written in "first_allele|second_allele" format.
                If False, genotype data is written in "first_allele/second_allele" format.
        """
        del n_jobs
 
        self.__snpobj = snpobj
        self.__filename = Path(filename)
        self.__phased = phased
 
    def write(
            self,
            chrom_partition: bool = False,
            rename_missing_values: bool = True,
            before: Union[int, float, str] = -1,
            after: Union[int, float, str] = '.',
            variants_info: Optional[Sequence[str]] = None,
        ):
        """
        Writes the SNP data to VCF file(s). If writing a sampleless VCF,
        the genotypes array must be an empty array with a variant axis.

        Args:
            chrom_partition (bool, optional):
                If True, individual VCF files are generated for each chromosome.
                If False, a single VCF file containing data for all chromosomes is created. Defaults to False.
            rename_missing_values (bool, optional):
                If True, renames potential missing values in `snpobj.genotypes` before writing.
                Defaults to True.
            before (int, float, or str, default=-1):
                The input missing allele marker. The default recognizes both numeric -1 and
                the -9 returned by phased PGEN reads. Custom markers can include '.' or NaN.
            after (int, float, or str, default='.'):
                The value to write for `before`, either '.' or an allele index defined at each
                record where the replacement is used.
            variants_info (sequence of str, optional):
                Per-variant INFO column values (e.g. ``["END=2000", "END=3000"]``). Length must match variant count.
                When provided, a ##INFO header line for END is written if any value contains ``END=``.
        """
        replacement = _vcf_allele_index(after, "after") if rename_missing_values else None
        data = copy(self.__snpobj)
        genotypes = np.asarray(data.genotypes)
        if data.n_samples:
            if genotypes.ndim != 3 or genotypes.shape[2] != 2:
                raise ValueError("VCFWriter requires diploid genotype arrays with shape (n_variants, n_samples, 2).")
            data.genotypes = _encode_vcf_genotypes(genotypes, before, replacement)
            _validate_vcf_allele_bounds(data)
        elif genotypes.ndim not in (2, 3):
            raise ValueError("Sampleless VCF writes require an empty genotype array with a variant axis.")

        file_extensions = (".vcf", ".bcf")
        suffixes = self.__filename.suffixes
        if len(suffixes) >= 2 and suffixes[-2:] == [".vcf", ".gz"]:
            self.__file_extension = ".vcf.gz"
            self.__filename = self.__filename.with_suffix("").with_suffix("")
        elif self.__filename.suffix in file_extensions:
            self.__file_extension = self.__filename.suffix
            self.__filename = self.__filename.with_suffix('')
        else:
            self.__file_extension = ".vcf"

        if chrom_partition:
            chroms = data.unique_chrom

            for chrom in chroms:
                data_chrom = data.filter_variants(chrom=chrom, inplace=False)
                if variants_info is not None:
                    mask = data.variants_chrom == chrom
                    info_chrom = [variants_info[i] for i in np.where(mask)[0]]
                else:
                    info_chrom = None
                log.debug(f'Storing chromosome {chrom}')
                self._write_chromosome_data(chrom, data_chrom, info_chrom)
        else:
            self._write_chromosome_data("All", data, variants_info)

    def _write_chromosome_data(
        self, chrom, data_chrom, variants_info: Optional[Sequence[str]] = None
    ):
        """
        Writes the SNP data for a specific chromosome to a VCF file.

        Args:
            chrom: The chromosome name.
            data_chrom: The SNPObject instance containing the data for the chromosome.
            variants_info: Optional per-variant INFO strings; length must match variant count.
        """
        genotypes = np.asarray(data_chrom.genotypes)
        n_windows = data_chrom.n_snps
        n_samples = data_chrom.n_samples
        has_samples = n_samples > 0

        info_values = _normalize_info_values(data_chrom, variants_info)
        qual_values = _normalize_qual_values(data_chrom)
        filter_values = _normalize_filter_values(data_chrom)

        if chrom == "All":
            file = self.__filename.with_suffix(self.__file_extension)
        else:
            file = self.__filename.parent / f"{self.__filename.stem}_{chrom}{self.__file_extension}"

        if self.__file_extension == ".vcf.gz":
            out = gzip.open(file, "wt", encoding="utf-8")
        else:
            out = open(file, "w")
        out.write("##fileformat=VCFv4.1\n")
        if has_samples:
            out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')
        if any("END=" in s for s in info_values):
            out.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the segment">\n')
        for c in set(data_chrom.variants_chrom):
            out.write(f"##contig=<ID={c}>\n")
        cols = ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO"]
        if has_samples:
            cols += ["FORMAT"] + list(data_chrom.samples)
        out.write("\t".join(cols) + "\n")

        sep = "|" if self.__phased else "/"
        for i in range(n_windows):
            chrom_val = data_chrom.variants_chrom[i]
            pos = data_chrom.variants_pos[i]
            vid = data_chrom.variants_id[i]
            ref = data_chrom.variants_ref[i]
            alt = _vcf_alt_text(data_chrom.variants_alt[i])
            fields = [
                str(chrom_val),
                str(pos),
                str(vid),
                str(ref),
                alt,
                str(qual_values[i]),
                str(filter_values[i]),
                str(info_values[i]),
            ]
            if has_samples:
                row = genotypes[i]
                missing = row == -1
                if np.any(missing):
                    row = np.where(missing, ".", row)
                sample_fields = [
                    f"{row[s,0]}{sep}{row[s,1]}"
                    for s in range(n_samples)
                ]
                fields.extend(["GT", *sample_fields])
            line = "\t".join(fields)
            out.write(line + "\n")
        out.close()
