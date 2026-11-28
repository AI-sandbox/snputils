import logging
import gzip
import struct
from copy import copy
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

import numpy as np

from snputils.snp.genobj import SNPObject

log = logging.getLogger(__name__)


def _bgzf_block(data: Union[bytes, bytearray]) -> bytes:
    member = gzip.compress(data, compresslevel=6, mtime=0)
    # Replace the ten-byte gzip header. The BGZF extra field adds eight bytes,
    # and BSIZE stores the size of the entire member minus one.
    header = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff"
    extra = struct.pack("<H2sHH", 6, b"BC", 2, len(member) + 8 - 1)
    return header + extra + member[10:]


def _write_bgzf_lines(file: Path, lines: Iterable[str]) -> None:
    # Leave room for the gzip framing and worst-case deflate overhead.
    block_size = 65280
    pending = bytearray()
    with file.open("wb") as out:
        for line in lines:
            pending.extend(line.encode("utf-8"))
            if len(pending) >= block_size:
                stop = len(pending) - len(pending) % block_size
                for start in range(0, stop, block_size):
                    out.write(_bgzf_block(pending[start:start + block_size]))
                del pending[:stop]
        if pending:
            out.write(_bgzf_block(pending))
        out.write(_bgzf_block(b""))


def _decode_vcf_bytes(value: object) -> object:
    return value.decode("ascii") if isinstance(value, bytes) else value


def _vcf_allele_index(value: object, field: str = "GT") -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
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
    elif genotypes.dtype == object:
        genotypes = np.frompyfunc(_decode_vcf_bytes, 1, 1)(genotypes)
    elif genotypes.dtype == np.bool_:
        genotypes = genotypes.astype(np.uint8)
    if replacement is None:
        missing = False
    elif isinstance(before, (float, np.floating)) and np.isnan(before):
        missing = np.not_equal(genotypes, genotypes)
    else:
        # Phased PGEN reads use -9 for the same missing allele that VCF reads encode as -1.
        if before == -1 and genotypes.dtype.kind in "ifO" and np.any(genotypes == -9):
            genotypes = np.where(genotypes == -9, -1, genotypes)
        missing = genotypes == _decode_vcf_bytes(before)

    native_integers = genotypes.dtype.kind in "iu"
    if native_integers:
        if np.any((genotypes < 0) & np.logical_not(missing)):
            raise ValueError("VCF GT requires '.' or nonnegative integer allele indexes.")
        # The default -1 missing marker already has the internal encoding for '.'.
        if before == replacement or not np.any(missing):
            return genotypes

    # Decoding already created a private buffer for object arrays.
    indices = genotypes.astype(object, copy=False)
    if not native_integers:
        np.frompyfunc(_vcf_allele_index, 1, 1)(indices, where=np.logical_not(missing), out=indices)
    indices[missing] = replacement
    return indices


def _vcf_alt_text(alt: object) -> str:
    alt = _decode_vcf_bytes(alt)
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

        file = self.__filename
        if file.name.lower().endswith(".vcf.gz"):
            extension = file.name[-len(".vcf.gz"):]
        elif file.suffix.lower() in (".vcf", ".bcf"):
            extension = file.suffix
        else:
            extension = ".vcf"
            file = file.with_name(file.name + extension)

        if chrom_partition:
            chroms = data.unique_chrom
            prefix = file.name[:-len(extension)]

            for chrom in chroms:
                data_chrom = data.filter_variants(chrom=chrom, inplace=False)
                if variants_info is not None:
                    mask = data.variants_chrom == chrom
                    info_chrom = [variants_info[i] for i in np.where(mask)[0]]
                else:
                    info_chrom = None
                log.debug(f'Storing chromosome {chrom}')
                chrom_file = file.with_name(f"{prefix}_{chrom}{extension}")
                self._write_chromosome_data(chrom_file, data_chrom, info_chrom)
        else:
            self._write_chromosome_data(file, data, variants_info)

    def _write_chromosome_data(
        self, file: Path, data_chrom, variants_info: Optional[Sequence[str]] = None
    ):
        """
        Writes the SNP data for a specific chromosome to a VCF file.

        Args:
            file: The output path.
            data_chrom: The SNPObject instance containing the data for the chromosome.
            variants_info: Optional per-variant INFO strings; length must match variant count.
        """
        info_values = _normalize_info_values(data_chrom, variants_info)
        qual_values = _normalize_qual_values(data_chrom)
        filter_values = _normalize_filter_values(data_chrom)
        lines = self._vcf_lines(data_chrom, info_values, qual_values, filter_values)
        if file.suffix.lower() == ".gz":
            _write_bgzf_lines(file, lines)
        else:
            with file.open("w", encoding="utf-8") as out:
                out.writelines(lines)

    def _vcf_lines(self, data_chrom, info_values, qual_values, filter_values) -> Iterator[str]:
        genotypes = np.asarray(data_chrom.genotypes)
        n_windows = data_chrom.n_snps
        n_samples = data_chrom.n_samples
        has_samples = n_samples > 0
        yield "##fileformat=VCFv4.1\n"
        if has_samples:
            yield '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n'
        if any("END=" in s for s in info_values):
            yield '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the segment">\n'
        for c in set(data_chrom.variants_chrom):
            yield f"##contig=<ID={c}>\n"
        cols = ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO"]
        if has_samples:
            cols += ["FORMAT"] + list(data_chrom.samples)
        yield "\t".join(cols) + "\n"

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
            yield "\t".join(fields) + "\n"
