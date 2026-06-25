import logging
import gzip
import struct
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from snputils.snp.genobj import SNPObject

log = logging.getLogger(__name__)

def _encode_typed_string(s: str) -> bytes:
    if not s:
        return b"\x07"
    s_bytes = s.encode("utf-8")
    n = len(s_bytes)
    if n < 15:
        return bytes([(n << 4) | 7]) + s_bytes
    if n <= 127:
        len_enc = b"\x11" + bytes([n])
    elif n <= 32767:
        len_enc = b"\x12" + struct.pack("<h", n)
    else:
        len_enc = b"\x13" + struct.pack("<i", n)
    return b"\xf7" + len_enc + s_bytes

def _encode_typed_int_list(ints: list[int]) -> bytes:
    n = len(ints)
    if n == 0:
        return b"\x01"
    max_val = max(ints)
    min_val = min(ints)
    if min_val >= -120 and max_val <= 127:
        type_code = 1
        val_bytes = bytes([x & 0xFF for x in ints])
    elif min_val >= -32760 and max_val <= 32767:
        type_code = 2
        val_bytes = b"".join(struct.pack("<h", x) for x in ints)
    else:
        type_code = 3
        val_bytes = b"".join(struct.pack("<i", x) for x in ints)
    
    if n < 15:
        return bytes([(n << 4) | type_code]) + val_bytes
    if n <= 127:
        len_enc = b"\x11" + bytes([n])
    elif n <= 32767:
        len_enc = b"\x12" + struct.pack("<h", n)
    else:
        len_enc = b"\x13" + struct.pack("<i", n)
    return bytes([(15 << 4) | type_code]) + len_enc + val_bytes

class BCFWriter:
    """
    A writer class for exporting SNP data from a `snputils.snp.genobj.SNPObject` 
    into a `.bcf` file.
    """
    def __init__(self, snpobj: SNPObject, filename: str, n_jobs: int = -1, phased: bool = False):
        """
        Args:
            snpobj (SNPObject):
                A SNPObject instance.
            filename (str or pathlib.Path): 
                Path to the file where the data will be saved. It should end with `.bcf`. 
                If the provided path does not have this extension, the `.bcf` extension will be appended.
            n_jobs: 
                Number of jobs to run in parallel. Unused, included for API consistency.
            phased: 
                If True, genotype data is written in phased format.  
                If False, genotype data is written in unphased format.
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
        Writes the SNP data to BCF file(s).

        Args:
            chrom_partition (bool, optional):
                If True, individual BCF files are generated for each chromosome.
                If False, a single BCF file containing data for all chromosomes is created. Defaults to False.
            rename_missing_values (bool, optional):
                If True, renames potential missing values in `snpobj.genotypes` before writing.
                Defaults to True.
            before (int, float, or str, default=-1):
                The current representation of missing values in `genotypes`. Common values might be -1, '.', or NaN.
                Default is -1.
            after (int, float, or str, default='.'):
                The value that will replace `before`. Default is '.'.
            variants_info (sequence of str, optional):
                Per-variant INFO column values (e.g. ``["END=2000", "END=3000"]``). Length must match variant count.
                When provided, a ##INFO header line for END is written if any value contains ``END=``.
        """
        self.__chrom_partition = chrom_partition

        if self.__filename.suffix != ".bcf":
            self.__filename = self.__filename.with_suffix(".bcf")

        # BCF binary format requires integer genotypes (where missing is -1),
        # so we do not rename missing values to string representations.
        pass

        data = self.__snpobj

        if self.__chrom_partition:
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
        self, chrom: str, data_chrom: SNPObject, variants_info: Optional[Sequence[str]] = None
    ):
        n_records = data_chrom.genotypes.shape[0] if data_chrom.genotypes is not None else len(data_chrom.variants_pos)
        n_samples = len(data_chrom.samples) if data_chrom.samples is not None else 0

        if chrom == "All":
            file = self.__filename
        else:
            file = self.__filename.parent / f"{self.__filename.stem}_{chrom}.bcf"

        # Unique chromosomes/contigs list
        unique_chroms = []
        seen = set()
        for c in data_chrom.variants_chrom:
            if c not in seen:
                seen.add(c)
                unique_chroms.append(c)

        # Build header lines
        header_lines = [
            "##fileformat=VCFv4.3",
            '##FILTER=<ID=PASS,Description="All filters passed",IDX=0>',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype",IDX=1>'
        ]
        if variants_info is not None and any("END=" in s for s in variants_info):
            header_lines.append('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the segment",IDX=2>')

        for c in unique_chroms:
            header_lines.append(f"##contig=<ID={c}>")

        cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + [str(sample) for sample in data_chrom.samples]
        header_lines.append("\t".join(cols))

        header_text = "\n".join(header_lines) + "\n"
        header_bytes = header_text.encode("utf-8") + b"\0"
        header_len = len(header_bytes)

        with gzip.open(file, "wb") as out:
            # 1. Magic
            out.write(b"BCF\x02\x02")
            # 2. Header length
            out.write(struct.pack("<I", header_len))
            # 3. Header text
            out.write(header_bytes)

            # 4. Records
            for i in range(n_records):
                chrom_val = data_chrom.variants_chrom[i]
                chrom_id = unique_chroms.index(chrom_val)
                pos = int(data_chrom.variants_pos[i]) - 1
                ref = data_chrom.variants_ref[i]
                rlen = len(ref)
                
                if data_chrom.variants_qual is not None and not np.isnan(data_chrom.variants_qual[i]):
                    qual_val = float(data_chrom.variants_qual[i])
                else:
                    # Reinterpret 0x7F800001 (BCF missing float sentinel bits) as a float
                    qual_val = struct.unpack("<f", b"\x01\x00\x80\x7f")[0]

                alt_str = data_chrom.variants_alt[i]
                if not alt_str or alt_str == ".":
                    alts = []
                else:
                    alts = alt_str.split(",")

                n_alleles = 1 + len(alts)

                # Parse END info
                n_info = 0
                info_bytes = b""
                if variants_info is not None:
                    info_str = variants_info[i]
                    if info_str and "END=" in info_str:
                        match = re.search(r"END=(\d+)", info_str)
                        if match:
                            end_val = int(match.group(1))
                            n_info = 1
                            # INFO key: END (IDX = 2)
                            key_bytes = _encode_typed_int_list([2])
                            val_bytes = _encode_typed_int_list([end_val])
                            info_bytes = key_bytes + val_bytes

                # Genotypes / FORMAT
                if data_chrom.genotypes is not None:
                    n_fmt = 1
                    gt_array = np.zeros((n_samples, 2), dtype=np.uint8)
                    row = data_chrom.genotypes[i]
                    missing = row < 0
                    gt_array[~missing] = (row[~missing] + 1) << 1
                    if self.__phased:
                        gt_array[~missing[:, 1], 1] |= 1
                    
                    # FORMAT key index: GT (IDX = 1)
                    fmt_key_bytes = _encode_typed_int_list([1])
                    # FORMAT value descriptor: (2 values per sample, type 1 int8)
                    fmt_desc_bytes = b"\x21"
                    indiv_bytes = fmt_key_bytes + fmt_desc_bytes + gt_array.tobytes()
                else:
                    n_fmt = 0
                    indiv_bytes = b""

                # Shared section fields
                id_bytes = _encode_typed_string(data_chrom.variants_id[i] if data_chrom.variants_id[i] != "." else "")
                ref_bytes = _encode_typed_string(ref)
                alt_bytes = b"".join(_encode_typed_string(alt) for alt in alts)
                filter_bytes = _encode_typed_int_list([0]) # PASS (IDX = 0)

                shared_bytes = struct.pack(
                    "<iiIfII",
                    chrom_id,
                    pos,
                    rlen,
                    qual_val,
                    (n_alleles << 16) | n_info,
                    (n_fmt << 24) | n_samples
                ) + id_bytes + ref_bytes + alt_bytes + filter_bytes + info_bytes

                l_shared = len(shared_bytes)
                l_indiv = len(indiv_bytes)

                out.write(struct.pack("<II", l_shared, l_indiv))
                out.write(shared_bytes)
                out.write(indiv_bytes)
