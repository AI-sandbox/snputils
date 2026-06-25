import logging
import gzip
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from snputils.snp.genobj import SNPObject

log = logging.getLogger(__name__)


class VCFWriter:
    """
    A writer class for exporting SNP data from a `snputils.snp.genobj.SNPObject` 
    into an `.vcf` file.
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
                If True, genotype data is written in "maternal|paternal" format.  
                If False, genotype data is written in "maternal/paternal" format.
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
        Writes the SNP data to VCF file(s).

        Args:
            chrom_partition (bool, optional):
                If True, individual VCF files are generated for each chromosome.
                If False, a single VCF file containing data for all chromosomes is created. Defaults to False.
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

        # Optionally rename potential missing values in `snpobj.genotypes` before writing
        if rename_missing_values:
            self.__snpobj.rename_missings(before=before, after=after, inplace=True)

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
        self, chrom, data_chrom, variants_info: Optional[Sequence[str]] = None
    ):
        """
        Writes the SNP data for a specific chromosome to a VCF file.

        Args:
            chrom: The chromosome name.
            data_chrom: The SNPObject instance containing the data for the chromosome.
            variants_info: Optional per-variant INFO strings; length must match variant count.
        """
        npy3 = data_chrom.genotypes
        n_windows, n_samples, _ = npy3.shape

        if chrom == "All":
            file = self.__filename.with_suffix(self.__file_extension)
        else:
            file = self.__filename.parent / f"{self.__filename.stem}_{chrom}{self.__file_extension}"

        if self.__file_extension == ".vcf.gz":
            out = gzip.open(file, "wt", encoding="utf-8")
        else:
            out = open(file, "w")
        out.write("##fileformat=VCFv4.1\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')
        if variants_info is not None and any("END=" in s for s in variants_info):
            out.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the segment">\n')
        for c in set(data_chrom.variants_chrom):
            out.write(f"##contig=<ID={c}>\n")
        cols = ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"] + list(data_chrom.samples)
        out.write("\t".join(cols) + "\n")

        sep = "|" if self.__phased else "/"
        for i in range(n_windows):
            chrom_val = data_chrom.variants_chrom[i]
            pos = data_chrom.variants_pos[i]
            vid = data_chrom.variants_id[i]
            ref = data_chrom.variants_ref[i]
            alt = data_chrom.variants_alt[i]
            info_str = variants_info[i] if variants_info is not None else "."
            row = npy3[i]
            genotypes = [
                f"{row[s,0]}{sep}{row[s,1]}"
                for s in range(n_samples)
            ]
            line = "\t".join([
                str(chrom_val), str(pos), vid, ref, alt,
                ".", "PASS", info_str, "GT", *genotypes
            ])
            out.write(line + "\n")
        out.close()
