from __future__ import annotations

import pathlib
from typing import Union


class SNPReader:
    def __new__(cls,
                filename: Union[str, pathlib.Path],
                vcf_backend: str = 'default') -> SNPReader:
        """
        Automatically detect the SNP file format from the file extension, and return its corresponding reader.

        Args:
            filename: Filename of the file to read.
            vcf_backend: Backend to use for reading the VCF file. Options are 'default' or 'polars'. Default is 'default'.

        Raises:
            ValueError: If the filename does not have an extension or the extension is not supported.
        """
        filename = pathlib.Path(filename)
        suffixes = filename.suffixes
        if not suffixes:
            raise ValueError("The filename should have an extension when using SNPReader.")

        extension = suffixes[-2] if suffixes[-1].lower() in (".zst", ".gz") else suffixes[-1]
        extension = extension.lower()

        if extension == ".vcf":
            if vcf_backend == 'default':
                from snputils.snp.io.read.vcf import VCFReader

                return VCFReader(filename)
            elif vcf_backend == 'polars':
                from snputils.snp.io.read.vcf import VCFReaderPolars

                return VCFReaderPolars(filename)
            else:
                raise ValueError(f"VCF backend not supported: {vcf_backend}")
        elif extension in (".bed", ".bim", ".fam"):
            from snputils.snp.io.read.bed import BEDReader

            return BEDReader(filename)
        elif extension in (".pgen", ".pvar", ".psam", ".pvar.zst"):
            from snputils.snp.io.read.pgen import PGENReader

            return PGENReader(filename)
        elif extension == ".bgen":
            from snputils.snp.io.read.bgen import BGENReader

            return BGENReader(filename)
        else:
            raise ValueError(f"File format not supported: {filename}")
