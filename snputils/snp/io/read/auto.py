from __future__ import annotations

import pathlib
from typing import Union


class SNPReader:
    def __new__(_cls,
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

        compression_suffix = suffixes[-1].lower() if suffixes[-1].lower() in (".zst", ".gz") else None
        if compression_suffix is not None and len(suffixes) < 2:
            raise ValueError("A format extension must precede the compression extension.")
        extension = suffixes[-2] if compression_suffix is not None else suffixes[-1]
        extension = extension.lower()

        if compression_suffix is not None and extension in (".bed", ".pgen", ".bgen", ".bcf"):
            raise ValueError(
                f"Outer {compression_suffix} compression is not supported for native binary "
                f"{extension} files. Provide the native {extension} file directly."
            )

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
        elif extension in (".pgen", ".pvar", ".psam"):
            from snputils.snp.io.read.pgen import PGENReader

            return PGENReader(filename)
        elif extension == ".bgen":
            from snputils.snp.io.read.bgen import BGENReader

            return BGENReader(filename)
        elif extension == ".bcf":
            from snputils.snp.io.read.bcf import BCFReader

            return BCFReader(filename)
        else:
            raise ValueError(f"File format not supported: {filename}")
