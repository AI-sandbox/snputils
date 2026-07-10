from __future__ import annotations

import gzip
from pathlib import Path
from typing import TextIO, Union


_COMPRESSION_SUFFIXES = ('.gz', '.zst')


def _open_text(file: Path) -> TextIO:
    if file.suffix.lower() == '.gz':
        return gzip.open(file, 'rt', encoding='utf-8')
    if file.suffix.lower() == '.zst':
        import zstandard as zstd

        return zstd.open(file, 'rt', encoding='utf-8')
    return open(file, 'rt', encoding='utf-8')


class LAIReader:
    def __new__(
        _cls,
        file: Union[str, Path]
    ) -> object:
        """
        A factory class that automatically detects the local ancestry data file format from the
        file's extension and returns the corresponding reader object.

        **Supported formats:**

        Detect MSP (`.msp`, `.msp.tsv`), FLARE (`.anc.vcf`), or admix-kit
        LANC (`.lanc`).

        Args:
            file (str or pathlib.Path):
                Path to the file to be read.

        Returns:
            **object:** A reader object corresponding to the file format (e.g., `MSPReader`).
        """
        file = Path(file)
        suffixes = [suffix.lower() for suffix in file.suffixes]
        if not suffixes:
            raise ValueError(
                "The file must have an extension. Supported extensions are: "
                ".msp, .msp.tsv, .anc.vcf, and .lanc, optionally followed by .gz or .zst."
            )

        format_suffixes = suffixes[:-1] if suffixes[-1] in _COMPRESSION_SUFFIXES else suffixes
        if not format_suffixes:
            raise ValueError(
                f"Unsupported file extension: {suffixes[-1]}. A format extension must precede "
                "the compression extension."
            )

        if format_suffixes[-2:] == ['.msp', '.tsv'] or format_suffixes[-1] == '.msp':
            from snputils.ancestry.io.local.read.msp import MSPReader

            return MSPReader(file)
        if format_suffixes[-1] == '.lanc':
            from snputils.ancestry.io.local.read.lanc import LANCReader

            return LANCReader(file)
        if format_suffixes[-1] == '.vcf':
            if not _looks_like_flare_vcf(file):
                raise ValueError(
                    f"VCF file '{file}' does not look like FLARE local ancestry output. "
                    "Expected ##ANCESTRY metadata and AN1/AN2 FORMAT fields."
                )
            from snputils.ancestry.io.local.read.flare import FLAREReader

            return FLAREReader(file)
        else:
            raise ValueError(
                f"Unsupported file extension: {suffixes[-1]}. "
                "Supported extensions are: .msp, .msp.tsv, .anc.vcf, and .lanc, "
                "optionally followed by .gz or .zst."
            )


def _looks_like_flare_vcf(file: Path) -> bool:
    has_ancestry = False
    with _open_text(file) as handle:
        for raw_line in handle:
            if raw_line.startswith("##ANCESTRY="):
                has_ancestry = True
                continue
            if raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            return has_ancestry and len(fields) >= 10 and {"AN1", "AN2"}.issubset(set(fields[8].split(":")))
    return False
