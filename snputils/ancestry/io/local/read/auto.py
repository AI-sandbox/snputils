from __future__ import annotations

import gzip
from pathlib import Path
from typing import Union


class LAIReader:
    def __new__(
        _cls,
        file: Union[str, Path]
    ) -> object:
        """
        A factory class that automatically detects the local ancestry data file format from the 
        file's extension and returns the corresponding reader object.

        **Supported formats:**

        - `.msp`: Text-based MSP format.
        - `.msp.tsv`: Text-based MSP format with TSV extension.
        - `.anc.vcf` / `.anc.vcf.gz`: FLARE local ancestry VCF output.
        - `.lanc`: admix-kit local ancestry change-point format.

        Args:
            file (str or pathlib.Path): 
                Path to the file to be read. It should end with `.msp` or `.msp.tsv`.
        
        Returns:
            **object:** A reader object corresponding to the file format (e.g., `MSPReader`).
        """
        file = Path(file)
        suffixes = [suffix.lower() for suffix in file.suffixes]
        if not suffixes:
            raise ValueError(
                "The file must have an extension. Supported extensions are: "
                ".msp, .msp.tsv, .anc.vcf, .anc.vcf.gz, .lanc."
            )

        if suffixes[-2:] == ['.msp', '.tsv'] or suffixes[-1] == '.msp':
            from snputils.ancestry.io.local.read.msp import MSPReader

            return MSPReader(file)
        if suffixes[-1] == '.lanc':
            from snputils.ancestry.io.local.read.lanc import LANCReader

            return LANCReader(file)
        if suffixes[-3:] == ['.anc', '.vcf', '.gz'] or suffixes[-2:] == ['.anc', '.vcf'] or suffixes[-2:] == ['.vcf', '.gz'] or suffixes[-1] == '.vcf':
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
                "Supported extensions are: .msp, .msp.tsv, .anc.vcf, .anc.vcf.gz, .lanc."
            )


def _looks_like_flare_vcf(file: Path) -> bool:
    opener = gzip.open if file.name.endswith(".gz") else open
    has_ancestry = False
    with opener(file, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith("##ANCESTRY="):
                has_ancestry = True
                continue
            if raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            return has_ancestry and len(fields) >= 10 and {"AN1", "AN2"}.issubset(set(fields[8].split(":")))
    return False
