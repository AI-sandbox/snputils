from __future__ import annotations

from pathlib import Path



_PLINK1_BED_SUFFIXES = {".bed", ".bim", ".fam"}


def validate_phased_simulation_input(path: str | Path) -> None:
    """Reject formats that cannot preserve haplotype phase for mosaic simulation."""
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if suffixes and suffixes[-1] in _PLINK1_BED_SUFFIXES:
        raise ValueError(
            "The simulation pipeline requires phased haplotypes. PLINK1 BED/BIM/FAM "
            "files do not store phase, so they cannot be used as simulator input. "
            "Use phased VCF, PGEN, or BGEN input instead."
        )
