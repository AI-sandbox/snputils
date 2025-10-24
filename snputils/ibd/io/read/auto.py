from __future__ import annotations

from pathlib import Path
from typing import Union


class IBDReader:
    def __new__(
        cls,
        file: Union[str, Path]
    ) -> object:
        """
        A factory class that attempts to detect the IBD file format and returns the corresponding reader.

        Supported detections:
        - Hap-IBD: *.ibd or *.ibd.gz (headerless, 8 columns)
        - ancIBD: directories with `ch_all.tsv`/`ch*.tsv` or files *.tsv / *.tsv.gz with ancIBD schema
        """
        file = Path(file)
        suffixes = [s.lower() for s in file.suffixes]

        # Directory-based detection for ancIBD
        if file.is_dir():
            if (file / 'ch_all.tsv').exists() or (file / 'ch_all.tsv.gz').exists():
                from snputils.ibd.io.read.anc_ibd import AncIBDReader
                return AncIBDReader(file)
            has_chr_files = list(file.glob('ch*.tsv')) or list(file.glob('ch*.tsv.gz'))
            if has_chr_files:
                from snputils.ibd.io.read.anc_ibd import AncIBDReader
                return AncIBDReader(file)
            # Fallback to HapIBD if nothing matches
            from snputils.ibd.io.read.hap_ibd import HapIBDReader
            return HapIBDReader(file)

        # File-based detection
        if suffixes[-2:] == ['.ibd', '.gz'] or suffixes[-1:] == ['.ibd']:
            from snputils.ibd.io.read.hap_ibd import HapIBDReader
            return HapIBDReader(file)
        if suffixes[-2:] == ['.tsv', '.gz'] or suffixes[-1:] == ['.tsv']:
            from snputils.ibd.io.read.anc_ibd import AncIBDReader
            return AncIBDReader(file)

        # Default to HapIBDReader (most tools use .ibd[.gz])
        from snputils.ibd.io.read.hap_ibd import HapIBDReader
        return HapIBDReader(file)


