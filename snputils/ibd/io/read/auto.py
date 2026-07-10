from __future__ import annotations

from pathlib import Path
from typing import Union


class IBDReader:
    def __new__(
        _cls,
        file: Union[str, Path]
    ) -> object:
        """
        Detect hap-IBD or ancIBD input and return the corresponding reader.
        """
        file = Path(file)
        suffixes = [s.lower() for s in file.suffixes]

        # Directory-based detection for ancIBD
        if file.is_dir():
            if any((file / name).exists() for name in ('ch_all.tsv', 'ch_all.tsv.gz', 'ch_all.tsv.zst')):
                from snputils.ibd.io.read.anc_ibd import AncIBDReader
                return AncIBDReader(file)
            has_chr_files = (
                list(file.glob('ch*.tsv'))
                or list(file.glob('ch*.tsv.gz'))
                or list(file.glob('ch*.tsv.zst'))
            )
            if has_chr_files:
                from snputils.ibd.io.read.anc_ibd import AncIBDReader
                return AncIBDReader(file)
            # Fallback to HapIBD if nothing matches
            from snputils.ibd.io.read.hap_ibd import HapIBDReader
            return HapIBDReader(file)

        # File-based detection
        if (
            suffixes[-1:] == ['.ibd']
            or suffixes[-2:] in (['.ibd', '.gz'], ['.ibd', '.zst'])
        ):
            from snputils.ibd.io.read.hap_ibd import HapIBDReader
            return HapIBDReader(file)
        if (
            suffixes[-1:] == ['.tsv']
            or suffixes[-2:] in (['.tsv', '.gz'], ['.tsv', '.zst'])
        ):
            from snputils.ibd.io.read.anc_ibd import AncIBDReader
            return AncIBDReader(file)

        # Default to HapIBDReader (most tools use .ibd[.gz])
        from snputils.ibd.io.read.hap_ibd import HapIBDReader
        return HapIBDReader(file)
