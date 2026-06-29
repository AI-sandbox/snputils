import logging
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from .base import LAIBaseWriter

log = logging.getLogger(__name__)


class LANCWriter(LAIBaseWriter):
    """
    Writer for admix-kit `.lanc` local ancestry files.

    The format stores SNP-level diploid local ancestry only. Sample IDs,
    ancestry labels, chromosomes, and positions are not encoded and are
    therefore not preserved in the output.
    """

    def __init__(self, laiobj, file: Union[str, Path]) -> None:
        self.__laiobj = laiobj
        self.__file = Path(file)

    @property
    def laiobj(self):
        return self.__laiobj

    @property
    def file(self) -> Path:
        return self.__file

    @file.setter
    def file(self, x: Union[str, Path]):
        self.__file = Path(x)

    def _ensure_path(self) -> None:
        if self.file.suffix.lower() != ".lanc":
            self.file = self.file.with_name(self.file.name + ".lanc")

    def _coerce_lai(self) -> np.ndarray:
        lai = np.asarray(self.laiobj.lai)
        if lai.ndim != 2 or lai.shape[1] % 2 != 0:
            raise ValueError("LocalAncestryObject.lai must have shape (n_windows, 2 * n_samples).")

        if not np.issubdtype(lai.dtype, np.integer):
            if not np.issubdtype(lai.dtype, np.floating):
                raise ValueError("LocalAncestryObject.lai must contain integer ancestry codes.")
            if not np.all(np.isfinite(lai)):
                raise ValueError("LocalAncestryObject.lai must contain finite ancestry codes.")
            if not np.all(np.equal(lai, np.floor(lai))):
                raise ValueError("LocalAncestryObject.lai must contain integer ancestry codes.")

        lai_int = lai.astype(np.int64, copy=False)
        if np.any(lai_int < 0) or np.any(lai_int > 9):
            raise ValueError(
                ".lanc output requires single-digit ancestry codes in the inclusive range [0, 9]."
            )
        return lai_int

    def write(self) -> None:
        self._ensure_path()
        if self.file.exists():
            warnings.warn(f"File '{self.file}' already exists and will be overwritten.")

        lai = self._coerce_lai()
        n_windows, n_haplotypes = lai.shape
        n_samples = n_haplotypes // 2

        log.info("Writing LANC local ancestry to '%s'...", self.file)
        lines = [f"{n_windows} {n_samples}"]
        for sample_idx in range(n_samples):
            sample_lai = lai[:, (2 * sample_idx):(2 * sample_idx + 2)]
            if n_windows == 0:
                lines.append("")
                continue

            change_mask = np.any(sample_lai[1:] != sample_lai[:-1], axis=1)
            break_ends = np.concatenate([np.where(change_mask)[0] + 1, np.array([n_windows])])
            segment_values = sample_lai[break_ends - 1]
            tokens = [
                f"{int(stop)}:{int(value[0])}{int(value[1])}"
                for stop, value in zip(break_ends.tolist(), segment_values.tolist())
            ]
            lines.append(" ".join(tokens))

        with open(self.file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


LAIBaseWriter.register(LANCWriter)
