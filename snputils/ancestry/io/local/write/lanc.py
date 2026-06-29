import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .base import LAIBaseWriter

log = logging.getLogger(__name__)


class LANCWriter(LAIBaseWriter):
    """
    Writer for admix-kit `.lanc` local ancestry files.

    The `.lanc` stream stores SNP-level diploid local ancestry only. To make
    snputils round-trips self-contained, the writer emits matching `.psam` and
    `.pvar` sidecars by default when enough metadata is available on the
    `LocalAncestryObject`.
    """

    def __init__(
        self,
        laiobj,
        file: Union[str, Path],
        *,
        write_sidecars: bool = True,
        pvar_file: Optional[Union[str, Path]] = None,
        psam_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.__laiobj = laiobj
        self.__file = Path(file)
        self.__write_sidecars = bool(write_sidecars)
        self.__pvar_file = None if pvar_file is None else Path(pvar_file)
        self.__psam_file = None if psam_file is None else Path(psam_file)

    @property
    def laiobj(self):
        return self.__laiobj

    @property
    def file(self) -> Path:
        return self.__file

    @file.setter
    def file(self, x: Union[str, Path]):
        self.__file = Path(x)

    @property
    def write_sidecars(self) -> bool:
        return self.__write_sidecars

    def _ensure_path(self) -> None:
        if self.file.suffix.lower() != ".lanc":
            self.file = self.file.with_name(self.file.name + ".lanc")

    def _sidecar_path(self, explicit: Optional[Path], suffix: str) -> Path:
        if explicit is not None:
            return explicit
        return self.file.with_suffix(suffix)

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

    def _resolve_samples(self) -> List[str]:
        if self.laiobj.samples is not None:
            return [str(sample) for sample in self.laiobj.samples]
        if self.laiobj.haplotypes is not None:
            return [str(hap).rsplit(".", 1)[0] for hap in self.laiobj.haplotypes[0::2]]
        return [f"sample_{i}" for i in range(self.laiobj.n_samples)]

    def _write_psam(self, samples: List[str]) -> None:
        psam_path = self._sidecar_path(self.__psam_file, ".psam")
        if psam_path.exists():
            warnings.warn(f"File '{psam_path}' already exists and will be overwritten.")

        log.info("Writing LANC PSAM sidecar to '%s'...", psam_path)
        lines = ["#IID"] + samples
        with open(psam_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def _write_pvar(self, n_windows: int) -> None:
        if self.laiobj.chromosomes is None or self.laiobj.physical_pos is None:
            warnings.warn(
                "LANCWriter could not emit a .pvar sidecar because LocalAncestryObject "
                "is missing chromosomes and/or physical_pos. The .lanc file was written, "
                "but SNP coordinate metadata will not round-trip unless you provide it separately."
            )
            return

        chromosomes = np.asarray(self.laiobj.chromosomes, dtype=object)
        physical_pos = np.asarray(self.laiobj.physical_pos)
        if chromosomes.shape[0] != n_windows:
            raise ValueError("LocalAncestryObject.chromosomes length must match n_windows.")
        if physical_pos.shape != (n_windows, 2):
            raise ValueError("LocalAncestryObject.physical_pos must have shape (n_windows, 2).")

        positions = physical_pos[:, 0].astype(np.int64, copy=False)
        ids = [f"{chrom}:{int(pos)}" for chrom, pos in zip(chromosomes.tolist(), positions.tolist())]

        centimorgan_pos = None
        if self.laiobj.centimorgan_pos is not None:
            centimorgan_pos = np.asarray(self.laiobj.centimorgan_pos)
            if centimorgan_pos.shape != (n_windows, 2):
                raise ValueError("LocalAncestryObject.centimorgan_pos must have shape (n_windows, 2).")

        pvar_path = self._sidecar_path(self.__pvar_file, ".pvar")
        if pvar_path.exists():
            warnings.warn(f"File '{pvar_path}' already exists and will be overwritten.")

        log.info("Writing LANC PVAR sidecar to '%s'...", pvar_path)
        lines = ["##fileformat=VCFv4.2", "##source=snputils"]
        if centimorgan_pos is not None:
            lines.append("#CHROM\tPOS\tID\tREF\tALT\tCM")
            for chrom, pos, vid, cm in zip(
                chromosomes.tolist(),
                positions.tolist(),
                ids,
                centimorgan_pos[:, 0].tolist(),
            ):
                cm_text = "." if cm is None or (isinstance(cm, float) and np.isnan(cm)) else str(cm)
                lines.append(f"{chrom}\t{int(pos)}\t{vid}\tN\t.\t{cm_text}")
        else:
            lines.append("#CHROM\tPOS\tID\tREF\tALT")
            for chrom, pos, vid in zip(chromosomes.tolist(), positions.tolist(), ids):
                lines.append(f"{chrom}\t{int(pos)}\t{vid}\tN\t.")

        with open(pvar_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

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

        if self.write_sidecars:
            samples = self._resolve_samples()
            if len(samples) != n_samples:
                raise ValueError("Resolved sample identifiers must match LocalAncestryObject.n_samples.")
            self._write_psam(samples)
            self._write_pvar(n_windows)


LAIBaseWriter.register(LANCWriter)
