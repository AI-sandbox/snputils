import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import LAIBaseReader
from snputils.ancestry.genobj.local import LocalAncestryObject

log = logging.getLogger(__name__)


def _open_textfile(path: Path):
    if path.suffix.lower() == ".zst":
        import zstandard as zstd

        return zstd.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _detect_separator(line: str) -> str:
    return "\t" if "\t" in line else r"\s+"


@dataclass
class LANCMetadata:
    n_windows: int
    samples: List[str]
    haplotypes: List[str]
    ancestry_map: Optional[Dict[str, str]]
    chromosomes: Optional[np.ndarray]
    physical_pos: Optional[np.ndarray]
    centimorgan_pos: Optional[np.ndarray]
    window_sizes: np.ndarray


class LANCReader(LAIBaseReader):
    """
    Reader for admix-kit `.lanc` local ancestry files.

    By default the reader looks for sibling `.pvar`/`.pvar.zst` and `.psam`
    files with the same prefix as the `.lanc` file. When present, those tables
    are used to recover SNP coordinates and sample IDs. If either table is not
    available, the reader falls back to loading the local ancestry matrix alone
    and warns that the missing metadata can be provided explicitly.
    """

    def __init__(
        self,
        file: Union[str, Path],
        *,
        pvar_file: Optional[Union[str, Path]] = None,
        psam_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.__file = Path(file)
        self._default_pvar_file = None if pvar_file is None else Path(pvar_file)
        self._default_psam_file = None if psam_file is None else Path(psam_file)
        self._metadata: Optional[LANCMetadata] = None
        self._metadata_key: Optional[Tuple[Optional[Path], Optional[Path]]] = None
        self._segments: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None

    @property
    def file(self) -> Path:
        return self.__file

    def _fallback_samples(self, n_samples: int) -> List[str]:
        return [f"sample_{i}" for i in range(n_samples)]

    def _make_haplotypes(self, samples: List[str]) -> List[str]:
        return [f"{sample}.{phase}" for sample in samples for phase in (0, 1)]

    def _resolve_sidecar_paths(
        self,
        *,
        pvar_file: Optional[Union[str, Path]] = None,
        psam_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        explicit_pvar = self._default_pvar_file if pvar_file is None else Path(pvar_file)
        explicit_psam = self._default_psam_file if psam_file is None else Path(psam_file)

        if explicit_pvar is not None:
            pvar_path = explicit_pvar if explicit_pvar.exists() else None
        else:
            candidate = self.file.with_suffix(".pvar")
            if candidate.exists():
                pvar_path = candidate
            else:
                candidate_zst = self.file.with_suffix(".pvar.zst")
                pvar_path = candidate_zst if candidate_zst.exists() else None

        if explicit_psam is not None:
            psam_path = explicit_psam if explicit_psam.exists() else None
        else:
            candidate = self.file.with_suffix(".psam")
            psam_path = candidate if candidate.exists() else None

        return pvar_path, psam_path

    def _warn_missing_metadata(
        self,
        *,
        pvar_path: Optional[Path],
        psam_path: Optional[Path],
    ) -> None:
        missing: List[str] = []
        if pvar_path is None:
            missing.append("pvar")
        if psam_path is None:
            missing.append("psam")
        if not missing:
            return

        missing_text = " and ".join(missing)
        warnings.warn(
            f"No {missing_text} sidecar file found for '{self.file}'. Please specify "
            f"{', '.join(f'{name}_file' for name in missing)} to reconstruct that info. "
            "Loading LAI calls without the missing SNP-level and/or sample-level metadata.",
            stacklevel=3,
        )

    def _read_psam_samples(self, path: Path, n_samples_expected: int) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        has_header = first_line.startswith(("#FID", "FID", "#IID", "IID"))

        psam = pd.read_csv(
            path,
            sep="\t",
            header=0 if has_header else None,
            names=None if has_header else ["FID", "IID", "PAT", "MAT", "SEX", "PHENO1"],
            dtype=str,
        )

        if "#IID" in psam.columns:
            psam = psam.rename(columns={"#IID": "IID"})
        if "IID" not in psam.columns:
            raise ValueError(f"PSAM file '{path}' does not contain an IID column.")

        samples = psam["IID"].astype(str).tolist()
        if len(samples) != n_samples_expected:
            raise ValueError(
                f"PSAM sample count ({len(samples)}) must match .lanc n_indiv ({n_samples_expected})."
            )
        return samples

    def _read_pvar_metadata(
        self,
        path: Path,
        n_windows_expected: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        header_line_num = 0
        has_header = True
        separator = "\t"
        header: List[str] = []

        with _open_textfile(path) as handle:
            for line_num, line in enumerate(handle):
                if line.startswith("##"):
                    continue
                separator = _detect_separator(line)
                if line.startswith("#CHROM"):
                    header_line_num = line_num
                    header = line.strip().split()
                    has_header = True
                    break
                if not line.startswith("#"):
                    has_header = False
                    cols = len(line.strip().split("\t" if separator == "\t" else None))
                    if cols >= 6:
                        header = ["#CHROM", "POS", "ID", "REF", "ALT", "CM"][:cols]
                    elif cols == 5:
                        header = ["#CHROM", "POS", "ID", "REF", "ALT"]
                    else:
                        raise ValueError(f"PVAR file '{path}' is not a valid .pvar file.")
                    break

        read_kwargs = {
            "sep": separator,
            "skiprows": header_line_num,
            "header": 0 if has_header else None,
            "names": None if has_header else header,
            "dtype": str,
        }
        if separator != "\t":
            read_kwargs["engine"] = "python"
        if path.suffix.lower() == ".zst":
            read_kwargs["compression"] = "zstd"

        pvar = pd.read_csv(path, **read_kwargs)
        if "#CHROM" not in pvar.columns or "POS" not in pvar.columns:
            raise ValueError(f"PVAR file '{path}' must contain '#CHROM' and 'POS' columns.")
        if len(pvar) != n_windows_expected:
            raise ValueError(
                f"PVAR variant count ({len(pvar)}) must match .lanc n_snp ({n_windows_expected})."
            )

        chromosomes = pvar["#CHROM"].astype(str).to_numpy(dtype=object)
        positions = pd.to_numeric(pvar["POS"], errors="raise").to_numpy(dtype=np.int64, copy=False)
        physical_pos = np.column_stack([positions, positions]).astype(np.int64, copy=False)

        centimorgan_pos: Optional[np.ndarray]
        cm_col = "CM" if "CM" in pvar.columns else None
        if cm_col is not None:
            cm = pd.to_numeric(pvar[cm_col], errors="coerce").to_numpy(dtype=float, copy=False)
            if np.isnan(cm).all():
                centimorgan_pos = None
            else:
                centimorgan_pos = np.column_stack([cm, cm]).astype(float, copy=False)
        else:
            centimorgan_pos = None

        return chromosomes, physical_pos, centimorgan_pos

    def read_metadata(
        self,
        *,
        pvar_file: Optional[Union[str, Path]] = None,
        psam_file: Optional[Union[str, Path]] = None,
    ) -> LANCMetadata:
        pvar_path, psam_path = self._resolve_sidecar_paths(pvar_file=pvar_file, psam_file=psam_file)
        metadata_key = (pvar_path, psam_path)
        if self._metadata is not None and self._metadata_key == metadata_key:
            return self._metadata

        with open(self.file, "r", encoding="utf-8") as handle:
            header = handle.readline().strip().split()

        if len(header) != 2:
            raise ValueError(
                "Malformed .lanc header: expected '<n_snp> <n_indiv>' on the first line."
            )

        try:
            n_windows = int(header[0])
            n_samples = int(header[1])
        except ValueError as exc:
            raise ValueError(
                "Malformed .lanc header: n_snp and n_indiv must be integers."
            ) from exc

        if n_windows < 0:
            raise ValueError("Malformed .lanc header: n_snp must be >= 0.")
        if n_samples < 0:
            raise ValueError("Malformed .lanc header: n_indiv must be >= 0.")

        if pvar_path is None or psam_path is None:
            self._warn_missing_metadata(pvar_path=pvar_path, psam_path=psam_path)

        samples = (
            self._read_psam_samples(psam_path, n_samples)
            if psam_path is not None
            else self._fallback_samples(n_samples)
        )
        haplotypes = self._make_haplotypes(samples)

        if pvar_path is not None:
            chromosomes, physical_pos, centimorgan_pos = self._read_pvar_metadata(
                pvar_path, n_windows
            )
        else:
            chromosomes = None
            physical_pos = None
            centimorgan_pos = None

        self._metadata = LANCMetadata(
            n_windows=n_windows,
            samples=samples,
            haplotypes=haplotypes,
            ancestry_map=None,
            chromosomes=chromosomes,
            physical_pos=physical_pos,
            centimorgan_pos=centimorgan_pos,
            window_sizes=np.ones(n_windows, dtype=np.int64),
        )
        self._metadata_key = metadata_key
        return self._metadata

    def _parse_token(self, token: str, *, line_no: int, n_windows: int) -> Tuple[int, np.ndarray]:
        stop_str, sep, value_str = token.partition(":")
        if sep != ":":
            raise ValueError(
                f"Malformed .lanc token at line {line_no}: expected '<stop>:<anc0><anc1>', got {token!r}."
            )

        try:
            stop = int(stop_str)
        except ValueError as exc:
            raise ValueError(
                f"Malformed .lanc token at line {line_no}: stop must be an integer, got {stop_str!r}."
            ) from exc

        if stop < 0:
            raise ValueError(
                f"Malformed .lanc token at line {line_no}: stop must be >= 0, got {stop}."
            )
        if stop > n_windows:
            raise ValueError(
                f"Malformed .lanc token at line {line_no}: stop {stop} exceeds n_snp={n_windows}."
            )
        if len(value_str) != 2 or not value_str.isdigit():
            raise ValueError(
                f"Malformed .lanc token at line {line_no}: ancestry payload must be two digits, got {value_str!r}."
            )

        return stop, np.array([int(value_str[0]), int(value_str[1])], dtype=np.uint8)

    def _read_segments(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self._segments is not None:
            return self._segments

        metadata = self.read_metadata()
        breaks: List[np.ndarray] = []
        values: List[np.ndarray] = []

        with open(self.file, "r", encoding="utf-8") as handle:
            _ = handle.readline()
            for line_no, raw_line in enumerate(handle, start=2):
                line = raw_line.strip()
                if not line:
                    continue

                indiv_breaks: List[int] = []
                indiv_values: List[np.ndarray] = []
                last_stop = 0
                for token in line.split():
                    stop, anc = self._parse_token(token, line_no=line_no, n_windows=metadata.n_windows)
                    if stop < last_stop:
                        raise ValueError(
                            f"Malformed .lanc line {line_no}: stop positions must be non-decreasing."
                        )
                    indiv_breaks.append(stop)
                    indiv_values.append(anc)
                    last_stop = stop

                if not indiv_breaks:
                    raise ValueError(f"Malformed .lanc line {line_no}: expected at least one segment.")
                if indiv_breaks[-1] != metadata.n_windows:
                    raise ValueError(
                        f"Malformed .lanc line {line_no}: final stop must equal n_snp={metadata.n_windows}."
                    )

                breaks.append(np.asarray(indiv_breaks, dtype=np.int64))
                values.append(np.vstack(indiv_values).astype(np.uint8, copy=False))

        if len(breaks) != len(metadata.samples):
            raise ValueError(
                "Malformed .lanc file: number of individual lines does not match header "
                f"n_indiv={len(metadata.samples)}."
            )

        self._segments = (breaks, values)
        return self._segments

    def iter_windows(
        self,
        chunk_size: int = 1024,
        sample_indices: Optional[np.ndarray] = None,
        *,
        pvar_file: Optional[Union[str, Path]] = None,
        psam_file: Optional[Union[str, Path]] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        metadata = self.read_metadata(pvar_file=pvar_file, psam_file=psam_file)
        breaks, values = self._read_segments()

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        n_samples = len(metadata.samples)
        if sample_indices is None:
            selected_samples = np.arange(n_samples, dtype=np.int64)
        else:
            selected_samples = np.asarray(sample_indices, dtype=np.int64)
            if selected_samples.size == 0:
                raise ValueError("sample_indices cannot be empty.")
            if np.any(selected_samples < 0) or np.any(selected_samples >= n_samples):
                raise ValueError("sample_indices contain out-of-bounds sample indexes.")

        n_selected_haps = int(selected_samples.size * 2)
        for start in range(0, metadata.n_windows, int(chunk_size)):
            stop = min(start + int(chunk_size), metadata.n_windows)
            chunk_len = stop - start
            lai_chunk = np.empty((chunk_len, n_selected_haps), dtype=np.uint8)
            out_col = 0

            for sample_idx in selected_samples.tolist():
                indiv_breaks = breaks[int(sample_idx)]
                indiv_values = values[int(sample_idx)]
                seg_idx = int(np.searchsorted(indiv_breaks, start, side="right"))
                pos = start

                while pos < stop:
                    if seg_idx >= indiv_breaks.size:
                        raise ValueError(
                            f"Malformed .lanc data for sample index {sample_idx}: segments ended before n_snp."
                        )
                    seg_stop = min(int(indiv_breaks[seg_idx]), stop)
                    fill_start = pos - start
                    fill_stop = seg_stop - start
                    lai_chunk[fill_start:fill_stop, out_col] = indiv_values[seg_idx, 0]
                    lai_chunk[fill_start:fill_stop, out_col + 1] = indiv_values[seg_idx, 1]
                    pos = seg_stop
                    seg_idx += 1

                out_col += 2

            yield {
                "window_indexes": np.arange(start, stop, dtype=np.int64),
                "chromosomes": (
                    metadata.chromosomes[start:stop]
                    if metadata.chromosomes is not None
                    else np.full(chunk_len, ".", dtype=object)
                ),
                "physical_pos": (
                    metadata.physical_pos[start:stop]
                    if metadata.physical_pos is not None
                    else None
                ),
                "lai": lai_chunk,
            }

    def read(
        self,
        *,
        pvar_file: Optional[Union[str, Path]] = None,
        psam_file: Optional[Union[str, Path]] = None,
    ) -> LocalAncestryObject:
        log.info("Reading LANC local ancestry '%s'...", self.file)
        metadata = self.read_metadata(pvar_file=pvar_file, psam_file=psam_file)

        lai_rows: List[np.ndarray] = []
        for chunk in self.iter_windows(pvar_file=pvar_file, psam_file=psam_file):
            lai_rows.append(chunk["lai"])

        if metadata.n_windows == 0:
            lai = np.empty((0, len(metadata.haplotypes)), dtype=np.uint8)
        elif not lai_rows:
            raise ValueError("Malformed .lanc file: no individual ancestry data found.")
        else:
            lai = np.vstack(lai_rows).astype(np.uint8, copy=False)

        return LocalAncestryObject(
            haplotypes=metadata.haplotypes,
            lai=lai,
            samples=metadata.samples,
            ancestry_map=None,
            window_sizes=metadata.window_sizes,
            centimorgan_pos=metadata.centimorgan_pos,
            chromosomes=metadata.chromosomes,
            physical_pos=metadata.physical_pos,
        )


LAIBaseReader.register(LANCReader)
