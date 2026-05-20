import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO, Union

import numpy as np

from .base import LAIBaseReader
from snputils.ancestry.genobj.local import LocalAncestryObject

log = logging.getLogger(__name__)


@dataclass
class FLAREMetadata:
    header: List[str]
    samples: List[str]
    haplotypes: List[str]
    ancestry_map: Optional[Dict[str, str]]
    format_fields: List[str]
    an1_index: int
    an2_index: int


def _open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class FLAREReader(LAIBaseReader):
    """
    Reader for FLARE local ancestry output VCF files (`.anc.vcf.gz`).

    FLARE stores hard-call local ancestry in the `AN1` and `AN2` FORMAT
    subfields for each sample. The reader exposes those marker-level records as
    rows in a :class:`snputils.ancestry.genobj.LocalAncestryObject`, using
    `[POS, POS]` as physical positions.
    """

    def __init__(self, file: Union[str, Path]) -> None:
        self.__file = Path(file)

    @property
    def file(self) -> Path:
        return self.__file

    def _parse_ancestry_map(self, line: str) -> Dict[str, str]:
        match = re.match(r"##ANCESTRY=<(.+)>", line.strip())
        if match is None:
            return {}

        ancestry_map: Dict[str, str] = {}
        for token in match.group(1).split(","):
            if "=" not in token:
                continue
            name, code = token.split("=", 1)
            ancestry_map[code.strip()] = name.strip()
        return ancestry_map

    def read_metadata(self) -> FLAREMetadata:
        ancestry_map: Optional[Dict[str, str]] = None
        header: Optional[List[str]] = None
        first_data_format: Optional[List[str]] = None

        with _open_text(self.file, "rt") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("##ANCESTRY="):
                    parsed_map = self._parse_ancestry_map(line)
                    ancestry_map = parsed_map if parsed_map else None
                    continue
                if line.startswith("#CHROM"):
                    header = line.split("\t")
                    continue
                if line.startswith("#"):
                    continue
                if header is None:
                    raise ValueError("FLARE VCF header line '#CHROM' was not found before data records.")
                fields = line.split("\t")
                if len(fields) < 10:
                    raise ValueError("FLARE VCF records must contain FORMAT and sample columns.")
                first_data_format = fields[8].split(":")
                break

        if header is None:
            raise ValueError("FLARE VCF header line '#CHROM' was not found.")
        if len(header) < 10:
            raise ValueError("FLARE VCF must contain at least one sample column.")
        if first_data_format is None:
            raise ValueError("FLARE VCF contains no data records.")
        if "AN1" not in first_data_format or "AN2" not in first_data_format:
            raise ValueError("FLARE VCF FORMAT must contain AN1 and AN2 local ancestry fields.")

        samples = header[9:]
        haplotypes = [f"{sample}.{phase}" for sample in samples for phase in (0, 1)]
        return FLAREMetadata(
            header=header,
            samples=samples,
            haplotypes=haplotypes,
            ancestry_map=ancestry_map,
            format_fields=first_data_format,
            an1_index=first_data_format.index("AN1"),
            an2_index=first_data_format.index("AN2"),
        )

    def _sample_indices_to_columns(
        self,
        n_samples: int,
        sample_indices: Optional[np.ndarray],
    ) -> np.ndarray:
        if sample_indices is None:
            return np.arange(n_samples, dtype=np.int64)

        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if sample_indices.size == 0:
            raise ValueError("sample_indices cannot be empty.")
        if np.any(sample_indices < 0) or np.any(sample_indices >= n_samples):
            raise ValueError("sample_indices contain out-of-bounds sample indexes.")
        return sample_indices

    def iter_windows(
        self,
        chunk_size: int = 1024,
        sample_indices: Optional[np.ndarray] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        metadata = self.read_metadata()
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        selected_samples = self._sample_indices_to_columns(len(metadata.samples), sample_indices)
        n_selected_haps = int(selected_samples.size * 2)

        row_in_chunk = 0
        window_start = 0
        chromosomes_chunk = np.empty(int(chunk_size), dtype=object)
        physical_pos_chunk = np.empty((int(chunk_size), 2), dtype=np.int64)
        lai_chunk = np.empty((int(chunk_size), n_selected_haps), dtype=np.uint8)

        with _open_text(self.file, "rt") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if not raw_line or raw_line.startswith("#"):
                    continue
                line = raw_line.rstrip("\n")
                if not line:
                    continue

                fields = line.split("\t")
                if len(fields) < 9 + len(metadata.samples):
                    raise ValueError(
                        f"Malformed FLARE VCF row at line {line_no}: expected "
                        f"{9 + len(metadata.samples)} fields, got {len(fields)}."
                    )

                fmt = fields[8].split(":")
                try:
                    an1_index = fmt.index("AN1")
                    an2_index = fmt.index("AN2")
                except ValueError as exc:
                    raise ValueError(f"FLARE VCF row at line {line_no} is missing AN1/AN2 FORMAT fields.") from exc

                chromosomes_chunk[row_in_chunk] = fields[0]
                pos = int(fields[1])
                physical_pos_chunk[row_in_chunk, 0] = pos
                physical_pos_chunk[row_in_chunk, 1] = pos

                out_col = 0
                for sample_idx in selected_samples:
                    sample_field = fields[9 + int(sample_idx)].split(":")
                    if len(sample_field) <= max(an1_index, an2_index):
                        raise ValueError(
                            f"Malformed FLARE sample field at line {line_no}: "
                            "not enough FORMAT values for AN1/AN2."
                        )
                    lai_chunk[row_in_chunk, out_col] = np.uint8(sample_field[an1_index])
                    lai_chunk[row_in_chunk, out_col + 1] = np.uint8(sample_field[an2_index])
                    out_col += 2

                row_in_chunk += 1
                if row_in_chunk == chunk_size:
                    yield {
                        "window_indexes": np.arange(window_start, window_start + row_in_chunk, dtype=np.int64),
                        "chromosomes": chromosomes_chunk,
                        "physical_pos": physical_pos_chunk,
                        "lai": lai_chunk,
                    }
                    window_start += row_in_chunk
                    row_in_chunk = 0
                    chromosomes_chunk = np.empty(int(chunk_size), dtype=object)
                    physical_pos_chunk = np.empty((int(chunk_size), 2), dtype=np.int64)
                    lai_chunk = np.empty((int(chunk_size), n_selected_haps), dtype=np.uint8)

        if row_in_chunk > 0:
            yield {
                "window_indexes": np.arange(window_start, window_start + row_in_chunk, dtype=np.int64),
                "chromosomes": chromosomes_chunk[:row_in_chunk],
                "physical_pos": physical_pos_chunk[:row_in_chunk],
                "lai": lai_chunk[:row_in_chunk],
            }

    def read(self) -> LocalAncestryObject:
        log.info("Reading FLARE local ancestry VCF '%s'...", self.file)
        metadata = self.read_metadata()

        chromosomes: List[str] = []
        positions: List[int] = []
        lai_rows: List[np.ndarray] = []

        for chunk in self.iter_windows():
            chromosomes.extend(chunk["chromosomes"].astype(str).tolist())
            positions.extend(chunk["physical_pos"][:, 0].astype(int).tolist())
            lai_rows.append(chunk["lai"])

        if not lai_rows:
            raise ValueError("FLARE VCF contains no local ancestry records.")

        lai = np.vstack(lai_rows).astype(np.uint8, copy=False)
        physical_pos = np.column_stack([positions, positions]).astype(np.int64, copy=False)

        return LocalAncestryObject(
            haplotypes=metadata.haplotypes,
            lai=lai,
            samples=metadata.samples,
            ancestry_map=metadata.ancestry_map,
            window_sizes=np.ones(lai.shape[0], dtype=np.int64),
            centimorgan_pos=None,
            chromosomes=np.asarray(chromosomes, dtype=object),
            physical_pos=physical_pos,
        )


LAIBaseReader.register(FLAREReader)
