import logging
from io import StringIO
from pathlib import Path
from typing import Optional, Sequence, Union

import polars as pl
import numpy as np

from snputils.ibd.genobj.ibdobj import IBDObject
from snputils.ibd.io.read.base import IBDBaseReader


log = logging.getLogger(__name__)


class AncIBDReader(IBDBaseReader):
    """
    Reads IBD data from ancIBD outputs (TSV), accepting a file (`ch_all.tsv` or `ch*.tsv`) or a directory.
    """

    def read(
        self,
        path: Optional[Union[str, Path]] = None,
        include_segment_types: Optional[Sequence[str]] = ("IBD1", "IBD2"),
    ) -> IBDObject:
        """
        Read ancIBD outputs and convert to `IBDObject`.

        Inputs accepted:
        - A single TSV (optionally gzipped), e.g. `ch_all.tsv[.gz]` or `ch{CHR}.tsv[.gz]`.
        - A directory containing per-chromosome TSVs or `ch_all.tsv`.

        Column schema (tab-separated with header):
        iid1, iid2, ch, Start, End, length, StartM, EndM, lengthM, StartBP, EndBP, segment_type

        Notes:
        - Haplotype indices are not provided by ancIBD; set to -1.
        - Positions in IBDObject use base-pair StartBP/EndBP.
        - Length uses centiMorgan as `lengthM * 100`.

        Args:
            path (str or Path, optional): Override input path. Defaults to `self.file`.
            include_segment_types (sequence of str, optional): Filter by `segment_type` (e.g., IBD1, IBD2). None to disable.

        Returns:
            **IBDObject**: An IBDObject instance.
        """
        p = Path(path) if path is not None else Path(self.file)
        log.info(f"Reading ancIBD from {p}")

        files: list[Path]
        if p.is_dir():
            # Prefer combined file if present, else gather per-chromosome files
            combined = p / "ch_all.tsv"
            combined_gz = p / "ch_all.tsv.gz"
            if combined.exists():
                files = [combined]
            elif combined_gz.exists():
                files = [combined_gz]
            else:
                files = sorted(list(p.glob("ch*.tsv")) + list(p.glob("ch*.tsv.gz")))
                if not files:
                    raise FileNotFoundError("No ancIBD output files found in directory.")
        else:
            files = [p]

        frames = []
        schema_overrides = {
            "iid1": pl.Utf8,
            "iid2": pl.Utf8,
            "ch": pl.Utf8,
            "Start": pl.Int64,
            "End": pl.Int64,
            "length": pl.Int64,  # marker span; not used
            "StartM": pl.Float64,
            "EndM": pl.Float64,
            "lengthM": pl.Float64,
            "StartBP": pl.Int64,
            "EndBP": pl.Int64,
            "segment_type": pl.Utf8,
        }

        for f in files:
            frame = pl.read_csv(str(f), separator="\t", has_header=True, schema_overrides=schema_overrides)
            frames.append(frame)

        df = pl.concat(frames, how="vertical") if len(frames) > 1 else frames[0]

        if include_segment_types is not None:
            df = df.filter(pl.col("segment_type").is_in(list(include_segment_types)))

        # Map columns to IBDObject schema
        sample_id_1 = df["iid1"].to_numpy()
        sample_id_2 = df["iid2"].to_numpy()
        chrom = df["ch"].to_numpy()
        start_bp = df["StartBP"].to_numpy()
        end_bp = df["EndBP"].to_numpy()
        length_cm = (df["lengthM"] * 100.0).to_numpy()

        # ancIBD doesn't include haplotype indices; set to -1
        hap1 = np.full(sample_id_1.shape[0], -1, dtype=np.int8)
        hap2 = np.full(sample_id_2.shape[0], -1, dtype=np.int8)

        ibdobj = IBDObject(
            sample_id_1=sample_id_1,
            haplotype_id_1=hap1,
            sample_id_2=sample_id_2,
            haplotype_id_2=hap2,
            chrom=chrom,
            start=start_bp,
            end=end_bp,
            length_cm=length_cm,
            segment_type=df["segment_type"].to_numpy(),
        )

        log.info(f"Finished reading ancIBD from {p}")
        return ibdobj


