import logging
import re
from io import StringIO
from typing import Optional

import numpy as np
import polars as pl

from snputils.ibd.genobj.ibdobj import IBDObject
from snputils.ibd.io.read.base import IBDBaseReader, open_text


log = logging.getLogger(__name__)


class HapIBDReader(IBDBaseReader):
    """
    Reads an IBD file in Hap-IBD format and processes it into an `IBDObject`.
    """

    def read(self, separator: Optional[str] = None) -> IBDObject:
        """
        Read a Hap-IBD file into an `IBDObject`.

        The Hap-IBD format is a delimited text without a header with columns:
        sample_id_1, haplotype_id_1, sample_id_2, haplotype_id_2, chromosome, start, end, length_cm

        Notes:
        - Haplotype identifiers are 1-based and take values in {1, 2}.

        Args:
            separator (str, optional): Field delimiter. If None, whitespace (any number of spaces or tabs) is assumed.

        Returns:
            **IBDObject**: An IBDObject instance.
        """
        log.info(f"Reading {self.file}")

        # Column names for Hap-IBD files (no header present in input)
        col_names = [
            'sample_id_1', 'haplotype_id_1', 'sample_id_2', 'haplotype_id_2',
            'chrom', 'start', 'end', 'length_cm'
        ]

        if separator is None:
            # Polars does not support regex separators, so normalize arbitrary
            # whitespace to tabs before parsing.
            with open_text(self.file) as handle:
                lines = [
                    re.sub(r"\s+", "\t", line.strip())
                    for line in handle
                    if line.strip()
                ]
            source = StringIO("\n".join(lines))
            parse_separator = "\t"
        else:
            source = str(self.file)
            parse_separator = separator

        df = pl.read_csv(
            source=source,
            has_header=False,
            separator=parse_separator,
            new_columns=col_names,
            schema_overrides={
                'sample_id_1': pl.Utf8,
                'haplotype_id_1': pl.Int8,
                'sample_id_2': pl.Utf8,
                'haplotype_id_2': pl.Int8,
                'chrom': pl.Utf8,
                'start': pl.Int64,
                'end': pl.Int64,
                'length_cm': pl.Float64,
            },
        )

        ibdobj = IBDObject(
            sample_id_1=df['sample_id_1'].to_numpy(),
            haplotype_id_1=df['haplotype_id_1'].to_numpy(),
            sample_id_2=df['sample_id_2'].to_numpy(),
            haplotype_id_2=df['haplotype_id_2'].to_numpy(),
            chrom=df['chrom'].to_numpy(),
            start=df['start'].to_numpy(),
            end=df['end'].to_numpy(),
            length_cm=df['length_cm'].to_numpy(),
            segment_type=np.array(["IBD1"] * df.height),  # hap-IBD does not distinguish; treat as IBD1
        )

        log.info(f"Finished reading {self.file}")

        return ibdobj
