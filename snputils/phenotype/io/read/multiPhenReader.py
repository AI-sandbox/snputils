import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .base import PhenotypeBaseReader
from .phenotypeReader import PhenotypeReader
from snputils.phenotype.genobj import MultiPhenotypeObject

log = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = (".xlsx", ".csv", ".tsv", ".txt", ".phe", ".pheno", ".map", ".smap", ".phen")


class MultiPhenReader(PhenotypeBaseReader):
    """Reader for headered multi-phenotype tables with an ``IID`` column."""

    def __init__(self, file: Union[str, Path]) -> None:
        super().__init__(file)

    @property
    def file(self) -> Path:
        return Path(self._file)

    def _read_raw_table(self, sep: str, header: Optional[int]) -> pd.DataFrame:
        file_extension = os.path.splitext(self.file)[1]
        log.info("Reading '%s' file from '%s'...", file_extension, self.file)

        if file_extension == ".xlsx":
            return pd.read_excel(self.file, header=header, index_col=None)
        if file_extension == ".csv":
            return pd.read_csv(self.file, sep=sep, header=header)
        if file_extension in [".map", ".smap"]:
            return pd.read_csv(self.file, sep=sep, header=header)
        if file_extension == ".tsv":
            return pd.read_csv(self.file, sep="\t", header=header)
        if file_extension in [".txt", ".phe", ".pheno"]:
            return pd.read_csv(self.file, sep=r"\s+", header=header)
        if file_extension == ".phen":
            with open(self.file, "r", encoding="utf-8") as handle:
                contents = [line.split() for line in handle if line.strip()]
            if len(contents) < 2:
                raise ValueError("Empty phenotype file.")
            return pd.DataFrame(contents[1:], columns=["IID", "PHENO"])
        raise ValueError(
            f"Unsupported file extension {file_extension}. Supported extensions: {SUPPORTED_EXTENSIONS}."
        )

    def read(
        self,
        samples_idx: int = 0,
        phen_names: Optional[List[str]] = None,
        sep: str = ",",
        header: Optional[int] = 0,
        drop: bool = False,
    ) -> "MultiPhenotypeObject":
        """Read a multi-phenotype table using the same ``IID`` convention as `read_pheno()`."""
        if not self.file.exists():
            raise FileNotFoundError(f"Phenotype file not found: '{self.file}'")
        if drop:
            raise ValueError(
                "`drop` is not supported in the IID-based reader API. "
                "Select columns before writing the file or after reading the object."
            )

        has_iid_header = PhenotypeReader._has_header_with_iid(self.file)
        if header is None and has_iid_header:
            raise ValueError("header=None is not supported for headered phenotype files with IID.")
        if not has_iid_header:
            raise ValueError("Phenotype file must include an IID column in the header.")

        phen_df = self._read_raw_table(sep=sep, header=header)

        if phen_df.empty:
            raise ValueError("Empty phenotype file.")

        columns = [str(col) for col in phen_df.columns]
        normalized_columns = [col.lstrip("#").upper() for col in columns]
        if "IID" not in normalized_columns:
            raise ValueError("Phenotype file must include an IID column in the header.")

        iid_idx = normalized_columns.index("IID")
        if samples_idx != 0 and samples_idx != iid_idx:
            raise ValueError(
                "`samples_idx` no longer selects an arbitrary sample column; "
                "the input must contain an IID column."
            )

        iid_col = columns[iid_idx]
        phenotype_candidates = columns[iid_idx + 1 :]
        if not phenotype_candidates:
            raise ValueError(
                "Phenotype file must include at least one phenotype column after IID."
            )

        out_df = phen_df.loc[:, [iid_col] + phenotype_candidates].copy()
        out_df.columns = ["IID"] + phenotype_candidates

        if phen_names is not None:
            renamed = [str(name) for name in phen_names]
            if len(renamed) != len(phenotype_candidates):
                raise ValueError(
                    f"Mismatch between number of phenotype columns ({len(phenotype_candidates)}) "
                    f"and length of `phen_names` ({len(renamed)})."
                )
            out_df.columns = ["IID"] + renamed

        return MultiPhenotypeObject(phen_df=out_df, sample_column="IID")


PhenotypeBaseReader.register(MultiPhenReader)
