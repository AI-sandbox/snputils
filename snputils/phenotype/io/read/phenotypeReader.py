import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .base import PhenotypeBaseReader
from snputils.phenotype.genobj import PhenotypeObject

class PhenotypeReader(PhenotypeBaseReader):
    """
    Reader for single-trait phenotype files (any extension; common: .txt, .phe, .pheno).

    Expected format (headered, whitespace-delimited):
      - Must include `IID` (optionally preceded by `FID`)
      - First phenotype column after `IID` is used by default
    """

    def __init__(self, file: Union[str, Path]) -> None:
        super().__init__(file)

    @property
    def file(self) -> Path:
        return Path(self._file)

    @staticmethod
    def _has_header_with_iid(file_path: Path) -> bool:
        with open(file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                tokens = line.split()
                return any(token.lstrip("#").upper() == "IID" for token in tokens)
        raise ValueError("Empty phenotype file.")

    @staticmethod
    def _resolve_column(columns, normalized_columns, requested: str) -> Optional[str]:
        requested_norm = str(requested).lstrip("#").upper()
        for col, col_norm in zip(columns, normalized_columns):
            if str(col) == str(requested) or col_norm == requested_norm:
                return str(col)
        return None

    def read(
        self,
        phenotype_col: Optional[str] = None,
        quantitative: Optional[bool] = None,
    ) -> PhenotypeObject:
        file_path = self.file
        if not file_path.exists():
            raise FileNotFoundError(f"Phenotype file not found: '{file_path}'")

        has_iid_header = self._has_header_with_iid(file_path)
        if has_iid_header:
            phen_df = pd.read_csv(file_path, sep=r"\s+", dtype=str)
        else:
            warnings.warn(
                (
                    "Phenotype file has no header/IID column. Legacy 3-column parsing "
                    "(FID IID PHENO) is deprecated; please switch to a headered format."
                ),
                UserWarning,
                stacklevel=2,
            )
            legacy = pd.read_csv(file_path, header=None, sep=r"\s+", dtype=str)
            if legacy.shape[1] < 3:
                raise ValueError(
                    "Legacy phenotype parsing expects at least 3 columns: FID IID PHENO."
                )
            phen_df = legacy.iloc[:, :3].copy()
            phen_df.columns = ["FID", "IID", "PHENO"]

        if phen_df.empty:
            raise ValueError("Empty phenotype file.")

        columns = [str(col) for col in phen_df.columns]
        normalized_columns = [col.lstrip("#").upper() for col in columns]
        if "IID" not in normalized_columns:
            raise ValueError("Phenotype file must include an IID column in the header.")
        iid_col = columns[normalized_columns.index("IID")]

        iid_series = phen_df[iid_col].astype(str).str.strip()
        if iid_series.eq("").any():
            raise ValueError("Phenotype IID column contains empty values.")
        if iid_series.duplicated().any():
            raise ValueError("Phenotype IID values must be unique.")

        if phenotype_col is not None:
            resolved = self._resolve_column(columns, normalized_columns, phenotype_col)
            if resolved is None:
                raise ValueError(
                    f"Phenotype column '{phenotype_col}' not found in header: {columns}"
                )
            target_col = resolved
        else:
            iid_idx = normalized_columns.index("IID")
            if iid_idx + 1 >= len(columns):
                raise ValueError(
                    "Phenotype file must include at least one phenotype column after IID."
                )
            target_col = columns[iid_idx + 1]

        values = pd.to_numeric(phen_df[target_col], errors="coerce")
        if values.isna().any():
            bad_examples = phen_df.loc[values.isna(), target_col].astype(str).head(5).tolist()
            raise ValueError(
                f"Phenotype column '{target_col}' contains non-numeric or missing values: "
                f"{bad_examples}"
            )

        phenotype_name = str(target_col).lstrip("#")
        return PhenotypeObject(
            samples=iid_series.tolist(),
            values=values.to_numpy(),
            phenotype_name=phenotype_name,
            quantitative=quantitative,
        )


PhenotypeBaseReader.register(PhenotypeReader)
