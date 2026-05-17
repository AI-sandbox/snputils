from pathlib import Path
from typing import Union

import pandas as pd


def read_labels(file: Union[str, Path], sep: str = "\t") -> pd.DataFrame:
    """Read an individual-label table used by dimensionality-reduction plots.

    The returned DataFrame has ``indID`` coerced to string and must include at
    least ``indID`` and ``label`` columns.
    """
    labels = pd.read_csv(file, sep=sep)
    required = {"indID", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels file is missing required column(s): {sorted(missing)}")
    labels = labels.copy()
    labels["indID"] = labels["indID"].astype(str)
    return labels
