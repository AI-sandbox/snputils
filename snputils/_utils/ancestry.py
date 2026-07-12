import numpy as np


def known_lai_values(lai: np.ndarray) -> np.ndarray:
    values = np.asarray(lai).ravel()
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype(float, copy=False)
        return values[np.isfinite(numeric) & (numeric >= 0)]

    normalized = np.char.lower(np.char.strip(values.astype(str)))
    missing = np.isin(normalized, ("", ".", "-1", "-1.0", "na", "nan", "none"))
    return values[~missing]
