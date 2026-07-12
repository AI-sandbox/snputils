import numpy as np


def known_lai_mask(lai: np.ndarray) -> np.ndarray:
    values = np.asarray(lai)
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype(float, copy=False)
        return np.isfinite(numeric) & (numeric >= 0)

    normalized = np.char.lower(np.char.strip(values.astype(str)))
    return ~np.isin(normalized, ("", ".", "-1", "-1.0", "na", "nan", "none"))


def known_lai_values(lai: np.ndarray) -> np.ndarray:
    values = np.asarray(lai).ravel()
    if np.issubdtype(values.dtype, np.unsignedinteger):
        return values
    return values[known_lai_mask(values)]
