from typing import Optional, Sequence, Union

import numpy as np


def coerce_sex_codes(
    sample_sex: Optional[Union[np.ndarray, Sequence[Union[str, int]]]],
    n_samples: int,
    *,
    missing_code: str,
) -> np.ndarray:
    if sample_sex is None:
        return np.repeat(missing_code, n_samples)
    sex = np.asarray(sample_sex)
    if sex.ndim == 0 or sex.shape[0] != n_samples:
        observed = 0 if sex.ndim == 0 else sex.shape[0]
        raise ValueError(f"sample_sex length ({observed}) must match number of samples ({n_samples}).")
    mapping = {
        "1": "1",
        "m": "1",
        "male": "1",
        "2": "2",
        "f": "2",
        "female": "2",
        "0": missing_code,
        "u": missing_code,
        "unknown": missing_code,
        "": missing_code,
        ".": missing_code,
        "nan": missing_code,
        "none": missing_code,
        "na": missing_code,
    }
    return np.asarray([mapping.get(str(value).strip().lower(), missing_code) for value in sex], dtype=object)
