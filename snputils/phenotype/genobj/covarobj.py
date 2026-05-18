from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from snputils._utils.printing import array_shape, format_repr


class CovariateObject:
    """Sample-aligned covariate matrix for association analyses."""

    def __init__(
        self,
        samples: Sequence[str],
        values: Sequence[Sequence[float]],
        covariate_names: Optional[Sequence[str]] = None,
    ) -> None:
        sample_ids = [str(sample) for sample in samples]
        if len(sample_ids) == 0:
            raise ValueError("CovariateObject contains no samples.")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Covariate sample IDs must be unique.")

        try:
            values_f64 = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Covariate values must be numeric.") from exc

        if values_f64.ndim != 2:
            raise ValueError("Covariate values must be a 2-dimensional matrix.")
        if values_f64.shape[0] != len(sample_ids):
            raise ValueError(
                "Covariate sample/value length mismatch: "
                f"{len(sample_ids)} samples but {values_f64.shape[0]} rows."
            )
        if values_f64.shape[1] == 0:
            raise ValueError("CovariateObject must contain at least one covariate column.")
        if not np.all(np.isfinite(values_f64)):
            raise ValueError("Covariates contain non-finite values (NaN/Inf).")

        if covariate_names is None:
            names = [f"COV{i + 1}" for i in range(values_f64.shape[1])]
        else:
            names = [str(name) for name in covariate_names]
            if len(names) != values_f64.shape[1]:
                raise ValueError(
                    "Covariate name/value width mismatch: "
                    f"{len(names)} names but {values_f64.shape[1]} columns."
                )
            if len(set(names)) != len(names):
                raise ValueError("Covariate names must be unique.")

        self._samples = sample_ids
        self._values = values_f64
        self._covariate_names = names

    def __repr__(self) -> str:
        return format_repr(
            self,
            shape=self.shape,
            n_samples=self.n_samples,
            n_covariates=self.n_covariates,
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def samples(self) -> list[str]:
        return self._samples

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def covariate_names(self) -> list[str]:
        return self._covariate_names

    @property
    def names(self) -> list[str]:
        return self._covariate_names

    @property
    def shape(self) -> tuple[int, ...]:
        return array_shape(self._values) or self._values.shape

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def n_covariates(self) -> int:
        return int(self._values.shape[1])
