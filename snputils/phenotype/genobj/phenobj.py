import copy
from typing import List, Optional, Sequence

import numpy as np


class PhenotypeObject:
    """
    Generic phenotype container for single-trait analyses.

    The object stores sample IDs, normalized phenotype values, inferred/declared
    trait type, and binary case/control convenience attributes.
    """

    def __init__(
        self,
        samples: Sequence[str],
        values: Sequence[float],
        phenotype_name: str = "PHENO",
        quantitative: Optional[bool] = None,
    ) -> None:
        sample_ids = [str(sample) for sample in samples]
        if len(sample_ids) == 0:
            raise ValueError("Phenotype file contains no samples.")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Phenotype sample IDs must be unique.")

        try:
            values_f64 = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Phenotype values must be numeric.") from exc

        if values_f64.ndim != 1:
            raise ValueError("Phenotype values must be a 1-dimensional array.")
        if values_f64.size != len(sample_ids):
            raise ValueError(
                "Phenotype sample/value length mismatch: "
                f"{len(sample_ids)} samples but {values_f64.size} values."
            )
        if not np.all(np.isfinite(values_f64)):
            raise ValueError("Phenotype contains non-finite values (NaN/Inf).")

        trait_is_quantitative = (
            self._infer_quantitative(values_f64)
            if quantitative is None
            else bool(quantitative)
        )

        if trait_is_quantitative:
            if float(np.var(values_f64)) <= 0.0:
                raise ValueError("Quantitative phenotype has zero variance.")
            normalized_values = values_f64
            cases: List[str] = []
            controls: List[str] = sample_ids.copy()
        else:
            normalized_values = self._normalize_binary(values_f64)
            case_mask = normalized_values == 1
            control_mask = normalized_values == 0
            cases = [sample_ids[idx] for idx in np.where(case_mask)[0].tolist()]
            controls = [sample_ids[idx] for idx in np.where(control_mask)[0].tolist()]
            if len(cases) == 0:
                raise ValueError("No case data available.")
            if len(controls) == 0:
                raise ValueError("No control data available.")

        self._samples = sample_ids
        self._values = normalized_values
        self._phenotype_name = str(phenotype_name)
        self._is_quantitative = trait_is_quantitative

        self._cases = cases
        self._controls = controls
        self._all_haplotypes = [f"{sample}.0" for sample in sample_ids] + [
            f"{sample}.1" for sample in sample_ids
        ]
        self._cases_haplotypes = [f"{sample}.0" for sample in cases] + [
            f"{sample}.1" for sample in cases
        ]
        self._controls_haplotypes = [f"{sample}.0" for sample in controls] + [
            f"{sample}.1" for sample in controls
        ]

    @staticmethod
    def _matches_binary_encoding(values_f64: np.ndarray, encoding: Sequence[float]) -> bool:
        unique_vals = np.unique(values_f64)
        if unique_vals.size != 2:
            return False
        target = np.asarray(sorted(float(v) for v in encoding), dtype=np.float64)
        observed = np.asarray(sorted(unique_vals.tolist()), dtype=np.float64)
        return bool(np.allclose(observed, target, rtol=0.0, atol=1e-8))

    @staticmethod
    def _infer_quantitative(values_f64: np.ndarray) -> bool:
        return not (
            PhenotypeObject._matches_binary_encoding(values_f64, (0.0, 1.0))
            or PhenotypeObject._matches_binary_encoding(values_f64, (1.0, 2.0))
        )

    @staticmethod
    def _normalize_binary(values_f64: np.ndarray) -> np.ndarray:
        unique_vals = np.unique(values_f64)
        if PhenotypeObject._matches_binary_encoding(values_f64, (1.0, 2.0)):
            return np.isclose(values_f64, 2.0, rtol=0.0, atol=1e-8).astype(np.int8)
        if PhenotypeObject._matches_binary_encoding(values_f64, (0.0, 1.0)):
            return values_f64.astype(np.int8)
        raise ValueError(
            "Binary phenotype must use exactly two levels encoded as {1,2} or {0,1}. "
            f"Observed unique values: {sorted(unique_vals.tolist())}"
        )

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(f"Invalid key: {key}") from exc

    def __setitem__(self, key, value):
        try:
            setattr(self, key, value)
        except AttributeError as exc:
            raise KeyError(f"Invalid key: {key}") from exc

    @property
    def samples(self) -> List[str]:
        return self._samples

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def y(self) -> np.ndarray:
        return self._values

    @property
    def phenotype_name(self) -> str:
        return self._phenotype_name

    @property
    def is_quantitative(self) -> bool:
        return self._is_quantitative

    @property
    def quantitative(self) -> bool:
        return self._is_quantitative

    @property
    def cases(self) -> List[str]:
        return self._cases

    @property
    def n_cases(self) -> int:
        return len(self._cases)

    @property
    def controls(self) -> List[str]:
        return self._controls

    @property
    def n_controls(self) -> int:
        return len(self._controls)

    @property
    def all_haplotypes(self) -> List[str]:
        return self._all_haplotypes

    @property
    def cases_haplotypes(self) -> List[str]:
        return self._cases_haplotypes

    @property
    def controls_haplotypes(self) -> List[str]:
        return self._controls_haplotypes

    def copy(self):
        return copy.copy(self)

    def keys(self) -> List[str]:
        return [
            "samples",
            "n_samples",
            "values",
            "y",
            "phenotype_name",
            "is_quantitative",
            "quantitative",
            "cases",
            "n_cases",
            "controls",
            "n_controls",
            "all_haplotypes",
            "cases_haplotypes",
            "controls_haplotypes",
        ]
