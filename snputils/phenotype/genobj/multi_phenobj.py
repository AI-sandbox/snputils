import copy
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from snputils._utils.printing import array_shape, format_repr


class MultiPhenotypeObject:
    """Sample-aligned table for multiple phenotypes or sample traits."""

    def __init__(
        self,
        phen_df: pd.DataFrame,
        sample_column: Optional[str] = None,
    ) -> None:
        normalized_df, resolved_sample_column = self._normalize_frame(
            phen_df,
            sample_column=sample_column,
        )
        self._phen_df = normalized_df
        self._sample_column = resolved_sample_column

    @staticmethod
    def _normalize_frame(
        phen_df: pd.DataFrame,
        *,
        sample_column: Optional[str],
    ) -> tuple[pd.DataFrame, str]:
        if not isinstance(phen_df, pd.DataFrame):
            raise TypeError("phen_df must be a pandas DataFrame.")
        if phen_df.empty:
            raise ValueError("MultiPhenotypeObject contains no samples.")
        if phen_df.shape[1] < 2:
            raise ValueError(
                "MultiPhenotypeObject requires a sample column and at least one phenotype column."
            )

        columns = [str(col) for col in phen_df.columns]
        if len(set(columns)) != len(columns):
            raise ValueError("Phenotype column names must be unique.")

        if sample_column is None:
            resolved = columns[0]
        else:
            resolved = str(sample_column)
            if resolved not in columns:
                raise ValueError(
                    f"Sample column '{resolved}' not found in phenotype table: {columns}"
                )

        ordered_columns = [resolved] + [col for col in columns if col != resolved]
        normalized = phen_df.loc[:, ordered_columns].copy()
        normalized.columns = ordered_columns

        sample_series = normalized.iloc[:, 0].astype(str).str.strip()
        if sample_series.eq("").any():
            raise ValueError("Phenotype sample IDs contain empty values.")
        if sample_series.duplicated().any():
            raise ValueError("Phenotype sample IDs must be unique.")
        normalized.iloc[:, 0] = sample_series

        return normalized.reset_index(drop=True), resolved

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

    def __repr__(self) -> str:
        return format_repr(
            self,
            shape=self.shape,
            n_samples=self.n_samples,
            n_phenotypes=self.n_phenotypes,
            sample_column=self.sample_column,
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def phen_df(self) -> pd.DataFrame:
        return self._phen_df

    @phen_df.setter
    def phen_df(self, x: pd.DataFrame):
        normalized_df, resolved_sample_column = self._normalize_frame(
            x,
            sample_column=self._sample_column,
        )
        self._phen_df = normalized_df
        self._sample_column = resolved_sample_column

    @property
    def sample_column(self) -> str:
        return self._sample_column

    @property
    def samples(self) -> List[str]:
        return self._phen_df.iloc[:, 0].astype(str).tolist()

    @property
    def n_samples(self) -> int:
        return len(self._phen_df)

    @property
    def phenotype_names(self) -> List[str]:
        return [str(col) for col in self._phen_df.columns[1:].tolist()]

    @property
    def n_phenotypes(self) -> int:
        return len(self.phenotype_names)

    @property
    def values(self) -> np.ndarray:
        return self._phen_df.iloc[:, 1:].to_numpy()

    @property
    def shape(self) -> tuple[int, int]:
        phen_shape = array_shape(self._phen_df)
        if phen_shape is None:
            return (self.n_samples, self.n_phenotypes + 1)
        return phen_shape

    def copy(self):
        return MultiPhenotypeObject(self._phen_df.copy(), sample_column=self._sample_column)

    def keys(self) -> List[str]:
        return [
            "phen_df",
            "sample_column",
            "samples",
            "n_samples",
            "phenotype_names",
            "n_phenotypes",
            "values",
        ]

    def filter_samples(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        indexes: Optional[Union[int, Sequence[int], np.ndarray]] = None,
        include: bool = True,
        reorder: bool = False,
        inplace: bool = False,
    ) -> Optional["MultiPhenotypeObject"]:
        """Filter rows by sample ID or row index."""
        if samples is None and indexes is None:
            raise ValueError("At least one of 'samples' or 'indexes' must be provided.")

        n_samples = self.n_samples
        sample_names = np.asarray(self.samples, dtype=object)

        if samples is not None:
            samples = np.asarray(samples).ravel().astype(object)
            mask_samples = np.isin(sample_names, samples)
        else:
            mask_samples = np.zeros(n_samples, dtype=bool)

        if indexes is not None:
            indexes = np.asarray(indexes).ravel()
            if np.any(indexes >= n_samples) or np.any(indexes < -n_samples):
                raise IndexError("One or more sample indexes are out of bounds.")
            indexes = np.mod(indexes, n_samples)
            mask_indexes = np.zeros(n_samples, dtype=bool)
            mask_indexes[indexes] = True
        else:
            mask_indexes = np.zeros(n_samples, dtype=bool)

        mask_combined = mask_samples | mask_indexes
        if not include:
            mask_combined = ~mask_combined

        ordered_indices = None
        if include and reorder:
            selected = np.where(mask_combined)[0]
            ordered_list = []
            added = np.zeros(n_samples, dtype=bool)

            if samples is not None:
                for sample_id in samples:
                    matches = np.where(sample_names == sample_id)[0]
                    for idx in matches:
                        if mask_combined[idx] and not added[idx]:
                            ordered_list.append(int(idx))
                            added[idx] = True

            if indexes is not None:
                for idx in np.mod(np.atleast_1d(indexes), n_samples):
                    if mask_combined[idx] and not added[idx]:
                        ordered_list.append(int(idx))
                        added[idx] = True

            for idx in selected:
                if not added[idx]:
                    ordered_list.append(int(idx))

            ordered_indices = np.asarray(ordered_list, dtype=int)

        result_df = (
            self._phen_df.iloc[ordered_indices].reset_index(drop=True)
            if ordered_indices is not None
            else self._phen_df.loc[mask_combined].reset_index(drop=True)
        )

        if inplace:
            self._phen_df = result_df
            return None
        return MultiPhenotypeObject(result_df, sample_column=self._sample_column)
