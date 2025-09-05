import copy
import warnings
import numpy as np
import pandas as pd
from typing import Union, Sequence, Optional


class MultiPhenotypeObject():
    """
    A class for multi-phenotype data.

    This class serves as a container for phenotype data, allowing for
    operations such as filtering samples and accessing phenotype information.
    It uses a DataFrame to store the data, with the first column reserved for the sample identifers.
    """
    def __init__(
        self,
        phen_df: pd.DataFrame
    ) -> None:
        """
        Args:
            phen_df (pd.DataFrame): 
                A Pandas DataFrame containing phenotype data, with the first column 
                representing sample identifiers.
        """
        self.__phen_df = phen_df

    def __getitem__(self, key):
        """
        To access an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            return getattr(self, key)
        except:
            raise KeyError(f'Invalid key: {key}')

    def __setitem__(self, key, value):
        """
        To set an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            setattr(self, key, value)
        except AttributeError:
            raise KeyError(f'Invalid key: {key}')

    @property
    def phen_df(self) -> pd.DataFrame:
        """
        Retrieve `phen_df`.

        Returns:
            pd.DataFrame: 
                A Pandas DataFrame containing phenotype data, with the first column 
                representing sample identifiers.
        """
        return self.__phen_df
    
    @phen_df.setter
    def phen_df(self, x: pd.DataFrame):
        """
        Update `phen_df`.
        """
        self.__phen_df = x
    
    @property
    def n_samples(self) -> int:
        """
        Retrieve `n_samples`.

        Returns:
            int: The total number of samples.
        """
        return len(self.phen_df)

    def copy(self):
        """
        Create and return a copy of the current `MultiPhenotypeObject` instance.

        Returns:
            MultiPhenotypeObject: A new instance of the current object.
        """
        return copy.copy(self)
    
    def filter_samples(
            self, 
            samples: Optional[Union[str, Sequence[str], np.ndarray]] = None, 
            indexes: Optional[Union[int, Sequence[int], np.ndarray]] = None, 
            include: bool = True, 
            reorder: bool = False, 
            inplace: bool = False
        ) -> Optional['MultiPhenotypeObject']:
        """
        Filter samples in the `MultiPhenotypeObject` based on sample names or indexes.

        This method allows you to include or exclude specific samples by their names,
        indexes, or both. When both samples and indexes are provided, the union of
        the specified samples is used. Negative indexes are supported and follow NumPy's indexing 
        conventions. Set `reorder=True` to match the ordering of the provided `samples` and/or
        `indexes` lists when including.

        Args:
            samples (str or array_like of str, optional): 
                 Names of the samples to include or exclude. Can be a single sample name or a
                 sequence of sample names. Default is None.
            indexes (int or array_like of int, optional):
                Indexes of the samples to include or exclude. Can be a single index or a sequence
                of indexes. Negative indexes are supported. Default is None.
            include (bool, default=True): 
                If True, includes only the specified samples. If False, excludes the specified
                samples. Default is True.
            inplace (bool, default=False): 
                If True, modifies the object in place. If False, returns a new
                `MultiPhenotypeObject` with the samples filtered. Default is False.

        Returns:
            Optional[MultiPhenotypeObject]: Returns a new MultiPhenotypeObject with the specified samples 
            filtered if `inplace=False`. If `inplace=True`, modifies the object in place and returns None.
        """
        # Ensure at least one of samples or indexes is provided
        if samples is None and indexes is None:
            raise ValueError("At least one of 'samples' or 'indexes' must be provided.")

        n_samples = self.n_samples

        # Create mask based on sample names
        if samples is not None:
            samples = np.asarray(samples).ravel()
            # Extract sample names from the DataFrame
            sample_names = self.__phen_df.iloc[:, 0].values
            # Create mask for samples belonging to specified names
            mask_samples = np.isin(sample_names, samples)
        else:
            mask_samples = np.zeros(n_samples, dtype=bool)

        # Create mask based on sample indexes
        if indexes is not None:
            indexes = np.asarray(indexes).ravel()
            # Adjust negative indexes
            indexes = np.mod(indexes, n_samples)
            if np.any((indexes < 0) | (indexes >= n_samples)):
                raise IndexError("One or more sample indexes are out of bounds.")
            # Create mask for samples at specified indexes
            mask_indexes = np.zeros(n_samples, dtype=bool)
            mask_indexes[indexes] = True
        else:
            mask_indexes = np.zeros(n_samples, dtype=bool)

        # Combine masks using logical OR (union of samples)
        mask_combined = mask_samples | mask_indexes

        if not include:
            # Invert mask if excluding samples
            mask_combined = ~mask_combined

        # If requested, compute an ordering of selected rows that follows the provided lists
        ordered_indices = None
        if include and reorder:
            sel_indices = np.where(mask_combined)[0]
            sample_names = self.__phen_df.iloc[:, 0].values
            ordered_list = []
            added = np.zeros(n_samples, dtype=bool)

            # Respect the order provided in `samples` (supports duplicate sample names)
            if samples is not None:
                for s in samples:
                    matches = np.where(sample_names == s)[0]
                    for idx in matches:
                        if mask_combined[idx] and not added[idx]:
                            ordered_list.append(int(idx))
                            added[idx] = True

            # Then respect the order in `indexes`
            if indexes is not None:
                adj_idx = np.mod(np.atleast_1d(indexes), n_samples)
                for idx in adj_idx:
                    if mask_combined[idx] and not added[idx]:
                        ordered_list.append(int(idx))
                        added[idx] = True

            # Finally, append any remaining selected rows in their original order
            for idx in sel_indices:
                if not added[idx]:
                    ordered_list.append(int(idx))

            ordered_indices = np.asarray(ordered_list, dtype=int)

        # Filter the phenotype DataFrame
        if inplace:
            if ordered_indices is not None:
                self['phen_df'] = self['phen_df'].iloc[ordered_indices].reset_index(drop=True)
            else:
                self['phen_df'] = self['phen_df'][mask_combined].reset_index(drop=True)
            return None
        else:
            phen_obj = self.copy()
            if ordered_indices is not None:
                phen_obj['phen_df'] = phen_obj['phen_df'].iloc[ordered_indices].reset_index(drop=True)
            else:
                phen_obj['phen_df'] = phen_obj['phen_df'][mask_combined].reset_index(drop=True)
            return phen_obj
