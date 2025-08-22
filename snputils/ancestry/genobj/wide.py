from typing import Sequence, Optional, Union, List
from pathlib import Path
import numpy as np
import copy

from .base import AncestryObject


class GlobalAncestryObject(AncestryObject):
    """
    A class for Global Ancestry Inference (GAI) data.
    """
    def __init__(
        self,
        Q: np.ndarray,
        P: Optional[np.ndarray] = None,
        samples: Optional[Sequence] = None,
        snps: Optional[Sequence] = None,
        ancestries: Optional[Sequence] = None
    ) -> None:
        """
        Args:
            Q (array of shape (n_samples, n_ancestries)):
                A 2D array containing per-sample ancestry proportions. Each row corresponds to a sample,
                and each column corresponds to an ancestry.
            P (array of shape (n_snps, n_ancestries)):
                A 2D array containing per-ancestry SNP frequencies. Each row corresponds to a SNP,
                and each column corresponds to an ancestry.
            samples (sequence of length n_samples, optional):
                A sequence containing unique identifiers for each sample. If None, sample identifiers 
                are assigned as integers from `0` to `n_samples - 1`.
            snps (sequence of length n_snps, optional):
                A sequence containing identifiers for each SNP. If None, SNPs are assigned as integers 
                from `0` to `n_snps - 1`.
            ancestries (sequence of length n_samples, optional):
                A sequence containing ancestry labels for each sample.
        """
        # Determine dimensions
        n_samples, n_ancestries_Q = Q.shape
        if P is not None:
            n_snps, n_ancestries_P = P.shape
            if n_ancestries_Q != n_ancestries_P:
                raise ValueError(
                    f"The number of ancestries in Q ({n_ancestries_Q}) and P ({n_ancestries_P}) must be the same."
                )

        n_ancestries = n_ancestries_Q

        # Assign default sample identifiers if none provided
        if samples is None:
            samples = list(range(n_samples))
        else:
            samples = list(samples)
            if len(samples) != n_samples:
                raise ValueError(
                    f"Length of samples ({len(samples)}) does not match number of samples ({n_samples})."
                )

        # Assign default SNP identifiers if none provided
        if P is None:
            snps = None
        else:
            if snps is None:
                snps = list(range(n_snps))
            else:
                snps = list(snps)
                if len(snps) != n_snps:
                    raise ValueError(
                        f"Length of snps ({len(snps)}) does not match number of SNPs ({n_snps})."
                    )

        if ancestries is not None:
            if len(ancestries) != n_samples:
                raise ValueError(
                    f"Length of ancestries ({len(ancestries)}) does not match number of samples ({n_samples})."
                )

        super().__init__(n_samples, n_ancestries)

        # Store attributes
        self.__Q = Q
        self.__P = P
        self.__samples = np.asarray(samples)
        self.__snps = np.asarray(snps) if snps is not None else None
        self.__ancestries = np.asarray(ancestries) if ancestries is not None else None

        # Perform sanity checks
        self._sanity_check()

    def __getitem__(self, key):
        """
        To access an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            return getattr(self, key)
        except AttributeError:
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
    def Q(self) -> np.ndarray:
        """
        Retrieve `Q`.

        Returns:
            **array of shape (n_samples, n_ancestries):** 
                A 2D array containing per-sample ancestry proportions. Each row corresponds to a sample,
                and each column corresponds to an ancestry.
        """
        return self.__Q
    
    @Q.setter
    def Q(self, x: np.ndarray):
        """
        Update `Q`.
        """
        if x.shape != (self.n_samples, self.n_ancestries):
            raise ValueError(
                f"Q must have shape ({self.n_samples}, {self.n_ancestries}); got {x.shape}."
            )
        self.__Q = x
    
    @property
    def P(self) -> np.ndarray:
        """
        Retrieve `P`.

        Returns:
            **array of shape (n_snps, n_ancestries):** 
                A 2D array containing per-ancestry SNP frequencies. Each row corresponds to a SNP,
                and each column corresponds to an ancestry.
        """
        return self.__P

    @P.setter
    def P(self, x: np.ndarray):
        """
        Update `P`.
        """
        if x.shape[1] != self.n_ancestries:
            raise ValueError(
                f"P must have {self.n_ancestries} columns (one per ancestry); got shape {x.shape}."
            )
        self.__P = x
        self._sanity_check()
    
    @property
    def F(self) -> np.ndarray:
        """
        Alias for `P`.

        Returns:
            **array of shape (n_snps, n_ancestries):** 
                A 2D array containing per-ancestry SNP frequencies. Each row corresponds to a SNP,
                and each column corresponds to an ancestry.
        """
        return self.P

    @F.setter
    def F(self, x: np.ndarray):
        """
        Update `F`.
        """
        if x.shape[1] != self.n_ancestries:
            raise ValueError(
                f"F must have {self.n_ancestries} columns (one per ancestry); got shape {x.shape}."
            )
        self.__P = x
    
    @property
    def samples(self) -> Optional[np.ndarray]:
        """
        Retrieve `samples`.

        Returns:
            **array of shape (n_samples,):** 
                An array containing unique identifiers for each sample. If None, sample 
                identifiers are assigned as integers from `0` to `n_samples - 1`.
        """
        return self.__samples
        
    @samples.setter
    def samples(self, x: Sequence):
        """
        Update `samples`.
        """
        x = list(x)
        if len(x) != self.n_samples:
            raise ValueError(
                f"samples must have length {self.n_samples}; got length {len(x)}."
            )
        self.__samples = x

    @property
    def snps(self) -> Optional[np.ndarray]:
        """
        Retrieve `snps`.

        Returns:
            **array of shape (n_snps,):** 
                An array containing identifiers for each SNP. If None, SNPs are assigned as integers 
                from `0` to `n_snps - 1`.
        """
        return self.__snps

    @snps.setter
    def snps(self, x: Sequence):
        """
        Update `snps`.
        """
        x = list(x)
        if len(x) != self.n_snps:
            raise ValueError(
                f"snps must have length {self.n_snps}; got length {len(x)}."
            )
        self.__snps = np.asarray(x)

    @property
    def ancestries(self) -> Optional[np.ndarray]:
        """
        Retrieve `ancestries`.

        Returns:
            **array of shape (n_samples,):** 
                An array containing ancestry labels for each sample.
        """
        return self.__ancestries
    
    @ancestries.setter
    def ancestries(self, x: Sequence):
        """
        Update `ancestries`.
        """
        x = list(x)
        num_x = len(x)
        num_unique_x = len(np.unique(x))

        if num_x != self.n_samples:
            raise ValueError(
                f"ancestries must have length {self.n_samples}; got length {num_x}."
            )
        if num_unique_x > self.n_ancestries:
            raise ValueError(
                f"Number of unique ancestry labels must be less than or equal to {self.n_ancestries}; got {num_unique_x} unique labels."
            )
        self.__ancestries = np.asarray(x)
    
    @property
    def n_samples(self) -> int:
        """
        Retrieve `n_samples`.

        Returns:
            **int:** The total number of samples.
        """
        return self.__Q.shape[0]

    @property
    def n_snps(self) -> int:
        """
        Retrieve `n_snps`.

        Returns:
            **int:** The total number of SNPs.
        """
        return 0 if self.__P is None else self.__P.shape[0]

    @property
    def n_ancestries(self) -> int:
        """
        Retrieve `n_ancestries`.

        Returns:
            **int:** The total number of unique ancestries.
        """
        return self.__Q.shape[1]

    def copy(self) -> 'GlobalAncestryObject':
        """
        Create and return a copy of `self`.

        Returns:
            **GlobalAncestryObject:** A new instance of the current object.
        """
        return copy.copy(self)

    def keys(self) -> List[str]:
        """
        Retrieve a list of public attribute names for `self`.

        Returns:
            **list of str:** 
                A list of attribute names, with internal name-mangling removed, 
                for easier reference to public attributes in the instance.
        """
        return [attr.replace('_GlobalAncestryObject__', '').replace('_AncestryObject__', '') for attr in vars(self)]

    def _sanity_check(self) -> None:
        """
        Perform sanity checks to ensure that matrix dimensions are consistent with expected sizes.
        
        Raises:
            **ValueError:** If any of the matrix dimensions do not match the expected sizes.
        """       
        # Check that the Q matrix has the correct shape
        if self.__Q.shape != (self.n_samples, self.n_ancestries):
            raise ValueError(
                f"Q must have shape ({self.n_samples}, {self.n_ancestries}); got {self.__Q.shape}."
            )

        # Check that the P matrix has the correct shape (if provided)
        if self.__P is not None:
            if self.__P.shape != (self.n_snps, self.n_ancestries):
                raise ValueError(
                    f"P must have shape ({self.n_snps}, {self.n_ancestries}); got {self.__P.shape}."
                )

        # Check that samples length matches n_samples
        if self.samples is not None:
            if len(self.__samples) != self.n_samples:
                raise ValueError(
                    f"samples must have length {self.n_samples}; got length {len(self.__samples)}."
                )

        # Check that snps length matches n_snps
        if self.snps is not None:
            if len(self.__snps) != self.n_snps:
                raise ValueError(
                    f"snps must have length {self.n_snps}; got length {len(self.__snps)}."
                )

        # Check that ancestries length matches n_samples
        if self.ancestries is not None:
            if len(self.__ancestries) != self.n_samples:
                raise ValueError(
                    f"ancestries must have length {self.n_samples}; got length {len(self.__ancestries)}."
                )

            # Check number of unique ancestry labels
            num_unique_ancestries = len(np.unique(self.__ancestries))
            if num_unique_ancestries > self.n_ancestries:
                raise ValueError(
                    f"Number of unique ancestry labels must be less than or equal to {self.n_ancestries}; got {num_unique_ancestries} unique labels."
                )

    def save(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a specified file or set of files.

        The format of the saved file(s) is determined by the file extension provided in the `file` 
        argument. If the extension is `.pkl`, the object is serialized as a pickle file. Otherwise, 
        the file is treated as a prefix for saving ADMIXTURE files.

        **Supported formats:**

        - `.pkl`: Pickle format for saving `self` in serialized form.
        - Any other extension or no extension: Treated as a prefix for ADMIXTURE files.

        Args:
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. If the extension is `.pkl`, the object
                is serialized. Otherwise, it is treated as a prefix for ADMIXTURE files.
        """
        path = Path(file)
        suffix = path.suffix.lower()

        if suffix == '.pkl':
            self.save_pickle(path)
        else:
            self.save_admixture(path)

    def save_admixture(self, file_prefix: Union[str, Path]) -> None:
        """
        Save the data stored in `self` into multiple ADMIXTURE files.
        If the file already exists, it will be overwritten.

        **Output files:**

        - `<file_prefix>.K.Q`: Q matrix file. The file uses space (' ') as the delimiter.
        - `<file_prefix>.K.P`: P matrix file. The file uses space (' ') as the delimiter.
        - `<file_prefix>.sample_ids.txt`: Sample IDs file (if sample IDs are available).
        - `<file_prefix>.snp_ids.txt`: SNP IDs file (if SNP IDs are available).
        - `<file_prefix>.map`: Ancestry file (if ancestries information is available).
        
        Args:
            file_prefix (str or pathlib.Path): 
                The base prefix for output file names, including directory path but excluding file extensions. 
                The prefix is used to generate specific file names for each output, with file-specific 
                suffixes appended as described above (e.g., `file_prefix.n_ancestries.Q` for the Q matrix file).
        """
        from snputils.ancestry.io.wide.write.admixture import AdmixtureWriter

        AdmixtureWriter(self, file_prefix).write()

    def save_pickle(self, file: Union[str, Path]) -> None:
        """
        Save `self` in serialized form to a `.pkl` file.
        If the file already exists, it will be overwritten.

        Args:
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. It should end with `.pkl`. 
                If the provided path does not have this extension, it will be appended.
        """
        import pickle
        with open(file, 'wb') as file:
            pickle.dump(self, file)
