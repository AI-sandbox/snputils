from __future__ import annotations

import warnings
import numpy as np
import copy
from typing import TYPE_CHECKING, Tuple, Optional, Union, List
from sklearn.decomposition import PCA as skPCA 

from snputils.snp.genobj.snpobj import SNPObject

if TYPE_CHECKING:
    import torch


_TORCH_INSTALL_MESSAGE = (
    "PyTorch-backed PCA requires optional dependency 'torch'. "
    "Install it with `pip install snputils[torch]` or `pip install torch`."
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_TORCH_INSTALL_MESSAGE) from exc
    return torch


def _svd_flip(u, v, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    torch = _require_torch()

    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), axis=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]

    return u, v


class TorchPCA:
    """
    A class for GPU-based Principal Component Analysis (PCA) with PyTorch tensors.
    
    This implementation leverages GPU acceleration to achieve significant performance improvements,
    being up to 25 times faster than `sklearn.decomposition.PCA` when running on a compatible GPU.
    """
    def __init__(self, n_components: int = 2, fitting: str = 'reduced'):
        """
        Args:
            n_components (int, default=2): 
                The number of principal components. If None, defaults to the minimum of `n_samples` and `n_snps`.
            fitting (str, default='reduced'): 
                The fitting approach for the SVD computation. Options:
                - `'full'`: Full Singular Value Decomposition (SVD).
                - `'reduced'`: Economy SVD.
                - `'lowrank'`: Low-rank approximation, which provides a faster but approximate solution.
                Default is 'reduced'.
        """
        _require_torch()

        self.__n_components = n_components
        self.__fitting = fitting
        self.__n_components_ = None
        self.__components_ = None
        self.__mean_ = None
        self.__X_new_ = None  # Store transformed SNP data

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
    def n_components(self) -> Optional[int]:
        """
        Retrieve `n_components`.

        Returns:
            **int:** The number of principal components.
        """
        return self.__n_components
    
    @n_components.setter
    def n_components(self, x: Optional[int]) -> None:
        """
        Update `n_components`.
        """
        if x is not None and (not isinstance(x, int) or x <= 0):
            raise ValueError("n_components must be a positive integer or None.")
        self.__n_components = x

    @property
    def fitting(self) -> str:
        """
        Retrieve `fitting`.

        Returns:
            **str:** 
                The fitting approach for the SVD computation.
        """
        return self.__fitting

    @fitting.setter
    def fitting(self, x: str) -> None:
        """
        Update `fitting`.
        """
        allowed = {"full", "reduced", "lowrank"}
        if x not in allowed:
            raise ValueError(f"fitting must be one of {sorted(allowed)}")
        self.__fitting = x

    @property
    def n_components_(self) -> Optional[int]:
        """
        Retrieve `n_components_`.
        
        Returns:
            **int:** 
                The effective number of components retained after fitting, 
                calculated as `min(self.n_components, min(n_samples, n_snps))`.
        """
        return self.__n_components_

    @n_components_.setter
    def n_components_(self, x: int) -> None:
        """
        Update `n_components_`.
        """
        self.__n_components_ = x

    @property
    def components_(self) -> Optional[torch.Tensor]:
        """
        Retrieve `components_`.
        
        Returns:
            **tensor of shape (n_components_, n_snps):** 
                Matrix of principal components, where each row is a principal component vector.
        """
        return self.__components_

    @components_.setter
    def components_(self, x: torch.Tensor) -> None:
        """
        Update `components_`.
        """
        self.__components_ = x

    @property
    def mean_(self) -> Optional[torch.Tensor]:
        """
        Retrieve `mean_`.

        Returns:
            **tensor of shape (n_snps,):** 
                Per-feature mean vector of the input data used for centering.
        """
        return self.__mean_

    @mean_.setter
    def mean_(self, x: torch.Tensor) -> None:
        """
        Update `mean_`.
        """
        self.__mean_ = x

    @property
    def X_new_(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Retrieve `X_new_`.

        Returns:
            **tensor of shape (n_samples, n_components_):** 
                The transformed SNP data projected onto the `n_components_` principal components.
        """
        return self.__X_new_

    @X_new_.setter
    def X_new_(self, x: torch.Tensor) -> None:
        """
        Update `X_new_`.
        """
        self.__X_new_ = x

    def copy(self) -> 'TorchPCA':
        """
        Create and return a copy of `self`.

        Returns:
            **TorchPCA:** 
                A new instance of the current object.
        """
        return copy.copy(self)

    def _fit(self, X: torch.Tensor) -> Tuple:
        """
        Internal method to fit the PCA model to the data `X`.

        Args:
            X (tensor of shape (n_samples, n_snps)): 
                Input SNP data used for fitting the model.

        Returns:
            Tuple: U, S, and Vt matrices from the SVD, where:
                - `U` has shape (n_samples, n_components).
                - `S` contains singular values (n_components).
                - `Vt` has shape (n_components, n_snps) and represents the principal components.
        """
        torch = _require_torch()

        n_samples, n_snps = X.shape
        self.n_components_ = min(self.n_components or min(X.shape), min(X.shape))

        if self.n_components_ > min(X.shape):
            raise ValueError(f"n_components should be <= min(n_samples: {n_samples}, n_snps: {n_snps})")

        # Compute the mean to center the data
        self.mean_ = torch.mean(X, dim=0)

        # Choose SVD method based on the fitting type
        if self.fitting == "full":
            U, S, Vt = torch.linalg.svd(X - self.mean_, full_matrices=True)
        elif self.fitting == "reduced":
            U, S, Vt = torch.linalg.svd(X - self.mean_, full_matrices=False)
        elif self.fitting == "lowrank":
            U, S, V = torch.svd_lowrank(X, q=self.n_components_, M=self.mean_)
            Vt = V.mT  # Transpose V to get Vt
        else:
            raise ValueError(f"Unrecognized fitting method: {self.fitting}")

        # Select the first `n_components` columns and singular values
        U = U[:, :self.n_components_]
        S = S[:self.n_components_]
        Vt = Vt[:self.n_components_]

        # Ensure deterministic output for U and Vt signs
        U, Vt = _svd_flip(U, Vt)
        self.components_ = Vt

        return U, S, Vt

    def fit(self, X: torch.Tensor) -> 'TorchPCA':
        """
        Fit the model to the input SNP data.

        Args:
            X (tensor of shape (n_samples, n_snps)): 
                The SNP data matrix to fit the model.

        Returns:
            **TorchPCA:** 
                The fitted instance of `self`.
        """
        self._fit(X)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply dimensionality reduction to the input SNP data using the fitted model.

        Args:
            X (tensor of shape (n_samples, n_snps)): 
                The SNP data matrix to be transformed.

        Returns:
            **tensor of shape (n_samples, n_components_):** 
                The transformed SNP data projected onto the `n_components_` principal components, 
                stored in `self.X_new_`.
        """
        if self.components_ is None or self.mean_ is None:
            raise ValueError("The PCA model must be fitted before calling `transform`.")

        torch = _require_torch()
        self.X_new_ = torch.matmul(X - self.mean_, self.components_.T)
        return self.X_new_

    def fit_transform(self, X):
        """
        Fit the model to the SNP data and apply dimensionality reduction on the same SNP data.

        Args:
            X (tensor of shape n_samples, n_snps): 
                The SNP data matrix used for both fitting and transformation.

        Returns:
            **tensor of shape (n_samples, n_components_):** 
                The transformed SNP data projected onto the `n_components_` principal components, 
                stored in `self.X_new_`.
        """
        U, S, _ = self._fit(X)
        self.X_new_ = U * S.unsqueeze(0)
        return self.X_new_


class PCA:
    """
    A class for Principal Component Analysis (PCA) on SNP data stored in a `snputils.snp.genobj.SNPObject`. 
    This class employs either 
    [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 
    or the custom `TorchPCA` implementation for efficient GPU-accelerated analysis.
    
    The PCA class supports both separate and averaged strand processing for SNP data. If the `snpobj` parameter is 
    provided during instantiation, the `fit_transform` method will be automatically called, 
    applying PCA to transform the data according to the selected configuration.
    """
    def __init__(
        self, 
        snpobj: Optional['SNPObject'] = None, 
        backend: str = 'sklearn',
        n_components: int = 2, 
        fitting: str = 'full', 
        device: str = 'cpu',
        average_strands: bool = True, 
        samples_subset: Optional[Union[int, List]] = None, 
        snps_subset: Optional[Union[int, List]] = None
    ):
        """
        Args:
            snpobj (SNPObject, optional): 
                A SNPObject instance.
            backend (str, default='sklearn'):
                The backend to use (`'sklearn'` or `'pytorch'`). Default is 'sklearn'.
            n_components (int, default=2): 
                The number of principal components. Default is 2.
            fitting (str, default='full'): 
                The fitting approach to use for the SVD computation (only for `backend='pytorch'`). 
                - `'full'`: Full Singular Value Decomposition (SVD).
                - `'reduced'`: Economy SVD.
                - `'lowrank'`: Low-rank approximation, which provides a faster but approximate solution.
                Default is 'full'.
            device (str, default='cpu'): 
                Device to use (`'cpu'`, `'gpu'`, `'cuda'`, or `'cuda:<index>'`). Default is 'cpu'.
            average_strands (bool, default=True): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
            samples_subset (int or list of int, optional): 
                Subset of samples to include, as an integer for the first samples or a list of sample indices.
            snps_subset (int or list of int, optional): 
                Subset of SNPs to include, as an integer for the first SNPs or a list of SNP indices.
        """
        self.__snpobj = snpobj
        self.__backend = backend.lower()
        self.__n_components = n_components
        self.__fitting = fitting
        self.__device = self._process_device_argument(device) if self.__backend == "pytorch" else device
        self.__average_strands = average_strands
        self.__samples_subset = samples_subset
        self.__snps_subset = snps_subset
        self.__X_ = None
        self.__X_new_ = None  # Store transformed SNP data
        self.__n_components_ = None
        self.__components_ = None
        self.__mean_ = None

        # Initialize PCA backend
        if self.backend == "pytorch":
            self.pca = TorchPCA(n_components=self.n_components, fitting=self.fitting)
        elif self.backend == "sklearn":
            self.pca = skPCA(n_components=self.n_components)
        else:
            raise ValueError("Unknown backend for PCA: ", backend)

        # Fit and transform if a `snpobj` is provided
        if self.snpobj is not None:
            self.fit_transform(snpobj)

    @property
    def snpobj(self) -> Optional['SNPObject']:
        """
        Retrieve `snpobj`.

        Returns:
            **SNPObject:** A SNPObject instance.
        """
        return self.__snpobj

    @snpobj.setter
    def snpobj(self, x) -> None:
        """
        Update `snpobj`.
        """
        self.__snpobj = x

    @property
    def backend(self) -> str:
        """
        Retrieve `backend`.
        
        Returns:
            **str:** The backend to use (`'sklearn'` or `'pytorch'`).
        """
        return self.__backend

    @backend.setter
    def backend(self, x: str) -> None:
        """
        Update `backend`.
        """
        self.__backend = x

    @property
    def n_components(self) -> int:
        """
        Retrieve `n_components`.
        
        Returns:
            **int:** The number of principal components.
        """
        return self.__n_components
    
    @n_components.setter
    def n_components(self, x: int) -> None:
        """
        Update `n_components`.
        """
        self.__n_components = x

    @property
    def fitting(self) -> str:
        """
        Retrieve `fitting`.

        Returns:
            **str:** 
                The fitting approach to use for the SVD computation (only for `backend='pytorch'`). 
                - `'full'`: Full Singular Value Decomposition (SVD).
                - `'lowrank'`: Low-rank approximation, which provides a faster but approximate solution.
        """
        return self.__fitting

    @fitting.setter
    def fitting(self, x: str) -> None:
        """
        Update `fitting`.
        """
        self.__fitting = x

    @property
    def device(self) -> torch.device:
        """
        Retrieve `device`.

        Returns:
            **torch.device:** Device to use (`'cpu'`, `'gpu'`, `'cuda'`, or `'cuda:<index>'`).
        """
        return self.__device

    @device.setter
    def device(self, x: str) -> None:
        """
        Update `device`.
        """
        self.__device = self._process_device_argument(x)

    @property
    def average_strands(self) -> bool:
        """
        Retrieve `average_strands`.

        Returns:
            **bool:** 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
        """
        return self.__average_strands

    @average_strands.setter
    def average_strands(self, x: bool) -> None:
        """
        Update `average_strands`.
        """
        self.__average_strands = x

    @property
    def samples_subset(self) -> Optional[Union[int, List[int]]]:
        """
        Retrieve `samples_subset`.

        Returns:
            **int or list of int:** 
                Subset of samples to include, as an integer for the first samples or a list of sample indices.
        """
        return self.__samples_subset

    @samples_subset.setter
    def samples_subset(self, x: Optional[Union[int, List[int]]]) -> None:
        """
        Update `samples_subset`.
        """
        self.__samples_subset = x

    @property
    def snps_subset(self) -> Optional[Union[int, List[int]]]:
        """
        Retrieve `snps_subset`.

        Returns:
            **int or list of int:** Subset of SNPs to include, as an integer for the first SNPs or a list of SNP indices.
        """
        return self.__snps_subset

    @snps_subset.setter
    def snps_subset(self, x: Optional[Union[int, List[int]]]) -> None:
        """
        Update `snps_subset`.
        """
        self.__snps_subset = x

    @property
    def n_components_(self) -> Optional[int]:
        """
        Retrieve `n_components_`.

        Returns:
            **int:** 
                The effective number of components retained after fitting, 
                calculated as `min(self.n_components, min(n_samples, n_snps))`.
        """
        return self.__n_components_

    @n_components_.setter
    def n_components_(self, x: int) -> None:
        """
        Update `n_components_`.
        """
        self.__n_components_ = x

    @property
    def components_(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Retrieve `components_`.

        Returns:
            **tensor or array of shape (n_components_, n_snps):** 
                Matrix of principal components, where each row is a principal component vector.
        """
        return self.__components_

    @components_.setter
    def components_(self, x: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Update `components_`.
        """
        self.__components_ = x

    @property
    def mean_(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Retrieve `mean_`.

        Returns:
            **tensor or array of shape (n_snps,):** 
                Per-feature mean vector of the input data used for centering.
        """
        return self.__mean_

    @mean_.setter
    def mean_(self, x: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Update `mean_`.
        """
        self.__mean_ = x

    @property
    def X_(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Retrieve `X_`.

        Returns:
            **tensor or array of shape (n_samples, n_snps):** 
                The SNP data matrix used to fit the model.
        """
        return self.__X_

    @X_.setter
    def X_(self, x: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Update `X_`.
        """
        self.__X_ = x

    @property
    def X_new_(self) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """
        Retrieve `X_new_`.

        Returns:
            **tensor or array of shape (n_samples, n_components_):** 
                The transformed SNP data projected onto the `n_components_` principal components.
        """
        return self.__X_new_

    @X_new_.setter
    def X_new_(self, x: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Update `X_new_`.
        """
        self.__X_new_ = x

    def copy(self) -> 'PCA':
        """
        Create and return a copy of `self`.

        Returns:
            **PCA:** 
                A new instance of the current object.
        """
        return copy.copy(self)

    def _process_device_argument(self, device: str):
        """
        Process the device argument to map user-friendly device names to PyTorch device specifications.

        Args:
            device (str): Device specified by the user.

        Returns:
            torch.device: PyTorch device object.
        """
        if isinstance(device, str):
            torch = _require_torch()
            device_lower = device.lower()
            if device_lower in ['cpu']:
                return torch.device('cpu')
            elif device_lower in ['gpu', 'cuda']:
                if torch.cuda.is_available():
                    return torch.device('cuda')
                else:
                    warnings.warn("CUDA is not available; using CPU instead.")
                    return torch.device('cpu')
            elif device_lower.startswith('cuda:'):
                if torch.cuda.is_available():
                    return torch.device(device_lower)
                else:
                    warnings.warn(f"CUDA is not available; requested device '{device}' is not available. Using CPU instead.")
                    return torch.device('cpu')
            else:
                raise ValueError(f"Unknown device type: '{device}'. Please use 'CPU', 'GPU', 'cuda', or 'cuda:<index>'.")
        else:
            torch = _require_torch()
            if isinstance(device, torch.device):
                return device
            raise TypeError(f"Device must be a string or torch.device, got {type(device)}.")

    def _get_data_from_snpobj(
            self, 
            snpobj: Optional['SNPObject'] = None, 
            average_strands: Optional[bool] = None, 
            samples_subset: Optional[Union[int, List]] = None, 
            snps_subset: Optional[Union[int, List]] = None
        ) -> Union[np.ndarray, torch.Tensor]:
        """
        Retrieve and prepare SNP data for PCA analysis, with options for selecting subsets and handling strands.

        This method processes SNP data stored in an `SNPObject`, which may include averaging of paternal 
        and maternal strands or selecting subsets of samples and SNPs. The prepared data is formatted 
        for use in PCA, optionally converted to a PyTorch tensor if the backend is set to 'pytorch'.

        Args:
            snpobj (SNPObject, optional): 
                A SNPObject object instance. If None, defaults to `self.snpobj`.
            average_strands (bool, optional): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
                If None, defaults to `self.average_strands`.
            samples_subset (int or list of int, optional): 
                Subset of samples to include, as an integer for the first samples or a list of sample indices.
                If None, defaults to `self.samples_subset`.
            snps_subset (int or list of int, optional): 
                Subset of SNPs to include, as an integer for the first SNPs or a list of SNP indices.
                If None, defaults to `self.snps_subset`.

            Returns:
                numpy.ndarray or torch.Tensor: 
                    The processed SNP data. If `backend` is set to 'pytorch', the data is returned as a 
                    PyTorch tensor on the specified `device`.
        """
        if snpobj is None:
            snpobj = self.snpobj
        if average_strands is None:
            average_strands = self.average_strands
        if samples_subset is None:
            samples_subset = self.samples_subset
        if snps_subset is None:
            snps_subset = self.snps_subset
            
        if snpobj.calldata_gt.ndim == 2:
            X = np.transpose(snpobj.calldata_gt.astype(float), (1,0))
        elif snpobj.calldata_gt.ndim == 3:
            X = np.transpose(snpobj.calldata_gt.astype(float), (1,0,2))
        
            if average_strands:
                X = np.mean(X, axis=2)
            else:
                X = np.reshape(X, (-1, X.shape[1]))
        else:
            raise ValueError(f"Invalid shape for `calldata_gt`: expected a 2D or 3D array, but got {snpobj.calldata_gt.ndim}D array.")
    
        # Handle sample and SNP subsets
        if isinstance(samples_subset, int):
            X = X[:samples_subset]
        elif isinstance(samples_subset, list):
            X = X[samples_subset]
        
        if isinstance(snps_subset, int):
            X = X[:, :snps_subset]
        elif isinstance(snps_subset, list):
            X = X[:, snps_subset]
        
        if self.backend == "pytorch":
            torch = _require_torch()
            print(f"Converting data to PyTorch tensor on device {self.device}")
            X = torch.from_numpy(X).to(self.device)

        return X

    def fit(
            self, 
            snpobj: Optional['SNPObject'] = None, 
            average_strands: Optional[bool] = None, 
            samples_subset: Optional[Union[int, List]] = None, 
            snps_subset: Optional[Union[int, List]] = None
        ) -> 'PCA':
        """
        Fit the model to the input SNP data stored in the provided `snpobj`.

        Args:
            snpobj (SNPObject, optional): 
                A SNPObject instance. If None, defaults to `self.snpobj`.
            average_strands (bool, optional): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
                If None, defaults to `self.average_strands`.
            samples_subset (int or list of int, optional): 
                Subset of samples to include, as an integer for the first samples or a list of sample indices.
                If None, defaults to `self.samples_subset`.
            snps_subset (int or list of int, optional): 
                Subset of SNPs to include, as an integer for the first SNPs or a list of SNP indices.
                If None, defaults to `self.snps_subset`.

        Returns:
            **PCA:** 
                The fitted instance of `self`.
        """
        self.X_ = self._get_data_from_snpobj(snpobj, average_strands, samples_subset, snps_subset)
        self.pca.fit(self.X_)

        # Update attributes based on the fitted model
        self.n_components_ = self.pca.n_components_
        self.components_ = self.pca.components_
        self.mean_ = self.pca.mean_
        
        return self

    def transform(
            self, 
            snpobj: Optional['SNPObject'] = None, 
            average_strands: Optional[bool] = None, 
            samples_subset: Optional[Union[int, List]] = None, 
            snps_subset: Optional[Union[int, List]] = None
        ):
        """
        Apply dimensionality reduction to the input SNP data stored in the provided `snpobj` using the fitted model.

        Args:
            snpobj (SNPObject, optional): 
                A SNPObject instance. If None, defaults to `self.snpobj`.
            average_strands (bool, optional): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
                If None, defaults to `self.average_strands`.
            samples_subset (int or list of int, optional): 
                Subset of samples to include, as an integer for the first samples or a list of sample indices.
                If None, defaults to `self.samples_subset`.
            snps_subset (int or list of int, optional): 
                Subset of SNPs to include, as an integer for the first SNPs or a list of SNP indices.
                If None, defaults to `self.snps_subset`.

        Returns:
            **tensor or array of shape (n_samples, n_components):**
                The transformed SNP data projected onto the `n_components_` principal components,
                stored in `self.X_new_`.
        """
        # Retrieve or update the data to transform
        if snpobj is not None or self.X_ is None:
            self.X_ = self._get_data_from_snpobj(snpobj, average_strands, samples_subset, snps_subset)
        
        # Apply transformation using the fitted PCA model
        return self.pca.transform(self.X_)

    def fit_transform(self, snpobj: Optional['SNPObject'] = None, average_strands: Optional[bool] = None, 
                      samples_subset: Optional[Union[int, List]] = None, snps_subset: Optional[Union[int, List]] = None):
        """
        Fit the model to the SNP data stored in the provided `snpobj` and apply the dimensionality reduction 
        on the same SNP data.

        Args:
            snpobj (SNPObject, optional): 
                A SNPObject instance. If None, defaults to `self.snpobj`.
            average_strands (bool, optional): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
                If None, defaults to `self.average_strands`.
            samples_subset (int or list of int, optional): 
                Subset of samples to include, as an integer for the first samples or a list of sample indices.
                If None, defaults to `self.samples_subset`.
            snps_subset (int or list of int, optional): 
                Subset of SNPs to include, as an integer for the first SNPs or a list of SNP indices.
                If None, defaults to `self.snps_subset`.

        Returns:
            **tensor or array of shape (n_samples, n_components):** 
                The transformed SNP data projected onto the `n_components_` principal components,
                stored in `self.X_new_`.
        """
        self.X_ = self._get_data_from_snpobj(snpobj, average_strands, samples_subset, snps_subset)
        self.X_new_ = self.pca.fit_transform(self.X_)

        # Update attributes based on the fitted model
        self.n_components_ = self.pca.n_components_
        self.components_ = self.pca.components_
        self.mean_ = self.pca.mean_

        return self.X_new_
