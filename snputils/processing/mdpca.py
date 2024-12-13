import pathlib
import os
import time
import gc
import logging
import logging.config
import numpy as np
import copy
from typing import Optional, Dict, List, Union
from sklearn.decomposition import TruncatedSVD

from snputils.snp.genobj.snpobj import SNPObject
from snputils.ancestry.genobj.window import WindowLevelAncestryObject
from ._utils.gen_tools import array_process, process_labels_weights
from ._utils.iterative_svd import IterativeSVD


class mdPCA:
    """
    A class for missing data principal component analysis (mdPCA).

    This class supports both separate and averaged strand processing for SNP data. If the `snpobj`, 
    `laiobj`, `labels_file`, and `ancestry` parameters are all provided during instantiation, 
    the `fit_transform` method will be automatically called, applying the specified mdPCA method to transform 
    the data upon instantiation.
    """
    def __init__(
        self,
        method: str = 'weighted_cov_pca',
        snpobj: Optional['SNPObject'] = None,
        laiobj: Optional['WindowLevelAncestryObject'] = None,
        labels_file: Optional[str] = None,
        ancestry: Optional[str] = None,
        is_masked: bool = True,
        prob_thresh: float = 0,
        average_strands: bool = False,
        is_weighted: bool = False,
        groups_to_remove: Dict[int, List[str]] = {},
        min_percent_snps: float = 4,
        save_masks: bool = False,
        load_masks: bool = False,
        masks_file: Union[str, pathlib.Path] = 'masks.npz',
        output_file: Union[str, pathlib.Path] = 'output.tsv',
        covariance_matrix_file: Optional[str] = None,
        n_components: int = 2,
        rsid_or_chrompos: int = 2,
        percent_vals_masked: float = 0
    ):
        """
        Args:
            method (str, default='weighted_cov_pca'): 
                The PCA method to use for dimensionality reduction. Options include:
                - `'weighted_cov_pca'`: 
                    Simple covariance-based PCA, weighted by sample strengths.
                - `'regularized_optimization_ils'`: 
                    Regularized optimization followed by iterative, weighted (via the strengths) least squares projection of 
                    missing samples using the original covariance matrix (considering only relevant elements not missing in 
                    the original covariance matrix for those samples).
                - `'cov_matrix_imputation'`: 
                    Eigen-decomposition of the covariance matrix after first imputing the covariance matrix missing values 
                    using the Iterative SVD imputation method.
                - `'cov_matrix_imputation_ils'`: 
                    The method of 'cov_matrix_imputation', but where afterwards missing samples are re-projected onto the space 
                    given by 'cov_matrix_imputation' using the same iterative method on the original covariance matrix just 
                    as done in 'regularized_optimization_ils'.
                - `'nonmissing_pca_ils'`: 
                    The method of 'weighted_cov_pca' on the non-missing samples, followed by the projection of missing samples onto 
                    the space given by 'weighted_cov_pca' using the same iterative method on the original covariance matrix just as 
                    done in 'regularized_optimization_ils'.
            snpobj (SNPObject, optional): 
                A SNPObject instance.
            laiobj (LAIObject, optional): 
                A LAIObject instance.
            labels_file (str, optional): 
                Path to the labels file in .tsv format. The first column, `indID`, contains the individual identifiers, and the second 
                column, `label`, specifies the groups for all individuals. If `is_weighted=True`, a `weight` column with individual 
                weights is required. Optionally, `combination` and `combination_weight` columns can specify sets of individuals to be 
                combined into groups, with respective weights.
            ancestry (str, optional): 
                Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
            is_masked (bool, default=True): 
                True if an ancestry file is passed for ancestry-specific masking, or False otherwise.
            prob_thresh (float, default=0): 
                Minimum probability threshold for a SNP to belong to an ancestry.
            average_strands (bool, default=False): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
            is_weighted (bool, default=False): 
                True if weights are provided in the labels file, or False otherwise.
            groups_to_remove (dict of int to list of str, default={}): 
                Dictionary specifying groups to exclude from analysis. Keys are array numbers, and values are 
                lists of groups to remove for each array.
                Example: `{1: ['group1', 'group2'], 2: [], 3: ['group3']}`.
            min_percent_snps (float, default=4): 
                Minimum percentage of SNPs that must be known for an individual to be included in the analysis.
                All individuals with fewer percent of unmasked SNPs than this threshold will be excluded.
            save_masks (bool, default=False): 
                True if the masked matrices are to be saved in a `.npz` file, or False otherwise.
            load_masks (bool, default=False): 
                True if the masked matrices are to be loaded from a pre-existing `.npz` file specified by `masks_file`, or False otherwise.
            masks_file (str or pathlib.Path, default='masks.npz'): 
                Path to the `.npz` file used for saving/loading masked matrices.
            output_file (str or pathlib.Path, default='output.tsv'): 
                Path to the output `.tsv` file where mdPCA results are saved.
            covariance_matrix_file (str, optional): 
                Path to save the covariance matrix file in `.npy` format. If None, the covariance matrix is not saved. Default is None.
            n_components (int, default=2): 
                The number of principal components.
            rsid_or_chrompos (int, default=2): 
                Format indicator for SNP IDs in the SNP data. Use 1 for `rsID` format or 2 for `chromosome_position`.
            percent_vals_masked (float, default=0): 
                Percentage of values in the covariance matrix to be masked and then imputed. Only applicable if `method` is 
                `'cov_matrix_imputation'` or `'cov_matrix_imputation_ils'`.
        """
        self.__snpobj = snpobj
        self.__laiobj = laiobj
        self.__labels_file = labels_file
        self.__ancestry = ancestry
        self.__method = method
        self.__is_masked = is_masked
        self.__prob_thresh = prob_thresh
        self.__average_strands = average_strands
        self.__is_weighted = is_weighted
        self.__groups_to_remove = groups_to_remove
        self.__min_percent_snps = min_percent_snps
        self.__save_masks = save_masks
        self.__load_masks = load_masks
        self.__masks_file = masks_file
        self.__output_file = output_file
        self.__covariance_matrix_file = covariance_matrix_file
        self.__n_components = n_components
        self.__rsid_or_chrompos = rsid_or_chrompos
        self.__percent_vals_masked = percent_vals_masked
        self.__X_new_ = None  # Store transformed SNP data

        # Fit and transform if a `snpobj`, `laiobj`, `labels_file`, and `ancestry` are provided
        if self.snpobj is not None and self.laiobj is not None and self.labels_file is not None and self.ancestry is not None:
            self.fit_transform(snpobj, laiobj, labels_file, ancestry)

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
    def method(self) -> str:
        """
        Retrieve `method`.
        
        Returns:
            **str:** The PCA method to use for dimensionality reduction.
        """
        return self.__method

    @method.setter
    def method(self, x: str) -> None:
        """
        Update `method`.
        """
        self.__method = x

    @property
    def snpobj(self) -> Optional['SNPObject']:
        """
        Retrieve `snpobj`.
        
        Returns:
            **SNPObject:** A SNPObject instance.
        """
        return self.__snpobj

    @snpobj.setter
    def snpobj(self, x: 'SNPObject') -> None:
        """
        Update `snpobj`.
        """
        self.__snpobj = x

    @property
    def laiobj(self) -> Optional['WindowLevelAncestryObject']:
        """
        Retrieve `laiobj`.
        
        Returns:
            **WindowLevelAncestryObject:** A WindowLevelAncestryObject instance.
        """
        return self.__laiobj

    @laiobj.setter
    def laiobj(self, x: 'WindowLevelAncestryObject') -> None:
        """
        Update `laiobj`.
        """
        self.__laiobj = x

    @property
    def labels_file(self) -> Optional[str]:
        """
        Retrieve `labels_file`.
        
        Returns:
            **str:** Path to the labels file in `.tsv` format.
        """
        return self.__labels_file

    @labels_file.setter
    def labels_file(self, x: str) -> None:
        """
        Update `labels_file`.
        """
        self.__labels_file = x

    @property
    def ancestry(self) -> Optional[str]:
        """
        Retrieve `ancestry`.
        
        Returns:
            **str:** Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at 0.
        """
        return self.__ancestry

    @ancestry.setter
    def ancestry(self, x: str) -> None:
        """
        Update `ancestry`.
        """
        self.__ancestry = x

    @property
    def is_masked(self) -> bool:
        """
        Retrieve `is_masked`.
        
        Returns:
            **bool:** True if an ancestry file is passed for ancestry-specific masking, or False otherwise.
        """
        return self.__is_masked

    @is_masked.setter
    def is_masked(self, x: bool) -> None:
        """
        Update `is_masked`.
        """
        self.__is_masked = x

    @property
    def prob_thresh(self) -> float:
        """
        Retrieve `prob_thresh`.
        
        Returns:
            **float:** Minimum probability threshold for a SNP to belong to an ancestry.
        """
        return self.__prob_thresh

    @prob_thresh.setter
    def prob_thresh(self, x: float) -> None:
        """Update `prob_thresh`.
        """
        self.__prob_thresh = x

    @property
    def average_strands(self) -> bool:
        """
        Retrieve `average_strands`.
        
        Returns:
            **bool:** True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
        """
        return self.__average_strands

    @average_strands.setter
    def average_strands(self, x: bool) -> None:
        """
        Update `average_strands`.
        """
        self.__average_strands = x

    @property
    def is_weighted(self) -> bool:
        """
        Retrieve `is_weighted`.
        
        Returns:
            **bool:** True if weights are provided in the labels file, or False otherwise.
        """
        return self.__is_weighted

    @is_weighted.setter
    def is_weighted(self, x: bool) -> None:
        """
        Update `is_weighted`.
        """
        self.__is_weighted = x

    @property
    def groups_to_remove(self) -> Dict[int, List[str]]:
        """
        Retrieve `groups_to_remove`.
        
        Returns:
            **dict of int to list of str:** Dictionary specifying groups to exclude from analysis. Keys are array numbers, and values are 
                lists of groups to remove for each array. Example: `{1: ['group1', 'group2'], 2: [], 3: ['group3']}`.
        """
        return self.__groups_to_remove

    @groups_to_remove.setter
    def groups_to_remove(self, x: Dict[int, List[str]]) -> None:
        """
        Update `groups_to_remove`.
        """
        self.__groups_to_remove = x

    @property
    def min_percent_snps(self) -> float:
        """
        Retrieve `min_percent_snps`.
        
        Returns:
            **float:** 
                Minimum percentage of SNPs that must be known for an individual to be included in the analysis.
                All individuals with fewer percent of unmasked SNPs than this threshold will be excluded.
        """
        return self.__min_percent_snps

    @min_percent_snps.setter
    def min_percent_snps(self, x: float) -> None:
        """
        Update `min_percent_snps`.
        """
        self.__min_percent_snps = x

    @property
    def save_masks(self) -> bool:
        """
        Retrieve `save_masks`.
        
        Returns:
            **bool:** True if the masked matrices are to be saved in a `.npz` file, or False otherwise.
        """
        return self.__save_masks

    @save_masks.setter
    def save_masks(self, x: bool) -> None:
        """
        Update `save_masks`.
        """
        self.__save_masks = x

    @property
    def load_masks(self) -> bool:
        """
        Retrieve `load_masks`.
        
        Returns:
            **bool:** 
                True if the masked matrices are to be loaded from a pre-existing `.npz` file specified 
                by `masks_file`, or False otherwise.
        """
        return self.__load_masks

    @load_masks.setter
    def load_masks(self, x: bool) -> None:
        """
        Update `load_masks`.
        """
        self.__load_masks = x

    @property
    def masks_file(self) -> Union[str, pathlib.Path]:
        """
        Retrieve `masks_file`.
        
        Returns:
            **str or pathlib.Path:** Path to the `.npz` file used for saving/loading masked matrices.
        """
        return self.__masks_file

    @masks_file.setter
    def masks_file(self, x: Union[str, pathlib.Path]) -> None:
        """
        Update `masks_file`.
        """
        self.__masks_file = x

    @property
    def output_file(self) -> Union[str, pathlib.Path]:
        """
        Retrieve `output_file`.
        
        Returns:
            **str or pathlib.Path:** Path to the output `.tsv` file where mdPCA results are saved.
        """
        return self.__output_file

    @output_file.setter
    def output_file(self, x: Union[str, pathlib.Path]) -> None:
        """
        Update `output_file`.
        """
        self.__output_file = x

    @property
    def covariance_matrix_file(self) -> Optional[str]:
        """
        Retrieve `covariance_matrix_file`.
        
        Returns:
            **str:** Path to save the covariance matrix file in `.npy` format.
        """
        return self.__covariance_matrix_file
    
    @covariance_matrix_file.setter
    def covariance_matrix_file(self, x: Optional[str]) -> None:
        """
        Update `covariance_matrix_file`.
        """
        self.__covariance_matrix_file = x

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
    def rsid_or_chrompos(self) -> int:
        """
        Retrieve `rsid_or_chrompos`.
        
        Returns:
            **int:** Format indicator for SNP IDs in the SNP data. Use 1 for `rsID` format or 2 for `chromosome_position`.
        """
        return self.__rsid_or_chrompos

    @rsid_or_chrompos.setter
    def rsid_or_chrompos(self, x: int) -> None:
        """
        Update `rsid_or_chrompos`.
        """
        self.__rsid_or_chrompos = x

    @property
    def percent_vals_masked(self) -> float:
        """
        Retrieve `percent_vals_masked`.
        
        Returns:
            **float:** 
                Percentage of values in the covariance matrix to be masked and then imputed. Only applicable if `method` is 
                `'cov_matrix_imputation'` or `'cov_matrix_imputation_ils'`.
        """
        return self.__percent_vals_masked

    @percent_vals_masked.setter
    def percent_vals_masked(self, x: float) -> None:
        """
        Update `percent_vals_masked`.
        """
        self.__percent_vals_masked = x

    @property
    def X_new_(self) -> Optional[np.ndarray]:
        """
        Retrieve `X_new_`.

        Returns:
            **array of shape (n_samples, n_components):** 
                The transformed SNP data projected onto the `n_components` principal components.
        """
        return self.__X_new_

    @X_new_.setter
    def X_new_(self, x: np.ndarray) -> None:
        """
        Update `X_new_`.
        """
        self.__X_new_ = x

    def copy(self) -> 'mdPCA':
        """
        Create and return a copy of `self`.

        Returns:
            **mdPCA:** 
                A new instance of the current object.
        """
        return copy.copy(self)

    def _process_masks(self, masks, rs_ID_list, ind_ID_list):
        masked_matrix = masks[0][self.ancestry].T
        rs_IDs = rs_ID_list[0]
        ind_IDs = ind_ID_list[0]
        return masked_matrix, rs_IDs, ind_IDs

    def _load_mask_file(self):
        mask_files = np.load(self.masks_file, allow_pickle=True)
        masks = mask_files['masks']
        rs_ID_list = mask_files['rs_ID_list']
        ind_ID_list = mask_files['ind_ID_list']
        labels = mask_files['labels']
        weights = mask_files['weights']
        return masks, rs_ID_list, ind_ID_list, labels, weights

    @staticmethod
    def _compute_strength_vector(X):
        strength_vector = np.sum(~np.isnan(X), axis=1) / X.shape[1]
        return strength_vector

    @staticmethod
    def _compute_strength_matrix(X):
        notmask = (~np.isnan(X)).astype(np.float32)
        strength_matrix = np.dot(notmask, notmask.T)
        strength_matrix /= X.shape[1]
        return strength_matrix

    def _cov(self, x):
        ddof = 1
        x = np.ma.array(x, ndmin=2, copy=True, dtype=np.float32)
        xmask = np.ma.getmaskarray(x)
        xnotmask = np.logical_not(xmask).astype(np.float32)
        fact = np.dot(xnotmask, xnotmask.T) * 1. - ddof
        del xnotmask
        gc.collect()
        result = (np.ma.dot(x, x.T, strict=False) / fact).squeeze()
        x = x.data
        strength_vec = self._compute_strength_vector(x)
        strength_mat = self._compute_strength_matrix(x)
        return result.data, strength_vec, strength_mat

    @staticmethod
    def _demean(S, w):
        w_sum = np.matmul(w, S)
        w_rowsum = w_sum.reshape((1, S.shape[0]))
        w_colsum = w_sum.reshape((S.shape[0], 1))
        w_totalsum = np.dot(w_sum, w)
        S -= (w_rowsum + w_colsum) - w_totalsum
        return S

    @staticmethod
    def _iterative_svd_matrix_completion(cov):
        num_masked = np.isnan(cov).sum()
        if num_masked > 0:
            start_time = time.time()
            start_rank = 1
            end_rank = 10
            rank = 5
            choose_best = True
            num_cores = 5
            cov_complete = IterativeSVD(start_rank=start_rank, end_rank=end_rank, rank=rank, choose_best=choose_best, num_cores=num_cores).fit_transform(cov)
            logging.info("Iterative SVD --- %s seconds ---" % (time.time() - start_time))
        else:
            cov_complete = cov
        return cov_complete

    def _weighted_cov_pca(self, X_incomplete, weights, covariance_matrix_file, n_components):
        start_time = time.time()
        X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
        S, _, _ = self._cov(X_incomplete)
        logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
        weights_normalized = weights / weights.sum()
        S = self._demean(S, weights_normalized)
        W = np.diag(weights)
        WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
        if covariance_matrix_file:
            np.save(covariance_matrix_file, S.data)
        svd = TruncatedSVD(n_components, algorithm="arpack")
        svd.fit(WSW)
        U = np.matmul(svd.components_, np.linalg.inv(np.sqrt(W))).T
        D = np.diag(np.sqrt(svd.singular_values_))
        T = np.matmul(U, D)
        total_var = np.trace(S)
        for i in range(n_components):
            pc_percentvar = 100 * svd.singular_values_[i] / total_var
            logging.info("Percent variance explained by the principal component %d: %s", i + 1, pc_percentvar)
        return T

    @staticmethod
    def _create_validation_mask(X_incomplete, percent_inds):
        np.random.seed(1)
        masked_rows = np.isnan(X_incomplete).any(axis=1)
        masked_inds = np.flatnonzero(masked_rows)
        X_masked = X_incomplete[masked_rows] 
        percent_masked = 100 * np.isnan(X_masked).sum() / (X_masked.shape[0] * X_masked.shape[1])
        unmasked_rows = ~masked_rows
        X_unmasked = X_incomplete[unmasked_rows]
        masked_rows = np.random.choice(range(X_unmasked.shape[0]), size=int(X_unmasked.shape[0] * percent_inds / 100), replace=False)
        X_masked_rows = X_unmasked[masked_rows,:]
        mask = np.zeros(X_masked_rows.shape[0] * X_masked_rows.shape[1], dtype=np.int8)
        mask[:int(X_masked_rows.shape[0] * X_masked_rows.shape[1] * percent_masked / 100)] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)
        mask = mask.reshape(X_masked_rows.shape)
        X_masked_rows[mask] = np.nan
        X_unmasked[masked_rows] = X_masked_rows
        X_incomplete[unmasked_rows] = X_unmasked
        masked_rows_new = np.isnan(X_incomplete).any(axis=1)
        masked_inds_new = np.flatnonzero(masked_rows_new)
        masked_inds_val = sorted(list(set(masked_inds_new) - set(masked_inds)))
        return X_incomplete, masked_inds_val

    @staticmethod    
    def _create_cov_mask(cov, strength_mat, percent_vals_masked):
        cov_flattened = cov.reshape(-1).copy()
        num_vals_masked = int(percent_vals_masked * len(cov_flattened) / 100)
        pos_masked = strength_mat.reshape(-1).argsort()[:num_vals_masked]
        cov_flattened[pos_masked] = np.nan
        cov_masked = cov_flattened.reshape(cov.shape)
        return cov_masked

    def _regularized_optimization_ils(self, X_incomplete, covariance_matrix_file, n_components):
        def run_cov_matrix_regularized_optimization_ils(X_incomplete, covariance_matrix_file):
            start_time = time.time()
            X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
            has_missing = np.isnan(X_incomplete.data).sum() > 0
            S, _, strength_mat = self._cov(X_incomplete)
            logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
            S = self._demean(S, np.ones(S.shape[0]) / S.shape[0])
            if has_missing: 
                logging.info("Starting matrix completion. This will take a few minutes...")
                start_time = time.time()
                percent_inds_val = 20 # Percent of unmasked individuals to be masked for cross-validation 
                X_incomplete, masked_inds_val = self._create_validation_mask(X_incomplete.data, percent_inds_val) # masked_inds_val is the list of indices of the individuals masked for validation
                X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
                S_prime, _, w_mat_prime = self._cov(X_incomplete)
                del X_incomplete
                gc.collect()
                S_prime = self._demean(S_prime, np.ones(S_prime.shape[0]) / S_prime.shape[0])
                S_robust, lam = self.matrix_completion(S, strength_mat, S_prime, w_mat_prime, lams=None, method="NN", 
                                                cv_inds=masked_inds_val)
                logging.info(f"Covariance Matrix --- %{time.time() - start_time:.2}s seconds ---")
            else:
                S_robust = S.copy()
            if covariance_matrix_file:
                np.save(covariance_matrix_file, S.data)
                if has_missing:
                    base, ext = os.path.splitext(covariance_matrix_file)
                    np.save(f"{base}_completed_{lam}{ext}", S_robust.data)
            return S, S_robust

        def _compute_projection_regularized_optimization_ils(X_incomplete, S, S_robust, n_components):
            _, _, strength_mat = self._cov(np.ma.array(X_incomplete, mask=np.isnan(X_incomplete)))
            nonmissing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) == 0)
            missing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) > 0)
            missing_sorted = sorted(missing, key=lambda x: (~np.isnan(X_incomplete[x])).sum(), reverse=True)
            svd = TruncatedSVD(n_components, algorithm="arpack")
            svd.fit(S_robust)
            U = svd.components_.T
            U_nonmissing = U[nonmissing]
            D_nonmissing = np.diag(np.sqrt(svd.singular_values_))
            T = np.zeros((X_incomplete.shape[0], n_components))
            T_filled = np.matmul(U_nonmissing, D_nonmissing)
            i_filled = nonmissing
            T[nonmissing] = T_filled
            for i in missing_sorted:
                S_i = S[i][i_filled]
                W_i = np.diag(strength_mat[i][i_filled])
                A = np.matmul(W_i, T_filled)
                b = np.matmul(W_i, S_i)
                t = np.linalg.lstsq(A, b, rcond=None)[0]
                T[i] = t
                i_filled = np.append(i_filled, i)
                T_filled = np.append(T_filled, [t], axis=0)
            total_var = np.trace(S)
            for i in range(n_components):
                pc_percentvar = 100 * svd.singular_values_[i] / total_var
                logging.info("Percent variance explained by the principal component %d: %s", i + 1, pc_percentvar)
            return T
        
        S, S_robust = run_cov_matrix_regularized_optimization_ils(X_incomplete, covariance_matrix_file)
        T = _compute_projection_regularized_optimization_ils(X_incomplete, S, S_robust, n_components)
        return T

    def _cov_matrix_imputation(self, X_incomplete, percent_vals_masked, covariance_matrix_file, n_components):
        start_time = time.time()
        X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
        S, strength_vec, strength_mat = self._cov(X_incomplete)
        logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
        S_masked = self._create_cov_mask(S, strength_mat, percent_vals_masked)
        S = self._iterative_svd_matrix_completion(S_masked)
        weights = np.ones(len(strength_vec))
        weights_normalized = weights / weights.sum()
        S = self._demean(S, weights_normalized)
        W = np.diag(weights)
        WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
        if covariance_matrix_file:
            np.save(covariance_matrix_file, S.data)
        svd = TruncatedSVD(n_components, algorithm="arpack")
        svd.fit(WSW)
        U = np.matmul(svd.components_, np.linalg.inv(np.sqrt(W))).T
        D = np.diag(np.sqrt(svd.singular_values_))
        T = np.matmul(U, D)
        total_var = np.trace(S)
        for i in range(n_components):
            pc_percentvar = 100 * svd.singular_values_[i] / total_var
            logging.info("Percent variance explained by the principal component %d: %s", i + 1, pc_percentvar)
        return T

    def _cov_matrix_imputation_ils(self, X_incomplete, percent_vals_masked, covariance_matrix_file, n_components):
        
        def run_cov_matrix_cov_matrix_imputation_ils(X_incomplete, covariance_matrix_file):
            start_time = time.time()
            X_incomplete = np.ma.array(X_incomplete, mask=np.isnan(X_incomplete))
            S, strength_vec, strength_mat = self._cov(X_incomplete)
            logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
            S_masked = self._create_cov_mask(S, strength_mat, percent_vals_masked)
            S = self._iterative_svd_matrix_completion(S_masked)
            weights = np.ones(len(strength_vec))
            weights_normalized = weights / weights.sum()
            S = self._demean(S, weights_normalized)
            W = np.diag(weights)
            if covariance_matrix_file:
                np.save(covariance_matrix_file, S.data)
            return S, W

        def compute_projection_cov_matrix_imputation_ils(X_incomplete, S, W, n_components):
            S, _, strength_mat = self._cov(np.ma.array(X_incomplete, mask=np.isnan(X_incomplete)))
            S = self._demean(S, np.ones(S.shape[0]) / S.shape[0])
            nonmissing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) == 0)
            missing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) > 0)
            missing_sorted = sorted(missing, key=lambda x: (~np.isnan(X_incomplete[x])).sum(), reverse=True)
            svd = TruncatedSVD(n_components, algorithm="arpack")
            WSW = np.matmul(np.matmul(np.sqrt(W), S), np.sqrt(W))
            svd.fit(WSW)
            U = np.matmul(svd.components_, np.linalg.inv(np.sqrt(W))).T
            U_nonmissing = U[nonmissing]
            D_nonmissing = np.diag(np.sqrt(svd.singular_values_))
            T = np.zeros((X_incomplete.shape[0], n_components))
            T_filled = np.matmul(U_nonmissing, D_nonmissing)
            i_filled = nonmissing
            T[nonmissing] = T_filled
            for i in missing_sorted:
                S_i = S[i][i_filled]
                W_i = np.diag(strength_mat[i][i_filled])
                A = np.matmul(W_i, T_filled)
                b = np.matmul(W_i, S_i)
                t = np.linalg.lstsq(A, b, rcond=None)[0]
                T[i] = t
                i_filled = np.append(i_filled, i)
                T_filled = np.append(T_filled, [t], axis=0)
            total_var = np.trace(S)
            for i in range(n_components):
                pc_percentvar = 100 * svd.singular_values_[i] / total_var
                logging.info("Percent variance explained by the principal component %d: %s", i + 1, pc_percentvar)
            return T
        
        S, W = run_cov_matrix_cov_matrix_imputation_ils(X_incomplete, covariance_matrix_file)
        T = compute_projection_cov_matrix_imputation_ils(X_incomplete, S, W, n_components)
        return T

    def _nonmissing_pca_ils(self, X_incomplete, covariance_matrix_file, n_components):
    
        def run_cov_matrix_nonmissing_pca_ils(X_incomplete, covariance_matrix_file):
            start_time = time.time()
            nonmissing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) == 0)
            X_incomplete_nonmissing = X_incomplete[nonmissing]
            X_incomplete_nonmissing = np.ma.array(X_incomplete_nonmissing, mask=np.isnan(X_incomplete_nonmissing))
            S_nonmissing, _, _ = self._cov(X_incomplete_nonmissing)
            logging.info("Covariance Matrix --- %s seconds ---" % (time.time() - start_time))
            S_nonmissing = self._demean(S_nonmissing, np.ones(S_nonmissing.shape[0]) / S_nonmissing.shape[0])    
            if covariance_matrix_file:
                np.save(covariance_matrix_file, S_nonmissing.data)
            return S_nonmissing
        
        def compute_projection_nonmissing_pca_ils(X_incomplete, S_nonmissing, n_components):
            S, _, strength_mat = self._cov(np.ma.array(X_incomplete, mask=np.isnan(X_incomplete)))
            S = self._demean(S, np.ones(S.shape[0]) / S.shape[0])
            nonmissing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) == 0)
            missing = np.flatnonzero(np.isnan(X_incomplete).sum(axis=1) > 0)
            missing_sorted = sorted(missing, key=lambda x: (~np.isnan(X_incomplete[x])).sum(), reverse=True)
            svd = TruncatedSVD(n_components, algorithm="arpack")
            svd.fit(S_nonmissing)
            U_nonmissing = svd.components_.T
            D_nonmissing = np.diag(np.sqrt(svd.singular_values_))
            T = np.zeros((X_incomplete.shape[0], n_components))
            T_filled = np.matmul(U_nonmissing, D_nonmissing)
            i_filled = nonmissing
            T[nonmissing] = T_filled
            for i in missing_sorted:
                S_i = S[i][i_filled]
                W_i = np.diag(strength_mat[i][i_filled])
                A = np.matmul(W_i, T_filled)
                b = np.matmul(W_i, S_i)
                t = np.linalg.lstsq(A, b, rcond=None)[0]
                T[i] = t
                i_filled = np.append(i_filled, i)
                T_filled = np.append(T_filled, [t], axis=0)
            total_var = np.trace(S)
            for i in range(n_components):
                pc_percentvar = 100 * svd.singular_values_[i] / total_var
                logging.info("Percent variance explained by the principal component %d: %s", i + 1, pc_percentvar)
            return T
        
        S_nonmissing = run_cov_matrix_nonmissing_pca_ils(X_incomplete, covariance_matrix_file)
        T = compute_projection_nonmissing_pca_ils(X_incomplete, S_nonmissing, n_components)
        return T

    def _run_cov_matrix(self, X_incomplete, weights):
        """
        Runs the specified PCA method on the incomplete data.
        """
        if self.method == "weighted_cov_pca":
            return self._weighted_cov_pca(X_incomplete, weights, self.covariance_matrix_file, self.n_components)
        elif self.method == "regularized_optimization_ils":
            return self._regularized_optimization_ils(X_incomplete, self.covariance_matrix_file, self.n_components)
        elif self.method == "cov_matrix_imputation":
            return self._cov_matrix_imputation(X_incomplete, self.percent_vals_masked, self.covariance_matrix_file, self.n_components)
        elif self.method == "cov_matrix_imputation_ils":
            return self._cov_matrix_imputation_ils(X_incomplete, self.percent_vals_masked, self.covariance_matrix_file, self.n_components)
        elif self.method == "nonmissing_pca_ils":
            return self._nonmissing_pca_ils(X_incomplete, self.covariance_matrix_file, self.n_components)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def fit_transform(
            self,
            snpobj: Optional['SNPObject'] = None, 
            laiobj: Optional['WindowLevelAncestryObject'] = None,
            labels_file: Optional[str] = None,
            ancestry: Optional[str] = None,
            average_strands: Optional[bool] = None
        ) -> np.ndarray:
        """
        Fit the model to the SNP data stored in the provided `snpobj` and apply the dimensionality reduction on the same SNP data.
        
        This method starts by loading or updating SNP and ancestry data. Then, it manages missing values by applying 
        masks based on ancestry, either by loading a pre-existing mask or generating new ones. After processing these 
        masks to produce an incomplete SNP data matrix, it applies the chosen mdPCA method to reduce dimensionality 
        while handling missing data as specified.

        Args:
            snpobj (SNPObject, optional): 
                A SNPObject instance.
            laiobj (LAIObject, optional): 
                A LAIObject instance.
            labels_file (str, optional): 
                Path to the labels file in .tsv format. The first column, `indID`, contains the individual identifiers, and the second 
                column, `label`, specifies the groups for all individuals. If `is_weighted=True`, a `weight` column with individual 
                weights is required. Optionally, `combination` and `combination_weight` columns can specify sets of individuals to be 
                combined into groups, with respective weights.
            ancestry (str, optional): 
                Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at 0.
            average_strands (bool, optional): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
                If None, defaults to `self.average_strands`.

        Returns:
            **array of shape (n_samples, n_components):** 
                The transformed SNP data projected onto the `n_components` principal components, stored in `self.X_new_`.
        """
        if snpobj is None:
            snpobj = self.snpobj
        if laiobj is None:
            laiobj = self.laiobj
        if labels_file is None:
            labels_file = self.labels_file
        if ancestry is None:
            ancestry = self.ancestry
        if average_strands is None:
            average_strands = self.average_strands
        
        if self.load_masks:
            masks, rs_id_list, ind_id_list, _, weights = self._load_mask_file()
        else:
            masks, rs_id_list, ind_id_list = array_process(
                self.snpobj,
                self.laiobj,
                self.average_strands,
                self.prob_thresh,
                self.is_masked,
                self.rsid_or_chrompos
            )

            masks, ind_id_list, _, weights = process_labels_weights(
                self.labels_file,
                masks,
                rs_id_list,
                ind_id_list,
                self.average_strands,
                self.ancestry,
                self.min_percent_snps,
                self.groups_to_remove,
                self.is_weighted,
                self.save_masks,
                self.masks_file
            )

        X_incomplete, _, _ = self._process_masks(masks, rs_id_list, ind_id_list)

        # Call run_cov_matrix with the specified method
        self.X_new_ = self._run_cov_matrix(
            X_incomplete,
            weights
        )

        return self.X_new_
