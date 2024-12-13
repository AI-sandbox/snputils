import pathlib
import numpy as np
import copy
from typing import Optional, Dict, List, Union

from snputils.snp.genobj.snpobj import SNPObject
from snputils.ancestry.genobj.window import WindowLevelAncestryObject
from ._utils.mds_distance import distance_mat, mds_transform
from ._utils.gen_tools import array_process, process_labels_weights


class maasMDS:
    """
    A class for multiple array ancestry-specific multidimensional scaling (maasMDS).

    This class supports both separate and averaged strand processing for SNP data. If the `snpobj`, 
    `laiobj`, `labels_file`, and `ancestry` parameters are all provided during instantiation, 
    the `fit_transform` method will be automatically called, applying the specified maasMDS method to transform 
    the data upon instantiation.
    """
    def __init__(
            self, 
            snpobj, 
            laiobj,
            labels_file,
            ancestry,
            is_masked: bool = True,
            prob_thresh: float = 0,
            average_strands: bool = False,
            is_weighted: bool = False,
            groups_to_remove: Dict[int, List[str]] = {},
            min_percent_snps: float = 4,
            save_masks: bool = False,
            load_masks: bool = False,
            masks_file: Union[str, pathlib.Path] = 'masks.npz',
            distance_type: str = 'AP',
            n_components: int = 2,
            rsid_or_chrompos: int = 2
        ):
        """
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
                Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
            is_masked (bool, default=True): 
                True if an ancestry file is passed for ancestry-specific masking, or False otherwise.
            prob_thresh (float, default=0.0): 
                Minimum probability threshold for a SNP to belong to an ancestry.
            average_strands (bool, default=False): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
            is_weighted (bool, default=False): 
                True if weights are provided in the labels file, or False otherwise.
            groups_to_remove (dict of int to list of str, default={}): 
                Dictionary specifying groups to exclude from analysis. Keys are array numbers, and values are 
                lists of groups to remove for each array.
                Example: `{1: ['group1', 'group2'], 2: [], 3: ['group3']}`.
            min_percent_snps (float, default=4.0): 
                Minimum percentage of SNPs to be known in an individual for an individual to be included in the analysis. 
                All individuals with fewer percent of unmasked SNPs than this threshold will be excluded.
            save_masks (bool, default=False): 
                True if the masked matrices are to be saved in a `.npz` file, or False otherwise.
            load_masks (bool, default=False): 
                True if the masked matrices are to be loaded from a pre-existing `.npz` file specified by `masks_file`, or False otherwise.
            masks_file (str or pathlib.Path, default='masks.npz'): 
                Path to the `.npz` file used for saving/loading masked matrices.
            distance_type (str, default='AP'): 
                Distance metric to use. Options to choose from are: 'Manhattan', 'RMS' (Root Mean Square), 'AP' (Average Pairwise).
                If `average_strands=True`, use 'distance_type=AP'.
            n_components (int, default=2): 
                The number of principal components.
            rsid_or_chrompos (int, default=2): 
                Format indicator for SNP IDs in the SNP data. Use 1 for `rsID` format or 2 for `chromosome_position`.
        """
        self.__snpobj = snpobj
        self.__laiobj = laiobj
        self.__labels_file = labels_file
        self.__ancestry = ancestry
        self.__is_masked = is_masked
        self.__prob_thresh = prob_thresh
        self.__average_strands = average_strands
        self.__groups_to_remove = groups_to_remove
        self.__min_percent_snps = min_percent_snps
        self.__is_weighted = is_weighted
        self.__save_masks = save_masks
        self.__load_masks = load_masks
        self.__masks_file = masks_file
        self.__distance_type = distance_type
        self.__n_components = n_components
        self.__rsid_or_chrompos = rsid_or_chrompos
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
        
    def copy(self) -> 'maasMDS':
        """
        Create and return a copy of `self`.

        Returns:
            **maasMDS:** 
                A new instance of the current object.
        """
        return copy.copy(self)

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
            **WindowLevelAncestryObject:** A LAIObject instance.
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
            **str:** 
                Path to the labels file in `.tsv` format.
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
            **str:** Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
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
        """
        Update `prob_thresh`.
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
                Minimum percentage of SNPs to be known in an individual for an individual to be included in the analysis. 
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
    def distance_type(self) -> str:
        """
        Retrieve `distance_type`.
        
        Returns:
            **str:** 
                Distance metric to use. Options to choose from are: 'Manhattan', 'RMS' (Root Mean Square), 'AP' (Average Pairwise).
                If `average_strands=True`, use 'distance_type=AP'.
        """
        return self.__distance_type

    @distance_type.setter
    def distance_type(self, x: str) -> None:
        """
        Update `distance_type`.
        """
        self.__distance_type = x

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

    @staticmethod
    def _load_masks_file(masks_file):
        mask_files = np.load(masks_file, allow_pickle=True)
        masks = mask_files['masks']
        rs_ID_list = mask_files['rs_ID_list']
        ind_ID_list = mask_files['ind_ID_list']
        groups = mask_files['labels']
        weights = mask_files['weights']
        return masks, rs_ID_list, ind_ID_list, groups, weights

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
        
        if not self.is_masked:
            self.ancestry = '1'
        if self.load_masks:
            masks, rs_ID_list, ind_ID_list, groups, weights = self._load_masks_file(self.masks_file)
        else:
            masks, rs_ID_list, ind_ID_list = array_process(
                self.snpobj,
                self.laiobj,
                self.average_strands,
                self.prob_thresh, 
                self.is_masked, 
                self.rsid_or_chrompos
            )

            masks, ind_ID_list, groups, weights = process_labels_weights(
                self.labels_file, 
                masks, 
                rs_ID_list,
                ind_ID_list, 
                self.average_strands, 
                self.ancestry, 
                self.min_percent_snps, 
                self.groups_to_remove,
                self.is_weighted, 
                self.save_masks, 
                self.masks_file
            )
        
        distance_list = [[distance_mat(first=masks[0][self.ancestry], dist_func=self.distance_type)]]
        
        self.X_new_ = mds_transform(distance_list, groups, weights, ind_ID_list, self.n_components)
