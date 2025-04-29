import pathlib
import numpy as np
import copy
from typing import Optional, Dict, List, Union

from snputils.snp.genobj.snpobj import SNPObject
from snputils.ancestry.genobj.local import LocalAncestryObject
from ._utils.mds_distance import distance_mat, mds_transform
from ._utils.gen_tools import process_calldata_gt, process_labels_weights


class maasMDS:
    """
    A class for performing multiple array ancestry-specific multidimensional scaling (maasMDS) on SNP data.

    The maasMDS class focuses on genotype segments from the ancestry of interest when the `is_masked` flag is set to `True`. It offers 
    flexible processing options, allowing either separate handling of masked haplotype strands or combining (averaging) strands into a 
    single composite representation for each individual. Moreover, the analysis can be performed on individual-level data, group-level SNP 
    frequencies, or a combination of both.

    This class supports both separate and averaged strand processing for SNP data. If the `snpobj`, 
    `laiobj`, `labels_file`, and `ancestry` parameters are all provided during instantiation, 
    the `fit_transform` method will be automatically called, applying the specified maasMDS method to transform 
    the data upon instantiation.
    """
    def __init__(
            self, 
            snpobj: Optional['SNPObject'] = None,
            laiobj: Optional['LocalAncestryObject'] = None,
            labels_file: Optional[str] = None,
            ancestry: Optional[Union[int, str]] = None,
            is_masked: bool = True,
            average_strands: bool = False,
            force_nan_incomplete_strands: bool = False,
            is_weighted: bool = False,
            groups_to_remove: Dict[int, List[str]] = {},
            min_percent_snps: float = 4,
            group_snp_frequencies_only: bool = True,
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
            ancestry (int or str, optional): 
                Ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at `0`. The ancestry input can be:
                - An integer (e.g., 0, 1, 2).
                - A string representation of an integer (e.g., '0', '1').
                - A string matching one of the ancestry map values (e.g., 'Africa').
            is_masked (bool, default=True): 
                If `True`, applies ancestry-specific masking to the genotype matrix, retaining only genotype data 
                corresponding to the specified `ancestry`. If `False`, uses the full, unmasked genotype matrix.
            average_strands (bool, default=False): 
                True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
            force_nan_incomplete_strands (bool): 
                If `True`, sets the result to NaN if either haplotype in a pair is NaN. 
                Otherwise, computes the mean while ignoring NaNs (e.g., 0|NaN -> 0, 1|NaN -> 1).
            is_weighted (bool, default=False): 
                True if weights are provided in the labels file, or False otherwise.
            groups_to_remove (dict of int to list of str, default={}): 
                Dictionary specifying groups to exclude from analysis. Keys are array numbers, and values are 
                lists of groups to remove for each array.
                Example: `{1: ['group1', 'group2'], 2: [], 3: ['group3']}`.
            min_percent_snps (float, default=4.0): 
                Minimum percentage of SNPs to be known in an individual for an individual to be included in the analysis. 
                All individuals with fewer percent of unmasked SNPs than this threshold will be excluded.
            group_snp_frequencies_only (bool, default=True):
                If True, mdPCA is performed exclusively on group-level SNP frequencies, ignoring individual-level data. This applies when `is_weighted` is 
                set to True and a `combination` column is provided in the `labels_file`,  meaning individuals are aggregated into groups based on their assigned 
                labels. If False, mdPCA is performed on individual-level SNP data alone or on both individual-level and group-level SNP frequencies when 
                `is_weighted` is True and a `combination` column is provided.
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
        self.__ancestry = self._define_ancestry(ancestry, laiobj.ancestry_map)
        self.__is_masked = is_masked
        self.__average_strands = average_strands
        self.__force_nan_incomplete_strands = force_nan_incomplete_strands
        self.__groups_to_remove = groups_to_remove
        self.__min_percent_snps = min_percent_snps
        self.__group_snp_frequencies_only = group_snp_frequencies_only
        self.__is_weighted = is_weighted
        self.__save_masks = save_masks
        self.__load_masks = load_masks
        self.__masks_file = masks_file
        self.__distance_type = distance_type
        self.__n_components = n_components
        self.__rsid_or_chrompos = rsid_or_chrompos
        self.__X_new_ = None  # Store transformed SNP data
        self.__haplotypes_ = None  # Store haplotypes after filtering if min_percent_snps > 0
        self.__samples_ = None  # Store samples after filtering if min_percent_snps > 0
        self.__variants_id_ = None  # Store variants ID (after filtering SNPs not in laiobj)

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
    def laiobj(self) -> Optional['LocalAncestryObject']:
        """
        Retrieve `laiobj`.
        
        Returns:
            **LocalAncestryObject:** A LAIObject instance.
        """
        return self.__laiobj

    @laiobj.setter
    def laiobj(self, x: 'LocalAncestryObject') -> None:
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
    def force_nan_incomplete_strands(self) -> bool:
        """
        Retrieve `force_nan_incomplete_strands`.
        
        Returns:
            **bool**: If `True`, sets the result to NaN if either haplotype in a pair is NaN.
                      Otherwise, computes the mean while ignoring NaNs (e.g., 0|NaN -> 0, 1|NaN -> 1).
        """
        return self.__force_nan_incomplete_strands

    @force_nan_incomplete_strands.setter
    def force_nan_incomplete_strands(self, x: bool) -> None:
        """
        Update `force_nan_incomplete_strands`.
        """
        self.__force_nan_incomplete_strands = x

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
    def group_snp_frequencies_only(self) -> bool:
        """
        Retrieve `group_snp_frequencies_only`.
        
        Returns:
            **bool:** 
                If True, mdPCA is performed exclusively on group-level SNP frequencies, ignoring individual-level data. This applies 
                when `is_weighted` is set to True and a `combination` column is provided in the `labels_file`,  meaning individuals are 
                aggregated into groups based on their assigned labels. If False, mdPCA is performed on individual-level SNP data alone 
                or on both individual-level and group-level SNP frequencies when `is_weighted` is True and a `combination` column is provided.
        """
        return self.__group_snp_frequencies_only

    @group_snp_frequencies_only.setter
    def group_snp_frequencies_only(self, x: bool) -> None:
        """
        Update `group_snp_frequencies_only`.
        """
        self.__group_snp_frequencies_only = x

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
            **array of shape (n_haplotypes_, n_components):** 
                The transformed SNP data projected onto the `n_components` principal components.
                n_haplotypes_ is the number of haplotypes, potentially reduced if filtering is applied 
                (`min_percent_snps > 0`). For diploid individuals without filtering, the shape is 
                `(n_samples * 2, n_components)`.
        """
        return self.__X_new_

    @X_new_.setter
    def X_new_(self, x: np.ndarray) -> None:
        """
        Update `X_new_`.
        """
        self.__X_new_ = x

    @property
    def haplotypes_(self) -> Optional[List[str]]:
        """
        Retrieve `haplotypes_`.

        Returns:
            list of str:
                A list of unique haplotype identifiers.
        """
        if isinstance(self.__haplotypes_, np.ndarray):
            return self.__haplotypes_.ravel().tolist()  # Flatten and convert NumPy array to a list
        elif isinstance(self.__haplotypes_, list):
            if len(self.__haplotypes_) == 1 and isinstance(self.__haplotypes_[0], np.ndarray):
                return self.__haplotypes_[0].ravel().tolist()  # Handle list containing a single array
            return self.__haplotypes_  # Already a flat list
        elif self.__haplotypes_ is None:
            return None  # If no haplotypes are set
        else:
            raise TypeError("`haplotypes_` must be a list or a NumPy array.")

    @haplotypes_.setter
    def haplotypes_(self, x: Union[np.ndarray, List[str]]) -> None:
        """
        Update `haplotypes_`.
        """
        if isinstance(x, np.ndarray):
            self.__haplotypes_ = x.ravel().tolist()  # Flatten and convert to a list
        elif isinstance(x, list):
            if len(x) == 1 and isinstance(x[0], np.ndarray):  # Handle list containing a single array
                self.__haplotypes_ = x[0].ravel().tolist()
            else:
                self.__haplotypes_ = x  # Use directly if already a list
        else:
            raise TypeError("`x` must be a list or a NumPy array.")

    @property
    def samples_(self) -> Optional[List[str]]:
        """
        Retrieve `samples_`.

        Returns:
            list of str:
                A list of sample identifiers based on `haplotypes_` and `average_strands`.
        """
        haplotypes = self.haplotypes_
        if haplotypes is None:
            return None
        if self.__average_strands:
            return haplotypes
        else:
            return [x[:-2] for x in haplotypes]

    @property
    def variants_id_(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_id_`.

        Returns:
            **array of shape (n_snp,):** 
                An array containing unique identifiers (IDs) for each SNP,
                potentially reduced if there are SNPs not present in the `laiboj`.
                The format will depend on `rsid_or_chrompos`.
        """
        return self.__variants_id_

    @variants_id_.setter
    def variants_id_(self, x: np.ndarray) -> None:
        """
        Update `variants_id_`.
        """
        self.__variants_id_ = x

    @property
    def n_haplotypes(self) -> Optional[int]:
        """
        Retrieve `n_haplotypes`.

        Returns:
            **int:**
                The total number of haplotypes, potentially reduced if filtering is applied 
                (`min_percent_snps > 0`).
        """
        return len(self.__haplotypes_)

    @property
    def n_samples(self) -> Optional[int]:
        """
        Retrieve `n_samples`.

        Returns:
            **int:**
                The total number of samples, potentially reduced if filtering is applied 
                (`min_percent_snps > 0`).
        """
        return len(np.unique(self.samples_))

    @staticmethod
    def _define_ancestry(ancestry, ancestry_map):
        """
        Determine the ancestry index based on different input types.

        Args:
            ancestry (int or str): The ancestry input, which can be:
                - An integer (e.g., 0, 1, 2).
                - A string representation of an integer (e.g., '0', '1').
                - A string matching one of the ancestry map values (e.g., 'Africa').
            ancestry_map (dict): A dictionary mapping ancestry indices (as strings) to ancestry names.

        Returns:
            int: The corresponding ancestry index.
        """
        if isinstance(ancestry, int):  
            return ancestry  
        elif isinstance(ancestry, str) and ancestry.isdigit():  
            return int(ancestry)  
        elif ancestry in ancestry_map.values():  
            return int(next(key for key, value in ancestry_map.items() if value == ancestry))  
        else:  
            raise ValueError(f"Invalid ancestry input: {ancestry}")

    @staticmethod
    def _load_masks_file(masks_file):
        mask_files = np.load(masks_file, allow_pickle=True)
        mask = mask_files['mask']
        rs_ID_list = mask_files['rs_ID_list']
        ind_ID_list = mask_files['ind_ID_list']
        groups = mask_files['labels']
        weights = mask_files['weights']
        return mask, rs_ID_list, ind_ID_list, groups, weights

    def fit_transform(
            self,
            snpobj: Optional['SNPObject'] = None, 
            laiobj: Optional['LocalAncestryObject'] = None,
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
            self.ancestry = 1
        if self.load_masks:
            # Load precomputed ancestry-based masked genotype matrix, SNP identifiers, haplotype identifiers, and weights
            mask, variants_id, haplotypes, _, weights = self._load_masks_file(self.masks_file)
        else:
            # Process genotype data with optional ancestry-based masking and return the corresponding SNP and individual identifiers
            mask, variants_id, haplotypes = process_calldata_gt(
                self.snpobj,
                self.laiobj,
                self.ancestry,
                self.average_strands,
                self.force_nan_incomplete_strands,
                self.is_masked, 
                self.rsid_or_chrompos
            )

            # Process individual genomic labels and weights, aligning them with a masked genotype matrix by 
            # filtering out low-coverage individuals, reordering data to match the matrix structure, and 
            # handling group-based adjustments
            mask, haplotypes, groups, weights = process_labels_weights(
                self.labels_file,
                mask,
                variants_id,
                haplotypes,
                self.average_strands,
                self.ancestry,
                self.min_percent_snps,
                self.group_snp_frequencies_only,
                self.groups_to_remove,
                self.is_weighted,
                self.save_masks,
                self.masks_file
            )
        
        distance_list = [[distance_mat(first=mask[self.ancestry], dist_func=self.distance_type)]]
        
        self.X_new_ = mds_transform(distance_list, groups, weights, haplotypes, self.n_components)
        
        self.haplotypes_ = haplotypes
        self.variants_id_ = variants_id
