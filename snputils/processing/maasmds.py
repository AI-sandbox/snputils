import pathlib
import numpy as np
import copy
from typing import Dict, Optional, List, Sequence, Union

from snputils.snp.genobj.snpobj import SNPObject
from snputils.ancestry.genobj.local import LocalAncestryObject
from ._utils.mds_distance import (
    binary_intersection,
    conversion_metrics,
    distance_mat,
    distance_overlap,
    mds_transform,
    overlap_blocks,
)
from ._utils.gen_tools import process_calldata_gt, process_labels_weights


class maasMDS:
    """Multi-array, ancestry-specific multidimensional scaling (maasMDS) on SNP data.

    When ``is_masked`` is True, genotype entries not attributed to the chosen ancestry are set to
    missing so distances reflect the ancestry segment of interest. You can keep haplotypes separate
    or average parental strands (see ``average_strands``). The workflow supports individual-level
    genotypes, group-level allele frequencies, or a mixture when weighting and ``combination``
    columns are used in the labels file.

    **Multiple arrays.** Pass a sequence of :class:`~snputils.snp.genobj.snpobj.SNPObject` and a
    parallel sequence of :class:`~snputils.ancestry.genobj.local.LocalAncestryObject` instances
    (one LAI object per array). Genotypes are harmonized to a shared reference allele across
    arrays where possible; pairwise distances between arrays use overlapping SNPs and linear
    calibration so embeddings are comparable. After :meth:`fit_transform`, ``array_labels_`` holds
    the array index for each row of ``X_new_``.

    If ``snpobj``, ``laiobj``, ``labels_file``, and ``ancestry`` are all provided at construction
    time, :meth:`fit_transform` runs immediately.
    """
    def __init__(
            self, 
            snpobj: Optional[Union['SNPObject', Sequence['SNPObject']]] = None,
            laiobj: Optional[Union['LocalAncestryObject', Sequence['LocalAncestryObject']]] = None,
            labels_file: Optional[str] = None,
            ancestry: Optional[Union[int, str]] = None,
            is_masked: bool = True,
            average_strands: bool = False,
            force_nan_incomplete_strands: bool = False,
            is_weighted: bool = False,
            groups_to_remove: Optional[Union[Dict[int, List[str]], List[str], Sequence[List[str]]]] = None,
            min_percent_snps: float = 4,
            group_snp_frequencies_only: bool = True,
            save_masks: bool = False,
            load_masks: bool = False,
            masks_file: Union[str, pathlib.Path] = 'masks.npz',
            distance_type: str = 'AP',
            n_components: int = 2,
            rsid_or_chrompos: int = 2,
            embedding_table_path: Optional[Union[str, pathlib.Path]] = None,
        ):
        """
        Args:
            snpobj (SNPObject or sequence of SNPObject, optional):
                One SNP object or a list/tuple of objects, one per genotyping array.
            laiobj (LocalAncestryObject or sequence of LocalAncestryObject, optional):
                Local ancestry object(s) parallel to ``snpobj`` when ``is_masked`` is True.
            labels_file (str, optional):
                Path to a TSV with at least columns ``indID`` and ``label``. If ``is_weighted`` is
                True, a ``weight`` column is required. Optional ``combination`` and
                ``combination_weight`` columns define merged groups.
            ancestry (int or str, optional):
                Target ancestry index or name. Indices start at ``0``. Accepts an ``int``, a
                numeric string (e.g. ``\"0\"``), or a string equal to a value in the LAI ancestry map.
            is_masked (bool, optional):
                If True (default), keep only genotypes assigned to ``ancestry``; otherwise use the
                full matrix.
            average_strands (bool, optional):
                If True, average the two haplotypes per individual.
            force_nan_incomplete_strands (bool, optional):
                If True, strand pairs with any missing value become NaN; if False, average while
                ignoring NaNs (e.g. ``0`` with NaN yields ``0``).
            is_weighted (bool, optional):
                If True, read per-individual weights from the labels file.
            groups_to_remove (optional):
                Labels to drop before analysis: ``None``; a dict mapping **1-based** array index to
                a list of labels; a single list of labels applied to every array; or a sequence of
                length ``num_arrays`` with one list of labels per array.
            min_percent_snps (float, optional):
                Minimum fraction of non-missing SNPs per individual (default ``4`` means 4%).
            group_snp_frequencies_only (bool, optional):
                If True, use only group-level frequencies when combinations are defined; if False,
                keep individual-level (and optionally group-level) inputs.
            save_masks (bool, optional):
                If True, write masks and sidecar arrays to ``masks_file``.
            load_masks (bool, optional):
                If True, read precomputed masks from ``masks_file`` instead of genotypes.
            masks_file (str or pathlib.Path, optional):
                Path for the compressed ``.npz`` mask archive.
            distance_type (str, optional):
                ``\"Manhattan\"``, ``\"RMS\"``, or ``\"AP\"`` (average pairwise). With
                ``average_strands=True``, ``\"AP\"`` is appropriate.
            n_components (int, optional):
                Embedding dimension (default ``2``).
            rsid_or_chrompos (int, optional):
                ``1`` for rsID-style IDs, ``2`` for chromosome/position encoding (default ``2``).
            embedding_table_path (path, optional):
                If set, :meth:`fit_transform` writes ``X_new_`` with row metadata to this TSV/CSV path
                (see :mod:`snputils.processing.dimred_tabular`).
        """
        self.__snpobj = snpobj
        self.__laiobj = laiobj
        self.__labels_file = labels_file
        ancestry_map = self._resolve_ancestry_map(laiobj)
        self.__ancestry = self._define_ancestry(ancestry, ancestry_map) if ancestry_map is not None and ancestry is not None else ancestry
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
        self.__embedding_table_path = (
            pathlib.Path(embedding_table_path) if embedding_table_path is not None else None
        )
        self.__X_new_ = None  # Store transformed SNP data
        self.__haplotypes_ = None  # Store haplotypes after filtering if min_percent_snps > 0
        self.__samples_ = None  # Store samples after filtering if min_percent_snps > 0
        self.__variants_id_ = None  # Store variants ID (after filtering SNPs not in laiobj)
        self.array_labels_ = None  # Store per-individual array membership after filtering

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
            maasMDS: 
                A new instance of the current object.
        """
        return copy.copy(self)

    @property
    def snpobj(self) -> Optional[Union['SNPObject', Sequence['SNPObject']]]:
        """
        Retrieve `snpobj`.
        
        Returns:
            SNPObject: A SNPObject instance.
        """
        return self.__snpobj

    @snpobj.setter
    def snpobj(self, x: Union['SNPObject', Sequence['SNPObject']]) -> None:
        """
        Update `snpobj`.
        """
        self.__snpobj = x

    @property
    def laiobj(self) -> Optional[Union['LocalAncestryObject', Sequence['LocalAncestryObject']]]:
        """
        Retrieve `laiobj`.
        
        Returns:
            LocalAncestryObject or sequence thereof: Local ancestry object(s) for masking.
        """
        return self.__laiobj

    @laiobj.setter
    def laiobj(self, x: Union['LocalAncestryObject', Sequence['LocalAncestryObject']]) -> None:
        """
        Update `laiobj`.
        """
        self.__laiobj = x

    @property
    def labels_file(self) -> Optional[str]:
        """
        Retrieve `labels_file`.
        
        Returns:
            str: 
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
    def ancestry(self) -> Optional[int]:
        """
        Retrieve `ancestry`.
        
        Returns:
            int: Ancestry index for which dimensionality reduction is to be performed. Ancestry counter starts at `0`.
        """
        return self.__ancestry

    @ancestry.setter
    def ancestry(self, x: Union[int, str]) -> None:
        """
        Update `ancestry`.
        """
        ancestry_map = self._resolve_ancestry_map(self.laiobj)
        self.__ancestry = self._define_ancestry(x, ancestry_map) if ancestry_map is not None else x

    @property
    def is_masked(self) -> bool:
        """
        Retrieve `is_masked`.
        
        Returns:
            bool: True if an ancestry file is passed for ancestry-specific masking, or False otherwise.
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
            bool: True if the haplotypes from the two parents are to be combined (averaged) for each individual, or False otherwise.
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
            bool: If `True`, sets the result to NaN if either haplotype in a pair is NaN.
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
            bool: True if weights are provided in the labels file, or False otherwise.
        """
        return self.__is_weighted

    @is_weighted.setter
    def is_weighted(self, x: bool) -> None:
        """
        Update `is_weighted`.
        """
        self.__is_weighted = x

    @property
    def groups_to_remove(self) -> Optional[Union[Dict[int, List[str]], List[str], Sequence[List[str]]]]:
        """
        Retrieve `groups_to_remove`.
        
        Returns:
            A flat removal list or a per-array mapping of labels to remove.
        """
        return self.__groups_to_remove

    @groups_to_remove.setter
    def groups_to_remove(self, x: Optional[Union[Dict[int, List[str]], List[str], Sequence[List[str]]]]) -> None:
        """
        Update `groups_to_remove`.
        """
        self.__groups_to_remove = x

    @property
    def min_percent_snps(self) -> float:
        """
        Retrieve `min_percent_snps`.
        
        Returns:
            float: 
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
            bool: 
                If True, maasMDS is performed exclusively on group-level SNP frequencies, ignoring individual-level data. This applies 
                when `is_weighted` is set to True and a `combination` column is provided in the `labels_file`,  meaning individuals are 
                aggregated into groups based on their assigned labels. If False, maasMDS is performed on individual-level SNP data alone 
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
            bool: True if the masked matrices are to be saved in a `.npz` file, or False otherwise.
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
            bool: 
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
            str or pathlib.Path: Path to the `.npz` file used for saving/loading masked matrices.
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
            str: 
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
            int: The number of principal components.
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
            int: Format indicator for SNP IDs in the SNP data. Use 1 for `rsID` format or 2 for `chromosome_position`.
        """
        return self.__rsid_or_chrompos

    @rsid_or_chrompos.setter
    def rsid_or_chrompos(self, x: int) -> None:
        """
        Update `rsid_or_chrompos`.
        """
        self.__rsid_or_chrompos = x

    @property
    def embedding_table_path(self) -> Optional[pathlib.Path]:
        """Optional path for the tabular embedding written by :meth:`fit_transform`."""
        return self.__embedding_table_path

    @embedding_table_path.setter
    def embedding_table_path(self, x: Optional[Union[str, pathlib.Path]]) -> None:
        self.__embedding_table_path = pathlib.Path(x) if x is not None else None

    @property
    def X_new_(self) -> Optional[np.ndarray]:
        """
        Retrieve `X_new_`.

        Returns:
            array: 
                The transformed SNP data projected onto the `n_components` principal components.
                ``n_haplotypes_`` is the number of haplotypes, potentially reduced if filtering is applied 
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
    def variants_id_(self) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """
        Retrieve `variants_id_`.

        Returns:
            numpy.ndarray or list of numpy.ndarray:
                Per-SN identifiers after LAI alignment. For a single array this is a 1-D array; for
                multiple arrays, a list with one array of IDs per array (overlap sets can differ).
                Interpretation follows ``rsid_or_chrompos``.
        """
        return self.__variants_id_

    @variants_id_.setter
    def variants_id_(self, x: Union[np.ndarray, List[np.ndarray]]) -> None:
        """
        Update `variants_id_`.
        """
        self.__variants_id_ = x

    @property
    def n_haplotypes(self) -> Optional[int]:
        """
        Retrieve `n_haplotypes`.

        Returns:
            int or None:
                The total number of haplotypes, potentially reduced if filtering is applied 
                (`min_percent_snps > 0`). ``None`` before :meth:`fit_transform` has been called.
        """
        haplotypes = self.haplotypes_
        return None if haplotypes is None else len(haplotypes)

    @property
    def n_samples(self) -> Optional[int]:
        """
        Retrieve ``n_samples``.

        Returns:
            int or None:
                The total number of samples, potentially reduced if filtering is applied 
                (`min_percent_snps > 0`). ``None`` before :meth:`fit_transform` has been called.
        """
        samples = self.samples_
        return None if samples is None else len(np.unique(samples))

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
    def _resolve_ancestry_map(laiobj):
        if laiobj is None:
            return None
        if isinstance(laiobj, (list, tuple)):
            if len(laiobj) == 0:
                return None
            return laiobj[0].ancestry_map
        return laiobj.ancestry_map

    @staticmethod
    def _as_object_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _normalize_groups_to_remove(
        groups_to_remove: Optional[Union[Dict[int, List[str]], List[str], Sequence[List[str]]]],
        num_arrays: int,
    ) -> List[List[str]]:
        if groups_to_remove is None:
            return [[] for _ in range(num_arrays)]
        if isinstance(groups_to_remove, dict):
            return [list(groups_to_remove.get(i + 1, [])) for i in range(num_arrays)]
        if isinstance(groups_to_remove, (list, tuple)):
            if len(groups_to_remove) == 0 or all(isinstance(x, str) for x in groups_to_remove):
                return [list(groups_to_remove) for _ in range(num_arrays)]
            if len(groups_to_remove) != num_arrays:
                raise ValueError(
                    "Per-array `groups_to_remove` sequences must match the number of arrays."
                )
            return [list(x) for x in groups_to_remove]
        raise TypeError(
            "`groups_to_remove` must be None, a dict keyed by 1-based array number, "
            "a flat list of labels, or a per-array sequence of label lists."
        )

    @staticmethod
    def _coerce_loaded_value(value):
        if isinstance(value, np.ndarray) and value.dtype == object:
            if value.shape == ():
                return value.item()
            return value.tolist()
        return value

    @staticmethod
    def _normalize_mask_keys(masks):
        """Convert string ancestry keys (e.g. ``'3'``) to ``int`` in mask dicts."""
        for i, m in enumerate(masks):
            if isinstance(m, dict):
                masks[i] = {
                    (int(k) if isinstance(k, str) and k.isdigit() else k): v
                    for k, v in m.items()
                }
        return masks

    @staticmethod
    def _load_masks_file(masks_file):
        mask_files = np.load(masks_file, allow_pickle=True)
        groups = np.asarray(mask_files['labels'])
        weights = np.asarray(mask_files['weights'])

        if 'masks' in mask_files:
            masks = maasMDS._coerce_loaded_value(mask_files['masks'])
            masks = maasMDS._normalize_mask_keys(list(masks))
            variants_key = 'variants_id_list' if 'variants_id_list' in mask_files else 'rs_ID_list'
            haplotypes_key = 'haplotypes_list' if 'haplotypes_list' in mask_files else 'ind_ID_list'
            variants_id_list = maasMDS._coerce_loaded_value(mask_files[variants_key])
            haplotypes_list = maasMDS._coerce_loaded_value(mask_files[haplotypes_key])
            return (
                masks,
                [np.asarray(x) for x in variants_id_list],
                [np.asarray(x) for x in haplotypes_list],
                groups,
                weights,
            )

        mask = maasMDS._coerce_loaded_value(mask_files['mask'])
        if isinstance(mask, dict):
            mask = {(int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in mask.items()}
        variants_key = 'variants_id' if 'variants_id' in mask_files else 'rs_ID_list'
        haplotypes_key = 'haplotypes' if 'haplotypes' in mask_files else 'ind_ID_list'
        variants_id = maasMDS._coerce_loaded_value(mask_files[variants_key])
        haplotypes = maasMDS._coerce_loaded_value(mask_files[haplotypes_key])
        return [mask], [np.asarray(variants_id)], [np.asarray(haplotypes)], groups, weights

    @staticmethod
    def _save_masks_file(masks_file, masks, variants_id_list, haplotypes_list, groups, weights):
        save_payload = {
            'masks': np.asarray(masks, dtype=object),
            'variants_id_list': np.asarray(variants_id_list, dtype=object),
            'haplotypes_list': np.asarray(haplotypes_list, dtype=object),
            'rs_ID_list': np.asarray(variants_id_list, dtype=object),
            'ind_ID_list': np.asarray(haplotypes_list, dtype=object),
            'labels': np.asarray(groups),
            'weights': np.asarray(weights),
        }
        if len(masks) == 1:
            single_mask = np.empty((), dtype=object)
            single_mask[()] = masks[0]
            save_payload['mask'] = single_mask
            save_payload['variants_id'] = np.asarray(variants_id_list[0])
            save_payload['haplotypes'] = np.asarray(haplotypes_list[0])
        np.savez_compressed(masks_file, **save_payload)

    def _process_input_arrays(self, snpobjs, laiobjs, labels_file, ancestry, average_strands):
        if len(snpobjs) == 0:
            raise ValueError("At least one `snpobj` must be provided.")
        if self.is_masked and len(snpobjs) != len(laiobjs):
            raise ValueError("`snpobj` and `laiobj` must contain the same number of arrays.")
        if not self.is_masked and len(laiobjs) == 0:
            laiobjs = [None] * len(snpobjs)
        if labels_file is None:
            raise ValueError("`labels_file` is required unless `load_masks=True`.")
        groups_to_remove = self._normalize_groups_to_remove(
            self.groups_to_remove,
            len(snpobjs),
        )

        masks = []
        variants_id_list = []
        haplotypes_list = []
        groups = []
        weights = []
        variants_ref_map = {}

        for array_index, current_snpobj in enumerate(snpobjs):
            current_laiobj = laiobjs[array_index] if array_index < len(laiobjs) else None
            mask, variants_id, haplotypes, variants_ref_map = process_calldata_gt(
                current_snpobj,
                current_laiobj,
                ancestry,
                average_strands,
                self.force_nan_incomplete_strands,
                self.is_masked,
                self.rsid_or_chrompos,
                variants_ref_map=variants_ref_map,
            )
            mask, haplotypes, current_groups, current_weights = process_labels_weights(
                labels_file,
                mask,
                variants_id,
                haplotypes,
                average_strands,
                ancestry,
                self.min_percent_snps,
                self.group_snp_frequencies_only,
                groups_to_remove[array_index],
                self.is_weighted,
                False,
                self.masks_file,
            )
            masks.append(mask)
            variants_id_list.append(np.asarray(variants_id))
            haplotypes_list.append(np.asarray(haplotypes))
            groups.append(np.asarray(current_groups))
            weights.append(np.asarray(current_weights))

        return (
            masks,
            variants_id_list,
            haplotypes_list,
            np.concatenate(groups) if groups else np.array([]),
            np.concatenate(weights) if weights else np.array([]),
        )

    def fit_transform(
            self,
            snpobj: Optional[Union['SNPObject', Sequence['SNPObject']]] = None, 
            laiobj: Optional[Union['LocalAncestryObject', Sequence['LocalAncestryObject']]] = None,
            labels_file: Optional[str] = None,
            ancestry: Optional[Union[int, str]] = None,
            average_strands: Optional[bool] = None
        ) -> np.ndarray:
        """
        Estimate the MDS embedding and store it on the instance.

        Omitted arguments fall back to attributes set on the object or in ``__init__``.

        Args:
            snpobj (SNPObject or sequence of SNPObject, optional):
                Input genotype container(s).
            laiobj (LocalAncestryObject or sequence of LocalAncestryObject, optional):
                Matching LAI object(s) when masking is enabled.
            labels_file (str, optional):
                TSV path with ``indID`` / ``label`` (and optional weight / combination columns).
            ancestry (int or str, optional):
                Same conventions as in ``__init__``.
            average_strands (bool, optional):
                If omitted, uses ``self.average_strands``.

        Returns:
            numpy.ndarray:
                Embedding of shape ``(n_rows, n_components)`` with ``n_rows`` equal to the number of
                haplotypes (or samples if strands are averaged) after ``min_percent_snps`` filtering.
                Also assigned to ``X_new_``; row-wise array indices are in ``array_labels_`` when
                multiple arrays are combined.
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

        self.__snpobj = snpobj
        self.__laiobj = laiobj
        self.__labels_file = labels_file
        self.__average_strands = average_strands

        ancestry_map = self._resolve_ancestry_map(laiobj)
        if ancestry is not None and ancestry_map is not None:
            ancestry = self._define_ancestry(ancestry, ancestry_map)
        analysis_ancestry = ancestry if self.is_masked else 1
        self.__ancestry = analysis_ancestry

        snpobjs = self._as_object_list(snpobj)
        laiobjs = self._as_object_list(laiobj)

        if self.load_masks:
            masks, variants_id_list, haplotypes_list, groups, weights = self._load_masks_file(
                self.masks_file
            )
        else:
            masks, variants_id_list, haplotypes_list, groups, weights = self._process_input_arrays(
                snpobjs,
                laiobjs,
                labels_file,
                analysis_ancestry,
                average_strands,
            )
            if self.save_masks:
                self._save_masks_file(
                    self.masks_file,
                    masks,
                    variants_id_list,
                    haplotypes_list,
                    groups,
                    weights,
                )

        num_arrays = len(masks)
        if num_arrays == 0:
            raise ValueError("No arrays available for maasMDS processing.")

        if num_arrays > 1:
            ref_row = 0
            ref_col = 1
            binary = binary_intersection(variants_id_list)
            overlap = overlap_blocks(
                analysis_ancestry,
                ref_col,
                ref_row,
                num_arrays,
                variants_id_list,
                binary,
                masks,
            )
            conversion, intercept = conversion_metrics(
                analysis_ancestry,
                ref_col,
                ref_row,
                num_arrays,
                variants_id_list,
                binary,
                masks,
                self.distance_type,
            )
            distance_list = distance_overlap(
                ref_col,
                ref_row,
                num_arrays,
                overlap,
                conversion,
                intercept,
                self.distance_type,
            )
            ind_id_arg = haplotypes_list
        else:
            distance_list = [[distance_mat(first=masks[0][analysis_ancestry], dist_func=self.distance_type)]]
            ind_id_arg = haplotypes_list[0]

        transformed, haplotypes, _, array_labels = mds_transform(
            distance_list,
            np.asarray(groups),
            np.asarray(weights),
            ind_id_arg,
            self.n_components,
            num_arrays=num_arrays,
            imputation_method="mean",
            return_metadata=True,
        )

        self.X_new_ = transformed
        self.haplotypes_ = haplotypes
        self.variants_id_ = variants_id_list[0] if num_arrays == 1 else [np.asarray(x) for x in variants_id_list]
        self.array_labels_ = np.asarray(array_labels)

        from .dimred_tabular import try_save_embedding_table

        try_save_embedding_table(self, self.__embedding_table_path)

        return self.X_new_
