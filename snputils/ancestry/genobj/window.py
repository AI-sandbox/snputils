import logging
from pathlib import Path
import numpy as np
import copy
import warnings
from typing import Union, List, Dict, Sequence, Optional

from .base import AncestryObject
from snputils.snp.genobj.snpobj import SNPObject

log = logging.getLogger(__name__)


class WindowLevelAncestryObject(AncestryObject):
    """
    A class for window-level Local Ancestry Inference (LAI) data.
    """
    def __init__(
        self,
        haplotypes: List[str],
        lai: np.ndarray,
        samples: Optional[List[str]] = None,
        ancestry_map: Optional[Dict[str, str]] = None,
        window_sizes: Optional[np.ndarray] = None,
        centimorgan_pos: Optional[np.ndarray] = None,
        chromosomes: Optional[np.ndarray] = None,
        physical_pos: Optional[np.ndarray] = None
    ) -> None:
        """
        Args:
            haplotypes (list of str of length n_haplotypes): 
                A list of unique haplotype identifiers.
            lai (array of shape (n_windows, n_haplotypes)): 
                A 2D array containing local ancestry inference values, where each row represents a 
                genomic window, and each column corresponds to a haplotype phase for each sample.
            samples (list of str of length n_samples, optional): 
                A list of unique sample identifiers.
            ancestry_map (dict of str to str, optional): 
                A dictionary mapping ancestry codes to region names.
            window_sizes (array of shape (n_windows,), optional): 
                An array specifying the number of SNPs in each genomic window.
            centimorgan_pos (array of shape (n_windows, 2), optional): 
                A 2D array containing the start and end centimorgan positions for each window.
            chromosomes (array of shape (n_windows,), optional): 
                An array with chromosome numbers corresponding to each genomic window.
            physical_pos (array of shape (n_windows, 2), optional): 
                A 2D array containing the start and end physical positions for each window.
        """
        if lai.ndim != 2:
            raise ValueError("`lai` must be a 2D array with shape (n_windows, n_haplotypes).")
        
        # Determine the number of unique ancestries and samples from the LAI array
        n_ancestries = len(np.unique(lai))
        n_haplotypes = lai.shape[1]
        n_samples = n_haplotypes // 2

        super(WindowLevelAncestryObject, self).__init__(n_samples, n_ancestries)

        self.__haplotypes = haplotypes
        self.__lai = lai
        self.__window_sizes = window_sizes
        self.__centimorgan_pos = centimorgan_pos
        self.__samples = samples
        self.__chromosomes = chromosomes
        self.__physical_pos = physical_pos
        self.__ancestry_map = ancestry_map

        # Perform sanity check to ensure all unique ancestries in LAI data are represented in the ancestry map
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
    def haplotypes(self) -> List[str]:
        """
        Retrieve `haplotypes`.

        Returns:
            **list of length n_haplotypes:** A list of unique haplotype identifiers.
        """
        return self.__haplotypes

    @haplotypes.setter
    def haplotypes(self, x):
        """
        Update `haplotypes`.
        """
        self.__haplotypes = x

    @property
    def lai(self) -> np.ndarray:
        """
        Retrieve `lai`.

        Returns:
            **array of shape (n_windows, n_haplotypes):** 
                A 2D array containing local ancestry inference values, where each row represents a 
                genomic window, and each column corresponds to a haplotype phase for each sample.
        """
        return self.__lai

    @lai.setter
    def lai(self, x):
        """
        Update `lai`.
        """
        self.__lai = x

    @property
    def samples(self) -> Optional[List[str]]:
        """
        Retrieve `samples`.

        Returns:
            **list of str:** A list of unique sample identifiers.
        """
        if self.__samples is not None:
            return self.__samples
        elif self.__haplotypes is not None:
            return [hap.split('.')[0] for hap in self.__haplotypes][::2]
        else:
            return None
    
    @samples.setter
    def samples(self, x):
        """
        Update `samples`.
        """
        self.__samples = x

    @property
    def ancestry_map(self) -> Optional[Dict[str, str]]:
        """
        Retrieve `ancestry_map`.

        Returns:
            **dict of str to str:** A dictionary mapping ancestry codes to region names.
        """
        return self.__ancestry_map

    @ancestry_map.setter
    def ancestry_map(self, x):
        """
        Update `ancestry_map`.
        """
        self.__ancestry_map = x

    @property
    def window_sizes(self) -> Optional[np.ndarray]:
        """
        Retrieve `window_sizes`.

        Returns:
            **array of shape (n_windows,):** 
                An array specifying the number of SNPs in each genomic window.
        """
        return self.__window_sizes
        
    @window_sizes.setter
    def window_sizes(self, x):
        """
        Update `window_sizes`.
        """
        self.__window_sizes = x

    @property
    def centimorgan_pos(self) -> Optional[np.ndarray]:
        """
        Retrieve `centimorgan_pos`.

        Returns:
            **array of shape (n_windows, 2):** 
                A 2D array containing the start and end centimorgan positions for each window.
        """
        return self.__centimorgan_pos

    @centimorgan_pos.setter
    def centimorgan_pos(self, x):
        """
        Update `centimorgan_pos`.
        """
        self.__centimorgan_pos = x

    @property
    def chromosomes(self) -> Optional[np.ndarray]:
        """
        Retrieve `chromosomes`.

        Returns:
            **array of shape (n_windows,):** 
                An array with chromosome numbers corresponding to each genomic window.
        """
        return self.__chromosomes
        
    @chromosomes.setter
    def chromosomes(self, x):
        """
        Update `chromosomes`.
        """
        self.__chromosomes = x

    @property
    def physical_pos(self) -> Optional[np.ndarray]:
        """
        Retrieve `physical_pos`.

        Returns:
            **array of shape (n_windows, 2):** 
                A 2D array containing the start and end physical positions for each window.
        """
        return self.__physical_pos

    @physical_pos.setter
    def physical_pos(self, x):
        """
        Update `physical_pos`.
        """
        self.__physical_pos = x

    @property
    def n_samples(self) -> int:
        """
        Retrieve `n_samples`.

        Returns:
            **int:** 
                The total number of samples.
        """
        if self.__samples is not None:
            return len(self.__samples)
        elif self.__haplotypes is not None:
            # Divide by 2 because each sample has two associated haplotypes
            return len(self.__haplotypes) // 2
        else:
            # Divide by 2 because columns represent haplotypes
            return self.__lai.shape[1] // 2

    @property
    def n_ancestries(self) -> int:
        """
        Retrieve `n_ancestries`.

        Returns:
            **int:** The total number of unique ancestries.
        """
        return len(np.unique(self.__lai))
    
    @property
    def n_haplotypes(self) -> int:
        """
        Retrieve `n_haplotypes`.

        Returns:
            **int:** The total number of haplotypes.
        """
        if self.__haplotypes is not None:
            return len(self.__haplotypes)
        else:
            return self.__lai.shape[1]

    @property
    def n_windows(self) -> int:
        """
        Retrieve `n_windows`.

        Returns:
            **int:** The total number of genomic windows.
        """
        return self.__lai.shape[0]

    def copy(self) -> 'WindowLevelAncestryObject':
        """
        Create and return a copy of `self`.

        Returns:
            **WindowLevelAncestryObject:** 
                A new instance of the current object.
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
        return [attr.replace('_WindowLevelAncestryObject__', '').replace('_AncestryObject__', '') for attr in vars(self)]

    def filter_windows(
            self,
            indexes: Union[int, Sequence[int], np.ndarray],
            include: bool = True,
            inplace: bool = False
        ) -> Optional['WindowLevelAncestryObject']:
        """
        Filter genomic windows based on specified indexes. 

        This method updates the `lai` attribute to include or exclude the specified genomic windows. 
        Attributes such as `window_sizes`, `centimorgan_pos`, `chromosomes`, and `physical_pos` will also be 
        updated accordingly if they are not None. The order of genomic windows is preserved.

        Negative indexes are supported and follow 
        [NumPy's indexing conventions](https://numpy.org/doc/stable/user/basics.indexing.html). 

        Args:
            indexes (int or array-like of int): 
                Index(es) of the windows to include or exclude. Can be a single integer or a
                sequence of integers. Negative indexes are supported.
            include (bool, default=True): 
                If True, includes only the specified windows. If False, excludes the specified
                windows. Default is True.
            inplace (bool, default=False): 
                If True, modifies `self` in place. If False, returns a new `WindowLevelAncestryObject` with 
                the windows filtered. Default is False.

        Returns:
            **Optional[WindowLevelAncestryObject]:** 
                A new `WindowLevelAncestryObject` with the specified windows filtered if `inplace=False`. 
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Convert indexes to a NumPy array
        indexes = np.atleast_1d(indexes)

        # Get total number of windows
        n_windows = self.n_windows

        # Validate indexes, allowing negative indexes
        if np.any((indexes < -n_windows) | (indexes >= n_windows)):
            raise IndexError("One or more indexes are out of bounds.")

        # Create boolean mask
        mask = np.zeros(n_windows, dtype=bool)
        mask[indexes] = True

        # Invert mask if `include=False`
        if not include:
            mask = ~mask
        
        # Filter `lai`
        filtered_lai = self['lai'][mask, :] 
        
        # Filter `window_sizes`, `chromosomes`, `centimorgan_pos`, and `physical_pos`, checking if they are None before filtering
        filtered_window_sizes = self['window_sizes'][mask] if self['window_sizes'] is not None else None
        filtered_chromosomes = self['chromosomes'][mask] if self['chromosomes'] is not None else None
        filtered_centimorgan_pos = self['centimorgan_pos'][mask, :] if self['centimorgan_pos'] is not None else None
        filtered_physical_pos = self['physical_pos'][mask, :] if self['physical_pos'] is not None else None

        # Modify the original object if `inplace=True`, otherwise create and return a copy
        if inplace:
            self['lai'] = filtered_lai
            if filtered_window_sizes is not None:
                self['window_sizes'] = filtered_window_sizes
            if filtered_chromosomes is not None:
                self['chromosomes'] = filtered_chromosomes
            if filtered_centimorgan_pos is not None:
                self['centimorgan_pos'] = filtered_centimorgan_pos
            if filtered_physical_pos is not None:
                self['physical_pos'] = filtered_physical_pos
            return None
        else:
            laiobj = self.copy()
            laiobj['lai'] = filtered_lai
            if filtered_window_sizes is not None:
                laiobj['window_sizes'] = filtered_window_sizes
            if filtered_chromosomes is not None:
                laiobj['chromosomes'] = filtered_chromosomes
            if filtered_centimorgan_pos is not None:
                laiobj['centimorgan_pos'] = filtered_centimorgan_pos
            if filtered_physical_pos is not None:
                laiobj['physical_pos'] = filtered_physical_pos
            return laiobj

    def filter_samples(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray, None]] = None,
        indexes: Optional[Union[int, Sequence[int], np.ndarray, None]] = None,
        include: bool = True,
        inplace: bool = False
    ) -> Optional['WindowLevelAncestryObject']:
        """
        Filter samples based on specified names or indexes.

        This method updates the `lai`, `haplotypes`, and `samples` attributes to include or exclude the specified 
        samples. Each sample is associated with two haplotypes, which are included or excluded together.
        The order of the samples is preserved.

        If both samples and indexes are provided, any sample matching either a name in samples or an index in 
        indexes will be included or excluded.
        
        Negative indexes are supported and follow 
        [NumPy's indexing conventions](https://numpy.org/doc/stable/user/basics.indexing.html).

        Args:
            samples (str or array_like of str, optional): 
                 Name(s) of the samples to include or exclude. Can be a single sample name or a
                 sequence of sample names. Default is None.
            indexes (int or array_like of int, optional):
                Index(es) of the samples to include or exclude. Can be a single index or a sequence
                of indexes. Negative indexes are supported. Default is None.
            include (bool, default=True): 
                If True, includes only the specified samples. If False, excludes the specified
                samples. Default is True.
            inplace (bool, default=False): 
                If True, modifies `self` in place. If False, returns a new `WindowLevelAncestryObject` with the 
                samples filtered. Default is False.

        Returns:
            **Optional[WindowLevelAncestryObject]:** 
                A new `WindowLevelAncestryObject` with the specified samples filtered if `inplace=False`. 
                If `inplace=True`, modifies `self` in place and returns None.
        """
        if samples is None and indexes is None:
            raise UserWarning("At least one of 'samples' or 'indexes' must be provided.")

        n_haplotypes = self.n_haplotypes
        n_samples = self.n_samples

        # Create mask based on sample names
        if samples is not None:
            samples = np.atleast_1d(samples)
            # Extract sample names from haplotype identifiers
            haplotype_ids = np.array(self['haplotypes'])
            sample_names = np.array([hap.split('.')[0] for hap in haplotype_ids])
            # Create mask for haplotypes belonging to specified samples
            mask_samples = np.isin(sample_names, samples)
        else:
            mask_samples = np.zeros(n_haplotypes, dtype=bool)

        # Create mask based on sample indexes
        if indexes is not None:
            indexes = np.atleast_1d(indexes)

            # Validate indexes, allowing negative indexes
            out_of_bounds_indexes = indexes[(indexes < -n_samples) | (indexes >= n_samples)]
            if out_of_bounds_indexes.size > 0:
                raise ValueError(f"One or more sample indexes are out of bounds.")

            # Adjust negative indexes
            indexes = np.mod(indexes, n_samples)
            
            # Get haplotype indexes for the specified sample indexes
            haplotype_indexes = np.concatenate([2*indexes, 2*indexes+1])
            # Create mask for haplotypes
            mask_indexes = np.zeros(n_haplotypes, dtype=bool)
            mask_indexes[haplotype_indexes] = True
        else:
            mask_indexes = np.zeros(n_haplotypes, dtype=bool)

        # Combine masks using logical OR (union of samples)
        mask_combined = mask_samples | mask_indexes

        if not include:
            mask_combined = ~mask_combined

        # Filter `lai`
        filtered_lai = self['lai'][:, mask_combined]

        # Filter `haplotypes`
        filtered_haplotypes = np.array(self['haplotypes'])[mask_combined].tolist()

        # Filter `samples`, checking if they are None before filtering
        sample_mask = mask_combined.reshape(-1, 2).any(axis=1)
        filtered_samples = np.array(self['samples'])[sample_mask].tolist() if self['samples'] is not None else None

        if inplace:
            self['haplotypes'] = filtered_haplotypes
            self['samples'] = filtered_samples
            self['lai'] = filtered_lai
            return None
        else:
            laiobj = self.copy()
            laiobj['haplotypes'] = filtered_haplotypes
            laiobj['samples'] = filtered_samples
            laiobj['lai'] = filtered_lai
            return laiobj

    def convert_to_snp_level(
        self,
        snpobject: Optional['SNPObject'] = None,
        variants_chrom: Optional[np.ndarray] = None,
        variants_pos: Optional[np.ndarray] = None,
        variants_ref: Optional[np.ndarray] = None,
        variants_alt: Optional[np.ndarray] = None,
        variants_filter_pass: Optional[np.ndarray] = None,
        variants_id: Optional[np.ndarray] = None,
        variants_qual: Optional[np.ndarray] = None
    ) -> 'SNPObject':
        """
        Convert `self` into a `snputils.snp.genobj.SNPObject` SNP-level Local Ancestry Information (LAI), 
        with optional support for Single Nucleotide Polymorphism (SNP) data.
        
        If SNP positions (`variants_pos`) and chromosomes (`variants_chrom`) are not specified, the method generates 
        SNPs uniformly across the start and end positions of each genomic window. Otherwise, the provided SNP 
        coordinates are used to assign ancestry values based on their respective windows.

        If an `SNPObject` is provided, its attributes are used unless explicitly overridden by the function arguments.

        Args:
            snpobject (SNPObject, optional):
                An existing `SNPObject` to extract SNP attributes from.
            variants_chrom (array of shape (n_snps,), optional): 
                An array containing the chromosome for each SNP.
            variants_pos (array of shape (n_snps,), optional): 
                An array containing the chromosomal positions for each SNP.
            variants_ref (array of shape (n_snps,), optional): 
                An array containing the reference allele for each SNP.
            variants_alt (array of shape (n_snps,), optional): 
                An array containing the alternate allele for each SNP.
            variants_filter_pass (array of shape (n_snps,), optional): 
                An array indicating whether each SNP passed control checks.
            variants_id (array of shape (n_snps,), optional): 
                An array containing unique identifiers (IDs) for each SNP.
            variants_qual (array of shape (n_snps,), optional): 
                An array containing the Phred-scaled quality score for each SNP.

        Returns:
            SNPObject: 
                A `SNPObject` containing SNP-level ancestry data, along with optional metadata.
        """
        # Extract attributes from SNPObject if provided
        if snpobject is not None:
            variants_chrom = variants_chrom if variants_chrom is not None else snpobject.variants_chrom
            variants_pos = variants_pos if variants_pos is not None else snpobject.variants_pos
            variants_ref = variants_ref if variants_ref is not None else snpobject.variants_ref
            variants_alt = variants_alt if variants_alt is not None else snpobject.variants_alt
            variants_filter_pass = variants_filter_pass if variants_filter_pass is not None else snpobject.variants_filter_pass
            variants_id = variants_id if variants_id is not None else snpobject.variants_id
            variants_qual = variants_qual if variants_qual is not None else snpobject.variants_qual

        n_samples = self.n_samples

        # Reshape lai to (n_windows, n_samples, 2)
        lai_reshaped = self.lai.reshape(self.n_windows, n_samples, 2)

        if variants_pos is None or variants_chrom is None:
            # Generate SNP positions and chromosomes from windows
            variants_pos_list = []
            variants_chrom_list = []
            ancestry_list = []

            for i in range(self.n_windows):
                start = int(self.physical_pos[i, 0])
                end = int(self.physical_pos[i, 1])
                chrom = self.chromosomes[i]
                ancestry = lai_reshaped[i, :, :]  # Shape: (n_samples, 2)

                # Create SNP positions between start and end with the given step size
                positions_in_window = np.arange(start, end + 1)
                n_positions = len(positions_in_window)

                if n_positions == 0:
                    continue  # Skip windows with no positions

                variants_pos_list.append(positions_in_window)
                variants_chrom_list.append(np.full(n_positions, chrom))

                # Repeat ancestry for each SNP position in the window
                ancestry_repeated = np.repeat(ancestry[np.newaxis, :, :], n_positions, axis=0)
                ancestry_list.append(ancestry_repeated)

                # Concatenate all SNP positions, chromosomes, and ancestries
                variants_pos = np.concatenate(variants_pos_list)
                variants_chrom = np.concatenate(variants_chrom_list)
                calldata_lai = np.concatenate(ancestry_list)

        else:
            # Use provided SNP positions and chromosomes
            n_snps = len(variants_pos)
            if len(variants_chrom) != n_snps:
                raise ValueError("`variants_pos` and `variants_chrom` must have the same length.")

            # Map SNPs to windows
            window_starts = self.physical_pos[:, 0]
            window_ends = self.physical_pos[:, 1]
            window_chromosomes = self.chromosomes
            
            # Initialize array to store window indices for each SNP
            snp_to_window_indices = np.full(n_snps, -1, dtype=int)
            unique_chroms = np.unique(variants_chrom)

            for chrom in unique_chroms:
                # Indices of SNPs on this chromosome
                snp_indices = np.where(variants_chrom == chrom)[0]
                snp_pos_chr = variants_pos[snp_indices]
                
                # Indices of windows on this chromosome
                window_indices = np.where(window_chromosomes == chrom)[0]
                window_starts_chr = window_starts[window_indices]
                window_ends_chr = window_ends[window_indices]
                
                # Check if windows are defined
                if len(window_starts_chr) == 0:
                    continue
                
                # For SNP positions, find where they would be inserted in window_starts to maintain order
                inds = np.searchsorted(window_starts_chr, snp_pos_chr, side='right') - 1
                
                # Ensure indices are within valid range
                valid_inds = (inds >= 0) & (inds < len(window_starts_chr))
                snp_inds_valid = snp_indices[valid_inds]
                inds_valid = inds[valid_inds]
                snp_pos_valid = snp_pos_chr[valid_inds]
                
                # Check if SNP positions are within window ranges
                within_window = snp_pos_valid <= window_ends_chr[inds_valid]
                final_snp_indices = snp_inds_valid[within_window]
                final_window_indices = window_indices[inds_valid[within_window]]
                log.debug(f"Number of SNPs within window ranges for chromosome {chrom}: {len(final_snp_indices)}")
                
                # Assign window indices to SNPs
                snp_to_window_indices[final_snp_indices] = final_window_indices

            # Initialize SNP-level ancestry array
            calldata_lai = np.full((n_snps, n_samples, 2), np.nan)

            # Create a boolean mask for valid SNP indices (where window_idx != -1)
            valid_snp_mask = (snp_to_window_indices != -1)

            # Get the array of valid SNP indices
            valid_snp_indices = np.where(valid_snp_mask)[0]

            # Get the corresponding window indices for valid SNPs
            valid_window_indices = snp_to_window_indices[valid_snp_indices]

            # Assign lai_values to calldata_lai for all valid SNPs at once
            calldata_lai[valid_snp_indices] = lai_reshaped[valid_window_indices]
        
        return SNPObject(
            calldata_lai=calldata_lai,
            samples=self.samples,
            variants_ref=variants_ref,
            variants_alt=variants_alt,
            variants_filter_pass=variants_filter_pass,
            variants_chrom=variants_chrom,
            variants_id=variants_id,
            variants_pos=variants_pos,
            variants_qual=variants_qual,
            ancestry_map=self.ancestry_map
        )

    def _sanity_check(self) -> None:
        """
        Perform sanity checks on the parsed data to ensure data integrity.

        This method checks that all unique ancestries in LAI are represented 
        in the ancestry map.

        Args:
            lai (np.ndarray): The LAI data array.
            ancestry_map (dict, optional): A dictionary mapping ancestry codes to region names, if available.
        """
        # Get unique ancestries from LAI data
        unique_ancestries = np.unique(self.lai)

        if self.ancestry_map is not None:
            # Check if all unique ancestries in the LAI are present in the ancestry map
            for ancestry in unique_ancestries:
                if str(ancestry) not in self.ancestry_map:
                    warnings.warn(
                        f"Ancestry '{ancestry}' found in LAI data is not represented in the ancestry map."
                    )

    def save(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a specified file.
        If the file already exists, it will be overwritten.

        The format of the saved file is determined by the file extension provided in the `file` 
        argument.

        **Supported formats:**

        - `.msp`: Text-based MSP format.
        - `.msp.tsv`: Text-based MSP format with TSV extension.
        - `.pkl`: Pickle format for saving `self` in serialized form.

        Args:
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. The extension of the file determines the save format. 
                Supported extensions: `.msp`, `.msp.tsv`, `.pkl`.
        """
        path = Path(file)
        suffixes = [suffix.lower() for suffix in path.suffixes]

        if suffixes[-2:] == ['.msp', '.tsv'] or suffixes[-1] == '.msp':
            self.save_msp(file)
        elif suffixes[-1] == '.pkl':
            self.save_pickle(file)
        else:
            raise ValueError(
                f"Unsupported file extension: {suffixes[-1]}"
                "Supported extensions are: .msp, .msp.tsv, .pkl."
            )

    def save_msp(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a `.msp` file.
        If the file already exists, it will be overwritten.

        Args:
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. It should end with `.msp` or `.msp.tsv`. 
                If the provided path does not have one of these extensions, the `.msp` extension will be appended.
        """
        from snputils.ancestry.io.local.write import MSPWriter

        MSPWriter(self, file).write()

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
