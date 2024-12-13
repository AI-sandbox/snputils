from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
import warnings
import numpy as np
import pandas as pd

from .base import LAIBaseReader
from snputils.ancestry.genobj.window import WindowLevelAncestryObject

log = logging.getLogger(__name__)


class MSPReader(LAIBaseReader):
    """
    A reader class for parsing Local Ancestry Inference (LAI) data from an `.msp` or `msp.tsv` file
    and constructing a `snputils.ancestry.genobj.WindowLevelAncestryObject`.
    """
    def __init__(self, file: Union[str, Path]) -> None:
        """
        Args:
            file (str or pathlib.Path): 
                Path to the file to be read. It should end with `.msp` or `.msp.tsv`.
        """
        self.__file = Path(file)

    @property
    def file(self) -> Path:
        """
        Retrieve `file`.

        Returns:
            **pathlib.Path:** 
                Path to the file to be read. It should end with `.msp` or `.msp.tsv`.
        """
        return self.__file

    def _get_samples(self, msp_df: pd.DataFrame, first_lai_col_indx: int) -> List[str]:
        """
        Extract unique sample identifiers from the pandas DataFrame.

        Args:
            msp_df (pd.DataFrame): 
                The DataFrame representing the `.msp` data, including LAI columns.
            first_lai_col_indx (int): 
                Index of the first column containing LAI data.

        Returns:
            **list:** List of unique sample identifiers.
        """
        # Get all columns starting from the first LAI data column
        query_samples_dub = msp_df.columns[first_lai_col_indx:]

        # Select only one of the maternal/paternal samples by taking every second sample
        single_ind_idx = np.arange(0, len(query_samples_dub), 2)
        query_samples_sing = query_samples_dub[single_ind_idx]

        # Remove the suffix from sample names to get clean identifiers
        query_samples = [qs[:-2] for qs in query_samples_sing]

        return query_samples

    def _get_ancestry_map_from_comment(self, comment: str) -> Dict[str, str]:
        """
        Construct an ancestry map from the comment line of the `.msp` file.

        This method parses the comment string to create a mapping of ancestry numerical identifiers 
        to their corresponding ancestry names.

        Args:
            comment (str): 
                The comment line containing ancestry mapping information.

        Returns:
            dict: A dictionary mapping names to ancestry numbers (as strings).
        """
        # Split the comment to extract relevant ancestry mapping data
        comment = comment.split(' ')[-1].replace('\n', '').split('\t')
        ancestry_map = {}

        # Populate the ancestry map with parsed values
        for elem in comment:
            x, y = elem.split('=')
            ancestry_map[y] = x

        return ancestry_map

    def _replace_nan_with_none(self, array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Replace arrays that are fully NaN with `None`.

        Args:
            array (np.ndarray): Array to check.

        Returns:
            Optional[np.ndarray]: Returns `None` if the array is fully NaN, otherwise returns the original array.
        """
        if array is not None:
            if array.size == 0:  # Check if the array is empty
                return None
            if np.issubdtype(array.dtype, np.number):  # Check for numeric types
                if np.isnan(array).all():  # Fully NaN numeric array
                    return None
            elif array.dtype == np.object_ or np.issubdtype(array.dtype, np.str_):  # String or object types
                if all(elem == '' or elem is None for elem in array):  # Empty or None strings
                    return None
        return array

    def read(self) -> 'WindowLevelAncestryObject':
        """
        Read data from the provided `.msp` or `msp.tsv` `file` and construct a 
        `snputils.ancestry.genobj.WindowLevelAncestryObject`.

        **Expected MSP content:**

        The `.msp` file should contain local ancestry assignments for each haplotype across genomic windows.
        Each row should correspond to a genomic window and include the following columns:

        - `#chm`: Chromosome numbers corresponding to each genomic window.
        - `spos`: Start physical position for each window.
        - `epos`: End physical position for each window.
        - `sgpos`: Start centimorgan position for each window.
        - `egpos`: End centimorgan position for each window.
        - `n snps`: Number of SNPs in each genomic window.
        - `SampleID.0`: Local ancestry for the first haplotype of the sample for each window.
        - `SampleID.1`: Local ancestry for the second haplotype of the sample for each window.

        Returns:
            **WindowLevelAncestryObject:**
                A WindowLevelAncestryObject instance.
        """
        log.info(f"Reading '{self.file}'...")

        # Open the file and read the first two lines to identify the header and comment
        with open(self.file) as f:
            first_line = f.readline()
            second_line = f.readline()

        first_line_ = [h.replace('\n', '') for h in first_line.split("\t")]
        second_line_ = [h.replace('\n', '') for h in second_line.split("\t")]

        # Determine which line contains the #chm reference value for the header
        if "#chm" in first_line_:
            comment = None
            header = first_line_
        elif "#chm" in second_line_:
            comment = first_line
            header = second_line_
        else:
            raise ValueError(
                f"Header not found. Expected '#chm' in the first two lines. "
                f"First line: {first_line.strip()} | Second line: {second_line.strip()}"
            )

        # Ensure no duplicate columns exist in the header
        if len(header) != len(set(header)):
            raise ValueError("Duplicate columns detected in the header.")

        # Read the main data into a DataFrame, skipping comment lines
        msp_df = pd.read_csv(self.file, sep="\t", comment="#", names=header)

        # Extract chromosomes data
        chromosomes = msp_df['#chm'].astype(str).to_numpy()

        # Extract physical positions (if available)
        column_counter = 1
        try:
            physical_pos = msp_df[['spos', 'epos']].to_numpy()
            column_counter += 2
        except KeyError:
            physical_pos = None
            log.warning("Physical positions ('spos' and 'epos') not found.")
        
        # Extract centimorgan positions (if available)
        try:
            centimorgan_pos = msp_df[['sgpos', 'egpos']].to_numpy()
            column_counter += 2
        except KeyError:
            centimorgan_pos = None
            log.warning("Genetic (centimorgan) positions ('sgpos' and 'egpos') not found.")
        
        # Extract window sizes (if available)
        try:
            window_sizes = msp_df['n snps'].to_numpy()
            column_counter += 1
        except KeyError:
            window_sizes = None
            log.warning("Window sizes ('n snps') not found.")
        
        # Extract LAI data (haplotype-level)
        lai = msp_df[msp_df.columns[column_counter:]].to_numpy()

        # Extract haplotype identifiers
        haplotypes = msp_df.columns[column_counter:].to_list()

        # Extract haplotype identifiers and sample identifiers
        samples = self._get_samples(msp_df, column_counter)

        # Validate the number of samples matches the LAI data dimensions
        n_samples = len(samples)
        if n_samples != int(lai.shape[1] / 2):
            raise ValueError(
                "Mismatch between the number of sample identifiers and the expected number of samples in the LAI array. "
                f"Expected {int(lai.shape[1] / 2)} samples (derived from LAI data); found {n_samples}."
            )
        
        # Count number of unique ancestries in the LAI data
        n_ancestries = len(np.unique(lai))

        # Parse ancestry map from the comment (if available)
        ancestry_map = None
        if comment is not None:
            ancestry_map = self._get_ancestry_map_from_comment(comment)
            if len(ancestry_map) != n_ancestries:
                warnings.warn(
                    "Mismatch between the number of unique ancestries in the LAI data "
                    f"({n_ancestries}) and the number of classes in the ancestry map "
                    f"({len(ancestry_map)})."
                )
        else:
            # Provide default ancestry mapping if no comment is provided
            ancestry_map = None
            warnings.warn(
                "Ancestry map not found. It is recommended to provide an .msp file that contains the ancestry "
                "map as a comment in the first line."
            )

        # Replace fully NaN attributes with None
        window_sizes = self._replace_nan_with_none(window_sizes)
        centimorgan_pos = self._replace_nan_with_none(centimorgan_pos)
        chromosomes = self._replace_nan_with_none(chromosomes)
        physical_pos = self._replace_nan_with_none(physical_pos)

        return WindowLevelAncestryObject(
            haplotypes=haplotypes,
            lai=lai,
            samples=samples,
            ancestry_map=ancestry_map,
            window_sizes=window_sizes,
            centimorgan_pos=centimorgan_pos,
            chromosomes=chromosomes,
            physical_pos=physical_pos
        )

LAIBaseReader.register(MSPReader)
