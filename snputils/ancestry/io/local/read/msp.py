import gc
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import numpy as np
import pandas as pd
import re
from .base import LAIBaseReader
from snputils.ancestry.genobj.local import LocalAncestryObject

log = logging.getLogger(__name__)


@dataclass
class MSPMetadata:
    header: List[str]
    comment: Optional[str]
    first_lai_col_indx: int
    haplotypes: List[str]
    samples: List[str]
    ancestry_map: Optional[Dict[str, str]]
    has_physical_pos: bool
    has_centimorgan_pos: bool
    has_window_sizes: bool


class MSPReader(LAIBaseReader):
    """
    A reader class for parsing Local Ancestry Inference (LAI) data from an `.msp` or `msp.tsv` file
    and constructing a `snputils.ancestry.genobj.LocalAncestryObject`.
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

    def _get_samples_from_haplotypes(self, haplotypes: List[str]) -> List[str]:
        query_samples_dub = np.asarray(haplotypes, dtype=object)
        single_ind_idx = np.arange(0, len(query_samples_dub), 2)
        query_samples_sing = query_samples_dub[single_ind_idx]
        return [str(qs)[:-2] for qs in query_samples_sing]

    def _parse_header_and_comment(self) -> tuple[Optional[str], List[str]]:
        with open(self.file) as f:
            first_line = f.readline()
            second_line = f.readline()

        first_line_ = [h.strip() for h in first_line.split("\t")]
        second_line_ = [h.strip() for h in second_line.split("\t")]

        if "#chm" in first_line_:
            return None, first_line_
        if "#chm" in second_line_:
            return first_line, second_line_

        raise ValueError(
            f"Header not found. Expected '#chm' in the first two lines. "
            f"First line: {first_line.strip()} | Second line: {second_line.strip()}"
        )

    def _get_first_lai_col_indx(self, header: List[str]) -> int:
        column_counter = 1
        if "spos" in header and "epos" in header:
            column_counter += 2
        if "sgpos" in header and "egpos" in header:
            column_counter += 2
        if "n snps" in header:
            column_counter += 1
        return column_counter

    def read_metadata(self) -> MSPMetadata:
        comment, header = self._parse_header_and_comment()

        if len(header) != len(set(header)):
            raise ValueError("Duplicate columns detected in the header.")

        first_lai_col_indx = self._get_first_lai_col_indx(header)
        haplotypes = header[first_lai_col_indx:]
        samples = self._get_samples_from_haplotypes(haplotypes)
        ancestry_map = self._get_ancestry_map_from_comment(comment) if comment is not None else None

        return MSPMetadata(
            header=header,
            comment=comment,
            first_lai_col_indx=first_lai_col_indx,
            haplotypes=haplotypes,
            samples=samples,
            ancestry_map=ancestry_map,
            has_physical_pos=("spos" in header and "epos" in header),
            has_centimorgan_pos=("sgpos" in header and "egpos" in header),
            has_window_sizes=("n snps" in header),
        )

    def iter_windows(
        self,
        chunk_size: int = 1024,
        sample_indices: Optional[np.ndarray] = None,
    ) -> Iterator[Dict[str, np.ndarray]]:
        metadata = self.read_metadata()

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        header = metadata.header
        first_lai_col_indx = metadata.first_lai_col_indx
        column_index = {name: i for i, name in enumerate(header)}
        chrom_col_idx = column_index["#chm"]

        spos_col_idx: Optional[int] = None
        epos_col_idx: Optional[int] = None
        if metadata.has_physical_pos:
            spos_col_idx = column_index["spos"]
            epos_col_idx = column_index["epos"]

        if sample_indices is None:
            hap_col_indices = list(range(first_lai_col_indx, len(header)))
        else:
            sample_indices = np.asarray(sample_indices, dtype=np.int64)
            if sample_indices.size == 0:
                raise ValueError("sample_indices cannot be empty.")
            if np.any(sample_indices < 0) or np.any(sample_indices >= len(metadata.samples)):
                raise ValueError("sample_indices contain out-of-bounds sample indexes.")

            hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
            hap_indices[0::2] = 2 * sample_indices
            hap_indices[1::2] = 2 * sample_indices + 1
            hap_col_indices = (first_lai_col_indx + hap_indices).astype(np.int64).tolist()

        n_selected_haps = len(hap_col_indices)
        n_total_haps = len(metadata.haplotypes)
        all_haps_selected = (
            n_selected_haps == n_total_haps
            and n_selected_haps > 0
            and hap_col_indices[0] == first_lai_col_indx
            and hap_col_indices[-1] == (len(header) - 1)
        )

        # Pre-compute relative indices for the sample-subset path so the
        # inner loop can use np.fromstring (C-level) + numpy fancy indexing
        # instead of a Python for-loop over potentially millions of columns.
        if not all_haps_selected:
            _relative_hap_idx = np.array(hap_col_indices, dtype=np.intp) - first_lai_col_indx
        else:
            _relative_hap_idx = None

        row_in_chunk = 0
        window_start = 0
        chromosomes_chunk = np.empty(int(chunk_size), dtype=object)
        lai_chunk = np.empty((int(chunk_size), n_selected_haps), dtype=np.uint8)
        physical_pos_chunk = (
            np.empty((int(chunk_size), 2), dtype=np.int64)
            if metadata.has_physical_pos
            else None
        )

        with open(self.file, "r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if not raw_line:
                    continue
                if raw_line.startswith("#"):
                    continue

                line = raw_line.rstrip("\n")
                if not line:
                    continue

                # Both paths split only at the metadata/haplotype boundary,
                # then use np.fromstring (C parser) for the haplotype tail.
                fields = line.split("\t", first_lai_col_indx)
                if len(fields) != (first_lai_col_indx + 1):
                    raise ValueError(
                        f"Malformed MSP row at line {line_no}: expected {first_lai_col_indx + 1} "
                        f"prefix segments when parsing haplotypes."
                    )

                chromosomes_chunk[row_in_chunk] = fields[chrom_col_idx]
                if physical_pos_chunk is not None and spos_col_idx is not None and epos_col_idx is not None:
                    physical_pos_chunk[row_in_chunk, 0] = int(fields[spos_col_idx])
                    physical_pos_chunk[row_in_chunk, 1] = int(fields[epos_col_idx])

                lai_row = np.fromstring(fields[first_lai_col_indx], sep="\t", dtype=np.uint8)

                if all_haps_selected:
                    if lai_row.size != n_selected_haps:
                        raise ValueError(
                            f"Malformed MSP haplotype row at line {line_no}: expected "
                            f"{n_selected_haps} haplotype values, got {lai_row.size}."
                        )
                    lai_chunk[row_in_chunk, :] = lai_row
                else:
                    if lai_row.size < n_total_haps:
                        raise ValueError(
                            f"Malformed MSP haplotype row at line {line_no}: expected at least "
                            f"{n_total_haps} haplotype values, got {lai_row.size}."
                        )
                    lai_chunk[row_in_chunk, :] = lai_row[_relative_hap_idx]

                row_in_chunk += 1
                if row_in_chunk == chunk_size:
                    window_indexes = np.arange(window_start, window_start + row_in_chunk, dtype=np.int64)
                    yield {
                        "window_indexes": window_indexes,
                        "chromosomes": chromosomes_chunk,
                        "physical_pos": physical_pos_chunk,
                        "lai": lai_chunk,
                    }

                    window_start += row_in_chunk
                    row_in_chunk = 0
                    chromosomes_chunk = np.empty(int(chunk_size), dtype=object)
                    lai_chunk = np.empty((int(chunk_size), n_selected_haps), dtype=np.uint8)
                    if metadata.has_physical_pos:
                        physical_pos_chunk = np.empty((int(chunk_size), 2), dtype=np.int64)
                    else:
                        physical_pos_chunk = None

        if row_in_chunk > 0:
            window_indexes = np.arange(window_start, window_start + row_in_chunk, dtype=np.int64)
            yield {
                "window_indexes": window_indexes,
                "chromosomes": chromosomes_chunk[:row_in_chunk],
                "physical_pos": (
                    physical_pos_chunk[:row_in_chunk]
                    if physical_pos_chunk is not None
                    else None
                ),
                "lai": lai_chunk[:row_in_chunk],
            }

    def _get_ancestry_map_from_comment(self, comment: str) -> Dict[str, str]:
        """
        Construct an ancestry map from the comment line of the `.msp` file.

        This method parses the comment string to create a mapping of ancestry numerical identifiers 
        to their corresponding ancestry names (e.g., '0': 'African').

        Args:
            comment (str): 
                The comment line containing ancestry mapping information.

        Returns:
            dict: A dictionary mapping ancestry codes (as strings) to ancestry names.
        """
        comment = comment.strip()

        # Remove everything before the colon, if present
        if ':' in comment:
            comment = comment.split(':', 1)[1].strip()

        ancestry_map: Dict[str, str] = {}

        # Split on tabs, spaces, commas, semicolons or any combination of them
        tokens = [tok.strip() for tok in re.split(r'[,\t; ]+', comment) if tok]

        for tok in tokens:
            if '=' not in tok:
                continue  # Skip invalid pieces

            left, right = (p.strip() for p in tok.split('=', 1))

            # Detect whether format is "Pop=0" or "0=Pop"
            if left.isdigit() and not right.isdigit():
                ancestry_map[left] = right       # 0=Africa
            elif right.isdigit() and not left.isdigit():
                ancestry_map[right] = left       # Africa=0
            else:
                # Fallback (if both sides are digits or both are pops, keep left as code)
                ancestry_map[left] = right

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
                if np.all((array == '') | (array == None)):  # Empty or None strings
                    return None
        return array

    def read(self) -> 'LocalAncestryObject':
        """
        Read data from the provided `.msp` or `msp.tsv` `file` and construct a 
        `snputils.ancestry.genobj.LocalAncestryObject`.

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
            **LocalAncestryObject:**
                A LocalAncestryObject instance.
        """
        log.info(f"Reading '{self.file}'...")
        metadata = self.read_metadata()
        comment = metadata.comment
        header = metadata.header

        # Read the main data into a DataFrame, skipping comment lines
        msp_df = pd.read_csv(self.file, sep="\t", comment="#", names=header)

        # Extract chromosomes data
        chromosomes = msp_df['#chm'].astype(str).to_numpy()

        # Extract physical positions (if available)
        column_counter = metadata.first_lai_col_indx
        if metadata.has_physical_pos:
            physical_pos = msp_df[['spos', 'epos']].to_numpy()
        else:
            physical_pos = None
            log.warning("Physical positions ('spos' and 'epos') not found.")
        
        # Extract centimorgan positions (if available)
        if metadata.has_centimorgan_pos:
            centimorgan_pos = msp_df[['sgpos', 'egpos']].to_numpy()
        else:
            centimorgan_pos = None
            log.warning("Genetic (centimorgan) positions ('sgpos' and 'egpos') not found.")

        # Extract window sizes (if available)
        if metadata.has_window_sizes:
            window_sizes = msp_df['n snps'].to_numpy()
        else:
            window_sizes = None
            log.warning("Window sizes ('n snps') not found.")
        
        # Extract LAI data (haplotype-level)
        lai = msp_df.iloc[:, column_counter:].to_numpy(dtype=np.uint8, copy=False)

        # Extract haplotype identifiers
        haplotypes = metadata.haplotypes

        # Extract haplotype identifiers and sample identifiers
        samples = metadata.samples
        del msp_df
        gc.collect()

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
            ancestry_map = metadata.ancestry_map
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

        return LocalAncestryObject(
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
