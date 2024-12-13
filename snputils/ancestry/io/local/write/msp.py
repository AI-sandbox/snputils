import logging
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import warnings

from .base import LAIBaseWriter
from snputils.ancestry.genobj.window import WindowLevelAncestryObject

log = logging.getLogger(__name__)


class MSPWriter(LAIBaseWriter):
    """
    A writer class for exporting local ancestry data from a `snputils.ancestry.genobj.WindowLevelAncestryObject` 
    into an `.msp` or `.msp.tsv` file.
    """
    def __init__(self, laiobj: WindowLevelAncestryObject, file: Union[str, Path]) -> None:
        """
        Args:
            laiobj (WindowLevelAncestryObject):
                A WindowLevelAncestryObject instance.
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. It should end with `.msp` or `.msp.tsv`. 
                If the provided path does not have one of these extensions, the `.msp` extension will be appended.
        """
        self.__laiobj = laiobj
        self.__file = Path(file)

    @property
    def laiobj(self) -> WindowLevelAncestryObject:
        """
        Retrieve `laiobj`. 

        Returns:
            **WindowLevelAncestryObject:** 
                A WindowLevelAncestryObject instance.
        """
        return self.__laiobj

    @property
    def file(self) -> Path:
        """
        Retrieve `file`.

        Returns:
            **pathlib.Path:** 
                Path to the file where the data will be saved. It should end with `.msp` or `.msp.tsv`. 
                If the provided path does not have one of these extensions, the `.msp` extension will be appended.
        """
        return self.__file
    
    def write(self) -> None:
        """
        Write the data contained in the `laiobj` instance to the specified output `file`. 
        If the file already exists, it will be overwritten.

        **Output MSP content:**

        The output `.msp` file will contain local ancestry assignments for each haplotype across genomic windows.
        Each row corresponds to a genomic window and includes the following columns:

        - `#chm`: Chromosome numbers corresponding to each genomic window.
        - `spos`: Start physical position for each window.
        - `epos`: End physical position for each window.
        - `sgpos`: Start centimorgan position for each window.
        - `egpos`: End centimorgan position for each window.
        - `n snps`: Number of SNPs in each genomic window.
        - `SampleID.0`: Local ancestry for the first haplotype of the sample for each window.
        - `SampleID.1`: Local ancestry for the second haplotype of the sample for each window.
        """
        log.info(f"LAI object contains: {self.laiobj.n_samples} samples, {self.laiobj.n_ancestries} ancestries.")

        # Define the valid file extensions
        valid_extensions = ('.msp', '.msp.tsv')

        # Append '.msp' extension if not already present
        if not self.file.name.endswith(valid_extensions):
            self.file = self.file.with_name(self.file.name + '.msp')

        # Check if file already exists
        if self.file.exists():
            warnings.warn(f"File '{self.file}' already exists and will be overwritten.")

        # Compute the number of windows and haplotypes
        n_windows = self.laiobj.n_windows
        n_haplotypes = self.laiobj.n_haplotypes

        # Initialize attributes with NaN where they are None
        chromosomes = self.laiobj.chromosomes if self.laiobj.chromosomes is not None else np.full(n_windows, np.nan)
        physical_pos = self.laiobj.physical_pos if self.laiobj.physical_pos is not None else np.full((n_windows, 2), np.nan)
        centimorgan_pos = self.laiobj.centimorgan_pos if self.laiobj.centimorgan_pos is not None else np.full((n_windows, 2), np.nan)
        window_sizes = self.laiobj.window_sizes if self.laiobj.window_sizes is not None else np.full(n_windows, np.nan)
        
        haplotypes = self.laiobj.haplotypes
        if haplotypes is None:
            # Generate haplotypes from samples or default identifiers
            if self.laiobj.samples is not None:
                haplotypes = [f"{sample}.{i}" for sample in self.laiobj.samples for i in range(2)]
                warnings.warn(
                    "Haplotype data is missing. Haplotypes have been automatically generated "
                    "from the provided sample identifiers."
                )
            else:
                haplotypes = [f"sample_{i//2}.{i%2}" for i in range(n_haplotypes)]
                warnings.warn(
                    "Haplotype data and sample identifiers are missing. Default haplotype identifiers have been generated "
                    "as `sample_<index>.0` and `sample_<index>.1`."
                )

        # Prepare columns for the DataFrame
        columns = ["spos", "epos", "sgpos", "egpos", "n snps"]
        lai_dic = {
            "#chm": chromosomes,
            "spos": physical_pos[:, 0],
            "epos": physical_pos[:, 1],
            "sgpos": centimorgan_pos[:, 0],
            "egpos": centimorgan_pos[:, 1],
            "n snps": window_sizes,
        }

        # Populate the dictionary with haplotype data
        for ilai, haplotype in enumerate(haplotypes):
            lai_dic[haplotype] = self.laiobj.lai[:, ilai]
            columns.append(haplotype)
            
        # Check if DataFrame is empty
        if len(lai_dic["#chm"]) == 0:
            raise ValueError("No data to write: all columns are empty or missing.")

        # Create a DataFrame from the dictionary containing all data
        lai_df = pd.DataFrame(lai_dic)

        log.info(f"Writing MSP file to '{self.file}'...")

        # Save the DataFrame to the .msp file in tab-separated format
        lai_df.to_csv(self.file, sep="\t", index=False, header=False)
        
        # Construct the second line for the output file containing the column headers
        second_line = "#chm" + "\t" + "\t".join(columns)
        
        # If an ancestry map is available, prepend it to the output file
        if self.laiobj.ancestry_map is not None:
            ancestries_codes = list(self.laiobj.ancestry_map.keys()) # Get corresponding codes
            ancestries = list(self.laiobj.ancestry_map.values()) # Get ancestry names
            
            # Create the first line for the ancestry information, detailing subpopulation codes
            first_line = "#Subpopulation order/codes: " + "\t".join(
                f"{a}={ancestries_codes[ai]}" for ai, a in enumerate(ancestries)
            )

            # Open the file for reading and prepend the first line       
            with open(self.__file, "r+") as f:
                content = f.read()
                f.seek(0,0)
                f.write(first_line.rstrip('\r\n') + '\n' + second_line + '\n' + content)

        log.info(f"Finished writing MSP file to '{self.file}'.")

        return None

LAIBaseWriter.register(MSPWriter)
