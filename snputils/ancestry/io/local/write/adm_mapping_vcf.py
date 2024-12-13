import logging
import warnings
import numpy as np
from typing import Dict, Union, Optional
from pathlib import Path

from snputils.ancestry.genobj.window import WindowLevelAncestryObject
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.write.vcf import VCFWriter

log = logging.getLogger(__name__)


class AdmixtureMappingVCFWriter:
    """
    A writer class for converting and writing local ancestry data into ancestry-specific 
    VCF/BCF files for ADMIXTURE mapping.
    """
    def __init__(
            self, 
            laiobj: WindowLevelAncestryObject, 
            file: Union[str, Path], 
            ancestry_map: Optional[Dict[str, str]] = None
        ):
        """
        Args:
            laiobj (WindowLevelAncestryObject): 
                A WindowLevelAncestryObject instance.
            file (str or pathlib.Path): 
                Path to the file where the data will be saved. It should end with `.vcf` or `.bcf`. 
                If the provided path does not have one of these extensions, the `.vcf` extension will be appended.
            ancestry_map (dict of str to str, optional): 
                A dictionary mapping ancestry codes to region names. If not explicitly 
                provided, it will default to the `ancestry_map` from `laiobj`.
        """
        self.__laiobj = laiobj
        self.__file = Path(file)
        self.__ancestry_map = ancestry_map

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
                Path to the file where the data will be saved. It should end with `.vcf` or `.bcf`. 
                If the provided path does not have one of these extensions, the `.vcf` extension will be appended.
        """
        return self.__file

    @property
    def ancestry_map(self) -> Dict[str, str]:
        """
        Retrieve `ancestry_map`.

        Returns:
            **dict of str to str:** 
                A dictionary mapping ancestry codes to region names. If not explicitly 
                provided, it will default to the `ancestry_map` from `laiobj`.
        """
        if self.__ancestry_map is not None:
            return self.__ancestry_map
        elif self.laiobj.ancestry_map is not None:
            return self.laiobj.ancestry_map
        else:
            raise ValueError(
                "Ancestry mapping is required but missing. Provide `ancestry_map` "
                "during initialization or ensure `laiobj.ancestry_map` is set."
            )

    def write(self) -> None:
        """
        Write VCF or BCF files for each ancestry type defined in the ancestry map.
        If the file already exists, it will be overwritten.

        **Output VCF/BCF content:**
        
        For each ancestry, this method converts LAI data to SNP alleles and writes it in a VCF-compatible format.
        SNPs are encoded as follows:

        - `1`: Indicates positions that match the specified ancestry.
        - `0`: Indicates positions that do not match the specified ancestry.

        The VCF/BCF files will contain the following fields:

        - `CHROM`: Chromosome for each variant.
        - `POS`: Chromosomal positions for each variant.
        - `ID`: Unique identifier for each variant.
        - `REF`: Reference allele for each variant.
        - `ALT`: Alternate allele for each variant.
        - `QUAL`: Phred-scaled quality score for each variant.
        - `FILTER`: Status indicating whether each SNP passed control checks.
        - `INFO`: Additional information field. Defaults to `'.'` indicating no extra metadata.
        - `FORMAT`: Genotype format. Set to `'GT'`, representing the genotype as phased alleles.
        - `<SampleID>`: One column per sample, containing the genotype data (`1|0`, `0|1`, etc.).

        **Output files:**

        - A separate VCF file is written for each ancestry type, with filenames formatted as:
        `<filename>_<ancestry>.vcf` (e.g., `output_African.vcf`).
        """
        # Process the list of positions to include both the start and end coordinates for each window
        # Iterate over each ancestry key in the ancestry mapping
        for key in self.ancestry_map:
            ancestry = int(key)
            anc_string = self.ancestry_map[key]

            # Define the output file format, ensuring it has the correct ancestry-specific suffix
            file_extension = (".vcf", ".bcf")
            
            # Check if file has one of the specified extensions
            if self.file.suffix not in file_extension:
                # If file does not have the correct extension, default to ".vcf"
                output_file = self.file.with_name(f"{self.file.stem}_{anc_string}.vcf")
            else:
                # If file has the correct extension, insert the ancestry string before the extension
                output_file = self.file.with_name(f"{self.file.stem}_{anc_string}{self.file.suffix}")

            # Check if file already exists
            if output_file.exists():
                warnings.warn(f"File '{output_file}' already exists and will be overwritten.")

            # Format start and end positions for the VCF file
            if self.laiobj.physical_pos is not None:
                pos_list = [f"{val1}_{val2}" for val1, val2 in self.laiobj.physical_pos]
            else:
                pos_list = None

            # Modify LAI data values to simulate a SNP file
            # The positions in LAI corresponding to the current ancestry key are mapped to 1, and the rest to 0
            match = (self.laiobj.lai == ancestry).astype(int)
            match = match.reshape(len(self.laiobj.lai),int(len(self.laiobj.lai[0])/2), 2 )

            # Set up VCF-related data
            calldata_gt = match
            samples = np.array(self.laiobj.samples)
            variants_chrom = self.laiobj.chromosomes
            variants_list = [str(i+1) for i in range(len(self.laiobj.lai))]
            variants_id = np.array(variants_list)
            variants_ref = np.full(calldata_gt.shape[0], 'A', dtype='U5')
            variants_alt = np.full(calldata_gt.shape[0], 'T', dtype='U1')

            # Create the SNPObject
            variant_data_obj = SNPObject(
                calldata_gt=calldata_gt,
                samples=samples,
                variants_chrom=variants_chrom,
                variants_id=variants_id,
                variants_ref = variants_ref,
                variants_alt = variants_alt,
                variants_pos = pos_list,
            )

            # Log the start of the VCF file writing process
            log.info(f"Writing VCF file for ancestry '{anc_string}' to '{output_file}'...")

            # Write the VCF file
            vcf_writer = VCFWriter(variant_data_obj, output_file)
            vcf_writer.write()

            log.info(f"Finished writing VCF file for ancestry '{anc_string}' to '{output_file}'.")

        return
