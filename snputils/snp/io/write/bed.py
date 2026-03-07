import logging
import pandas as pd
import numpy as np
import pgenlib as pg
from pathlib import Path
from typing import Union

from snputils.snp.genobj import SNPObject
from snputils.snp.io.write._genotype_encoding import phased_to_hardcalls

log = logging.getLogger(__name__)


class BEDWriter:
    """Writes an object in bed/bim/fam formats in the specified output path.

    Args:
        snpobj: The SNPObject to be written.
        file: The output file path.

    """

    def __init__(self, snpobj: SNPObject, filename: str):
        self.__snpobj = snpobj.copy()
        self.__filename = Path(filename)

    def write(
            self,
            rename_missing_values: bool = True, 
            before: Union[int, float, str] = -1, 
            after: Union[int, float, str] = '.'
        ):
        """
        Writes the SNPObject to bed/bim/fam formats.

        Args:
            rename_missing_values (bool, optional):
                If True, renames potential missing values in `snpobj.calldata_gt` before writing. 
                Defaults to True.
            before (int, float, or str, default=-1): 
                The current representation of missing values in `calldata_gt`. Common values might be -1, '.', or NaN.
                Default is -1.
            after (int, float, or str, default='.'): 
                The value that will replace `before`. Default is '.'.
        """
        # Save .bed file
        if self.__filename.suffix != '.bed':
            self.__filename = self.__filename.with_suffix('.bed')

        log.info(f"Writing .bed file: {self.__filename}")

        # pgenlib expects a numeric hardcall matrix in (samples, variants) order.
        hardcalls = phased_to_hardcalls(
            self.__snpobj.calldata_gt,
            rename_missing_values=rename_missing_values,
            before=before,
            after=after,
        )
        self.__snpobj.calldata_gt = hardcalls.T

        # Infer the number of samples and variants from the matrix
        samples, variants = self.__snpobj.calldata_gt.shape

        # Define the PgenWriter to save the data
        data_save = pg.PgenWriter(filename=str(self.__filename).encode('utf-8'),
                                  sample_ct=samples,
                                  variant_ct=variants,
                                  nonref_flags=True,
                                  hardcall_phase_present=False,
                                  dosage_present=True,
                                  dosage_phase_present=False)

        # Fill the data_save object with the matrix of individuals x variants
        for snp_i in range(0, variants):
            data_save.append_biallelic(np.ascontiguousarray(self.__snpobj.calldata_gt[:, snp_i]))

        # Save the .bed file
        data_save.close()

        log.info(f"Finished writing .bed file: {self.__filename}")

        # Remove .bed from the file name
        if self.__filename.suffix == '.bed':
            self.__filename = self.__filename.with_suffix('')

        # Save .fam file
        log.info(f"Writing .fam file: {self.__filename}")

        # Fill .fam file
        fam_file = pd.DataFrame(columns=['fid', 'iid', 'father', 'mother', 'gender', 'trait'])
        fam_file['iid'] = self.__snpobj.samples
        fam_file['fid'] = self.__snpobj.samples

        # Save .fam file
        fam_file.to_csv(self.__filename.with_suffix('.fam'), sep='\t', index=False, header=False)
        log.info(f"Finished writing .fam file: {self.__filename}")

        # Save .bim file
        log.info(f"Writing .bim file: {self.__filename}")

        # Fill .bim file
        bim_file = pd.DataFrame(columns=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1'])
        bim_file['chrom'] = self.__snpobj.variants_chrom
        bim_file['snp'] = self.__snpobj.variants_id
        bim_file['cm'] = 0  # TODO: read, save and write too if available?
        log.info("The .bim file is being saved with 0 cM values.")
        bim_file['pos'] = self.__snpobj.variants_pos
        bim_file['a0'] = self.__snpobj.variants_alt
        bim_file['a1'] = self.__snpobj.variants_ref

        # Save .bim file
        bim_file.to_csv(self.__filename.with_suffix('.bim'), sep='\t', index=False, header=False)
        log.info(f"Finished writing .bim file: {self.__filename}")
