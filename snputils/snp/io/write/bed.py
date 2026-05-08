import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Union

from snputils.snp.genobj import SNPObject

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
            after: Union[int, float, str] = '.',
            sample_fid: Optional[Union[np.ndarray, Sequence[str]]] = None,
            sample_sex: Optional[Union[np.ndarray, Sequence[Union[str, int]]]] = None,
            sample_phenotype: Optional[Union[np.ndarray, Sequence[Union[str, int, float]], str, int, float]] = None,
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
            sample_fid (optional): PLINK Family ID per sample (same order as ``snpobj.samples``).
                If None, uses ``snpobj.sample_fid`` when set; otherwise writes ``samples`` as both FID and IID.
            sample_sex (optional): PLINK sex code per sample. Accepted values include ``1``/``M``/``male``,
                ``2``/``F``/``female``, and ``0``/unknown. Defaults to ``0`` for all samples.
            sample_phenotype (optional): PLINK phenotype value per sample, or a scalar used for all samples.
                Defaults to ``-9`` for all samples.
        """
        # Save .bed file
        if self.__filename.suffix != '.bed':
            self.__filename = self.__filename.with_suffix('.bed')

        log.info(f"Writing .bed file: {self.__filename}")

        # Optionally rename potential missing values in `snpobj.calldata_gt` before writing
        if rename_missing_values:
            self.__snpobj.rename_missings(before=before, after=after, inplace=True)

        # PLINK BED stores variants-major packed 2-bit hard calls. Convert to an
        # individuals x variants dosage matrix first, then pack each variant row.
        if len(self.__snpobj.calldata_gt.shape) == 3:
            genotype_matrix = self.__snpobj.calldata_gt.transpose(1, 0, 2).sum(axis=2, dtype=np.int8)
        elif len(self.__snpobj.calldata_gt.shape) == 2:
            genotype_matrix = self.__snpobj.calldata_gt.T
        else:
            raise ValueError("`calldata_gt` must be a 2D or 3D array.")
        genotype_matrix = np.asarray(genotype_matrix)
        if not np.issubdtype(genotype_matrix.dtype, np.number):
            genotype_matrix = np.where(genotype_matrix == ".", -9, genotype_matrix)
        genotype_matrix = genotype_matrix.astype(np.int8, copy=False)
        genotype_matrix = np.where(genotype_matrix < 0, -9, genotype_matrix)

        # Infer the number of samples and variants from the matrix
        samples, variants = genotype_matrix.shape

        with self.__filename.open("wb") as handle:
            handle.write(bytes([0x6C, 0x1B, 0x01]))
            for snp_i in range(variants):
                for start in range(0, samples, 4):
                    byte = 0
                    for offset in range(4):
                        sample_i = start + offset
                        if sample_i >= samples:
                            code = 0b00
                        else:
                            dosage = int(genotype_matrix[sample_i, snp_i])
                            if dosage == -9:
                                code = 0b01
                            elif dosage == 0:
                                code = 0b11
                            elif dosage == 1:
                                code = 0b10
                            elif dosage == 2:
                                code = 0b00
                            else:
                                raise ValueError(f"Unexpected diploid dosage {dosage}")
                        byte |= code << (2 * offset)
                    handle.write(bytes([byte]))

        log.info(f"Finished writing .bed file: {self.__filename}")

        # Remove .bed from the file name
        if self.__filename.suffix == '.bed':
            self.__filename = self.__filename.with_suffix('')

        # Save .fam file
        log.info(f"Writing .fam file: {self.__filename}")

        # Fill .fam file
        fam_file = pd.DataFrame(columns=['fid', 'iid', 'father', 'mother', 'gender', 'trait'])
        fam_file['iid'] = self.__snpobj.samples
        fid = sample_fid
        if fid is None:
            fid = getattr(self.__snpobj, "sample_fid", None)
        if fid is None:
            fam_file['fid'] = self.__snpobj.samples
        else:
            fid_arr = np.asarray(fid)
            if fid_arr.shape[0] != len(self.__snpobj.samples):
                raise ValueError(
                    f"sample_fid length ({fid_arr.shape[0]}) must match number of samples ({len(self.__snpobj.samples)})."
                )
            fam_file['fid'] = fid_arr
        fam_file['father'] = 0
        fam_file['mother'] = 0
        fam_file['gender'] = self._coerce_sex_codes(sample_sex, len(self.__snpobj.samples))
        fam_file['trait'] = self._coerce_phenotypes(sample_phenotype, len(self.__snpobj.samples))

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
        log.warning("The .bim file is being saved with 0 cM values.")
        bim_file['pos'] = self.__snpobj.variants_pos
        bim_file['a0'] = self.__snpobj.variants_alt
        bim_file['a1'] = self.__snpobj.variants_ref

        # Save .bim file
        bim_file.to_csv(self.__filename.with_suffix('.bim'), sep='\t', index=False, header=False)
        log.info(f"Finished writing .bim file: {self.__filename}")

    @staticmethod
    def _coerce_sex_codes(
        sample_sex: Optional[Union[np.ndarray, Sequence[Union[str, int]]]],
        n_samples: int,
    ) -> np.ndarray:
        if sample_sex is None:
            return np.repeat("0", n_samples)
        sex = np.asarray(sample_sex)
        if sex.shape[0] != n_samples:
            raise ValueError(f"sample_sex length ({sex.shape[0]}) must match number of samples ({n_samples}).")
        mapping = {
            "1": "1",
            "m": "1",
            "male": "1",
            "2": "2",
            "f": "2",
            "female": "2",
            "0": "0",
            "u": "0",
            "unknown": "0",
            "": "0",
            ".": "0",
            "nan": "0",
            "none": "0",
        }
        return np.asarray([mapping.get(str(value).strip().lower(), "0") for value in sex], dtype=object)

    @staticmethod
    def _coerce_phenotypes(
        sample_phenotype: Optional[Union[np.ndarray, Sequence[Union[str, int, float]], str, int, float]],
        n_samples: int,
    ) -> np.ndarray:
        if sample_phenotype is None:
            return np.repeat("-9", n_samples)
        phenotype = np.asarray(sample_phenotype)
        if phenotype.ndim == 0:
            return np.repeat(str(phenotype.item()), n_samples)
        if phenotype.shape[0] != n_samples:
            raise ValueError(
                f"sample_phenotype length ({phenotype.shape[0]}) must match number of samples ({n_samples})."
            )
        return phenotype.astype(str)
