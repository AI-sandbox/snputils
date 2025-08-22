import logging
import numpy as np
from pathlib import Path
from typing import Union, Optional

log = logging.getLogger(__name__)

from .base import WideBaseReader
from snputils.ancestry.genobj.wide import GlobalAncestryObject


class AdmixtureReader(WideBaseReader):
    """
    A reader class for parsing ADMIXTURE files and constructing a `snputils.ancestry.genobj.GlobalAncestryObject`.
    """
    def __init__(
        self,
        Q_file: Union[str, Path],
        P_file: Optional[Union[str, Path]] = None,
        sample_file: Optional[Union[str, Path]] = None,
        snp_file: Optional[Union[str, Path]] = None,
        ancestry_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Args:
            Q_file (str or pathlib.Path):
                Path to the file containing the Q matrix (per-sample ancestry proportions).
                It should end with .Q or .txt.
                The file should use space (' ') as the delimiter.
            P_file (str or pathlib.Path, optional):
                Path to the file containing the P/F matrix (per-ancestry SNP frequencies).
                It should end with .P or .txt.
                The file should use space (' ') as the delimiter. If None, P is not loaded.
            sample_file (str or pathlib.Path, optional):
                Path to the single-column file containing sample identifiers. 
                It should end with .fam or .txt.
                If None, sample identifiers are not loaded.
            snp_file (str or pathlib.Path, optional):
                Path to the single-column file containing SNP identifiers. 
                It should end with .bim or .txt.
                If None, SNP identifiers are not loaded.
            ancestry_file (str or pathlib.Path, optional):
                Path to the single-column file containing ancestry labels for each sample.
                It should end with .map or .txt.
                If None, ancestries are not loaded.
        """
        self.__Q_file = Path(Q_file)
        self.__P_file = Path(P_file) if P_file is not None else None
        self.__sample_file = Path(sample_file) if sample_file is not None else None
        self.__snp_file = Path(snp_file) if snp_file is not None else None
        self.__ancestry_file = Path(ancestry_file) if ancestry_file is not None else None

    @property
    def Q_file(self) -> Path:
        """
        Retrieve Q_file.

        Returns:
            **pathlib.Path:** 
                Path to the file containing the Q matrix (per-sample ancestry proportions).
                It should end with .Q or .txt.
                The file should use space (' ') as the delimiter.
        """
        return self.__Q_file

    @property
    def P_file(self) -> Optional[Path]:
        """
        Retrieve P_file.

        Returns:
            **pathlib.Path or None:** 
                Path to the file containing the P/F matrix (per-ancestry SNP frequencies).
                It should end with .P or .txt.
                The file should use space (' ') as the delimiter. If None, P is not loaded.
        """
        return self.__P_file

    @property
    def sample_file(self) -> Optional[Path]:
        """
        Retrieve sample_file.

        Returns:
            **pathlib.Path:** 
                Path to the single-column file containing sample identifiers. 
                It should end with .fam or .txt.
                If None, sample identifiers are not loaded.
        """
        return self.__sample_file
    
    @property
    def snp_file(self) -> Optional[Path]:
        """
        Retrieve snp_file.

        Returns:
            **pathlib.Path:** 
                Path to the single-column file containing SNP identifiers. 
                It should end with .bim or .txt.
                If None, SNP identifiers are not loaded.
        """
        return self.__snp_file

    @property
    def ancestry_file(self) -> Optional[Path]:
        """
        Retrieve ancestry_file.

        Returns:
            **pathlib.Path:** 
                Path to the single-column file containing ancestry labels for each sample.
                It should end with .map or .txt.
                If None, ancestries are not loaded.
        """
        return self.__ancestry_file

    def read(self) -> 'GlobalAncestryObject':
        """
        Read data from the provided ADMIXTURE files and construct a 
        snputils.ancestry.genobj.GlobalAncestryObject instance.

        **Expected ADMIXTURE files content:**

        - **Q_file**: 
            A text file containing the Q matrix with per-sample ancestry proportions. 
             Each row corresponds to a sample, and each column corresponds to an ancestry.
        - **P_file**: 
            A text file containing the P matrix with per-ancestry SNP frequencies.
            Each row corresponds to a SNP, and each column corresponds to an ancestry.

        Optional files (if provided):
        - **sample_file**: A single-column text file containing sample identifiers in order.
        - **snp_file**: A single-column text file containing SNP identifiers in order.
        - **ancestry_file**: A single-column text file containing ancestry labels for each sample.

        Returns:
            **GlobalAncestryObject:** 
                A GlobalAncestryObject instance.
        """
        log.info(f"Reading Q matrix from '{self.Q_file}'...")
        Q_mat = np.genfromtxt(self.Q_file, delimiter=' ')
        if self.P_file is not None:
            log.info(f"Reading P matrix from '{self.P_file}'...")
            P_mat = np.genfromtxt(self.P_file, delimiter=' ')
        else:
            P_mat = None

        samples = self._read_sample_ids()
        snps = self._read_snps()
        ancestries = self._read_ancestries()

        return GlobalAncestryObject(
            Q_mat,
            P_mat,
            samples=samples,
            snps=snps,
            ancestries=ancestries
        )

WideBaseReader.register(AdmixtureReader)
