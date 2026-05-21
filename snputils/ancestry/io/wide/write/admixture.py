import logging
from pathlib import Path
from typing import Union, Optional
import numpy as np

from .base import WideBaseWriter
from snputils.ancestry.genobj.wide import GlobalAncestryObject

log = logging.getLogger(__name__)


def _append_admixture_suffix(path: Path, suffix: str) -> Path:
    return Path(f"{path}{suffix}")


class AdmixtureWriter(WideBaseWriter):
    """
    A writer class for exporting global ancestry data from a 
    `snputils.ancestry.genobj.GlobalAncestryObject` into multiple ADMIXTURE files.
    """
    def __init__(
        self, 
        wideobj: GlobalAncestryObject, 
        file_prefix: Union[str, Path]
    ) -> None:
        """
        Args:
            wideobj (GlobalAncestryObject): 
                A GlobalAncestryObject instance.
            file_prefix (str or pathlib.Path): 
                Prefix for output file names, including directory path but excluding file extensions. 
                The prefix is used to generate specific file names for each output, with file-specific 
                suffixes appended as described above (e.g., `file_prefix.n_ancestries.Q` for the Q matrix file).
        """
        super(AdmixtureWriter, self).__init__(wideobj, file_prefix)
        self.__Q_file = _append_admixture_suffix(
            self.file_prefix, f".{self.wideobj.n_ancestries}.Q"
        )
        self.__P_file = _append_admixture_suffix(
            self.file_prefix, f".{self.wideobj.n_ancestries}.P"
        )

        self.__sample_file = (
            _append_admixture_suffix(self.file_prefix, ".sample_ids.txt")
            if self.wideobj.samples is not None else None
        )
        self.__snp_file = (
            _append_admixture_suffix(self.file_prefix, ".snp_ids.txt")
            if self.wideobj.snps is not None else None
        )
        self.__ancestry_file = (
            _append_admixture_suffix(self.file_prefix, ".map")
            if self.wideobj.ancestries is not None else None
        )

    @property
    def wideobj(self) -> GlobalAncestryObject:
        """
        Retrieve `wideobj`.

        Returns:
            GlobalAncestryObject: A GlobalAncestryObject instance.
        """
        return super().wideobj

    @property
    def file_prefix(self) -> Path:
        """
        Retrieve `file_prefix`.

        Returns:
            pathlib.Path: 
                Prefix for output file names, including directory path but excluding file extensions. 
                The prefix is used to generate specific file names for each output, with file-specific 
                suffixes appended as described above (e.g., `file_prefix.n_ancestries.Q` for the Q matrix file).
        """
        return super().file_prefix

    @property
    def Q_file(self) -> Path:
        """
        Retrieve `Q_file`.

        Returns:
            pathlib.Path: 
                Path to the `.Q` file that will store the Q matrix (per-sample ancestry proportions).
        """
        return self.__Q_file
    
    @property
    def P_file(self) -> Path:
        """
        Retrieve `P_file`.

        Returns:
            pathlib.Path: 
                Path to the `.P` file that will store the P/F matrix (per-ancestry SNP frequencies).
        """
        return self.__P_file
    
    @property
    def sample_file(self) -> Optional[Path]:
        """
        Retrieve `sample_file`.

        Returns:
            pathlib.Path: 
                Path to the `.txt` the file that will store sample identifiers. 
                If None, sample identifiers are not saved.
        """
        return self.__sample_file
    
    @property
    def snp_file(self) -> Optional[Path]:
        """
        Retrieve `snp_file`.

        Returns:
            pathlib.Path: 
                Path to the `.txt` file that will store SNP identifiers. 
                If None, SNP identifiers are not saved.
        """
        return self.__snp_file
    
    @property
    def ancestry_file(self) -> Optional[Path]:
        """
        Retrieve `ancestry_file`.

        Returns:
            pathlib.Path: 
                Path to the `.map` file that will store ancestry labels for each sample. 
                If None, ancestries are not saved.
        """
        return self.__ancestry_file

    def _write_Q(self):
        log.info(f"Writing Q matrix to '{self.Q_file}'...")
        np.savetxt(self.Q_file, self.wideobj.Q, delimiter=" ")
        log.info(f"Finished writing Q matrix to '{self.Q_file}'.")

    def _write_P(self):
        log.info(f"Writing P matrix to '{self.P_file}'...")
        np.savetxt(self.P_file, self.wideobj.P, delimiter=" ")
        log.info(f"Finished writing P matrix to '{self.P_file}'.")

    def _write_sample_ids(self):
        if self.wideobj.samples is not None:
            log.info(f"Writing sample IDs to '{self.sample_file}'...")
            np.savetxt(self.sample_file, self.wideobj.samples, fmt="%s")
            log.info(f"Finished writing sample IDs to '{self.sample_file}'.")

    def _write_snps(self):
        if self.wideobj.snps is not None:
            log.info(f"Writing SNP IDs to '{self.snp_file}'...")
            np.savetxt(self.snp_file, self.wideobj.snps, fmt="%s")
            log.info(f"Finished writing SNP IDs to '{self.snp_file}'.")

    def _write_ancestries(self):
        if self.wideobj.ancestries is not None:
            log.info(f"Writing ancestry information to '{self.ancestry_file}'...")
            np.savetxt(self.ancestry_file, self.wideobj.ancestries, fmt="%s")
            log.info(f"Finished writing ancestry information to '{self.ancestry_file}'.")

    def write(self) -> None:
        """
        Write the data contained in the `wideobj` instance into the multiple ADMIXTURE files
        with the specified `file_prefix`. If the files already exist, they will be overwritten.

        Output files:

        - `<file_prefix>.K.Q`: Q matrix file. The file uses space (' ') as the delimiter.
        - `<file_prefix>.K.P`: P matrix file. The file uses space (' ') as the delimiter.
        - `<file_prefix>.sample_ids.txt`: Sample IDs file (if sample IDs are available).
        - `<file_prefix>.snp_ids.txt`: SNP IDs file (if SNP IDs are available).
        - `<file_prefix>.map`: Ancestry file (if ancestries information is available).

        where `K` is the total number of ancestries.
        """
        log.info(f"Preparing to write ADMIXTURE files with prefix '{self.file_prefix}'...")
        
        self.file_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        self._write_Q()
        self._write_P()
        self._write_sample_ids()
        self._write_snps()
        self._write_ancestries()

        log.info(f"Finished writing all ADMIXTURE files with prefix '{self.file_prefix}'.")

WideBaseWriter.register(AdmixtureWriter)
