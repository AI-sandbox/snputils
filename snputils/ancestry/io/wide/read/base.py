import abc
from pathlib import Path
from typing import Union, Optional
import logging
import numpy as np

log = logging.getLogger(__name__)


def _open_textfile(filename: Union[str, Path], mode: str = "rt"):
    filename = str(filename)
    if filename.endswith(".zst"):
        import zstandard as zstd
        return zstd.open(filename, mode, encoding="utf-8") if "t" in mode else zstd.open(filename, mode)
    elif filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode, encoding="utf-8") if "t" in mode else gzip.open(filename, mode)
    return open(filename, mode, encoding="utf-8") if "t" in mode else open(filename, mode)


def _get_format_suffix(path: Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        return ""
    if suffixes[-1] in (".zst", ".gz") and len(suffixes) >= 2:
        return suffixes[-2]
    return suffixes[-1]


class WideBaseReader(abc.ABC):
    """
    Abstract class for global ancestry readers.
    """
    def __init__(
        self,
        Q_file: Union[str, Path],
        P_file: Optional[Union[str, Path]] = None,
        sample_file: Optional[Union[str, Path]] = None,
        snp_file: Optional[Union[str, Path]] = None,
        ancestry_file: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Args:
            Q_file (str or pathlib.Path):
                Path to the file containing the Q matrix (per-sample ancestry proportions).
                It should end with `.Q` or `.txt`.
                The file should use space (' ') as the delimiter.
            P_file (str or pathlib.Path, optional):
                Path to the file containing the P/F matrix (per-ancestry SNP frequencies).
                It should end with `.P` or `.txt`.
                The file should use space (' ') as the delimiter. If None, P is not loaded.
            sample_file (str or pathlib.Path, optional):
                Path to the single-column file containing sample identifiers. 
                It should end with `.fam` or `.txt`.
                If None, sample identifiers are not loaded.
            snp_file (str or pathlib.Path, optional):
                Path to the single-column file containing SNP identifiers. 
                It should end with `.bim` or `.txt`.
                If None, SNP identifiers are not loaded.
            ancestry_file (str or pathlib.Path, optional):
                Path to the single-column file containing ancestry labels for each sample.
                It should end with `.map` or `.txt`.
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
        Retrieve `Q_file`.

        Returns:
            **pathlib.Path:** 
                Path to the file containing the Q matrix (per-sample ancestry proportions).
                It should end with `.Q` or `.txt`.
                The file should use space (' ') as the delimiter.
        """
        return self.__Q_file

    @property
    def P_file(self) -> Optional[Path]:
        """
        Retrieve `P_file`.

        Returns:
            **pathlib.Path or None:** 
                Path to the file containing the P/F matrix (per-ancestry SNP frequencies).
                It should end with `.P` or `.txt`.
                The file should use space (' ') as the delimiter. If None, P is not loaded.
        """
        return self.__P_file

    @property
    def sample_file(self) -> Optional[Path]:
        """
        Retrieve `sample_file`.

        Returns:
            **pathlib.Path:** 
                Path to the single-column file containing sample identifiers. 
                It should end with `.fam` or `.txt`.
                If None, sample identifiers are not loaded.
        """
        return self.__sample_file
    
    @property
    def snp_file(self) -> Optional[Path]:
        """
        Retrieve `snp_file`.

        Returns:
            **pathlib.Path:** 
                Path to the single-column file containing SNP identifiers. 
                It should end with `.bim` or `.txt`.
                If None, SNP identifiers are not loaded.
        """
        return self.__snp_file

    @property
    def ancestry_file(self) -> Optional[Path]:
        """
        Retrieve `ancestry_file`.

        Returns:
            **pathlib.Path:** 
                Path to the single-column file containing ancestry labels for each sample.
                It should end with `.map` or `.txt`.
                If None, ancestries are not loaded.
        """
        return self.__ancestry_file

    def _read_sample_ids(self) -> Optional[np.ndarray]:
        """
        Read sample identifiers from `sample_file` file.
        """
        if self.sample_file is None:
            return None
        
        log.info(f"Reading sample identifiers from '{self.sample_file}'...")
        suffix = _get_format_suffix(self.sample_file)
        if suffix == ".txt":
            with _open_textfile(self.sample_file, "rt") as f:
                return np.genfromtxt(f, dtype=str)
        elif suffix == ".fam":
            with _open_textfile(self.sample_file, "rt") as f:
                return np.genfromtxt(f, dtype=str, usecols=1)
        else:
            raise ValueError(
                f"Unsupported file extension for sample identifiers: {self.sample_file.name} "
                "Supported extensions are: .txt, .fam (optionally compressed with .zst or .gz)."
            )
    
    def _read_snps(self) -> Optional[np.ndarray]:
        """
        Read SNP identifiers from `snp_file` file.
        """
        if self.snp_file is None:
            return None
        
        log.info(f"Reading SNP identifiers from '{self.snp_file}'...")
        suffix = _get_format_suffix(self.snp_file)
        if suffix == ".txt":
            with _open_textfile(self.snp_file, "rt") as f:
                return np.genfromtxt(f, dtype=str)
        elif suffix == ".bim":
            with _open_textfile(self.snp_file, "rt") as f:
                return np.genfromtxt(f, dtype=str, usecols=1)
        else:
            raise ValueError(
                f"Unsupported file extension for SNP identifiers: {self.snp_file.name} "
                "Supported extensions are: .txt, .bim (optionally compressed with .zst or .gz)."
            )

    def _read_ancestries(self) -> Optional[np.ndarray]:
        """
        Read ancestries for each sample from `ancestry_file`.
        """
        if self.ancestry_file is None:
            return None
        
        log.info(f"Reading ancestries for each sample from '{self.ancestry_file}'...")
        suffix = _get_format_suffix(self.ancestry_file)
        if suffix in (".map", ".txt"):
            with _open_textfile(self.ancestry_file, "rt") as f:
                return np.genfromtxt(f, dtype=str)
        else:
            raise ValueError(
                f"Unsupported file extension for ancestries: {self.ancestry_file.name} "
                "Supported extension: .map, .txt (optionally compressed with .zst or .gz)."
            )

    @abc.abstractmethod
    def read(self) -> None:
        """
        Abstract method to read data from the provided files.

        Subclasses must implement this method to read and parse the data from
        the Q and P files, as well as sample identifiers, SNPs, and ancestries if provided.
        The implementation should construct an instance of `snputils.ancestry.genobj.GlobalAncestryObject` 
        based on the read data.
        """
        pass
