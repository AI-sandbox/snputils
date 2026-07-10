import abc
import gzip
from pathlib import Path
from typing import TextIO, Union

from snputils.ibd.genobj.ibdobj import IBDObject


def open_text(file: Union[str, Path]) -> TextIO:
    path = Path(file)
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if suffix == ".zst":
        import zstandard as zstd

        return zstd.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


class IBDBaseReader(abc.ABC):
    """
    Abstract class for IBD readers.
    """

    def __init__(self, file: Union[str, Path]) -> None:
        """
        Args:
            file (str or pathlib.Path): Path to the IBD file to read.
        """
        self.__file = Path(file)

    @property
    def file(self) -> Path:
        """
        Retrieve `file`.

        Returns:
            pathlib.Path: Path to the IBD file to read.
        """
        return self.__file

    @abc.abstractmethod
    def read(self) -> IBDObject:
        """
        Abstract method to read data from the provided `file` and construct an `IBDObject`.
        """
        pass

