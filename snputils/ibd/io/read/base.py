import abc
from pathlib import Path
from typing import Union

from snputils.ibd.genobj.ibdobj import IBDObject


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


