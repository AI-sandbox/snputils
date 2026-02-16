import abc
from pathlib import Path
from typing import Union

from snputils.phenotype.genobj import MultiPhenotypeObject
from snputils.phenotype.genobj import PhenotypeObject


class PhenotypeBaseReader(abc.ABC):
    """
    Abstract class for phenotype data readers.
    """
    def __init__(self, file):
        self._file = file
    
    @abc.abstractmethod
    def read(self) -> Union['MultiPhenotypeObject', 'PhenotypeObject']:
        """
        Abstract method to read data from the provided `file`.

        Subclasses must implement this method to read and parse the data.
        The implementation should construct an instance of
        `snputils.phenotype.genobj.MultiPhenotypeObject` or
        `snputils.phenotype.genobj.PhenotypeObject` based on the read data.
        """
        pass
    
    @property
    def file(self) -> Path:
        """
        Retrieve `file`.

        Returns:
            pathlib.Path: Path to the file containing phenotype data.
        """
        return Path(self._file)
