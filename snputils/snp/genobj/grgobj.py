from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import copy
import warnings
import re
from typing import Any, Union, Tuple, List, Sequence, Dict, Optional
import pygrgl as pyg
log = logging.getLogger(__name__)

GRGType = Union[pyg.GRG | pyg.MutableGRG]
class GRGObject:
    """
    A class for Single Nucleotide Polymorphism (SNP) data.
    """
    def __init__(
        self,
        calldata_gt: Optional[GRGType] = None,
    ) -> None:
        """
        Args:
            calldata_gt (GRG | MutableGRG, optional): 
                A Genotype Representation Graph containing genotype data for each sample. 
        """
        self.__calldata_gt = calldata_gt
        

    def __getitem__(self, key: str) -> Any:
        """
        To access an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            return getattr(self, key)
        except:
            raise KeyError(f'Invalid key: {key}.')

    def __setitem__(self, key: str, value: Any):
        """
        To set an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            setattr(self, key, value)
        except:
            raise KeyError(f'Invalid key: {key}.')

    @property
    def calldata_gt(self) -> np.ndarray:
        """
        Retrieve `calldata_gt`.

        Returns:
            **GRG | MutableGRG:** 
                An GRG containing genotype data for all samples.
        """
        return self.__calldata_gt

    @calldata_gt.setter
    def calldata_gt(self, x: GRGType):
        """
        Update `calldata_gt`.
        """
        self.__calldata_gt = x

    def allele_freq(self) -> np.ndarray:
        # allele frequency array
        al_freq = np.ones(self.calldata_gt.num_mutations) / self.calldata_gt.num_mutations
        return pyg.dot_product(self.calldata_gt, al_freq, pyg.TraversalDirection.DOWN)

        


    def copy(self) -> GRGObject:
        """
        Create and return a copy of `self`.

        Returns:
            **GRGObject:** 
                A new instance of the current object.
        """
        return copy.deepcopy(self)

    def keys(self) -> List[str]:
        """
        Retrieve a list of public attribute names for `self`.

        Returns:
            **list of str:** 
                A list of attribute names, with internal name-mangling removed, 
                for easier reference to public attributes in the instance.
        """
        return [attr.replace('_GRGObject__', '') for attr in vars(self)]
