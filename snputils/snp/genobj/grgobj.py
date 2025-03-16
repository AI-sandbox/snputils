from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import copy
import warnings
import re
from typing import Any, Union, Tuple, List, Sequence, Dict, Optional
import pygrgl as pyg
import subprocess
log = logging.getLogger(__name__)
import tempfile

GRGType = Union[pyg.GRG | pyg.MutableGRG]
class GRGObject:
    """
    A class for Single Nucleotide Polymorphism (SNP) data.
    """
    def __init__(
        self,
        calldata_gt: Optional[GRGType] = None,
        filename: Optional[str] = None,
        mutable: Optional[bool] = None
    ) -> None:
        """
        Args:
            calldata_gt (GRG | MutableGRG, optional): 
                A Genotype Representation Graph containing genotype data for each sample. 
            filename (str, optional)
                File storing the GRG.
        """
        self.__calldata_gt = calldata_gt
        self.__filename = filename
        self.__mutable = mutable
        self.__latest   = False

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


    @property
    def filename(self) -> str:
        """
        Retrieve `filename`.

        Returns:
            **str** 
                A string containing the file name.
        """
        return self.__filename

    @filename.setter
    def calldata_gt(self, x: str):
        """
        Update `calldata_gt`.
        """
        self.__filename = x
    
    @property
    def mutable(self, x:bool):
        return self.__mutable

    def allele_freq(self) -> np.ndarray:
        # allele frequency array
        al_freq = np.ones(self.calldata_gt.num_samples) / self.calldata_gt.num_samples
        return pyg.dot_product(self.calldata_gt, al_freq, pyg.TraversalDirection.UP)

    def dot_product(self, array: np.ndarray, traversal_direction: pyg.TraversalDirection):
        return pyg.dot_product(self.calldata_gt, array, traversal_direction)
    
    # TODO: consider moving this elsewhere.
    def allele_freq_from_file(self, filename : Optional[str]) -> pd.DataFrame:
        newfile = filename if filename is not None else self.__filename
        assert newfile is not None, "Either pass in a filename, or store an existing GRG's filename."

        with tempfile.NamedTemporaryFile() as fp:

            subprocess.run(["grg", "process", "freq", f"{filename}"], stdout=fp)
            fp.seek(0) # set the file cursor
            return pd.read_csv(fp.name, sep="\t")
        
        
    def gwas(self, genotype_file: str, filename: str) -> pd.DataFrame:
        with tempfile.NamedTemporaryFile() as fp:
            subprocess.run(["grg", "process", "gwas", f"{filename}", "--phenotype", f"{genotype_file}"], stdout=fp)
            fp.seek(0) # set the file cursor
            return pd.read_csv(fp.name, sep="\t")
    
    def merge(self, combineNodes : bool = False, *args) -> Optional[GRGType]:
        # assert self.__mutable and isinstance(self.calldata_gt, pyg.MutableGRG), "GRG must be mutable"
        for arg in args:
            assert isinstance(arg, str), "argument must be string"
        # list of files, and combineNodes 
        self.__calldata_gt.merge(list(args), combineNodes)
        #pep8 be damned
        # if inplace: self.__calldata_gt = merged_data
        # else      : return merged_data

    def n_samples(self, ploidy = 2) -> int:
        """
        Get number of samples from GRG. Diploid by default. 
        """
        return self.__calldata_gt.num_samples / ploidy
    
    def n_snps(self) -> int:
        return self.__calldata_gt.num_mutations

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

    def to_grg(self, filename: str, 
                     allow_simplify: bool = True):
        pyg.save_grg(self.__calldata_gt, filename, allow_simplify)
