import pygrgl as pyg
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
# from snputils.common._utils import ITE
from typing import Optional, Union
from os.path import abspath, splitext
import pathlib


@SNPBaseReader.register
class GRGReader(SNPBaseReader):
    def read(self,
             mutable: Optional[bool] = None,
             load_up_edges: Optional[bool] = None,
             binary_mutations: Optional[bool] = None) -> Union[pyg.GRG | pyg.MutableGRG]:
        """
        Read in a GRG or TSKit File
        """

        name, _ext1 = splitext(self.filename)
        name, _ext2 = splitext(name)

        _ext = _ext1 + _ext2

        
        file    = abspath(self.filename) if isinstance(self.filename, pathlib.Path) else self.filename
        edges   = load_up_edges if load_up_edges is not None else True
        binmuts = binary_mutations if binary_mutations is not None else False
        print(_ext)
        if _ext in [".tskit", ".trees"]:
            return pyg.grg_from_trees(file, binmuts)
        if mutable:
            return pyg.load_mutable_grg(file)
        else:
            return pyg.load_immutable_grg(file, edges)
        # return ITE(mutable,
        #                 pyg.load_mutable_grg(file), 
        #                 pyg.load_immutable_grg(file, edges)
        #             )
                    
         
