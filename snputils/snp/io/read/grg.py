import pygrgl as pyg
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from snputils.common._utils import ITE
from typing import Optional, Union
from os.path import abspath
import pathlib


@SNPBaseReader.register
class GRGReader(SNPBaseReader):
    def read(self,
             mutable: Optional[bool] = None,
             load_up_edges: Optional[bool] = None) -> Union[pyg.GRG | pyg.MutableGRG]:
        
        file = ITE(isinstance(self.filename, pathlib.Path), abspath(self.filename), self.filename)
        return ITE(mutable, pyg.load_mutable_grg(file), pyg.load_immutable_grg(file, load_up_edges))
