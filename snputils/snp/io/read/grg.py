import pygrgl as pyg
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from typing import Optional, Union
from os.path import abspath
import pathlib


@SNPBaseReader.register
class GRGReader(SNPBaseReader):
    def read(self,
             mutable: Optional[bool] = None) -> Union[pyg.GRG | pyg.MutableGRG]:
        
        if isinstance(self.filename, pathlib.Path):
                file = abspath(self.filename)
        else:
                file = self.filename
        if mutable:
            return pyg.load_mutable_grg(file)
        else:
            return pyg.load_immutable_grg(file)
