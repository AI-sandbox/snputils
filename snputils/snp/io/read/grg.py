import pygrgl as pyg
from snputils.snp.io.read.base import SNPBaseReader
from snputils.snp.genobj.grgobj import GRGObject
from typing import Optional
import pathlib


@SNPBaseReader.register
class GRGReader(SNPBaseReader):
    def read(self,
             mutable: Optional[bool] = None,
             load_up_edges: Optional[bool] = None,
             binary_mutations: Optional[bool] = None) -> GRGObject:
        """
        Read in a GRG or TSKit File
        """
        file = str(pathlib.Path(self.filename).resolve())
        extension = pathlib.Path(file).suffix.lower()
        edges = load_up_edges if load_up_edges is not None else True
        binmuts = binary_mutations if binary_mutations is not None else False

        if extension == ".trees":
            return GRGObject(calldata_gt=pyg.grg_from_trees(file, binmuts), filename=file, mutable=True)
        if mutable:
            return GRGObject(calldata_gt=pyg.load_mutable_grg(file), filename=file, mutable=True)

        return GRGObject(calldata_gt=pyg.load_immutable_grg(file, edges), filename=file, mutable=False)
     
                    
         
