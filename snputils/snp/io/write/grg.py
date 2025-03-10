import pygrgl as pyg
from snputils.snp.genobj.snpobj import SNPObject


from typing import Optional, Union, List, Tuple
from os.path import abspath
import pathlib



class GRGWriter:
    def __init__(self, grgobj: Union[pyg.GRG, pyg.MutableGRG], filename: str):
        self.grgobj = grgobj
        self.mutability = False if isinstance(self.grgobj, pyg.GRG) else True
        self.filename = filename
    
    def write(self, allow_simplify : Optional[bool]                         = None, 
                    subset         : Optional[bool]                         = None,
                    direction      : Optional[pyg._grgl.TraversalDirection] = None,
                    seed_list      : Optional[List[int]]                    = None,
                    bp_range       : Optional[Tuple[int, int]]              = None):
        """
        """

        if subset:
            assert (direction is not None), "If subset is true, must specify direction."
            assert (seed_list is not None), "If subset is true, must specify seed list."
            _bp_range = (0,0) if bp_range is None else bp_range
            pyg.save_subset(self.grgobj, self.filename, direction, seed_list, _bp_range) 
        else:
            _allow_simplify = True if allow_simplify is None else allow_simplify 
            pyg.save_grg(self.grgobj, self.filename, _allow_simplify)

