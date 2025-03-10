import pyigd as pyi
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader

from typing import Optional, Union
from os.path import abspath, splitext
import pathlib


@SNPBaseReader.register
class IGDReader(SNPBaseReader):
    def __init__(self, **kwargs):
        super.__init__(kwargs)
    def read(self) -> SNPObject:
        name, _ext1 = splitext(self.filename)
        name, _ext2 = splitext(name)
        # may use _ext later. not sure. 
        _ext = _ext1 + _ext2