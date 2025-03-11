import pyigd as pyi
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader
from io import TextIOWrapper
from typing import Optional, Union, Any, List
from os.path import abspath, splitext
import multiprocessing
import pathlib
import time
import subprocess

@SNPBaseReader.register
class IGDReader(SNPBaseReader):
    def __init__(self, **kwargs):
        super.__init__(kwargs)
    def read(self) -> SNPObject:
        # TO BE IMPLEMENTED
        name, _ext1 = splitext(self.filename)
        name, _ext2 = splitext(name)
        # may use _ext later. not sure. 
        _ext = _ext1 + _ext2


    def to_grg(self,
               range : Optional[str] = None,
               parts : Optional[int] = None,
               jobs  : Optional[int] = None,
               trees : Optional[int] = None,
               binmuts : Optional[bool] = None,
               no_file_cleanup : Optional[bool] = None,
               maf_flip : Optional[bool] = None,
               shape_lf_filter : Optional[float] = None,
               population_ids : Optional[str] = None,
               bs_triplet : Optional[int] = None,
               out_file : Optional[str] = None,
               verbose : Optional[bool] = None,
               no_merge : Optional[bool] = None,
               logfile_out : Optional[str] = None,
               logfile_err : Optional[str] = None
               ) -> None:
        """
        Converts the IGD file to a GRG file. All of these parameters are passed into the grg command-line tool.
        The main difference is in parts and trees, which are designed for large files.
        Args:
            range: Restrict to the given range. Can be absolute (in base-pairs) or relative (0.0 to 1.0).
            parts: The number of parts to split the sequence into; Unlike grgl's default of 8, we default to 50.
            jobs: Number of jobs (threads/cores) to use. Defaults to 1.
            trees: Number of trees to use during shape construction. Unlike grgl, defaults to 16. 
            binary_muts: Use binary mutations (don't track specific alternate alleles).
            no_file_cleanup: Do not cleanup intermediate files (for debugging, e.g.).
            maf_flip: Switch the reference allele with the major allele when they differ
            shape_lf_filter: During shape construction ignore mutations with counts less than this.
                If value is <1.0 then it is treated as a frequency. Defaults to 10 (count).
            population_ids: Format: "filename:fieldname". Read population ids from the given 
                tab-separate file, using the given fieldname.
            bs_triplet: Run the triplet algorithm for this many iterations in BuildShape
            out_file: Specify an output file. If none is supplied, the default name is <current_vcf_name>.grg.
            verbose:Verbose output, including timing information.
            no_merge: Do not merge the resulting GRGs (so if you specified "parts = C" there will be C GRGs).
            logfile_out: The file to log standard output to. If None (default), no output will be logged (i.e., piped to dev null).
            logfile_err: The file to log standard error to. If None (default), no error will be logged (i.e., piped to dev null).

        """

        # for debugging only 
        if self.debug:
            start = time.time()
        self._to_igd(logfile_out, logfile_err)
        if self.debug:
            end = time.time()
            print("vcf -> igd ", end - start)
        name, _ext = splitext(self._igd_path)
        self._grg_path = name + ".grg"

        # set logfiles 
        # should I use subprocess.devnull? probably. I don't want to to keep the open call's type consistent
        lf_o : Union[int, TextIOWrapper] = subprocess.DEVNULL if logfile_out == None else open(logfile_out, "w")
        lf_e : Union[int, TextIOWrapper] = subprocess.DEVNULL if logfile_out == None else open(logfile_err, "w")
        if self.debug:
            start = time.time()
        args = ["grg", "construct"]
        args += self._setarg(range, "-r", None)
        args += self._setarg(parts, "-p", 50)
        args += self._setarg(jobs,  "-j", f"{multiprocessing.cpu_count()}")
        args += self._setarg(trees, "-t", 16)
        args += self._setarg(binmuts, "-b", None)
        args += self._setarg(no_file_cleanup, "-c", None)
        args += self._setarg(maf_flip, "--maf-flip", None)
        args += self._setarg(shape_lf_filter, "--shape-lf-filter", None)
        args += self._setarg(population_ids, "--population-ids", None)
        args += self._setarg(bs_triplet, "--bs_triplet", None)
        args += self._setarg(out_file, "--out-file", self._grg_path)
        args += self._setarg(verbose, "-v", None)
        args += self._setarg(no_merge, "--no-merge", None)
        # finally, add the infile
        args += [f"{self.filename}"]
        print(args)
        subprocess.run(args, stdout=lf_o, stderr=lf_e)
        
        if self.debug:
            end = time.time()
            print("igd -> grg ", end - start)

        # cleanup
     
        if not isinstance(lf_o, int):
            lf_o.close()
        if not isinstance(lf_e, int): 
            lf_e.close()
       
     

    def _setarg(self, x: Optional[Any], flag: str, default_arg: Optional[Any] = None) -> List[str]:
        if x is None and default_arg is not None:
            return [flag, f"{default_arg}"] 
        elif x is not None:
            return [flag, f"{x}"]
        else:
            return []
        