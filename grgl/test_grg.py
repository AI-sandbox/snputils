from snputils.snp.io import read_vcf
from snputils.snp.genobj.snpobj import SNPObject
from typing import Callable, Any
import subprocess
import multiprocessing
import pygrgl
import time

# functional programming has corrupted me
ITE : Callable[[bool, Any, Any], Any] = lambda x, y, z : y if x else z


from pathlib import Path
def vcf_to_igd(filename: str) -> str:
    myfile : Path = Path(filename)
    newfile: str = ITE(("vcf" in filename), filename.replace("vcf", "igd"), filename + ".igd")
    if myfile.is_file():
        subprocess.run(["grg", "convert", filename, newfile])
        return newfile
    else: 
        raise FileNotFoundError(f"File {filename} does not exist")

def igd_to_grg(filename: str, 
               ncores: int = multiprocessing.cpu_count()) -> str:
    myfile : Path = Path(filename)
    newfile: str = ITE(("igd" in filename), filename.replace("igd", "grg"), filename + ".grg")
    if myfile.is_file():
        subprocess.run(["grg", "construct", "--parts", "20", "-j", f"{ncores}", "--out-file", newfile, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return newfile
    else: 
        raise FileNotFoundError(f"File {filename} does not exist")
    

def main():
    vcf_file : str = "test-200-samples.vcf"
    vcf_start = time.time()
    test_vcf : SNPObject = read_vcf(vcf_file)
    vcf_end = time.time()
    print("VCF load time ", vcf_end - vcf_start)
    grg_start = time.time()
    igd_file : str = vcf_to_igd(vcf_file)
    grg_file : str = igd_to_grg(igd_file)
    grg_data : pygrgl._grgl.MutableGRG = pygrgl.load_mutable_grg(grg_file)
    grg_end = time.time()
    print("GRG load time ", grg_end - grg_start)


if __name__ == "__main__":
    main()