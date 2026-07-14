"""Smoke-test an installed snputils wheel without using the source checkout."""

from importlib import import_module
from pathlib import Path
import platform

import snputils


project_root = Path(__file__).resolve().parents[2]
installed_package = Path(snputils.__file__).resolve()
try:
    installed_package.relative_to(project_root)
except ValueError:
    pass
else:
    raise RuntimeError(f"snputils was imported from the source checkout: {installed_package}")

if snputils.__version__ == "unknown":
    raise RuntimeError("The installed wheel does not expose package version metadata.")

for module_name in (
    "snputils.snp.io.read._bcf",
    "snputils.snp.io.write._bcf",
    "snputils.snp.io._bgen",
):
    import_module(module_name)

for public_name in (
    "SNPObject",
    "VCFReader",
    "BCFReader",
    "BGENReader",
    "PGENReader",
    "VCFWriter",
    "BCFWriter",
    "BGENWriter",
    "PGENWriter",
):
    getattr(snputils, public_name)

snpobj = snputils.build_synthetic_snp_dataset(n_samples=4, n_snps=8, seed=1)
if snpobj.n_samples != 4 or snpobj.n_snps != 8:
    raise RuntimeError("The installed wheel failed the synthetic SNPObject smoke test.")

from snputils.tools.cli import main

if main(["version"]) != 0:
    raise RuntimeError("The installed wheel's CLI smoke test failed.")

print(
    f"Wheel smoke test passed: snputils {snputils.__version__}, "
    f"{platform.system()} {platform.machine()}"
)
