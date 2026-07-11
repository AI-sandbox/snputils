import pytest
import numpy as np
from .utils import create_benchmark_test


def read_pgen_snputils(path, genotype_mode="dosage"):
    """Read PGEN file using snputils"""
    import snputils
    return snputils.read_pgen(path, genotype_mode=genotype_mode, fields=["GT"], chromosome_ploidy="autosomal").genotypes


def read_pgen_pgenlib(path, genotype_mode="dosage"):
    """Read PGEN fileset using Pgenlib"""
    import pgenlib
    pgen = pgenlib.PgenReader(str.encode(path + '.pgen'))  # No need for raw_sample_ct with pgen
    variant_ct = pgen.get_variant_ct()
    sample_ct = pgen.get_raw_sample_ct()
    variant_idxs = np.arange(variant_ct, dtype=np.uint32)
    if genotype_mode == "dosage":
        genotypes = np.empty((variant_ct, sample_ct), dtype=np.int8)
        pgen.read_list(variant_idxs, genotypes)
    else:
        genotypes = np.empty((variant_ct, sample_ct * 2), dtype=np.int32)
        pgen.read_alleles_list(variant_idxs, genotypes)
        genotypes = genotypes.astype(np.int8).reshape((variant_ct, sample_ct, 2))
    pgen.close()
    return genotypes


READERS = [
    (read_pgen_snputils, "snputils"),
    (read_pgen_pgenlib, "pgenlib"),
]


@pytest.mark.benchmark(group="PGEN-readers", warmup=False)
@pytest.mark.parametrize("reader,name", READERS)
def test_pgen_readers(benchmark, reader, name, path, memory_profile, reader_name, genotype_mode):
    """Benchmark readers and verify output"""
    if reader_name is not None and name != reader_name:
        pytest.skip(f"Skipping {name}; --reader-name={reader_name} requested")
    ref_array = None if memory_profile else read_pgen_snputils(path, genotype_mode=genotype_mode)
    create_benchmark_test(
        benchmark,
        reader,
        path,
        name,
        ref_array,
        memory_profile,
        genotype_mode=genotype_mode,
        ref_reader_func=read_pgen_snputils,
    )
