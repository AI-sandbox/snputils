from pathlib import Path

import numpy as np
import pytest

from .utils import create_benchmark_test


def read_bcf_snputils(path, sum_strands=True):
    """Read BCF file using snputils"""
    import snputils
    return snputils.read_bcf(path, fields=["GT"], sum_strands=sum_strands).genotypes


def read_bcf_cyvcf2(path, sum_strands=True):
    """Read BCF file using cyvcf2"""
    import cyvcf2
    gt = np.stack([record.genotype.array()[:, :2] for record in cyvcf2.VCF(str(path))]).astype(np.int8)
    if sum_strands:
        return gt.sum(axis=2, dtype=np.int8)
    return gt


def read_bcf_pysam(path, sum_strands=True):
    """Read BCF file using pysam"""
    import pysam
    with pysam.VariantFile(str(path)) as bcf:
        if sum_strands:
            return np.array([[s.get("GT").count(1) for s in record.samples.values()] for record in bcf], dtype=np.uint8)
        return np.array(
            [[[-1 if allele is None else allele for allele in s.get("GT")[:2]] for s in record.samples.values()] for record in bcf],
            dtype=np.int8,
        )


READERS = [
    (read_bcf_snputils, "snputils"),
    (read_bcf_cyvcf2, "cyvcf2"),
    (read_bcf_pysam, "pysam"),
]


@pytest.mark.benchmark(group="BCF-readers", warmup=False)
@pytest.mark.parametrize("reader,name", READERS)
def test_bcf_readers(benchmark, reader, name, path, memory_profile, reader_name, sum_strands):
    """Benchmark readers and verify output"""
    if reader_name is not None and name != reader_name:
        pytest.skip(f"Skipping {name}; --reader-name={reader_name} requested")

    path = Path(path)
    if path.suffix != ".bcf":
        path = Path(str(path) + ".bcf")
    ref_array = None if memory_profile else read_bcf_snputils(path, sum_strands=sum_strands)
    create_benchmark_test(
        benchmark,
        reader,
        path,
        name,
        ref_array,
        memory_profile,
        sum_strands=sum_strands,
        ref_reader_func=read_bcf_snputils,
    )
