import pytest
import numpy as np
from pathlib import Path
import os
import gzip
from .utils import create_benchmark_test


def read_vcf_snputils(path, sum_strands=True):
    """Read VCF file using snputils"""
    import snputils
    return snputils.read_vcf(path, sum_strands=sum_strands).calldata_gt


def read_vcf_snputils_polars(path, sum_strands=True):
    """Read VCF file using snputils and polars"""
    from snputils.snp.io.read.vcf import VCFReaderPolars
    return VCFReaderPolars(path).read(fields=[], sum_strands=sum_strands).calldata_gt


def read_vcf_scikit_allel(path, sum_strands=True):
    """Read VCF file using scikit-allel"""
    import allel
    gt = allel.read_vcf(str(path), fields=['calldata/GT'])['calldata/GT']
    if sum_strands:
        return np.sum(gt > 0, axis=2, dtype=np.uint8)
    return gt.astype(np.int8)


def read_vcf_hail(path, sum_strands=True):
    """Read VCF file using hail"""
    import hail as hl
    spark_memory = os.environ.get("HAIL_SPARK_MEMORY", "192g")
    cpus = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1"))))
    hl.init(master=f"local[{cpus}]", spark_conf={
        "spark.driver.memory": spark_memory,
        "spark.driver.cores": str(cpus),
        "spark.executor.memory": spark_memory,
        "spark.executor.cores": str(cpus),
        "spark.default.parallelism": str(cpus),
        "spark.sql.shuffle.partitions": str(cpus),
    })
    mt = hl.import_vcf(str(path), force_bgz=True, reference_genome="GRCh38")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#CHROM") or line.startswith("CHROM"):
                n_samples = max(0, len(line.rstrip("\r\n").split("\t")) - 9)
                break
        else:
            raise ValueError(f"Could not find VCF header in {path}")
    if sum_strands:
        mt = np.array(hl.or_else(mt.GT.n_alt_alleles(), -2).collect(), dtype=np.int8).reshape((-1, n_samples))
    else:
        mt = np.array(
            hl.array([hl.or_else(mt.GT[0], -1), hl.or_else(mt.GT[1], -1)]).collect(),
            dtype=np.int8,
        ).reshape((-1, n_samples, 2))
    hl.stop()
    return mt


def read_vcf_pyvcf3(path, sum_strands=True):
    """Read VCF file using PyVCF3"""
    import vcf
    if sum_strands:
        return np.array([[s.data.GT.count('1') for s in record.samples] for record in vcf.Reader(filename=str(path))], dtype=np.uint8)
    records = []
    for record in vcf.Reader(filename=str(path)):
        row = []
        for sample in record.samples:
            gt = sample.data.GT
            alleles = [] if gt is None else str(gt).replace("|", "/").split("/")
            pair = [-1 if allele == "." else int(allele) for allele in alleles[:2]]
            pair.extend([-1] * (2 - len(pair)))
            row.append(pair)
        records.append(row)
    return np.array(records, dtype=np.int8)


def read_vcf_cyvcf2(path, sum_strands=True):
    """Read VCF file using cyvcf2"""
    import cyvcf2
    gt = np.stack([record.genotype.array()[:, :2] for record in cyvcf2.VCF(str(path))]).astype(np.int8)
    if sum_strands:
        return gt.sum(axis=2, dtype=np.int8)
    return gt


def read_vcf_pysam(path, sum_strands=True):
    """Read VCF file using pysam"""
    import pysam
    with pysam.VariantFile(str(path)) as vcf:
        if sum_strands:
            return np.array([[s.get('GT').count(1) for s in record.samples.values()] for record in vcf], dtype=np.uint8)
        return np.array(
            [[[-1 if allele is None else allele for allele in s.get('GT')[:2]] for s in record.samples.values()] for record in vcf],
            dtype=np.int8,
        )


# Benchmark Configuration
READERS = [
    (read_vcf_snputils, "snputils"),
    (read_vcf_snputils_polars, "snputils-polars"),
    (read_vcf_scikit_allel, "scikit-allel"),
    (read_vcf_hail, "hail"),
    (read_vcf_pyvcf3, "pyvcf3"),
    (read_vcf_cyvcf2, "cyvcf2"),
    (read_vcf_pysam, "pysam"),
]


@pytest.mark.benchmark(group="VCF-readers", warmup=False, min_rounds=3)
@pytest.mark.parametrize("reader,name", READERS)
def test_vcf_readers(benchmark, reader, name, path, memory_profile, reader_name, sum_strands):
    """Benchmark readers and verify output"""
    if reader_name is not None and name != reader_name:
        pytest.skip(f"Skipping {name}; --reader-name={reader_name} requested")

    path = Path(path)
    if path.suffixes[-2:] != ['.vcf', '.gz'] and path.suffix != ".vcf":
        gz_path = Path(str(path) + ".vcf.gz")
        path = gz_path if gz_path.exists() else Path(str(path) + ".vcf")
    ref_array = None if memory_profile else read_vcf_snputils(path, sum_strands=sum_strands)
    create_benchmark_test(
        benchmark,
        reader,
        path,
        name,
        ref_array,
        memory_profile,
        sum_strands=sum_strands,
        ref_reader_func=read_vcf_snputils,
    )
