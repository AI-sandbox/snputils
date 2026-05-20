import pytest
import numpy as np
import os
from .utils import create_benchmark_test


def read_bed_snputils(path, sum_strands=True):
    """Read BED fileset using snputils"""
    import snputils
    return snputils.read_bed(path, sum_strands=sum_strands, fields=["GT"]).calldata_gt


def read_bed_pgenlib(path, sum_strands=True):
    """Read BED fileset using Pgenlib"""
    import pgenlib
    with open(path + '.fam', 'r') as handle:
        sample_ct = sum(1 for _ in handle)
    pgen = pgenlib.PgenReader(str.encode(path + '.bed'), raw_sample_ct=sample_ct)
    variant_ct = pgen.get_variant_ct()
    variant_idxs = np.arange(variant_ct, dtype=np.uint32)
    if sum_strands:
        genotypes = np.empty((variant_ct, sample_ct), dtype=np.int8)
        pgen.read_list(variant_idxs, genotypes)
    else:
        genotypes = np.empty((variant_ct, sample_ct * 2), dtype=np.int32)
        pgen.read_alleles_list(variant_idxs, genotypes)
        genotypes = genotypes.astype(np.int8).reshape((variant_ct, sample_ct, 2))
    pgen.close()
    return genotypes


def read_bed_hail(path, sum_strands=True):
    """Read BED fileset using hail"""
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
    mt = hl.import_plink(path + '.bed', path + '.bim', path + '.fam')
    n_samples = mt.count_cols()
    if sum_strands:
        mt = np.array(hl.or_else(mt.GT.n_alt_alleles(), -2).collect(), dtype=np.int8).reshape((-1, n_samples))
    else:
        mt = np.array(
            hl.array([
                hl.or_else(mt.GT[0], -1),
                hl.or_else(mt.GT[1], -1),
            ]).collect(),
            dtype=np.int8,
        ).reshape((-1, n_samples, 2))
    hl.stop()
    return mt


def read_bed_sgkit(path, sum_strands=True):
    """Read BED fileset using sgkit"""
    from sgkit.io import plink
    genotypes = plink.read_plink(path=path).call_genotype.to_numpy()
    genotypes = np.where(genotypes < 0, -1, 1 - genotypes).astype(np.int8)
    if sum_strands:
        return np.sum(genotypes, axis=2, dtype=np.int8)
    return genotypes


def read_bed_pandas_plink(path, sum_strands=True):
    """Read BED fileset using pandas-plink"""
    if not sum_strands:
        pytest.skip("pandas-plink BED benchmark returns dosages only.")
    import pandas_plink
    _, _, genotypes = pandas_plink.read_plink(path)
    return 2 - genotypes.compute().astype(np.uint8)


def read_bed_plinkio(path, sum_strands=True):
    """Read BED fileset using plinkio"""
    if not sum_strands:
        pytest.skip("plinkio BED benchmark returns dosages only.")
    from plinkio import plinkfile
    return np.array([2 - np.array(row) for row in plinkfile.open(path)], dtype=np.uint8)


def read_bed_pysnptools(path, sum_strands=True):
    """Read BED fileset using pysnptools"""
    if not sum_strands:
        pytest.skip("pysnptools BED benchmark returns dosages only.")
    from pysnptools.snpreader import Bed
    bed_path = path if str(path).endswith(".bed") else path + ".bed"
    return (2 - Bed(bed_path).read().val.T.astype(np.uint8))


READERS = [
    (read_bed_snputils, "snputils"),
    (read_bed_pgenlib, "pgenlib"),
    (read_bed_hail, "hail"),
    (read_bed_sgkit, "sgkit"),
    (read_bed_pandas_plink, "pandas-plink"),
    (read_bed_plinkio, "plinkio"),
    (read_bed_pysnptools, "pysnptools"),
]


@pytest.mark.benchmark(group="BED-readers", warmup=False, min_rounds=3)
@pytest.mark.parametrize("reader,name", READERS)
def test_bed_readers(benchmark, reader, name, path, memory_profile, reader_name, sum_strands):
    """Benchmark readers and verify output"""
    if reader_name is not None and name != reader_name:
        pytest.skip(f"Skipping {name}; --reader-name={reader_name} requested")
    ref_array = None if memory_profile else read_bed_snputils(path, sum_strands=sum_strands)
    create_benchmark_test(
        benchmark,
        reader,
        path,
        name,
        ref_array,
        memory_profile,
        sum_strands=sum_strands,
        ref_reader_func=read_bed_snputils,
    )
