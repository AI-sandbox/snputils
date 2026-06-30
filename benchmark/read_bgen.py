from pathlib import Path

import numpy as np
import pytest

from .utils import create_benchmark_test


def _probabilities_to_dosage(probabilities):
    if probabilities.shape[-1] == 3:
        return probabilities @ np.array([0.0, 1.0, 2.0], dtype=np.float32)
    if probabilities.shape[-1] % 2 == 0:
        return np.nansum(probabilities[..., 1::2], axis=-1, dtype=np.float32)
    return probabilities @ np.arange(probabilities.shape[-1], dtype=np.float32)


def read_bgen_snputils(path, sum_strands=True):
    """Read BGEN file using snputils"""
    import snputils
    snpobj = snputils.read_bgen(path, fields=["GP"])
    if sum_strands:
        return _probabilities_to_dosage(snpobj.calldata_gp.astype(np.float32, copy=False))
    return snpobj.calldata_gp.astype(np.float32, copy=False)


def read_bgen_bgen(path, sum_strands=True):
    """Read BGEN file using bgen"""
    from bgen import BgenReader
    sample_path = str(Path(path).with_suffix(".sample")) if Path(path).with_suffix(".sample").exists() else ""
    with BgenReader(str(path), sample_path, delay_parsing=True) as bfile:
        first_probabilities = np.asarray(bfile[0].probabilities, dtype=np.float32)
        n_variants = len(bfile)
        n_samples, width = first_probabilities.shape
        if sum_strands:
            out = np.empty((n_variants, n_samples), dtype=np.float32)
        else:
            out = np.empty((n_variants, n_samples, width), dtype=np.float32)

        for i, variant in enumerate(bfile):
            probabilities = np.asarray(variant.probabilities, dtype=np.float32)
            if sum_strands:
                out[i] = _probabilities_to_dosage(probabilities)
            else:
                out[i] = probabilities
    return out


def read_bgen_pysnptools(path, sum_strands=True):
    """Read BGEN file using pysnptools"""
    from pysnptools.distreader import Bgen
    probabilities = Bgen(str(path)).read(order="C", dtype=np.float32).val.transpose(1, 0, 2)
    if sum_strands:
        return probabilities @ np.array([0.0, 1.0, 2.0], dtype=np.float32)
    return probabilities


def read_bgen_sgkit(path, sum_strands=True):
    """Read BGEN file using sgkit"""
    from sgkit.io.bgen import read_bgen
    sample_path = Path(path).with_suffix(".sample")
    kwargs = {"sample_path": str(sample_path)} if sample_path.exists() else {}
    ds = read_bgen(str(path), chunks="auto", gp_dtype="float32", **kwargs)
    if sum_strands:
        dosage = ds["call_dosage"]
        if "call_dosage_mask" in ds:
            dosage = dosage.where(~ds["call_dosage_mask"])
        return dosage.compute().values.astype(np.float32, copy=False)

    probabilities = ds["call_genotype_probability"]
    if "call_genotype_probability_mask" in ds:
        probabilities = probabilities.where(~ds["call_genotype_probability_mask"])
    return probabilities.compute().values.astype(np.float32, copy=False)


def read_bgen_hail(path, sum_strands=True):
    """Read BGEN file using hail"""
    import os
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
    try:
        path = Path(path)
        if not Path(str(path) + ".idx2").exists():
            hl.index_bgen(str(path), reference_genome="GRCh37")
        sample_path = path.with_suffix(".sample")
        sample_file = str(sample_path) if sample_path.exists() else None
        entry_fields = ["dosage"] if sum_strands else ["GP"]
        mt = hl.import_bgen(
            str(path),
            entry_fields=entry_fields,
            sample_file=sample_file,
            n_partitions=cpus,
        )
        n_samples = mt.count_cols()
        if sum_strands:
            return np.array(hl.or_else(mt.dosage, float("nan")).collect(), dtype=np.float32).reshape((-1, n_samples))
        return np.array(
            hl.or_else(mt.GP, hl.array([float("nan"), float("nan"), float("nan")])).collect(),
            dtype=np.float32,
        ).reshape((-1, n_samples, 3))
    finally:
        hl.stop()


READERS = [
    (read_bgen_snputils, "snputils"),
    (read_bgen_bgen, "bgen"),
    (read_bgen_pysnptools, "pysnptools"),
    (read_bgen_sgkit, "sgkit"),
    (read_bgen_hail, "hail"),
]


@pytest.mark.benchmark(group="BGEN-readers", warmup=False)
@pytest.mark.parametrize("reader,name", READERS)
def test_bgen_readers(benchmark, reader, name, path, memory_profile, reader_name, sum_strands):
    """Benchmark readers and verify output"""
    if reader_name is not None and name != reader_name:
        pytest.skip(f"Skipping {name}; --reader-name={reader_name} requested")

    path = Path(path)
    if path.suffix != ".bgen":
        path = Path(str(path) + ".bgen")
    ref_array = None
    create_benchmark_test(
        benchmark,
        reader,
        path,
        name,
        ref_array,
        memory_profile,
        sum_strands=sum_strands,
        ref_reader_func=read_bgen_snputils,
        assert_allclose=True,
        atol=1 / 255 + 1e-6,
        equal_nan=True,
        verify=not (name == "snputils" and not memory_profile),
    )
