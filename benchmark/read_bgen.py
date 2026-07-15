from pathlib import Path

import numpy as np
import pytest

from .utils import create_benchmark_test


def _probabilities_to_dosage(probabilities):
    if probabilities.shape[-1] == 3:
        return probabilities @ np.array([0.0, 1.0, 2.0], dtype=np.float32)
    if probabilities.ndim == 2:
        return _probabilities_to_dosage(probabilities[np.newaxis, :, :])[0]
    if probabilities.shape[-1] % 2 == 0:
        finite = np.isfinite(probabilities)
        widths = finite.sum(axis=-1)
        observed_widths = widths > 0
        even_widths = np.all((widths % 2 == 0) | ~observed_widths, axis=1)
        phased_like = np.zeros(probabilities.shape[0], dtype=bool)
        even_indices = np.flatnonzero(even_widths)
        if even_indices.size:
            pairs = probabilities[even_indices].reshape(
                even_indices.size,
                probabilities.shape[1],
                probabilities.shape[2] // 2,
                2,
            )
            pair_present = np.isfinite(pairs).any(axis=3)
            pair_sums = np.nansum(pairs, axis=3, dtype=np.float32)
            phased_like[even_indices] = np.all(
                np.isclose(pair_sums, 1.0, atol=1e-4, rtol=0) | ~pair_present,
                axis=(1, 2),
            )

        dosage = np.full(probabilities.shape[:2], np.nan, dtype=np.float32)
        if np.any(phased_like):
            dosage[phased_like] = np.nansum(probabilities[phased_like, :, 1::2], axis=-1, dtype=np.float32)
        if np.any(~phased_like):
            weights = np.arange(probabilities.shape[-1], dtype=np.float32)
            dosage[~phased_like] = np.nansum(probabilities[~phased_like] * weights, axis=-1, dtype=np.float32)
        dosage[~np.any(finite, axis=-1)] = np.nan
        return dosage
    finite = np.isfinite(probabilities)
    dosage = np.nansum(
        probabilities * np.arange(probabilities.shape[-1], dtype=np.float32),
        axis=-1,
        dtype=np.float32,
    )
    dosage[~np.any(finite, axis=-1)] = np.nan
    return dosage


def _probabilities_to_phased_calls(probabilities):
    """Hard-call diploid haplotypes from phased BGEN probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim not in (2, 3):
        raise ValueError("Phased BGEN probabilities must be a 2D or 3D array.")

    width = probabilities.shape[-1]
    if width < 4 or width % 2 != 0:
        raise ValueError(
            "Phased diploid BGEN probabilities must contain two equally sized "
            "haplotype probability vectors."
        )

    n_alleles = width // 2
    haplotype_probabilities = probabilities.reshape(
        probabilities.shape[:-1] + (2, n_alleles)
    )
    finite = np.isfinite(haplotype_probabilities)
    any_finite = np.any(finite, axis=-1)
    fully_finite = np.all(finite, axis=-1)
    if np.any(any_finite & ~fully_finite):
        raise ValueError("Each phased haplotype probability vector must be complete or missing.")
    if np.any(np.sum(fully_finite, axis=-1) == 1):
        raise ValueError("The phased-call benchmark requires diploid BGEN records.")

    probability_sums = np.sum(
        np.where(finite, haplotype_probabilities, 0.0),
        axis=-1,
        dtype=np.float32,
    )
    if np.any(fully_finite & ~np.isclose(probability_sums, 1.0, atol=1e-4, rtol=0)):
        raise ValueError("The probability vectors do not look like phased BGEN probabilities.")

    safe_probabilities = np.where(finite, haplotype_probabilities, -np.inf)
    calls = np.argmax(safe_probabilities, axis=-1).astype(np.int8)
    calls[~fully_finite] = -1
    return calls


def _phased_calls_or_skip(probabilities, reader_name):
    """Return phased calls, omitting readers that do not preserve BGEN phase."""
    try:
        return _probabilities_to_phased_calls(probabilities)
    except ValueError as exc:
        pytest.skip(f"{reader_name} does not expose phased BGEN probabilities: {exc}")


def read_bgen_snputils(path, genotype_mode="dosage"):
    """Read BGEN file using snputils"""
    from snputils.snp.io.read.bgen import BGENReader

    reader = BGENReader(path)
    if genotype_mode == "dosage":
        return reader.read_dosage()
    return reader.read(fields=["GT"], genotype_mode="phased").genotypes


def read_bgen_bgen(path, genotype_mode="dosage"):
    """Read BGEN file using bgen"""
    from bgen import BgenReader
    sample_path = str(Path(path).with_suffix(".sample")) if Path(path).with_suffix(".sample").exists() else ""
    with BgenReader(str(path), sample_path, delay_parsing=True) as bfile:
        n_variants = len(bfile)
        variants = iter(bfile)
        first_variant = next(variants)
        first_probabilities = np.asarray(first_variant.probabilities, dtype=np.float32)
        n_samples = first_probabilities.shape[0]
        if genotype_mode == "dosage":
            out = np.empty((n_variants, n_samples), dtype=np.float32)
            out[0] = _probabilities_to_dosage(first_probabilities)
        else:
            out = np.empty((n_variants, n_samples, 2), dtype=np.int8)
            out[0] = _probabilities_to_phased_calls(first_probabilities)

        for i, variant in enumerate(variants, start=1):
            probabilities = np.asarray(variant.probabilities, dtype=np.float32)
            if genotype_mode == "dosage":
                out[i] = _probabilities_to_dosage(probabilities)
            else:
                out[i] = _probabilities_to_phased_calls(probabilities)
    return out


def read_bgen_pysnptools(path, genotype_mode="dosage"):
    """Read BGEN file using pysnptools"""
    from pysnptools.distreader import Bgen
    probabilities = Bgen(str(path)).read(order="C", dtype=np.float32).val.transpose(1, 0, 2)
    if genotype_mode == "dosage":
        return _probabilities_to_dosage(probabilities)
    return _phased_calls_or_skip(probabilities, "pysnptools")


def read_bgen_sgkit(path, genotype_mode="dosage"):
    """Read BGEN file using sgkit"""
    from sgkit.io.bgen import read_bgen
    sample_path = Path(path).with_suffix(".sample")
    kwargs = {"sample_path": str(sample_path)} if sample_path.exists() else {}
    ds = read_bgen(str(path), chunks="auto", gp_dtype="float32", **kwargs)
    if genotype_mode == "dosage":
        dosage = ds["call_dosage"]
        if "call_dosage_mask" in ds:
            dosage = dosage.where(~ds["call_dosage_mask"])
        return dosage.compute().values.astype(np.float32, copy=False)

    probabilities = ds["call_genotype_probability"]
    if "call_genotype_probability_mask" in ds:
        probabilities = probabilities.where(~ds["call_genotype_probability_mask"])
    probabilities = probabilities.compute().values.astype(np.float32, copy=False)
    return _phased_calls_or_skip(probabilities, "sgkit")


def read_bgen_hail(path, genotype_mode="dosage"):
    """Read BGEN file using hail"""
    if genotype_mode == "phased":
        pytest.skip("Hail does not preserve phase when importing BGEN files.")
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
        mt = hl.import_bgen(
            str(path),
            entry_fields=["dosage"],
            sample_file=sample_file,
            n_partitions=cpus,
        )
        n_samples = mt.count_cols()
        return np.array(
            hl.or_else(mt.dosage, float("nan")).collect(),
            dtype=np.float32,
        ).reshape((-1, n_samples))
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
def test_bgen_readers(benchmark, reader, name, path, memory_profile, reader_name, genotype_mode):
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
        genotype_mode=genotype_mode,
        ref_reader_func=read_bgen_bgen,
        assert_allclose=genotype_mode == "dosage",
        atol=(1 / 255 + 1e-6) if genotype_mode == "dosage" else 0.0,
        equal_nan=genotype_mode == "dosage",
    )
