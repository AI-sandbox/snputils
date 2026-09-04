from pathlib import Path

import numpy as np
import pytest

from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars


def _write_tiny_vcf(path: Path, chrom_header: str) -> None:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##source=snputils-test\n"
        f"{chrom_header}\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|0\n"
    )


@pytest.mark.parametrize("chrom_header", ["#CHROM", "CHROM"])
def test_vcf_polars_reads_sample_names_for_both_chrom_headers(tmp_path: Path, chrom_header: str) -> None:
    vcf_path = tmp_path / f"tiny_{chrom_header.replace('#', 'hash_')}.vcf"
    _write_tiny_vcf(vcf_path, chrom_header)

    snpobj = VCFReaderPolars(str(vcf_path)).read(samples=["HG00096"], genotype_mode="phased")

    expected = np.array([[[0, 1]], [[1, 1]]], dtype=np.int8)
    assert np.array_equal(snpobj.samples, np.array(["HG00096"]))
    assert np.array_equal(snpobj.genotypes, expected)
    assert np.array_equal(snpobj.variants_chrom, np.array(["1", "1"]))


@pytest.mark.parametrize(
    ("chrom_header", "requested_field"),
    [("#CHROM", "CHROM"), ("CHROM", "#CHROM")],
)
def test_vcf_polars_accepts_chrom_field_aliases(
    tmp_path: Path,
    chrom_header: str,
    requested_field: str,
) -> None:
    vcf_path = tmp_path / f"tiny_alias_{chrom_header.replace('#', 'hash_')}.vcf"
    _write_tiny_vcf(vcf_path, chrom_header)

    snpobj = VCFReaderPolars(str(vcf_path)).read(
        fields=[requested_field, "POS"],
        samples=["HG00096"],
    )

    assert np.array_equal(snpobj.variants_chrom, np.array(["1", "1"]))
    assert np.array_equal(snpobj.variants_pos, np.array([100, 200]))


def test_vcf_polars_passes_threads_to_csv_parser(tmp_path: Path, monkeypatch) -> None:
    vcf_path = tmp_path / "tiny_threads.vcf"
    _write_tiny_vcf(vcf_path, "#CHROM")
    observed_n_threads = []
    read_csv = __import__("polars").read_csv

    def recording_read_csv(*args, **kwargs):
        observed_n_threads.append(kwargs.get("n_threads"))
        return read_csv(*args, **kwargs)

    monkeypatch.setattr("snputils.snp.io.read.vcf.pl.read_csv", recording_read_csv)
    snpobj = VCFReaderPolars(vcf_path).read(genotype_mode="phased", threads=2)

    assert observed_n_threads == [2]
    assert snpobj.genotypes.shape == (2, 2, 2)


def test_vcf_polars_fallback_uses_single_native_reader_thread(tmp_path: Path, monkeypatch) -> None:
    vcf_path = tmp_path / "tiny_fallback.vcf"
    _write_tiny_vcf(vcf_path, "#CHROM")
    fallback_threads = []
    native_read = VCFReader.read

    def fail_polars_read(*args, **kwargs):
        raise RuntimeError("forced Polars failure")

    def recording_native_read(*args, **kwargs):
        fallback_threads.append(kwargs.get("threads"))
        return native_read(*args, **kwargs)

    monkeypatch.setattr("snputils.snp.io.read.vcf.pl.read_csv", fail_polars_read)
    monkeypatch.setattr(VCFReader, "read", recording_native_read)

    snpobj = VCFReaderPolars(vcf_path).read(threads=4)

    assert fallback_threads == [1]
    assert snpobj.genotypes.shape == (2, 2, 2)


@pytest.mark.parametrize("threads", [0, -1])
def test_vcf_polars_rejects_nonpositive_threads(threads: int) -> None:
    with pytest.raises(ValueError, match="threads must be at least 1"):
        VCFReaderPolars("unused.vcf").read(threads=threads)


def test_vcf_polars_rejects_noninteger_threads() -> None:
    with pytest.raises(TypeError, match="threads must be an integer or None"):
        VCFReaderPolars("unused.vcf").read(threads=1.5)
