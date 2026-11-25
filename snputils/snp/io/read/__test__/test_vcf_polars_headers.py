import inspect
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import polars as pl
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


@pytest.mark.parametrize("streaming", [False, True], ids=["eager", "streaming"])
@pytest.mark.parametrize("mode", ["phased", "dosage"])
def test_polars_preserves_records_without_gt(tmp_path, streaming, mode, caplog):
    path = tmp_path / "missing_gt.vcf"
    path.write_text(
        "##fileformat=VCFv4.4\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\t1|1\n"
        "1\t20\t.\tA\tC\t.\tPASS\t.\tDP\t.\t23\n"
        "1\t30\t.\tA\tC\t.\tPASS\t.\tGT:DP\t1|0:8\t0|0:9\n"
    )
    reader = VCFReaderPolars(path)
    kwargs = dict(fields=["POS"], genotype_mode=mode)
    chunks = list(reader.iter_read(chunk_size=1, **kwargs)) if streaming else [reader.read(**kwargs)]
    expected = (
        [[[0, 1], [1, 1]], [[-1, -1], [-1, -1]], [[1, 0], [0, 0]]]
        if mode == "phased" else [[1, 2], [-1, -1], [1, 0]]
    )
    np.testing.assert_array_equal(np.concatenate([chunk.variants_pos for chunk in chunks]), [10, 20, 30])
    np.testing.assert_array_equal(np.concatenate([chunk.genotypes for chunk in chunks]), expected)
    assert "Falling back to default VCF reader" not in caplog.text


def test_polars_auto_retries_dosage_with_missing_gt_and_unphased_call(tmp_path, caplog):
    path = tmp_path / "mixed_format.vcf"
    path.write_text(
        "##fileformat=VCFv4.4\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        "1\t10\t.\tA\tC\t.\tPASS\t.\tDP\t8\n"
        "1\t20\t.\tG\tT\t.\tPASS\t.\tGT\t/0/1\n"
    )
    reader = VCFReaderPolars(path)
    expected = [[-1], [1]]
    np.testing.assert_array_equal(reader.read(genotype_mode="dosage").genotypes, expected)
    np.testing.assert_array_equal(reader.read().genotypes, expected)
    with pytest.raises(ValueError, match="unphased VCF genotypes"):
        reader.read(genotype_mode="phased")
    assert "Falling back to default VCF reader" not in caplog.text


@pytest.mark.parametrize("chrom_header", ["#CHROM", "CHROM"])
def test_polars_region_filters_before_decoding(tmp_path, chrom_header, caplog):
    path = tmp_path / "regions.vcf"
    _write_tiny_vcf(path, chrom_header)
    with path.open("a") as out:
        out.write("2\t200\trs3\tG\tA,C\t.\tPASS\t.\tGT\t0/2\t0/1\n")
    reader = VCFReaderPolars(path)
    kwargs = dict(region="1:200-200", fields=["ID"], samples=["HG00096"])

    eager = reader.read(**kwargs)
    streamed, = reader.iter_read(chunk_size=1, **kwargs)
    for result in (eager, streamed):
        np.testing.assert_array_equal(result.variants_id, ["rs2"])
        np.testing.assert_array_equal(result.genotypes, [[[1, 1]]])
        assert result.variants_chrom.size == result.variants_pos.size == 0
    np.testing.assert_array_equal(reader.read(genotype_mode="dosage", **kwargs).genotypes, [[2]])
    assert "Falling back to default VCF reader" not in caplog.text


def test_polars_region_with_no_matches(tmp_path, caplog):
    path = tmp_path / "regions.vcf"
    _write_tiny_vcf(path, "#CHROM")
    reader = VCFReaderPolars(path)

    assert reader.read(region="X").genotypes.shape == (0, 2, 2)
    assert reader.read(region="X", genotype_mode="dosage").genotypes.shape == (0, 2)
    assert list(reader.iter_read(region="X")) == []
    assert "Falling back to default VCF reader" not in caplog.text


def test_polars_region_intersects_original_variant_indexes(tmp_path):
    path = tmp_path / "regions.vcf"
    _write_tiny_vcf(path, "#CHROM")
    reader = VCFReaderPolars(path)

    chunk, = reader.iter_read(region="1:200-", variant_idxs=[1], chunk_size=1)
    np.testing.assert_array_equal(chunk.variants_id, ["rs2"])
    assert list(reader.iter_read(region="1:200-", variant_idxs=[0], chunk_size=1)) == []


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


@pytest.mark.parametrize("threads", [None, 2])
def test_vcf_polars_csv_parser_threads(tmp_path: Path, monkeypatch, threads, caplog) -> None:
    vcf_path = tmp_path / "tiny_threads.vcf"
    _write_tiny_vcf(vcf_path, "#CHROM")
    supports_n_threads = "n_threads" in inspect.signature(pl.read_csv).parameters
    read_csv = Mock(wraps=pl.read_csv)
    monkeypatch.setattr(pl, "read_csv", read_csv)
    snpobj = VCFReaderPolars(vcf_path).read(genotype_mode="phased", threads=threads)

    read_csv.assert_called_once()
    if supports_n_threads:
        assert read_csv.call_args.kwargs["n_threads"] == threads
    else:
        assert "n_threads" not in read_csv.call_args.kwargs
    np.testing.assert_array_equal(snpobj.variants_filter_pass, ["PASS", "PASS"])
    assert snpobj.genotypes.shape == (2, 2, 2)
    assert ("POLARS_MAX_THREADS" in caplog.text) == (threads is not None and not supports_n_threads)


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


def test_assume_biallelic_empty_projection_keeps_variant_count(tmp_path: Path) -> None:
    path = tmp_path / "empty_projection.vcf"
    _write_tiny_vcf(path, "#CHROM")
    reader = VCFReaderPolars(path)
    kwargs = dict(fields=[], samples=[], chromosome_ploidy="autosomal",
                  genotype_mode="dosage", assume_biallelic=True)
    assert reader.read(**kwargs).genotypes.shape == (2, 0)
    assert [chunk.genotypes.shape for chunk in reader.iter_read(**kwargs)] == [(2, 0)]


@pytest.mark.parametrize(("trusted", "fields", "reads_alt"), [
    (False, [], True), (True, [], False), (True, ["ALT"], True),
])
def test_polars_projects_alt_only_when_needed(tmp_path, monkeypatch, trusted, fields, reads_alt):
    path = tmp_path / "projection.vcf"
    _write_tiny_vcf(path, "#CHROM")
    read_csv = Mock(wraps=pl.read_csv)
    monkeypatch.setattr(pl, "read_csv", read_csv)
    result = VCFReaderPolars(path).read(
        fields=fields, genotype_mode="dosage", assume_biallelic=trusted,
    )
    options = read_csv.call_args.kwargs
    alt_index = list(options["schema_overrides"]).index("ALT")
    assert (alt_index in options["columns"]) == reads_alt
    np.testing.assert_array_equal(result.variants_alt, ["G", "T"] if fields else [])
