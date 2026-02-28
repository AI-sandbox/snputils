from pathlib import Path

import numpy as np
import pytest

from snputils.snp.io.read.vcf import VCFReaderPolars


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

    snpobj = VCFReaderPolars(str(vcf_path)).read(samples=["HG00096"])

    expected = np.array([[[0, 1]], [[1, 1]]], dtype=np.int8)
    assert np.array_equal(snpobj.samples, np.array(["HG00096"]))
    assert np.array_equal(snpobj.calldata_gt, expected)
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
