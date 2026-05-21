import gzip
from pathlib import Path

import numpy as np

from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars


def test_vcf_reader_matches_polars_for_core_fields(data_path):
    path = data_path + "/vcf/subset.vcf"
    fields = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER"]

    default = VCFReader(path).read()
    polars = VCFReaderPolars(path).read(fields=fields)

    np.testing.assert_array_equal(default.genotypes, polars.genotypes)
    np.testing.assert_array_equal(default.samples, polars.samples)
    np.testing.assert_array_equal(default.variants_pos, polars.variants_pos)
    np.testing.assert_array_equal(default.variants_id.astype(str), polars.variants_id.astype(str))
    np.testing.assert_array_equal(default.variants_ref.astype(str), polars.variants_ref.astype(str))
    np.testing.assert_array_equal(default.variants_alt.astype(str), polars.variants_alt.astype(str))
    np.testing.assert_array_equal(default.variants_chrom.astype(str), polars.variants_chrom.astype(str))
    np.testing.assert_array_equal(default.variants_filter_pass, polars.variants_filter_pass.astype(str) == "PASS")
    expected_qual = np.array(
        [np.nan if value == "." else float(value) for value in polars.variants_qual.astype(str)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(default.variants_qual, expected_qual, equal_nan=True)


def test_vcf_reader_reads_info_when_requested(data_path):
    path = data_path + "/vcf/subset.vcf"

    default = VCFReader(path).read(fields="*")
    polars = VCFReaderPolars(path).read(fields="*")

    np.testing.assert_array_equal(default.genotypes, polars.genotypes)
    np.testing.assert_array_equal(default.variants_info.astype(str), polars.variants_info.astype(str))


def test_vcf_reader_falls_back_for_format_subfields(tmp_path: Path):
    vcf_path = tmp_path / "format_subfields.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\tAF=0.5\tGT:DP\t0|1:7\t1|0:8\n"
        "1\t200\trs2\tC\tT\t.\tPASS\tAF=0.25\tGT:DP\t1|1:9\t0|0:6\n"
    )

    snpobj = VCFReader(vcf_path).read(fields="*")

    expected = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)
    np.testing.assert_array_equal(snpobj.variants_info.astype(str), np.array(["AF=0.5", "AF=0.25"]))


def test_vcf_reader_reads_gt_when_format_field_is_not_first(tmp_path: Path):
    vcf_path = tmp_path / "gt_not_first.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t50\tPASS\t.\tDP:GT\t7:0|1\t8:1|1\n"
        "1\t200\trs2\tC\tT\t60\tPASS\t.\tDP:GT\t9:1|0\t6:0|0\n"
    )

    snpobj = VCFReader(vcf_path).read()

    expected = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)


def test_vcf_reader_normalizes_qual_and_filter_pass_like_vcf_reader(tmp_path: Path):
    vcf_path = tmp_path / "qual_filter.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\n"
        "1\t100\trs1\tA\tG\t50\tPASS\t.\tGT\t0|1\n"
        "1\t200\trs2\tC\tT\t.\tLowQual\t.\tGT\t0|0\n"
    )

    snpobj = VCFReader(vcf_path).read()

    assert np.issubdtype(snpobj.variants_qual.dtype, np.floating)
    np.testing.assert_allclose(snpobj.variants_qual, np.array([50.0, np.nan]), equal_nan=True)
    assert snpobj.variants_filter_pass.dtype == np.bool_
    np.testing.assert_array_equal(snpobj.variants_filter_pass, np.array([True, False]))


def test_vcf_reader_falls_back_when_later_record_has_format_subfields(tmp_path: Path):
    vcf_path = tmp_path / "mixed_format_subfields.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\tAF=0.5\tGT\t0|1\t1|0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\tAF=0.25\tGT:DP\t1|1:9\t0|0:6\n"
    )

    snpobj = VCFReader(vcf_path).read(fields="*")

    expected = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)


def test_vcf_reader_reads_gt_only_vcf_gz(tmp_path: Path):
    vcf_path = tmp_path / "gt_only.vcf.gz"
    with gzip.open(vcf_path, "wt", encoding="utf-8") as file:
        file.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
            "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
            "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|0\n"
        )

    snpobj = VCFReader(vcf_path).read()

    expected = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)
    np.testing.assert_array_equal(snpobj.samples, np.array(["HG00096", "HG00097"]))


def test_vcf_reader_read_supports_region(tmp_path: Path):
    vcf_path = tmp_path / "regions.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|0\n"
        "2\t200\trs3\tG\tA\t.\tPASS\t.\tGT\t0|0\t0|1\n"
    )

    snpobj = VCFReader(vcf_path).read(region="1:150-250")

    np.testing.assert_array_equal(snpobj.variants_id.astype(str), np.array(["rs2"]))
    np.testing.assert_array_equal(snpobj.variants_pos, np.array([200]))
    np.testing.assert_array_equal(snpobj.genotypes, np.array([[[1, 1], [0, 0]]], dtype=np.int8))


def test_vcf_reader_region_does_not_require_returning_chrom_or_pos(tmp_path: Path):
    vcf_path = tmp_path / "region_fields.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|0\n"
    )

    snpobj = VCFReader(vcf_path).read(fields=["ID"], region="1:200-200")

    np.testing.assert_array_equal(snpobj.variants_id.astype(str), np.array(["rs2"]))
    assert snpobj.variants_pos.size == 0
    assert snpobj.variants_chrom.size == 0


def test_vcf_reader_uses_pandas_fallback_for_non_tab_separator(tmp_path: Path):
    vcf_path = tmp_path / "comma.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM,POS,ID,REF,ALT,QUAL,FILTER,INFO,FORMAT,HG00096,HG00097\n"
        "1,100,rs1,A,G,.,PASS,.,GT,0|1,1|0\n"
        "1,200,rs2,C,T,.,PASS,.,GT,1|1,0|0\n"
    )

    snpobj = VCFReader(vcf_path).read(separator=",", region="1:100-100")

    np.testing.assert_array_equal(snpobj.variants_id.astype(str), np.array(["rs1"]))
    np.testing.assert_array_equal(snpobj.genotypes, np.array([[[0, 1], [1, 0]]], dtype=np.int8))


def test_vcf_reader_iter_read_yields_sampleless_metadata_only_chunks(tmp_path: Path):
    vcf_path = tmp_path / "metadata_only_stream.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|0\n"
        "1\t300\trs3\tG\tA\t.\tPASS\t.\tGT\t0|0\t0|1\n"
    )

    chunks = list(VCFReader(vcf_path).iter_read(fields=["REF"], samples=[], chunk_size=2))

    assert [chunk.n_snps for chunk in chunks] == [2, 1]
    assert all(chunk.samples.size == 0 for chunk in chunks)
    np.testing.assert_array_equal(
        np.concatenate([chunk.variants_ref.astype(str) for chunk in chunks]),
        np.array(["A", "C", "G"]),
    )
    assert all(chunk.genotypes.size == 0 for chunk in chunks)
