import gzip
from pathlib import Path

import numpy as np
import pytest

from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars


@pytest.mark.parametrize("reader_cls", [VCFReader, VCFReaderPolars])
@pytest.mark.parametrize("compressed", [False, True])
def test_multidigit_gt_eager_and_streaming(tmp_path, reader_cls, compressed, caplog):
    path = tmp_path / ("wide.vcf.gz" if compressed else "wide.vcf")
    alts = ",".join(f"<ALT{i}>" for i in range(1, 131))
    with (gzip.open(path, "wt") if compressed else path.open("w")) as out:
        out.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
            "1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\t1|0\n"
            f"1\t2\t.\tA\t{alts}\t.\tPASS\t.\tGT\t0|10\t0|20\n"
            f"1\t3\t.\tA\t{alts}\t.\tPASS\t.\tGT:DP\t20|.:4\t.|10:25\n"
            f"1\t4\t.\tA\t{alts}\t.\tPASS\t.\tDP:GT\t21:130|10\t2:20\n"
        )

    expected = np.array([[[0, 1], [1, 0]], [[0, 10], [0, 20]],
                         [[20, -1], [-1, 10]], [[130, 10], [20, -1]]])
    reader = reader_cls(path)
    eager = reader.read(fields=[], genotype_mode="phased")
    np.testing.assert_array_equal(eager.genotypes, expected)
    assert eager.genotypes.dtype == np.int16
    assert eager.variants_alt.size == 0
    chunks = list(reader.iter_read(fields=[], chunk_size=2, genotype_mode="phased"))
    np.testing.assert_array_equal(np.concatenate([chunk.genotypes for chunk in chunks]), expected)
    assert [chunk.genotypes.dtype for chunk in chunks] == [np.int8, np.int16]
    assert "Falling back to default VCF reader" not in caplog.text


def test_compressed_gt_and_alt_widen_across_blocks(tmp_path, monkeypatch):
    from snputils.snp.io.read import vcf

    path = tmp_path / "blocks.vcf.gz"
    alts = ",".join(f"<ALT{i}>" for i in range(1, 129))
    with gzip.open(path, "wt") as out:
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
            "1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\n"
            f"1\t2\t.\tA\t{alts}\t.\tPASS\t.\tGT\t128|0\n"
            "1\t3\t.\tG\tA\t.\tPASS\t.\tGT\t1|0\n"
        )

    # Yield one record per block and force the output to grow after promotion.
    read = gzip.GzipFile.read

    def read_one_record(self, size=-1):
        return self.readline() if size > 0 else read(self, size)

    monkeypatch.setattr(gzip.GzipFile, "read", read_one_record)
    monkeypatch.setattr(vcf, "_initial_stream_capacity", lambda _: 1)
    result = VCFReader(path).read(fields=["ALT"])
    np.testing.assert_array_equal(result.genotypes, [[[0, 1]], [[128, 0]], [[1, 0]]])
    assert result.genotypes.dtype == np.int16
    np.testing.assert_array_equal(result.variants_alt, ["C", alts, "A"])


@pytest.mark.parametrize("trusted", [False, True])
def test_compressed_empty_projection_counts_selected_variants(tmp_path, trusted):
    path = tmp_path / "region.vcf.gz"
    with gzip.open(path, "wt") as out:
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
            "1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\n"
            "1\t2\t.\tG\tA\t.\tPASS\t.\tGT\t1|0\n"
        )
    reader = VCFReader(path)
    kwargs = dict(fields=[], samples=[], region="1:2-2", genotype_mode="dosage", assume_biallelic=trusted)
    assert reader.read(**kwargs).genotypes.shape == (1, 0)
    assert [chunk.genotypes.shape for chunk in reader.iter_read(**kwargs)] == [(1, 0)]


@pytest.mark.parametrize("fmt,call", [("GT:DP", "0|20:9"), ("DP:GT", "9:0|20")])
def test_gt_subfield_preserves_final_multidigit_allele(tmp_path, fmt, call):
    path = tmp_path / "subfield.vcf"
    alts = ",".join(f"<ALT{i}>" for i in range(1, 21))
    path.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        f"1\t1\t.\tA\t{alts}\t.\tPASS\t.\t{fmt}\t{call}\n"
    )
    result = VCFReader(path).read(fields=["ALT"])
    np.testing.assert_array_equal(result.genotypes, [[[0, 20]]])
    np.testing.assert_array_equal(result.variants_alt, [alts])


@pytest.mark.parametrize("reader_cls", [VCFReader, VCFReaderPolars])
def test_gt_widths_can_cancel_within_a_row(tmp_path, reader_cls):
    path = tmp_path / "balanced.vcf"
    alts = ",".join(f"<ALT{i}>" for i in range(1, 101))
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        f"1\t1\t.\tA\t{alts}\t.\tPASS\t.\tGT\t1\t100|0\n"
    )
    reader = reader_cls(path)
    for result in (reader.read(fields=[], samples=["s2"]),
                   next(reader.iter_read(fields=[], samples=["s2"]))):
        np.testing.assert_array_equal(result.genotypes, [[[100, 0]]])


@pytest.mark.parametrize("reader_cls", [VCFReader, VCFReaderPolars])
def test_multidigit_unphased_gt_obeys_genotype_mode(tmp_path, reader_cls):
    path = tmp_path / "unphased.vcf"
    alts = ",".join(f"<ALT{i}>" for i in range(1, 11))
    path.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        f"1\t1\t.\tA\t{alts}\t.\tPASS\t.\tGT\t10/0\n"
    )
    reader = reader_cls(path)
    with pytest.raises(ValueError, match="unphased"):
        reader.read(fields=[], genotype_mode="phased")
    with pytest.raises(ValueError, match="biallelic"):
        reader.read(fields=[], genotype_mode="auto")


@pytest.mark.parametrize("reader_cls", [VCFReader, VCFReaderPolars])
def test_assume_biallelic_preserves_format_phase_and_ploidy(tmp_path, reader_cls):
    path = tmp_path / "biallelic.vcf"
    path.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t1\t.\tA\tC\t.\tPASS\t.\tGT:PS\t0|1:1\t.|1:1\n"
        "X\t2\t.\tG\tA\t.\tPASS\t.\tDP:GT\t15:1\t22:0/1\n"
    )
    reader = reader_cls(path)
    for trusted in (False, True):
        kwargs = dict(fields=[], assume_biallelic=trusted)
        expected = [[1, -1], [2, 1]]
        result = reader.read(genotype_mode="auto", **kwargs)
        np.testing.assert_array_equal(result.genotypes, expected)
        chunks = reader.iter_read(chunk_size=1, genotype_mode="dosage", **kwargs)
        np.testing.assert_array_equal(np.concatenate([c.genotypes for c in chunks]), expected)
        with pytest.raises(ValueError, match="unphased"):
            reader.read(genotype_mode="phased", **kwargs)
    with pytest.raises(TypeError, match="boolean"):
        reader.read(assume_biallelic="False")


@pytest.mark.parametrize("trusted", [False, True])
def test_assume_biallelic_native_dosage(tmp_path, trusted):
    path = tmp_path / "native.vcf.gz"
    with gzip.open(path, "wt") as out:
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
            "1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\n"
        )
    kwargs = dict(fields=[], genotype_mode="dosage", chromosome_ploidy="autosomal",
                  threads=2, assume_biallelic=trusted)
    reader = VCFReader(path)
    np.testing.assert_array_equal(reader.read(**kwargs).genotypes, [[1]])
    np.testing.assert_array_equal(next(reader.iter_read(**kwargs)).genotypes, [[1]])


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

    snpobj = VCFReader(vcf_path).read(fields="*", genotype_mode="phased")

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

    snpobj = VCFReader(vcf_path).read(genotype_mode="phased")

    expected = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)


def test_vcf_reader_sums_autosomal_partial_missing_as_missing(tmp_path: Path):
    vcf_path = tmp_path / "autosomal_partial_missing.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|.\t1|1\t.|.\n"
    )

    snpobj = VCFReader(vcf_path).read(genotype_mode="dosage")

    np.testing.assert_array_equal(snpobj.genotypes, np.array([[-1, -1, 2, -1]], dtype=np.int8))


def test_vcf_reader_sums_non_diploid_partial_missing_as_haploid_dosage(tmp_path: Path):
    vcf_path = tmp_path / "non_diploid_partial_missing.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\n"
        "X\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|.\t.|.\t1|1\n"
        "chrX\t150\trs1b\tA\tG\t.\tPASS\t.\tGT\t.|0\t.|1\t.|.\t0|0\n"
        "chrY\t200\trs2\tA\tG\t.\tPASS\t.\tGT\t.|0\t.|1\t.|.\t0|1\n"
        "MT\t250\trs2b\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|.\t.|.\t1|0\n"
        "chromMT\t300\trs3\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|.\t.|.\t0|0\n"
        "mitochondrial\t400\trs4\tA\tG\t.\tPASS\t.\tGT\t.|0\t.|1\t.|.\t1|0\n"
    )

    snpobj = VCFReader(vcf_path).read(genotype_mode="dosage")

    expected = np.array(
        [
            [0, 2, -1, 2],
            [0, 2, -1, 0],
            [0, 2, -1, 1],
            [0, 2, -1, 1],
            [0, 2, -1, 0],
            [0, 2, -1, 1],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)


def test_vcf_reader_polars_uses_internal_chrom_for_non_diploid_summing(tmp_path: Path):
    vcf_path = tmp_path / "polars_internal_chrom.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "chrX\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|.\n"
        "1\t200\trs2\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|.\n"
    )

    snpobj = VCFReaderPolars(vcf_path).read(fields=["ID"], genotype_mode="dosage")

    np.testing.assert_array_equal(snpobj.genotypes, np.array([[0, 2], [-1, -1]], dtype=np.int8))
    np.testing.assert_array_equal(snpobj.variants_id.astype(str), np.array(["rs1", "rs2"]))
    assert snpobj.variants_chrom.size == 0


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

    snpobj = VCFReader(vcf_path).read(fields="*", genotype_mode="phased")

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

    snpobj = VCFReader(vcf_path).read(genotype_mode="phased")

    expected = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snpobj.genotypes, expected)
    np.testing.assert_array_equal(snpobj.samples, np.array(["HG00096", "HG00097"]))

    serial_dosage = VCFReader(vcf_path).read(
        fields=[],
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
    )
    parallel_dosage = VCFReader(vcf_path).read(
        fields=["POS"],
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
        threads=2,
    )
    np.testing.assert_array_equal(parallel_dosage.genotypes, serial_dosage.genotypes)
    np.testing.assert_array_equal(parallel_dosage.variants_pos, np.array([100, 200], dtype=np.int32))
    np.testing.assert_array_equal(
        parallel_dosage.genotypes,
        np.array([[1, 1], [2, 0]], dtype=np.int8),
    )

    parallel_phased = VCFReader(vcf_path).read(
        fields=[],
        genotype_mode="phased",
        threads=2,
    )
    np.testing.assert_array_equal(parallel_phased.genotypes, snpobj.genotypes)

    unphased_path = tmp_path / "gt_only_unphased.vcf.gz"
    with gzip.open(unphased_path, "wt") as file:
        file.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\n"
            "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        )
    with np.testing.assert_raises_regex(ValueError, "unphased VCF genotypes"):
        VCFReader(unphased_path).read(fields=[], genotype_mode="phased", threads=2)

    with np.testing.assert_raises_regex(ValueError, "at least 1"):
        VCFReader(vcf_path).read(fields=[], genotype_mode="dosage", threads=0)
    with np.testing.assert_raises_regex(ValueError, "at most POS metadata"):
        VCFReader(vcf_path).read(genotype_mode="dosage", threads=2)


def test_vcf_reader_reads_sampleless_vcf_gz_with_threads(tmp_path: Path):
    vcf_path = tmp_path / "annotations_only.vcf.gz"
    with gzip.open(vcf_path, "wt", encoding="utf-8") as file:
        file.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t100\trs1\tA\tG\t.\tPASS\tANN=missense_variant\n"
            "1\t200\trs2\tC\tT\t.\tPASS\tANN=synonymous_variant\n"
        )

    phased = VCFReader(vcf_path).read(
        fields=["POS"],
        genotype_mode="phased",
        threads=2,
    )
    dosage = VCFReader(vcf_path).read(
        fields=[],
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
        threads=2,
    )

    assert phased.genotypes.shape == (2, 0, 2)
    assert dosage.genotypes.shape == (2, 0)
    np.testing.assert_array_equal(phased.variants_pos, np.array([100, 200], dtype=np.int32))
    np.testing.assert_array_equal(phased.samples, np.array([]))


def test_vcf_parallel_rejects_mixed_width_gt_before_sample_subsetting(tmp_path: Path):
    vcf_path = tmp_path / "mixed_width.vcf.gz"
    with gzip.open(vcf_path, "wt", encoding="utf-8") as file:
        file.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
            "1\t100\trs1\tA\tC,G,T,AA,AC,AG,AT,CA,CG,CT\t.\tPASS\t.\tGT\t10\t10|1\n"
        )

    with np.testing.assert_raises_regex(ValueError, "fixed-width diploid GT-only"):
        VCFReader(vcf_path).read(
            fields=[],
            samples=["S2"],
            genotype_mode="phased",
            threads=2,
        )


def test_vcf_parallel_rejects_haploid_gt_with_clear_error(tmp_path: Path):
    vcf_path = tmp_path / "haploid.vcf.gz"
    with gzip.open(vcf_path, "wt", encoding="utf-8") as file:
        file.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
            "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t1\n"
        )

    serial = VCFReader(vcf_path).read(fields=[], genotype_mode="phased")
    np.testing.assert_array_equal(serial.genotypes, np.array([[[1, -1]]], dtype=np.int8))

    with np.testing.assert_raises_regex(ValueError, "fixed-width diploid GT-only"):
        VCFReader(vcf_path).read(fields=[], genotype_mode="phased", threads=2)


def test_read_snp_dispatches_zstandard_vcf(tmp_path: Path):
    import zstandard as zstd

    from snputils import read_snp

    vcf_path = tmp_path / "gt_only.vcf.zst"
    with zstd.open(vcf_path, "wt", encoding="utf-8") as file:
        file.write(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
            "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
        )

    snpobj = read_snp(vcf_path, genotype_mode="phased")

    np.testing.assert_array_equal(snpobj.genotypes, np.array([[[0, 1]]], dtype=np.int8))
    np.testing.assert_array_equal(snpobj.samples, np.array(["S1"]))


def test_vcf_reader_read_supports_region(tmp_path: Path):
    vcf_path = tmp_path / "regions.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG00096\tHG00097\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|0\n"
        "2\t200\trs3\tG\tA\t.\tPASS\t.\tGT\t0|0\t0|1\n"
    )

    snpobj = VCFReader(vcf_path).read(region="1:150-250", genotype_mode="phased")

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

    snpobj = VCFReader(vcf_path).read(separator=",", region="1:100-100", genotype_mode="phased")

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


def test_vcf_reader_reads_variant_annotations_without_samples_or_genotypes_by_default(tmp_path: Path):
    vcf_path = tmp_path / "annotations_only.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##INFO=<ID=ANN,Number=.,Type=String,Description=\"Annotation\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t100\trs1\tA\tG\t.\tPASS\tANN=missense_variant\n"
        "1\t200\trs2\tC\tT\t.\tPASS\tANN=synonymous_variant\n"
    )

    snpobj = VCFReader(vcf_path).read()

    np.testing.assert_array_equal(snpobj.samples, np.array([]))
    assert snpobj.genotypes.shape == (2, 0, 2)
    assert snpobj.genotypes.size == 0
    assert snpobj.n_samples == 0
    assert snpobj.n_snps == 2
    np.testing.assert_array_equal(snpobj.variants_chrom.astype(str), np.array(["1", "1"]))
    np.testing.assert_array_equal(snpobj.variants_pos, np.array([100, 200]))
    np.testing.assert_array_equal(snpobj.variants_id.astype(str), np.array(["rs1", "rs2"]))
    np.testing.assert_array_equal(
        snpobj.variants_info.astype(str),
        np.array(["ANN=missense_variant", "ANN=synonymous_variant"]),
    )
