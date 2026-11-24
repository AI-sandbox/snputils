import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.vcf import VCFReader
from snputils.snp.io.write.vcf import VCFWriter


def test_vcf_writer_roundtrips_sampleless_annotation_only_vcf(tmp_path):
    path = tmp_path / "annotations_only.vcf"
    snpobj = SNPObject(
        genotypes=np.empty((2, 0, 2), dtype=np.int8),
        samples=np.array([], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([100, 200]),
        variants_qual=np.array([np.nan, 50.0], dtype=np.float32),
        variants_filter_pass=np.array([True, False]),
        variants_info=np.array(["ANN=missense_variant", "AF=0.125"], dtype=object),
    )

    VCFWriter(snpobj, path).write(rename_missing_values=False)

    lines = path.read_text().splitlines()
    assert lines[0] == "##fileformat=VCFv4.1"
    assert lines[1] == "##contig=<ID=1>"
    assert lines[2] == "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    assert lines[3] == "1\t100\trs1\tA\tG\t.\tPASS\tANN=missense_variant"
    assert lines[4] == "1\t200\trs2\tC\tT\t50.0\t.\tAF=0.125"

    observed = VCFReader(path).read(fields="*")

    assert observed.genotypes.shape == (2, 0, 2)
    np.testing.assert_array_equal(observed.samples, np.array([], dtype=object))
    np.testing.assert_array_equal(observed.variants_chrom.astype(str), snpobj.variants_chrom.astype(str))
    np.testing.assert_array_equal(observed.variants_pos, snpobj.variants_pos)
    np.testing.assert_array_equal(observed.variants_id.astype(str), snpobj.variants_id.astype(str))
    np.testing.assert_array_equal(observed.variants_ref.astype(str), snpobj.variants_ref.astype(str))
    np.testing.assert_array_equal(observed.variants_alt.astype(str), snpobj.variants_alt.astype(str))
    np.testing.assert_allclose(observed.variants_qual, np.array([np.nan, 50.0], dtype=np.float32), equal_nan=True)
    np.testing.assert_array_equal(observed.variants_filter_pass, np.array([True, False]))
    np.testing.assert_array_equal(observed.variants_info.astype(str), snpobj.variants_info.astype(str))


def test_vcf_writer_roundtrips_missing_genotypes(tmp_path):
    path = tmp_path / "missing.vcf"
    genotypes = np.array([[[0, 1], [-1, -1]], [[1, -1], [1, 1]]], dtype=np.int8)
    snpobj = SNPObject(
        genotypes=genotypes.copy(),
        samples=np.array(["S1", "S2"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([100, 200]),
    )

    VCFWriter(snpobj, path, phased=True).write()

    lines = path.read_text().splitlines()
    assert lines[-2] == "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t.|."
    assert lines[-1] == "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t1|.\t1|1"
    np.testing.assert_array_equal(snpobj.genotypes, genotypes)

    observed = VCFReader(path).read(genotype_mode="phased")

    np.testing.assert_array_equal(observed.genotypes, genotypes)


def _single_sample_snpobj(alleles):
    return SNPObject(
        genotypes=np.array([[alleles]]),
        samples=np.array(["S1"]),
        variants_ref=np.array(["A"]),
        variants_alt=np.array(["G"]),
        variants_chrom=np.array(["1"]),
        variants_id=np.array(["rs1"]),
        variants_pos=np.array([100]),
    )


def test_vcf_writer_writes_byte_string_alleles(tmp_path):
    path = tmp_path / "bytes.vcf"
    alleles = np.array([b"0", b"1"], dtype="S1")
    snpobj = _single_sample_snpobj(alleles)

    VCFWriter(snpobj, path).write()

    assert path.read_text().splitlines()[-1].split("\t")[-1] == "0/1"
    np.testing.assert_array_equal(snpobj.genotypes, alleles.reshape(1, 1, 2))


@pytest.mark.parametrize("after", ["oops", 0.5, -1, np.nan, np.inf])
def test_vcf_writer_rejects_invalid_missing_replacement(tmp_path, after):
    path = tmp_path / "invalid.vcf"
    snpobj = _single_sample_snpobj([0, -1])

    with pytest.raises(ValueError, match="after"):
        VCFWriter(snpobj, path).write(after=after)

    assert not path.exists()
    np.testing.assert_array_equal(snpobj.genotypes, [[[0, -1]]])


@pytest.mark.parametrize(
    ("alleles", "before", "after", "expected"),
    [
        (["1", "."], ".", "0", "1/0"),
        ([b"1", b"."], ".", "0", "1/0"),
        (["1", "NA"], "NA", ".", "1/."),
        ([1, -9], -9, 0.0, "1/0"),
        ([1.0, np.nan], np.nan, ".", "1/."),
    ],
)
def test_vcf_writer_encodes_custom_missing_marker(tmp_path, alleles, before, after, expected):
    path = tmp_path / "custom.vcf"
    snpobj = _single_sample_snpobj(alleles)
    original = snpobj.genotypes.copy()

    VCFWriter(snpobj, path).write(before=before, after=after)

    assert path.read_text().splitlines()[-1].split("\t")[-1] == expected
    np.testing.assert_array_equal(snpobj.genotypes, original)
