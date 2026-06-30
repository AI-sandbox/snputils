import numpy as np

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

    assert observed.genotypes.shape == (2, 0)
    np.testing.assert_array_equal(observed.samples, np.array([], dtype=object))
    np.testing.assert_array_equal(observed.variants_chrom.astype(str), snpobj.variants_chrom.astype(str))
    np.testing.assert_array_equal(observed.variants_pos, snpobj.variants_pos)
    np.testing.assert_array_equal(observed.variants_id.astype(str), snpobj.variants_id.astype(str))
    np.testing.assert_array_equal(observed.variants_ref.astype(str), snpobj.variants_ref.astype(str))
    np.testing.assert_array_equal(observed.variants_alt.astype(str), snpobj.variants_alt.astype(str))
    np.testing.assert_allclose(observed.variants_qual, np.array([np.nan, 50.0], dtype=np.float32), equal_nan=True)
    np.testing.assert_array_equal(observed.variants_filter_pass, np.array([True, False]))
    np.testing.assert_array_equal(observed.variants_info.astype(str), snpobj.variants_info.astype(str))
