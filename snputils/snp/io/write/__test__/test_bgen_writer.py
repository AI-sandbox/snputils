import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.bgen import BGENReader
from snputils.snp.io.write.bgen import BGENWriter


def test_bgen_writer_roundtrips_probabilities(tmp_path):
    path = tmp_path / "toy.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [np.nan, np.nan, np.nan]],
            [[0.0, 1.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.variants_id, snpobj.variants_id)
    np.testing.assert_array_equal(observed.variants_ref, snpobj.variants_ref)
    np.testing.assert_array_equal(observed.variants_alt, snpobj.variants_alt)
    np.testing.assert_array_equal(observed.variants_chrom, snpobj.variants_chrom)
    np.testing.assert_array_equal(observed.variants_pos, snpobj.variants_pos)
    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)


def test_bgen_writer_rejects_incompatible_phased_width(tmp_path):
    snpobj = SNPObject(
        calldata_gp=np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32),
        samples=np.array(["s1"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    with pytest.raises(ValueError, match="phased BGEN probabilities require 4 columns"):
        BGENWriter(snpobj, tmp_path / "bad.bgen").write(phased=True)


def test_bgen_writer_encodes_hardcalls_as_probabilities(tmp_path):
    path = tmp_path / "hardcalls.bgen"
    snpobj = SNPObject(
        genotypes=np.array([[0, 1, 2], [2, 0, -1]], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    snpobj.save_bgen(path)
    observed = BGENReader(path).read()

    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(observed.calldata_gp, expected, equal_nan=True)


def test_bgen_writer_roundtrips_mixed_probability_widths(tmp_path):
    path = tmp_path / "mixed.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0, np.nan], [0.0, 1.0, 0.0, np.nan]],
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)


def test_bgen_writer_roundtrips_variable_ploidy_unphased(tmp_path):
    path = tmp_path / "variable_ploidy_unphased.bgen"
    gp = np.array(
        [
            [
                [0.2, 0.8, np.nan],
                [0.2, 0.3, 0.5],
                [np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["haploid", "diploid", "missing"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)


def test_bgen_writer_roundtrips_variable_ploidy_phased(tmp_path):
    path = tmp_path / "variable_ploidy_phased.bgen"
    gp = np.array(
        [
            [
                [0.2, 0.8, np.nan, np.nan],
                [0.2, 0.8, 0.4, 0.6],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["haploid", "diploid", "missing"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)
