import numpy as np
import pytest

from snputils import BGENReader, read_bgen, read_snp
from snputils._utils.genotypes import sum_diploid_genotypes
from snputils.snp.genobj.snpobj import SNPObject


def test_bgen_reader_preserves_probabilities(snpobj_bgen, snpobj_vcf):
    assert snpobj_bgen.genotypes is None
    assert snpobj_bgen.calldata_gp is not None
    assert snpobj_bgen.calldata_gp.shape[:2] == (100, snpobj_vcf.n_samples)
    assert snpobj_bgen.calldata_gp.shape[2] in (3, 4)

    expected_dosage = sum_diploid_genotypes(snpobj_vcf.genotypes[:100], dtype=np.int16)
    np.testing.assert_allclose(snpobj_bgen.dosage(), expected_dosage, atol=1 / 255)


def test_bgen_reader_matches_vcf_metadata(snpobj_bgen, snpobj_vcf):
    np.testing.assert_array_equal(snpobj_bgen.samples, snpobj_vcf.samples)
    np.testing.assert_array_equal(snpobj_bgen.variants_ref, snpobj_vcf.variants_ref[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_alt, snpobj_vcf.variants_alt[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_chrom, snpobj_vcf.variants_chrom[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_id, snpobj_vcf.variants_id[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_pos, snpobj_vcf.variants_pos[:100])


def test_bgen_auto_reader_and_function(data_path):
    path = data_path + "/bgen/subset.bgen"
    by_auto = read_snp(path, variant_idxs=[0, 1, 2])
    by_function = read_bgen(path, variant_idxs=[0, 1, 2])

    np.testing.assert_allclose(by_auto.calldata_gp, by_function.calldata_gp, equal_nan=True)
    np.testing.assert_array_equal(by_auto.samples, by_function.samples)
    np.testing.assert_array_equal(by_auto.variants_id, by_function.variants_id)


def test_bgen_reader_sample_and_variant_selection(data_path):
    snpobj = BGENReader(data_path + "/bgen/subset.bgen").read(
        sample_ids=["HG00100", "HG00096"],
        variant_idxs=[0, 2],
    )

    np.testing.assert_array_equal(snpobj.samples, np.array(["HG00100", "HG00096"], dtype=object))
    assert snpobj.calldata_gp.shape[:2] == (2, 2)
    assert snpobj.calldata_gp.shape[2] in (3, 4)


def test_bgen_reader_does_not_hard_call(data_path):
    with pytest.raises(NotImplementedError, match="does not hard-call"):
        BGENReader(data_path + "/bgen/subset.bgen").read(fields=["GT"])


def test_bgen_to_dosage_populates_genotypes(snpobj_bgen):
    dosage_obj = snpobj_bgen.to_dosage()

    assert snpobj_bgen.genotypes is None
    assert dosage_obj.genotypes is not None
    np.testing.assert_allclose(dosage_obj.genotypes, snpobj_bgen.dosage(), equal_nan=True)
    np.testing.assert_allclose(dosage_obj.calldata_gp, snpobj_bgen.calldata_gp, equal_nan=True)


def test_bgen_dosage_handles_variable_ploidy_rows():
    gp = np.array(
        [
            [
                [0.2, 0.8, np.nan, np.nan],
                [0.1, 0.2, 0.3, 0.4],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.2, 0.8, np.nan, np.nan],
                [0.1, 0.9, 0.3, 0.7],
                [np.nan, np.nan, np.nan, np.nan],
            ],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        variants_ref=np.array(["A", "A"], dtype=object),
        variants_alt=np.array(["G", "G"], dtype=object),
    )

    expected = np.array(
        [
            [0.8, 2.0, np.nan],
            [0.8, 1.6, np.nan],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(snpobj.dosage(), expected, equal_nan=True)


def test_bgen_dosage_rejects_multiallelic_probabilities():
    snpobj = SNPObject(
        calldata_gp=np.zeros((1, 1, 6), dtype=np.float32),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["C,T"], dtype=object),
    )

    with pytest.raises(ValueError, match="biallelic"):
        snpobj.dosage()
