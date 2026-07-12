import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.write._genotype_encoding import (
    phased_to_flat_alleles,
    phased_to_hardcalls,
)
from snputils.snp.io.write.bgen import BGENWriter
from snputils.snp.io.write.vcf import VCFWriter


@pytest.mark.parametrize("value", [0.1, 0.9, 1.5])
def test_dosage_hardcalls_reject_fractional_values(value):
    with pytest.raises(ValueError, match="hard calls encoded as 0, 1, 2"):
        phased_to_hardcalls(
            np.array([[value]]),
            rename_missing_values=True,
            before=-1,
            after=".",
        )


def test_phased_alleles_reject_fractional_values_before_casting():
    with pytest.raises(ValueError, match="nonnegative integer allele indexes"):
        phased_to_flat_alleles(
            np.array([[[0.0, 0.9]]]),
            rename_missing_values=True,
            before=-1,
            after=".",
        )


def test_hardcall_conversion_preserves_nan_as_missing():
    observed = phased_to_hardcalls(
        np.array([[0.0, 1.0, 2.0, np.nan]]),
        rename_missing_values=True,
        before=-1,
        after=".",
    )

    np.testing.assert_array_equal(observed, np.array([[0, 1, 2, -9]], dtype=np.int8))


def test_flat_phased_alleles_keep_multiallelic_integer_indexes():
    observed = phased_to_flat_alleles(
        np.array([[[0, 2], [1, -1]]]),
        rename_missing_values=True,
        before=-1,
        after=".",
    )

    np.testing.assert_array_equal(observed, np.array([[0, 2, 1, -9]], dtype=np.int32))


def test_bgen_hardcalls_reject_fractional_values(tmp_path):
    snpobj = SNPObject(genotypes=np.array([[0.9]]))

    with pytest.raises(ValueError, match="hard calls encoded as 0, 1, 2"):
        BGENWriter(snpobj, tmp_path / "fractional.bgen")._probabilities(phased=False)


def test_vcf_hardcalls_reject_fractional_values(tmp_path):
    snpobj = SNPObject(
        genotypes=np.array([[[0.0, 0.9]]]),
        samples=np.array(["sample"]),
        variants_chrom=np.array(["1"]),
        variants_pos=np.array([10]),
        variants_id=np.array(["rs1"]),
        variants_ref=np.array(["A"]),
        variants_alt=np.array(["G"]),
    )

    with pytest.raises(ValueError, match="nonnegative integer allele indexes"):
        VCFWriter(snpobj, tmp_path / "fractional.vcf").write()
