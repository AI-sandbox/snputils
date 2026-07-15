import numpy as np
import pytest

from .read_bgen import (
    _probabilities_to_dosage,
    _probabilities_to_phased_calls,
    read_bgen_bgen,
    read_bgen_snputils,
)


def test_probabilities_to_dosage_handles_unphased_probabilities():
    probabilities = np.array(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]],
            [[np.nan, np.nan, np.nan], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    observed = _probabilities_to_dosage(probabilities)

    expected = np.array([[0.3, 1.6], [np.nan, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(observed, expected, equal_nan=True)


def test_probabilities_to_dosage_handles_phased_probabilities():
    probabilities = np.array(
        [
            [[0.9, 0.1, 0.2, 0.8], [0.4, 0.6, 0.7, 0.3]],
            [[np.nan, np.nan, np.nan, np.nan], [1.0, 0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    observed = _probabilities_to_dosage(probabilities)

    expected = np.array([[0.9, 0.9], [np.nan, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(observed, expected, equal_nan=True)


def test_probabilities_to_phased_calls_materializes_allele_codes():
    probabilities = np.array(
        [
            [
                [0.9, 0.1, 0.2, 0.8],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.2, 0.8, 0.7, 0.3],
                [0.4, 0.6, 0.9, 0.1],
            ],
        ],
        dtype=np.float32,
    )

    observed = _probabilities_to_phased_calls(probabilities)

    expected = np.array(
        [
            [[0, 1], [-1, -1]],
            [[1, 0], [1, 0]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(observed, expected)
    assert observed.shape == (2, 2, 2)
    assert observed.dtype == np.int8


def test_probabilities_to_phased_calls_rejects_unphased_probabilities():
    probabilities = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)

    with pytest.raises(ValueError, match="two equally sized haplotype"):
        _probabilities_to_phased_calls(probabilities)


def test_probabilities_to_phased_calls_rejects_partial_ploidy():
    probabilities = np.array([[0.1, 0.9, np.nan, np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="requires diploid"):
        _probabilities_to_phased_calls(probabilities)


def test_bgen_package_reference_matches_snputils(tmp_path):
    pytest.importorskip("bgen")
    from snputils import BGENWriter, SNPObject

    genotypes = np.array(
        [
            [[0, 0], [0, 1], [1, 0]],
            [[1, 1], [1, 0], [0, 1]],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(
        genotypes=genotypes,
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([100, 200]),
    )
    path = tmp_path / "phased.bgen"
    BGENWriter(snpobj, path).write(phased=True, bit_depth=16)

    np.testing.assert_allclose(
        read_bgen_snputils(path, genotype_mode="dosage"),
        read_bgen_bgen(path, genotype_mode="dosage"),
        atol=1 / 65535 + 1e-6,
    )
    np.testing.assert_array_equal(
        read_bgen_snputils(path, genotype_mode="phased"),
        read_bgen_bgen(path, genotype_mode="phased"),
    )
