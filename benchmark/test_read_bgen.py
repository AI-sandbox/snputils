import numpy as np
import pytest

from .read_bgen import _probabilities_to_phased_calls


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
