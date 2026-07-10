import numpy as np

from snputils._utils.genotypes import sum_diploid_alleles, sum_diploid_genotypes


def test_sum_diploid_alleles_missing_as_haploid_none_preserves_missing_behavior():
    first = np.array([0, 0, 1, 1, 1, -1], dtype=np.int8)
    second = np.array([0, -1, 0, -1, 1, -1], dtype=np.int8)

    observed = sum_diploid_alleles(first, second, missing_as_haploid=None)

    np.testing.assert_array_equal(observed, np.array([0, -1, 1, -1, 2, -1], dtype=np.int8))


def test_sum_diploid_alleles_missing_as_haploid_true_treats_one_missing_as_haploid():
    first = np.array([0, -1, 1, -1, -1], dtype=np.int8)
    second = np.array([-1, 0, -1, 1, -1], dtype=np.int8)

    observed = sum_diploid_alleles(first, second, missing_as_haploid=True)

    np.testing.assert_array_equal(observed, np.array([0, 0, 2, 2, -1], dtype=np.int8))


def test_sum_diploid_genotypes_missing_as_haploid_row_mask():
    genotypes = np.array(
        [
            [[0, -1], [1, -1], [-1, -1]],
            [[0, -1], [1, -1], [-1, -1]],
        ],
        dtype=np.int8,
    )

    observed = sum_diploid_genotypes(genotypes, missing_as_haploid=np.array([True, False]))

    np.testing.assert_array_equal(
        observed,
        np.array(
            [
                [0, 2, -1],
                [-1, -1, -1],
            ],
            dtype=np.int8,
        ),
    )
