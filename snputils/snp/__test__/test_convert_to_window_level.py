import numpy as np
import pytest

from snputils.snp.genobj import SNPObject


def _multi_chrom_snpobj() -> SNPObject:
    return SNPObject(
        genotypes=np.zeros((7, 2, 2), dtype=np.int8),
        samples=np.array(["S1", "S2"]),
        variants_chrom=np.array(["1", "1", "1", "1", "2", "2", "2"]),
        variants_pos=np.array([1000, 2000, 3000, 4000, 100, 200, 300]),
        calldata_lai=np.array(
            [
                [[0, 0], [1, 1]],
                [[0, 0], [1, 1]],
                [[0, 1], [1, 0]],
                [[0, 1], [1, 0]],
                [[1, 1], [0, 0]],
                [[1, 1], [0, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=np.int8,
        ),
    )


def test_convert_to_window_level_uses_per_chromosome_positions():
    snpobj = _multi_chrom_snpobj()

    lai = snpobj.convert_to_window_level(window_size=2)

    assert lai.chromosomes.tolist() == ["1", "1", "2"]
    assert lai.physical_pos.tolist() == [
        [1000, 2000],
        [3000, 4000],
        [100, 300],
    ]
    assert lai.window_sizes.tolist() == [2, 2, 3]
    np.testing.assert_array_equal(
        lai.lai,
        np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=float,
        ),
    )


def test_convert_to_window_level_when_chrom_has_fewer_snps_than_window_size():
    snpobj = _multi_chrom_snpobj()

    lai = snpobj.convert_to_window_level(window_size=5)

    assert lai.chromosomes.tolist() == ["1", "2"]
    assert lai.physical_pos.tolist() == [
        [1000, 4000],
        [100, 300],
    ]
    assert lai.window_sizes.tolist() == [4, 3]
    assert np.all(np.isfinite(lai.lai))
    np.testing.assert_array_equal(lai.lai[1], np.array([1, 1, 0, 0], dtype=float))


def test_convert_to_window_level_multi_window_second_chromosome():
    snpobj = SNPObject(
        genotypes=np.zeros((8, 1, 2), dtype=np.int8),
        samples=np.array(["S1"]),
        variants_chrom=np.array(["1", "1", "1", "2", "2", "2", "2", "2"]),
        variants_pos=np.array([500, 600, 700, 10, 20, 30, 40, 50]),
        calldata_lai=np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=np.int8,
        ),
    )

    lai = snpobj.convert_to_window_level(window_size=2)

    assert lai.chromosomes.tolist() == ["1", "2", "2"]
    assert lai.physical_pos.tolist() == [[500, 700], [10, 20], [30, 50]]
    assert lai.window_sizes.tolist() == [3, 2, 3]
    np.testing.assert_array_equal(
        lai.lai,
        np.array([[0, 0], [1, 0], [1, 1]], dtype=float),
    )


def test_convert_to_window_level_rejects_nonpositive_window_size():
    snpobj = _multi_chrom_snpobj()

    with pytest.raises(ValueError, match="window_size"):
        snpobj.convert_to_window_level(window_size=0)
