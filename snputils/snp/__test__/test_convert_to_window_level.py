import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("window_size", [5, np.uint32(5)])
def test_convert_to_window_level_when_chrom_has_fewer_snps_than_window_size(window_size):
    snpobj = _multi_chrom_snpobj()

    lai = snpobj.convert_to_window_level(window_size=window_size)

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


@pytest.mark.parametrize(
    ("n_chrom", "positions", "lai_shape"),
    [
        (2, [10, 20], (3, 2)),
        (3, [10, 20, np.nan], (2, 1, 2)),
        (1, [10, 20], (2, 2)),
        (2, [10], (2, 1, 2)),
    ],
    ids=["extra-ancestry", "short-ancestry-with-missing-position", "short-chromosomes", "short-positions"],
)
def test_window_conversion_rejects_mismatched_variant_counts(n_chrom, positions, lai_shape):
    snps = SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=np.repeat("1", n_chrom),
        variants_pos=np.array(positions),
        calldata_lai=np.zeros(lai_shape, dtype=int),
    )
    reference = _multi_chrom_snpobj().convert_to_window_level(window_size=2)
    counts = f"variants_chrom={n_chrom}, variants_pos={len(positions)}, calldata_lai={lai_shape[0]}"
    for window in ({"window_size": 2}, {"physical_pos": np.array([[10, 20]])}, {"laiobj": reference}):
        with pytest.raises(ValueError, match=counts):
            snps.convert_to_window_level(**window)


@pytest.mark.parametrize(
    "chromosomes",
    [np.array([2, 2, np.nan, 1]), np.array(["2", "2", None, "1"], dtype=object)],
)
@pytest.mark.parametrize("skipped_position", [15, None, pd.NA], ids=["known", "none", "pandas-na"])
def test_fixed_windows_skip_snps_with_missing_chromosomes(chromosomes, skipped_position):
    snpobj = SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=chromosomes,
        variants_pos=np.array([10, 20, skipped_position, 30]),
        calldata_lai=np.array([[1, 0], [0, 1], [1, 1], [1, 0]]),
    )

    with pytest.warns(UserWarning, match="missing chromosome"):
        lai = snpobj.convert_to_window_level(window_size=2)

    np.testing.assert_array_equal(lai.chromosomes, chromosomes[[0, 3]])
    np.testing.assert_array_equal(lai.physical_pos, [[10, 20], [30, 30]])
    np.testing.assert_array_equal(lai.window_sizes, [2, 1])
    np.testing.assert_array_equal(lai.lai, [[0, 0], [1, 0]])


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["1", "2", "1", "1"]),
        np.array([2**63, 2**63 + 1, 2**63, 2**63], dtype=np.uint64),
    ],
    ids=["string", "uint64"],
)
def test_fixed_windows_exclude_masked_chromosome_labels(labels):
    snpobj = SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=np.ma.array(labels, mask=[True, False, True, False]),
        variants_pos=np.array([10, 20, 30, 40]),
        calldata_lai=np.array([[1, 1], [1, 0], [1, 1], [0, 0]]),
    )

    with pytest.warns(UserWarning, match="missing chromosome"):
        lai = snpobj.convert_to_window_level(window_size=3)

    assert lai.chromosomes.dtype == labels.dtype
    np.testing.assert_array_equal(lai.chromosomes, labels[[1, 3]])
    np.testing.assert_array_equal(lai.physical_pos, [[20, 20], [40, 40]])
    np.testing.assert_array_equal(lai.window_sizes, [1, 1])
    np.testing.assert_array_equal(lai.lai, [[1, 0], [0, 0]])


@pytest.mark.parametrize("chromosomes", [np.full(7, np.nan), np.ma.array(np.ones(7, dtype=int), mask=True)])
def test_fixed_windows_with_only_missing_chromosomes_return_no_windows(chromosomes):
    snpobj = _multi_chrom_snpobj()
    snpobj.variants_chrom = chromosomes

    with pytest.warns(UserWarning, match="missing chromosome"):
        lai = snpobj.convert_to_window_level(window_size=2)

    assert lai.physical_pos.shape == (0, 2)
    assert lai.chromosomes.shape == lai.window_sizes.shape == (0,)
    assert lai.lai.shape == (0, 2 * snpobj.n_samples)


@pytest.mark.parametrize(
    "positions",
    [
        np.array([10, np.nan, 30]),
        np.array([10, None, 30], dtype=object),
        np.array([10, pd.NA, 30], dtype=object),
        np.ma.array([10, 20, 30], mask=[False, True, False]),
    ],
    ids=["nan", "none", "pandas-na", "masked-in-window"],
)
def test_missing_position_is_excluded_through_window_round_trip(positions):
    snps = SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=np.repeat("1", 3),
        variants_pos=positions,
        calldata_lai=np.array([[0, 1], [1, 1], [1, 0]]),
    )

    with pytest.warns(UserWarning, match="missing positions"):
        fixed = snps.convert_to_window_level(window_size=3)
    np.testing.assert_array_equal(fixed.physical_pos, [[10, 30]])
    np.testing.assert_array_equal(fixed.window_sizes, [2])
    np.testing.assert_array_equal(fixed.lai, [[0, 0]])

    with pytest.warns(UserWarning, match="missing positions"):
        reused = snps.convert_to_window_level(laiobj=fixed)
    with pytest.warns(UserWarning, match="missing positions"):
        explicit = snps.convert_to_window_level(physical_pos=fixed.physical_pos)
    np.testing.assert_array_equal(reused.lai, fixed.lai)
    np.testing.assert_array_equal(explicit.lai, fixed.lai)

    with pytest.warns(UserWarning, match="missing positions"):
        restored = fixed.convert_to_snp_level(snpobject=snps)
    np.testing.assert_array_equal(restored.calldata_lai, [[[0, 0]], [[-1, -1]], [[0, 0]]])


def test_all_missing_positions_return_no_windows_and_warn_once():
    snps = SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=np.array(["1", "2", None], dtype=object),
        variants_pos=np.ma.array([None, 20, np.nan], mask=[False, True, False]),
        calldata_lai=np.zeros((3, 2), dtype=int),
    )

    with pytest.warns(UserWarning, match="missing positions") as warnings:
        fixed = snps.convert_to_window_level(window_size=2)
    assert len(warnings) == 1
    assert fixed.physical_pos.shape == (0, 2)
    assert fixed.chromosomes.size == fixed.window_sizes.size == 0

    generated = fixed.convert_to_snp_level()
    assert generated.variants_pos.shape == generated.variants_chrom.shape == (0,)
    assert generated.calldata_lai.shape == (0, 1, 2)

    with pytest.warns(UserWarning, match="missing positions") as warnings:
        restored = fixed.convert_to_snp_level(snpobject=snps, lai_format="2D")
    assert len(warnings) == 1
    np.testing.assert_array_equal(restored.calldata_lai, np.full((3, 2), -1))


@pytest.fixture
def snps_at_shared_position():
    return SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=np.repeat("1", 6),
        variants_pos=np.array([10, 20, 30, 30, 30, 40]),
        calldata_lai=np.array([[0, 0], [0, 0], [1, 0], [1, 1], [1, 1], [1, 0]]),
    )


@pytest.mark.parametrize("shape", [(6, 2), (6, 1, 2)])
def test_fixed_windows_keep_shared_positions_together_on_reuse(snps_at_shared_position, shape):
    snps = snps_at_shared_position
    snps.calldata_lai = snps.calldata_lai.reshape(shape)

    result = snps.convert_to_window_level(window_size=3)
    reused = snps.convert_to_window_level(laiobj=result)

    np.testing.assert_array_equal(reused.lai, result.lai)
    np.testing.assert_array_equal(result.physical_pos, [[10, 20], [30, 40]])
    np.testing.assert_array_equal(result.window_sizes, [2, 4])
    np.testing.assert_array_equal(result.lai, [[0, 0], [1, 0]])


def test_single_position_spanning_multiple_windows_forms_one_window(snps_at_shared_position):
    snps = snps_at_shared_position
    snps.variants_pos[:] = 30

    result = snps.convert_to_window_level(window_size=1)

    np.testing.assert_array_equal(result.physical_pos, [[30, 30]])
    np.testing.assert_array_equal(result.window_sizes, [6])
    np.testing.assert_array_equal(result.lai, [[1, 0]])
    np.testing.assert_array_equal(snps.convert_to_window_level(laiobj=result).lai, result.lai)


def test_fixed_windows_sort_positions_within_each_chromosome():
    snps = SNPObject(
        samples=np.array(["S1"]),
        variants_chrom=np.array(["2", "1", "2", "1", "2", "2", "2"]),
        variants_pos=np.array([40, 80, 20, 60, 30, 10, 50]),
        calldata_lai=np.array([[1, 0], [0, 1], [1, 1], [0, 1], [1, 0], [1, 1], [1, 0]]),
    )

    result = snps.convert_to_window_level(window_size=2)

    np.testing.assert_array_equal(result.chromosomes, ["2", "2", "1"])
    np.testing.assert_array_equal(result.physical_pos, [[10, 20], [30, 50], [60, 80]])
    np.testing.assert_array_equal(result.window_sizes, [2, 3, 2])
    np.testing.assert_array_equal(result.lai, [[1, 1], [1, 0], [0, 1]])


def test_coordinate_windows_include_all_records_at_shared_endpoints(snps_at_shared_position):
    snps = snps_at_shared_position
    bounds = np.array([[10, 30], [30, 40], [50, 60]])

    explicit = snps.convert_to_window_level(physical_pos=bounds)
    referenced = snps.convert_to_window_level(laiobj=explicit)

    expected = [[1, 0], [1, 0], [np.nan, np.nan]]
    np.testing.assert_array_equal(explicit.lai, expected)
    np.testing.assert_array_equal(referenced.lai, expected)


def test_coordinate_windows_with_masked_metadata_have_missing_ancestry(snps_at_shared_position):
    snps = snps_at_shared_position
    bounds = np.ma.array([[30, 40]] * 4, mask=[[1, 0], [0, 1], [0, 0], [0, 0]])
    chromosomes = np.ma.array(["1"] * 4, mask=[False, False, True, False])

    explicit = snps.convert_to_window_level(physical_pos=bounds, chromosomes=chromosomes)
    reused = snps.convert_to_window_level(laiobj=explicit)

    for result in (explicit, reused):
        assert np.isnan(result.lai[:3]).all()
        np.testing.assert_array_equal(result.lai[3], [1, 0])
