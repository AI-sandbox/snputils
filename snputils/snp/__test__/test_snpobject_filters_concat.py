import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject


def _toy_snpobj() -> SNPObject:
    return SNPObject(
        calldata_gt=np.array(
            [
                [[0, 0], [0, 1], [1, 1]],
                [[0, 0], [-1, -1], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 1], [0, 0]],
                [[0, 0], [0, 1], [0, 0]],
            ],
            dtype=np.int8,
        ),
        samples=np.array(["s1", "s2", "s3"]),
        sample_fid=np.array(["A", "A", "B"]),
        variants_ref=np.array(["A", "C", "G", "AT", "T"], dtype=object),
        variants_alt=np.array(["G", "T", "A", "A", "C,G"], dtype=object),
        variants_chrom=np.array(["1", "1", "1", "1", "1"], dtype=object),
        variants_id=np.array(["v1", "v2", "v3", "v4", "v5"], dtype=object),
        variants_pos=np.array([10, 20, 30, 40, 50]),
    )


def test_variant_filters_cover_biallelic_complete_and_polymorphic():
    snpobj = _toy_snpobj()

    filtered = (
        snpobj
        .filter_biallelic_variants(snv_only=True)
        .filter_complete_genotypes()
        .filter_polymorphic_variants()
    )

    assert filtered.variants_id.tolist() == ["v1"]
    assert filtered.calldata_gt.shape == (1, 3, 2)


def test_filter_variants_accepts_boolean_mask():
    snpobj = _toy_snpobj()

    filtered = snpobj.filter_variants(mask=[True, False, True, False, False])

    assert filtered.variants_id.tolist() == ["v1", "v3"]


def test_filter_samples_by_index_works_without_sample_ids():
    calldata_gt = np.array(
        [
            [[0, 0], [0, 1], [1, 1]],
            [[1, 0], [1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(calldata_gt=calldata_gt)

    filtered = snpobj.filter_samples(indexes=[0, 2])

    assert filtered.samples is None
    np.testing.assert_array_equal(filtered.calldata_gt, calldata_gt[:, [0, 2], :])


def test_concat_variants_preserves_sample_metadata_and_validates_order():
    left = _toy_snpobj().filter_variants(indexes=[0, 1])
    right = _toy_snpobj().filter_variants(indexes=[2, 3])

    concatenated = SNPObject.concat_variants([left, right])

    assert concatenated.variants_id.tolist() == ["v1", "v2", "v3", "v4"]
    assert concatenated.samples.tolist() == ["s1", "s2", "s3"]
    assert concatenated.sample_fid.tolist() == ["A", "A", "B"]

    wrong_order = right.filter_samples(samples=["s3", "s2", "s1"], reorder=True)
    with pytest.raises(ValueError, match="sample IDs differ"):
        left.concat(wrong_order)


def test_merge_inplace_preserves_sample_sex_when_left_has_no_fid():
    left = SNPObject(
        calldata_gt=np.zeros((2, 1, 2), dtype=np.int8),
        samples=np.array(["s1"], dtype=object),
        sample_sex=np.array(["1"], dtype=object),
    )
    right = SNPObject(
        calldata_gt=np.ones((2, 1, 2), dtype=np.int8),
        samples=np.array(["s2"], dtype=object),
        sample_sex=np.array(["2"], dtype=object),
    )

    merged = left.merge(right, inplace=True)

    assert merged is left
    assert left.samples.tolist() == ["s1", "s2"]
    assert left.sample_sex.tolist() == ["1", "2"]
    assert left.calldata_gt.shape == (2, 2, 2)
