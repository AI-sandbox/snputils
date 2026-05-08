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
