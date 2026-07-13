import numpy as np

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.snp.genobj.snpobj import SNPObject


def test_sample_ids_preserve_periods_before_haplotype_suffix():
    laiobj = LocalAncestryObject(
        haplotypes=[
            "family.sample.0",
            "family.sample.1",
            "another.sample.0",
            "another.sample.1",
        ],
        lai=np.array([[0, 1, 1, 0]], dtype=np.uint8),
    )

    assert laiobj.samples == ["family.sample", "another.sample"]

    filtered = laiobj.filter_samples(samples=["family.sample"])

    assert filtered.samples == ["family.sample"]
    assert filtered.haplotypes == ["family.sample.0", "family.sample.1"]
    np.testing.assert_array_equal(filtered.lai, np.array([[0, 1]], dtype=np.uint8))


def test_missing_lai_states_are_not_counted_as_ancestries():
    lai = np.array([[0.0, 1.0, -1.0, np.nan]])
    laiobj = LocalAncestryObject(
        haplotypes=["s1.0", "s1.1", "s2.0", "s2.1"],
        lai=lai,
    )
    snpobj = SNPObject(calldata_lai=lai.reshape(1, 2, 2))

    assert laiobj.n_ancestries == 2
    assert snpobj.n_ancestries == 2


def test_reordering_samples_keeps_adjacent_haplotype_pairs():
    laiobj = LocalAncestryObject(
        haplotypes=["s1.0", "s1.1", "s2.0", "s2.1"],
        lai=np.array([[10, 11, 20, 21]]),
    )

    reordered = laiobj.filter_samples(samples=["s2", "s1"], reorder=True)

    assert reordered.samples == ["s2", "s1"]
    assert reordered.haplotypes == ["s2.0", "s2.1", "s1.0", "s1.1"]
    np.testing.assert_array_equal(reordered.lai, np.array([[20, 21, 10, 11]]))


def test_convert_to_snp_level_honors_numpy_array_overrides():
    laiobj = LocalAncestryObject(
        haplotypes=["s1.0", "s1.1"],
        lai=np.array([[0, 1], [1, 0]]),
        samples=["s1"],
        chromosomes=np.array(["1", "1"]),
        physical_pos=np.array([[10, 15], [20, 25]]),
    )
    snpobj = SNPObject(
        samples=np.array(["s1"]),
        variants_chrom=np.array(["2", "2"]),
        variants_pos=np.array([100, 200]),
        variants_ref=np.array(["A", "A"]),
        variants_alt=np.array(["C", "C"]),
        variants_filter_pass=np.array([True, True]),
        variants_id=np.array(["old1", "old2"]),
        variants_qual=np.array([99.0, 99.0]),
    )
    overrides = {
        "variants_chrom": np.array(["1", "1"]),
        "variants_pos": np.array([10, 20]),
        "variants_ref": np.array(["C", "G"]),
        "variants_alt": np.array(["T", "A"]),
        "variants_filter_pass": np.array([False, True]),
        "variants_id": np.array(["new1", "new2"]),
        "variants_qual": np.array([0.0, 5.0]),
    }

    converted = laiobj.convert_to_snp_level(snpobject=snpobj, **overrides)

    assert converted is snpobj
    for field, expected in overrides.items():
        np.testing.assert_array_equal(getattr(converted, field), expected)
    np.testing.assert_array_equal(
        converted.calldata_lai,
        np.array([[[0, 1]], [[1, 0]]]),
    )


def test_convert_to_snp_level_honors_falsey_single_value_overrides():
    laiobj = LocalAncestryObject(
        haplotypes=["s1.0", "s1.1"],
        lai=np.array([[0, 1]]),
        samples=["s1"],
        chromosomes=np.array(["1"]),
        physical_pos=np.array([[0, 0]]),
    )
    snpobj = SNPObject(
        samples=np.array(["s1"]),
        variants_chrom=np.array(["1"]),
        variants_pos=np.array([10]),
        variants_filter_pass=np.array([True]),
        variants_qual=np.array([99.0]),
    )

    converted = laiobj.convert_to_snp_level(
        snpobject=snpobj,
        variants_pos=np.array([0]),
        variants_filter_pass=np.array([False]),
        variants_qual=np.array([0.0]),
    )

    np.testing.assert_array_equal(converted.variants_pos, np.array([0]))
    np.testing.assert_array_equal(converted.variants_filter_pass, np.array([False]))
    np.testing.assert_array_equal(converted.variants_qual, np.array([0.0]))
    np.testing.assert_array_equal(converted.calldata_lai, np.array([[[0, 1]]]))
