import numpy as np

from snputils.ancestry.genobj.local import LocalAncestryObject


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
