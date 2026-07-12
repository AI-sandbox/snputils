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
