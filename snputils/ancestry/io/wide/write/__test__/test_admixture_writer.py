import numpy as np

from snputils.ancestry.genobj.wide import GlobalAncestryObject
from snputils.ancestry.io.wide.read import read_admixture
from snputils.ancestry.io.wide.write import AdmixtureWriter


def test_admixture_writer_appends_suffixes_to_dotted_prefix(tmp_path):
    prefix = tmp_path / "admix.10.10"
    q = np.array([[0.25, 0.75], [0.6, 0.4]])
    p = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    wideobj = GlobalAncestryObject(
        q,
        p,
        samples=["s1", "s2"],
        snps=["rs1", "rs2", "rs3"],
        ancestries=["EUR", "AFR"],
    )

    AdmixtureWriter(wideobj, prefix).write()

    assert (tmp_path / "admix.10.10.2.Q").exists()
    assert (tmp_path / "admix.10.10.2.P").exists()
    assert (tmp_path / "admix.10.10.sample_ids.txt").exists()
    assert (tmp_path / "admix.10.10.snp_ids.txt").exists()
    assert (tmp_path / "admix.10.10.map").exists()
    assert not (tmp_path / "admix.10.2.Q").exists()

    observed = read_admixture(tmp_path / "admix.10.10.2")
    np.testing.assert_allclose(observed.Q, q)
    np.testing.assert_allclose(observed.P, p)
