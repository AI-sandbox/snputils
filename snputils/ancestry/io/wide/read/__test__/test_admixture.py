import numpy as np

from snputils.ancestry.io.wide.read import read_admixture


def test_read_admixture_accepts_dotted_prefix_with_q_and_p(tmp_path):
    prefix = tmp_path / "admix.10.10"
    q = np.array([[0.25, 0.75], [0.6, 0.4]])
    p = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    np.savetxt(f"{prefix}.Q", q, delimiter=" ")
    np.savetxt(f"{prefix}.P", p, delimiter=" ")

    observed = read_admixture(prefix)

    np.testing.assert_allclose(observed.Q, q)
    np.testing.assert_allclose(observed.P, p)


def test_admixture_zst_and_gz(tmp_path):
    import gzip
    import zstandard as zstd
    from snputils.ancestry.io.wide.read import read_admixture
    from snputils.ancestry.genobj.wide import GlobalAncestryObject
    
    q = np.array([[0.25, 0.75], [0.6, 0.4]])
    p = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    samples = np.array(["S1", "S2"], dtype=object)
    snps = np.array(["rs1", "rs2", "rs3"], dtype=object)
    ancestries = np.array(["ANC1", "ANC2"], dtype=object)
    
    obj = GlobalAncestryObject(q, p, samples=samples, snps=snps, ancestries=ancestries)
    
    # 1. Save and read uncompressed
    prefix_uncompressed = tmp_path / "toy"
    obj.save(prefix_uncompressed)
    assert (tmp_path / "toy.2.Q").exists()
    assert (tmp_path / "toy.2.P").exists()
    assert (tmp_path / "toy.sample_ids.txt").exists()
    assert (tmp_path / "toy.snp_ids.txt").exists()
    assert (tmp_path / "toy.map").exists()
    
    loaded_uncompressed = read_admixture(
        tmp_path / "toy.2.Q",
        sample_file=tmp_path / "toy.sample_ids.txt",
        snp_file=tmp_path / "toy.snp_ids.txt",
        ancestry_file=tmp_path / "toy.map",
    )
    np.testing.assert_allclose(loaded_uncompressed.Q, q)
    np.testing.assert_allclose(loaded_uncompressed.P, p)
    np.testing.assert_array_equal(loaded_uncompressed.samples, samples)
    np.testing.assert_array_equal(loaded_uncompressed.snps, snps)
    np.testing.assert_array_equal(loaded_uncompressed.ancestries, ancestries)
    
    # 2. Save and read .zst
    prefix_zst = tmp_path / "toy.zst"
    obj.save(prefix_zst)
    assert (tmp_path / "toy.2.Q.zst").exists()
    assert (tmp_path / "toy.2.P.zst").exists()
    assert (tmp_path / "toy.sample_ids.txt.zst").exists()
    assert (tmp_path / "toy.snp_ids.txt.zst").exists()
    assert (tmp_path / "toy.map.zst").exists()
    
    loaded_zst = read_admixture(
        tmp_path / "toy.2.Q.zst",
        sample_file=tmp_path / "toy.sample_ids.txt.zst",
        snp_file=tmp_path / "toy.snp_ids.txt.zst",
        ancestry_file=tmp_path / "toy.map.zst",
    )
    np.testing.assert_allclose(loaded_zst.Q, q)
    np.testing.assert_allclose(loaded_zst.P, p)
    np.testing.assert_array_equal(loaded_zst.samples, samples)
    np.testing.assert_array_equal(loaded_zst.snps, snps)
    np.testing.assert_array_equal(loaded_zst.ancestries, ancestries)

    # 3. Save and read .gz
    prefix_gz = tmp_path / "toy.gz"
    obj.save(prefix_gz)
    assert (tmp_path / "toy.2.Q.gz").exists()
    assert (tmp_path / "toy.2.P.gz").exists()
    assert (tmp_path / "toy.sample_ids.txt.gz").exists()
    assert (tmp_path / "toy.snp_ids.txt.gz").exists()
    assert (tmp_path / "toy.map.gz").exists()
    
    loaded_gz = read_admixture(
        tmp_path / "toy.2.Q.gz",
        sample_file=tmp_path / "toy.sample_ids.txt.gz",
        snp_file=tmp_path / "toy.snp_ids.txt.gz",
        ancestry_file=tmp_path / "toy.map.gz",
    )
    np.testing.assert_allclose(loaded_gz.Q, q)
    np.testing.assert_allclose(loaded_gz.P, p)
    np.testing.assert_array_equal(loaded_gz.samples, samples)
    np.testing.assert_array_equal(loaded_gz.snps, snps)
    np.testing.assert_array_equal(loaded_gz.ancestries, ancestries)
