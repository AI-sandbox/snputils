import numpy as np
from snputils import BEDReader, PGENReader


# TODO: add tests for VCF


def test_bed_filter_sample_ids(data_path, snpobj_bed):
    sample_id = "HG00096"  # first sample (index 0)
    snpobj = BEDReader(data_path + "/bed/subset").read(sum_strands=True, sample_ids=[sample_id])
    assert snpobj.genotypes.shape == (snpobj_bed.n_snps, 1)
    assert np.array_equal(snpobj.genotypes, snpobj_bed.genotypes[:, 0].sum(axis=-1, keepdims=True))


def test_pgen_filter_sample_ids(data_path, snpobj_pgen):
    sample_id = "HG00096"  # first sample (index 0)
    snpobj = PGENReader(data_path + "/pgen/subset").read(sum_strands=False, sample_ids=[sample_id])
    assert snpobj.genotypes.shape == (snpobj_pgen.n_snps, 1, 2)
    assert np.array_equal(snpobj.genotypes.squeeze(), snpobj_pgen.genotypes[:, 0, :])


def test_bed_filter_variant_ids(data_path, snpobj_bed):
    variant_id = "22:10526445"  # third variant (index 2), for sample 0, it should be 1+1=2
    snpobj = BEDReader(data_path + "/bed/subset").read(sum_strands=True, variant_ids=[variant_id], sample_idxs=[0])
    assert snpobj.genotypes.shape == (1, 1)
    assert np.array_equal(snpobj.genotypes[0, 0], sum(snpobj_bed.genotypes[2, 0]))


def test_pgen_filter_variant_ids(data_path, snpobj_pgen):
    variant_id = "22:10526445"  # third variant (index 2), for sample 0, it should be 1,1
    snpobj = PGENReader(data_path + "/pgen/subset").read(sum_strands=False, variant_ids=[variant_id], sample_idxs=[0])
    assert snpobj.genotypes.shape == (1, 1, 2)
    assert np.array_equal(snpobj.genotypes[0, 0], snpobj_pgen.genotypes[2, 0, :])
