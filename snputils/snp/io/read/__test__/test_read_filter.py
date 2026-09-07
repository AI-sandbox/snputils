import numpy as np
from snputils import BEDReader, PGENReader


# TODO: add tests for VCF


def test_bed_filter_sample_ids(data_path, snpobj_bed):
    sample_id = "HG00096"  # first sample (index 0)
    snpobj = BEDReader(data_path + "/bed/subset").read(genotype_mode="dosage", sample_ids=[sample_id])
    assert snpobj.genotypes.shape == (snpobj_bed.n_snps, 1)
    assert np.array_equal(snpobj.genotypes, snpobj_bed.genotypes[:, [0]])


def test_pgen_filter_sample_ids(data_path, snpobj_pgen):
    sample_id = "HG00096"  # first sample (index 0)
    snpobj = PGENReader(data_path + "/pgen/subset").read(genotype_mode="phased", sample_ids=[sample_id])
    assert snpobj.genotypes.shape == (snpobj_pgen.n_snps, 1, 2)
    assert np.array_equal(snpobj.genotypes.squeeze(), snpobj_pgen.genotypes[:, 0, :])


def test_bed_filter_variant_ids(data_path, snpobj_bed):
    variant_id = "22:10526445"  # third variant (index 2), for sample 0, it should be 1+1=2
    snpobj = BEDReader(data_path + "/bed/subset").read(genotype_mode="dosage", variant_ids=[variant_id], sample_idxs=[0])
    assert snpobj.genotypes.shape == (1, 1)
    assert np.array_equal(snpobj.genotypes[0, 0], snpobj_bed.genotypes[2, 0])


def test_bed_parallel_dosage_preserves_filtered_order(data_path):
    reader = BEDReader(data_path + "/bed/subset")
    variant_idxs = np.array([5, 0, 3, 10, 2, 5, 3], dtype=np.uint32)
    sample_idxs = np.array([0, 1, 2], dtype=np.uint32)

    serial = reader.read(
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
        variant_idxs=variant_idxs,
        sample_idxs=sample_idxs,
        threads=1,
    )
    parallel = reader.read(
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
        variant_idxs=variant_idxs,
        sample_idxs=sample_idxs,
        threads=4,
    )

    np.testing.assert_array_equal(parallel.genotypes, serial.genotypes)
    np.testing.assert_array_equal(parallel.variants_id, serial.variants_id)
    np.testing.assert_array_equal(parallel.samples, serial.samples)


def test_pgen_filter_variant_ids(data_path, snpobj_pgen):
    variant_id = "22:10526445"  # third variant (index 2), for sample 0, it should be 1,1
    snpobj = PGENReader(data_path + "/pgen/subset").read(genotype_mode="phased", variant_ids=[variant_id], sample_idxs=[0])
    assert snpobj.genotypes.shape == (1, 1, 2)
    assert np.array_equal(snpobj.genotypes[0, 0], snpobj_pgen.genotypes[2, 0, :])
