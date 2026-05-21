import numpy as np

from snputils import BEDReader, PGENReader


# TODO: Fails with KeyError: 'calldata/GT' (genotypes = vcf_dict["calldata/GT"].astype(np.int8))
# def test_vcf_only_samples(data_path, snpobj_vcf):
#     snpobj_vcf_only_samples = VCFReader(data_path + "/subset.vcf").read(fields=["IID"])
#     assert np.array_equal(snpobj_vcf_only_samples.samples, snpobj_vcf.samples)


def test_bed_only_samples(data_path, snpobj_bed):
    snpobj_bed_only_samples = BEDReader(data_path + "/bed/subset").read(fields=["IID"])
    assert np.array_equal(snpobj_bed_only_samples.samples, snpobj_bed.samples)


def test_pgen_only_samples(data_path, snpobj_pgen):
    snpobj_pgen_only_samples = PGENReader(data_path + "/pgen/subset").read(fields=["IID"])
    assert np.array_equal(snpobj_pgen_only_samples.samples, snpobj_pgen.samples)


# TODO: Fails with KeyError: 'calldata/GT' (genotypes = vcf_dict["calldata/GT"].astype(np.int8))
# def test_vcf_only_variants(data_path, snpobj_vcf):
#     snpobj_vcf_only_variants = VCFReader(data_path + "/subset.vcf").read(fields=["ID"])
#     assert np.array_equal(snpobj_vcf_only_variants.variants_id, snpobj_vcf.variants_id)


def test_bed_only_variants(data_path, snpobj_bed):
    snpobj_bed_only_variants = BEDReader(data_path + "/bed/subset").read(fields=["ID"])
    assert np.array_equal(snpobj_bed_only_variants.variants_id, snpobj_bed.variants_id)


def test_pgen_only_variants(data_path, snpobj_pgen):
    snpobj_pgen_only_variants = PGENReader(data_path + "/pgen/subset").read(fields=["ID"])
    assert np.array_equal(snpobj_pgen_only_variants.variants_id, snpobj_pgen.variants_id)


# TODO: Fails with KeyError: 'samples' (samples=vcf_dict["samples"],)
# def test_vcf_only_gt(data_path, snpobj_vcf):
#     snpobj_vcf_only_gt = VCFReader(data_path + "/subset.vcf").read(fields=["GT"])
#     assert np.array_equal(snpobj_vcf_only_gt.genotypes, snpobj_vcf.genotypes)


def test_bed_only_gt(data_path, snpobj_bed):
    snpobj_bed_only_gt = BEDReader(data_path + "/bed/subset").read(fields=["GT"])
    assert np.array_equal(snpobj_bed_only_gt.genotypes, snpobj_bed.genotypes)


def test_pgen_only_gt(data_path, snpobj_pgen):
    snpobj_pgen_only_gt = PGENReader(data_path + "/pgen/subset").read(fields=["GT"])
    assert np.array_equal(snpobj_pgen_only_gt.genotypes, snpobj_pgen.genotypes)
