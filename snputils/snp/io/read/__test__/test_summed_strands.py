import numpy as np

from snputils import VCFReader, BEDReader, PGENReader


def test_vcf_summed_strands(snpobj_vcf, data_path):
    snpobj_vcf_summed_strands = VCFReader(data_path + "/vcf/subset.vcf").read(sum_strands=True)
    assert np.array_equal(snpobj_vcf.genotypes.sum(axis=2, dtype=np.int8), snpobj_vcf_summed_strands.genotypes)


def test_bed_summed_strands(snpobj_bed, data_path):
    snpojb_bed_summed_strands = BEDReader(data_path + "/bed/subset").read(sum_strands=True)
    assert np.array_equal(snpobj_bed.genotypes, snpojb_bed_summed_strands.genotypes)


def test_pgen_summed_strands(snpobj_pgen, data_path):
    snpobj_pgen_summed_strands = PGENReader(data_path + "/pgen/subset").read(sum_strands=True)
    assert np.array_equal(snpobj_pgen.genotypes.sum(axis=2, dtype=np.int8), snpobj_pgen_summed_strands.genotypes)
