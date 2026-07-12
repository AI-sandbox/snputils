import numpy as np

from snputils import VCFReader, BEDReader, PGENReader
from snputils._utils.genotypes import sum_diploid_genotypes


def test_vcf_dosage(snpobj_vcf, data_path):
    dosage = VCFReader(data_path + "/vcf/subset.vcf").read(genotype_mode="dosage")
    assert np.array_equal(sum_diploid_genotypes(snpobj_vcf.genotypes), dosage.genotypes)


def test_bed_dosage(snpobj_bed, data_path):
    dosage = BEDReader(data_path + "/bed/subset").read(genotype_mode="dosage")
    assert np.array_equal(snpobj_bed.genotypes, dosage.genotypes)


def test_pgen_dosage(snpobj_pgen, data_path):
    dosage = PGENReader(data_path + "/pgen/subset").read(genotype_mode="dosage")
    assert np.array_equal(sum_diploid_genotypes(snpobj_pgen.genotypes), dosage.genotypes)
