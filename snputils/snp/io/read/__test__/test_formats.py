import numpy as np
from snputils import PGENReader


# VCF - BED

def test_vcf_bed_samples(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.samples is not None
    assert snpobj_bed.samples is not None
    assert np.array_equal(snpobj_vcf.samples, snpobj_bed.samples)


def test_vcf_bed_variants_ref(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_ref is not None
    assert snpobj_bed.variants_ref is not None
    assert np.array_equal(snpobj_vcf.variants_ref, snpobj_bed.variants_ref)


def test_vcf_bed_variants_alt(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_alt is not None
    assert snpobj_bed.variants_alt is not None
    assert np.array_equal(snpobj_vcf.variants_alt, snpobj_bed.variants_alt)


def test_vcf_bed_variants_chrom(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_chrom is not None
    assert snpobj_bed.variants_chrom is not None
    assert np.array_equal(snpobj_vcf.variants_chrom, snpobj_bed.variants_chrom)


def test_vcf_bed_variants_id(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_id is not None
    assert snpobj_bed.variants_id is not None
    assert np.array_equal(snpobj_vcf.variants_id, snpobj_bed.variants_id)


def test_vcf_bed_variants_pos(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_pos is not None
    assert snpobj_bed.variants_pos is not None
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_bed.variants_pos)


# BED - PGEN


def test_bed_pgen_samples(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.samples is not None
    assert snpobj_pgen.samples is not None
    assert np.array_equal(snpobj_bed.samples, snpobj_pgen.samples)


def test_bed_pgen_variants_ref(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_ref is not None
    assert snpobj_pgen.variants_ref is not None
    assert np.array_equal(snpobj_bed.variants_ref, snpobj_pgen.variants_ref)


def test_bed_pgen_variants_alt(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_alt is not None
    assert snpobj_pgen.variants_alt is not None
    assert np.array_equal(snpobj_bed.variants_alt, snpobj_pgen.variants_alt)


def test_bed_pgen_variants_chrom(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_chrom is not None
    assert snpobj_pgen.variants_chrom is not None
    assert np.array_equal(snpobj_bed.variants_chrom, snpobj_pgen.variants_chrom)


def test_bed_pgen_variants_id(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_id is not None
    assert snpobj_pgen.variants_id is not None
    assert np.array_equal(snpobj_bed.variants_id, snpobj_pgen.variants_id)


def test_bed_pgen_variants_pos(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_pos is not None
    assert snpobj_pgen.variants_pos is not None
    assert np.array_equal(snpobj_bed.variants_pos, snpobj_pgen.variants_pos)


# VCF - PGEN


def test_vcf_pgen_genotypes(snpobj_vcf, snpobj_pgen):
    assert np.array_equal(snpobj_vcf.genotypes, snpobj_pgen.genotypes)


def test_vcf_pgen_samples(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.samples is not None
    assert snpobj_pgen.samples is not None
    assert np.array_equal(snpobj_vcf.samples, snpobj_pgen.samples)


def test_vcf_pgen_variants_ref(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_ref is not None
    assert snpobj_pgen.variants_ref is not None
    assert np.array_equal(snpobj_vcf.variants_ref, snpobj_pgen.variants_ref)


def test_vcf_pgen_variants_alt(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_alt is not None
    assert snpobj_pgen.variants_alt is not None
    assert np.array_equal(snpobj_vcf.variants_alt, snpobj_pgen.variants_alt)


def test_vcf_pgen_variants_chrom(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_chrom is not None
    assert snpobj_pgen.variants_chrom is not None
    assert np.array_equal(snpobj_vcf.variants_chrom, snpobj_pgen.variants_chrom)


def test_vcf_pgen_variants_id(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_id is not None
    assert snpobj_pgen.variants_id is not None
    assert np.array_equal(snpobj_vcf.variants_id, snpobj_pgen.variants_id)


def test_vcf_pgen_variants_pos(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_pos is not None
    assert snpobj_pgen.variants_pos is not None
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_pgen.variants_pos)


def test_variants_alt_shape(snpobj_vcf):
    assert snpobj_vcf.variants_alt.shape == (snpobj_vcf.n_snps,)


# Compressed VCF
def test_vcf_gz(data_path):
    pass  # TODO


# PGEN with compressed pvar
def test_pgen_pvar_zst(data_path, snpobj_pgen):
    snpobj = PGENReader(data_path + "/pgen_zst/subset").read(sum_strands=False)
    assert np.array_equal(snpobj_pgen.genotypes, snpobj.genotypes)
    assert np.array_equal(snpobj_pgen.variants_ref, snpobj.variants_ref)
    assert np.array_equal(snpobj_pgen.variants_alt, snpobj.variants_alt)
    assert np.array_equal(snpobj_pgen.variants_chrom, snpobj.variants_chrom)
    assert np.array_equal(snpobj_pgen.variants_id, snpobj.variants_id)
    assert np.array_equal(snpobj_pgen.variants_pos, snpobj.variants_pos)
    assert np.array_equal(snpobj_pgen.variants_filter_pass, snpobj.variants_filter_pass)
    assert np.array_equal(snpobj_pgen.variants_qual, snpobj.variants_qual)
