import numpy as np

from snputils import BCFReader, read_bcf, read_snp
from snputils._utils.genotypes import sum_diploid_genotypes


def test_bcf_reader_matches_vcf_genotypes_and_metadata(snpobj_bcf, snpobj_vcf):
    np.testing.assert_array_equal(snpobj_bcf.genotypes, snpobj_vcf.genotypes[:100])
    np.testing.assert_array_equal(snpobj_bcf.samples, snpobj_vcf.samples)
    np.testing.assert_array_equal(snpobj_bcf.variants_ref, snpobj_vcf.variants_ref[:100])
    np.testing.assert_array_equal(snpobj_bcf.variants_alt, snpobj_vcf.variants_alt[:100])
    np.testing.assert_array_equal(snpobj_bcf.variants_chrom, snpobj_vcf.variants_chrom[:100])
    np.testing.assert_array_equal(snpobj_bcf.variants_id, snpobj_vcf.variants_id[:100])
    np.testing.assert_array_equal(snpobj_bcf.variants_pos, snpobj_vcf.variants_pos[:100])


def test_bcf_auto_reader_and_function(data_path):
    path = data_path + "/bcf/subset.bcf"
    by_auto = read_snp(path, variant_idxs=[0, 1, 2])
    by_function = read_bcf(path, variant_idxs=[0, 1, 2])

    np.testing.assert_array_equal(by_auto.genotypes, by_function.genotypes)
    np.testing.assert_array_equal(by_auto.samples, by_function.samples)
    np.testing.assert_array_equal(by_auto.variants_id, by_function.variants_id)


def test_bcf_reader_sample_and_variant_selection(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(
        sum_strands=False,
        sample_ids=["HG00100", "HG00096"],
        variant_idxs=[0, 2],
    )

    np.testing.assert_array_equal(snpobj.samples, np.array(["HG00100", "HG00096"], dtype=object))
    np.testing.assert_array_equal(snpobj.genotypes, snpobj_vcf.genotypes[[0, 2]][:, [3, 0], :])


def test_bcf_reader_supports_region_filtering(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(
        sum_strands=False,
        region="22:10526445-10526445",
        sample_idxs=[0],
    )

    assert snpobj.genotypes.shape == (1, 1, 2)
    np.testing.assert_array_equal(snpobj.variants_pos, np.array([10526445], dtype=np.int64))
    np.testing.assert_array_equal(snpobj.genotypes[0, 0], snpobj_vcf.genotypes[2, 0])


def test_bcf_reader_supports_summed_strands(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(sum_strands=True, variant_idxs=[0, 1, 2])

    np.testing.assert_array_equal(snpobj.genotypes, sum_diploid_genotypes(snpobj_vcf.genotypes[:3]))


def test_bcf_reader_reads_info_qual_and_filter_when_requested(data_path):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(
        fields=["ID", "QUAL", "FILTER", "INFO"],
        variant_idxs=[0, 1, 2],
    )

    assert snpobj.genotypes is None
    np.testing.assert_array_equal(snpobj.variants_id.shape, (3,))
    np.testing.assert_array_equal(snpobj.variants_filter_pass, np.array([True, True, True]))
    assert snpobj.variants_qual.shape == (3,)
    assert snpobj.variants_info.shape == (3,)


def test_bcf_reader_reads_samples_without_genotypes(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(fields=["IID"])

    assert snpobj.genotypes is None
    np.testing.assert_array_equal(snpobj.samples, snpobj_vcf.samples)
