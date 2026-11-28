import numpy as np
import pytest
from snputils import BEDReader, PGENReader, SNPObject


# TODO: add tests for VCF


def _write_test_cohort(path):
    obj = SNPObject(
        genotypes=np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3"]),
        sample_fid=np.array(["F01", "F02", "F03"]),
        sample_sex=np.array(["2", "1", "1"]),
        variants_id=np.array(["rs1", "rs2"]),
        variants_chrom=np.array(["1", "1"]),
        variants_pos=np.array([100, 200]),
        variants_ref=np.array(["A", "C"]),
        variants_alt=np.array(["G", "T"]),
    )
    obj.save(path)
    return obj


@pytest.mark.parametrize("reader, extension", [(BEDReader, ".bed"), (PGENReader, ".pgen")])
@pytest.mark.parametrize("selectors, expected", [
    ({"sample_ids": ["s2"]}, [[1], [0]]),
    ({"variant_ids": ["rs2"]}, [[2, 0, 1]]),
    ({"sample_ids": ["s2"], "variant_ids": ["rs2"]}, [[0]]),
])
def test_gt_only_read_honors_id_filters(tmp_path, reader, extension, selectors, expected):
    path = tmp_path / f"toy{extension}"
    _write_test_cohort(path)

    result = reader(path).read(fields=["GT"], **selectors)
    np.testing.assert_array_equal(result.genotypes, expected)
    assert result.samples is None
    assert result.variants_id is None


@pytest.mark.parametrize("reader, extension, sidecar", [
    (BEDReader, ".bed", ".fam"), (PGENReader, ".pgen", ".psam"),
])
def test_sample_file_accepts_plink_whitespace(tmp_path, reader, extension, sidecar):
    path = tmp_path / ("cohort" + extension)
    cohort = _write_test_cohort(path)
    samples = path.with_suffix(sidecar)
    lines = samples.read_text().splitlines()
    samples.write_text("\n".join("  ".join(line.split()) for line in lines) + "\n")

    full = reader(path).read()
    np.testing.assert_array_equal(full.samples, cohort.samples)
    np.testing.assert_array_equal(full.sample_fid, cohort.sample_fid)
    np.testing.assert_array_equal(full.sample_sex, cohort.sample_sex)
    subset = reader(path).read(sample_ids=["s2"])
    np.testing.assert_array_equal(subset.samples, ["s2"])
    np.testing.assert_array_equal(subset.genotypes, cohort.genotypes[:, [1]])


@pytest.mark.parametrize("reader, extension", [(BEDReader, ".bed"), (PGENReader, ".pgen")])
def test_sample_indices_preserve_order_in_calls_and_metadata(tmp_path, reader, extension):
    path = tmp_path / ("cohort" + extension)
    cohort = _write_test_cohort(path)
    indices = [2, 0, 2]
    metadata = reader(path).read(fields=["IID"], sample_idxs=indices)
    np.testing.assert_array_equal(metadata.samples, cohort.samples[indices])
    block, = reader(path).iter_read(sample_idxs=indices, genotype_mode="dosage", threads=2)
    np.testing.assert_array_equal(block.samples, cohort.samples[indices])
    np.testing.assert_array_equal(block.sample_fid, cohort.sample_fid[indices])
    np.testing.assert_array_equal(block.sample_sex, cohort.sample_sex[indices])
    np.testing.assert_array_equal(block.genotypes, cohort.genotypes[:, indices])


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
