import numpy as np
import pytest

from snputils._utils.genotypes import sum_diploid_genotypes
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.bcf import _read_bgzf_or_gzip
from snputils.snp.io.read.bcf import BCFReader
from snputils.snp.io.write.bcf import BCFWriter

def test_bcf_writer_roundtrip(tmp_path):
    output_path = tmp_path / "toy.bcf"

    # 1. Create a toy SNPObject
    snpobj = SNPObject(
        genotypes=np.array([
            [[0, 0], [0, 1], [1, 1], [-1, -1]],
            [[1, 0], [0, 0], [-1, -1], [1, 1]]
        ], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3", "s4"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["22", "22"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([100, 200]),
        variants_qual=np.array([45.5, np.nan], dtype=np.float32)
    )

    # 2. Write unphased BCF
    BCFWriter(snpobj, str(output_path), phased=False).write()

    # 3. Read it back
    observed = BCFReader(str(output_path)).read()

    # 4. Verify reads return unphased dosages
    np.testing.assert_array_equal(observed.genotypes, sum_diploid_genotypes(snpobj.genotypes))
    with pytest.raises(ValueError, match="unphased BCF genotypes"):
        BCFReader(str(output_path)).read(genotype_mode="phased")
    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.variants_chrom, snpobj.variants_chrom)
    np.testing.assert_array_equal(observed.variants_pos, snpobj.variants_pos)
    np.testing.assert_array_equal(observed.variants_id, snpobj.variants_id)
    np.testing.assert_array_equal(observed.variants_ref, snpobj.variants_ref)
    np.testing.assert_array_equal(observed.variants_alt, snpobj.variants_alt)
    np.testing.assert_allclose(observed.variants_qual, snpobj.variants_qual, equal_nan=True)
    np.testing.assert_array_equal(observed.variants_filter_pass, np.array([True, True]))

def test_bcf_writer_phased(tmp_path):
    output_path = tmp_path / "phased.bcf"

    snpobj = SNPObject(
        genotypes=np.array([
            [[0, 1], [1, 0]],
            [[0, 0], [-1, -1]]
        ], dtype=np.int8),
        samples=np.array(["s1", "s2"]),
        variants_ref=np.array(["A", "G"], dtype=object),
        variants_alt=np.array(["T", "C"], dtype=object),
        variants_chrom=np.array(["X", "X"], dtype=object),
        variants_id=np.array([".", "rs3"], dtype=object),
        variants_pos=np.array([1000, 2000]),
        variants_qual=np.array([np.nan, 99.0], dtype=np.float32)
    )

    # Write phased BCF
    BCFWriter(snpobj, str(output_path), phased=True).write()

    observed = BCFReader(str(output_path)).read()

    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)
    np.testing.assert_array_equal(observed.variants_id, np.array([".", "rs3"], dtype=object))
    np.testing.assert_allclose(observed.variants_qual, snpobj.variants_qual, equal_nan=True)

def test_bcf_writer_chrom_partition(tmp_path):
    output_path = tmp_path / "partitioned.bcf"

    snpobj = SNPObject(
        genotypes=np.array([
            [[0, 0], [1, 1]],
            [[0, 1], [-1, -1]]
        ], dtype=np.int8),
        samples=np.array(["s1", "s2"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "2"], dtype=object),
        variants_id=np.array(["v1", "v2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BCFWriter(snpobj, str(output_path)).write(chrom_partition=True)

    # Check files exist
    assert (tmp_path / "partitioned_1.bcf").exists()
    assert (tmp_path / "partitioned_2.bcf").exists()

    # Read partition 1
    observed_1 = BCFReader(str(tmp_path / "partitioned_1.bcf")).read(genotype_mode="dosage")
    np.testing.assert_array_equal(observed_1.variants_chrom, np.array(["1"], dtype=object))
    np.testing.assert_array_equal(observed_1.genotypes, sum_diploid_genotypes(snpobj.genotypes[[0]]))

    # Read partition 2
    observed_2 = BCFReader(str(tmp_path / "partitioned_2.bcf")).read(genotype_mode="dosage")
    np.testing.assert_array_equal(observed_2.variants_chrom, np.array(["2"], dtype=object))
    np.testing.assert_array_equal(observed_2.genotypes, sum_diploid_genotypes(snpobj.genotypes[[1]]))

def test_bcf_writer_with_info(tmp_path):
    output_path = tmp_path / "info.bcf"

    snpobj = SNPObject(
        genotypes=np.array([
            [[0, 0], [1, 1]],
            [[0, 1], [-1, -1]]
        ], dtype=np.int8),
        samples=np.array(["s1", "s2"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["v1", "v2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    # Write with variants_info (END=...)
    BCFWriter(snpobj, str(output_path)).write(
        variants_info=["END=1000", "END=2000"]
    )

    observed = BCFReader(str(output_path)).read(fields=["INFO"])
    # BCFReader returns a parsed string for INFO, e.g. "END=1000"
    np.testing.assert_array_equal(observed.variants_info, np.array(["END=1000", "END=2000"], dtype=object))

def test_snpobj_save_bcf(tmp_path):
    output_path = tmp_path / "saved.bcf"
    snpobj = SNPObject(
        genotypes=np.array([[[0, 0]], [[1, 1]]], dtype=np.int8),
        samples=np.array(["s1"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["v1", "v2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    snpobj.save(output_path)
    assert output_path.exists()

    observed = BCFReader(str(output_path)).read(genotype_mode="dosage")
    np.testing.assert_array_equal(observed.genotypes, sum_diploid_genotypes(snpobj.genotypes))


def test_bcf_writer_roundtrips_2d_hardcall_dosages(tmp_path):
    output_path = tmp_path / "dosage.bcf"
    snpobj = SNPObject(
        genotypes=np.array([[0, 1, 2, -1], [2, 1, 0, -1]], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3", "s4"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["v1", "v2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BCFWriter(snpobj, output_path).write()

    observed_sum = BCFReader(output_path).read(genotype_mode="dosage")
    expected = np.array(
        [
            [[0, 0], [0, 1], [1, 1], [-1, -1]],
            [[1, 1], [0, 1], [0, 0], [-1, -1]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(observed_sum.genotypes, snpobj.genotypes)

    with pytest.raises(ValueError, match="unphased BCF genotypes"):
        BCFReader(output_path).read(genotype_mode="phased")

    BCFWriter(snpobj, output_path, phased=True).write()
    observed = BCFReader(output_path).read(genotype_mode="phased")
    np.testing.assert_array_equal(observed.genotypes, expected)


def test_bcf_writer_roundtrips_sampleless_annotation_only_bcf(tmp_path):
    output_path = tmp_path / "annotations.bcf"
    snpobj = SNPObject(
        genotypes=np.empty((2, 0, 2), dtype=np.int8),
        samples=np.array([], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([100, 200]),
        variants_qual=np.array([np.nan, 50.0], dtype=np.float32),
        variants_filter_pass=np.array([True, False]),
        variants_info=np.array(["ANN=missense_variant;DB", "AF=0.125"], dtype=object),
    )

    BCFWriter(snpobj, output_path).write(rename_missing_values=False)

    observed = BCFReader(output_path).read(fields="*")
    assert observed.genotypes.shape == (2, 0, 2)
    np.testing.assert_array_equal(observed.samples, np.array([], dtype=object))
    np.testing.assert_array_equal(observed.variants_filter_pass, np.array([True, False]))
    np.testing.assert_array_equal(
        observed.variants_info.astype(str),
        np.array(["ANN=missense_variant;DB", "AF=0.125"], dtype=object),
    )


def test_bcf_writer_roundtrips_general_info_filter_and_gp(tmp_path):
    output_path = tmp_path / "metadata_gp.bcf"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]],
            [[np.nan, np.nan, np.nan], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        genotypes=np.array([[[0, 0], [1, 1]], [[-1, -1], [0, 1]]], dtype=np.int8),
        calldata_gp=gp,
        samples=np.array(["s1", "s2"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T,C"], dtype=object),
        variants_chrom=np.array(["2", "2"], dtype=object),
        variants_id=np.array(["v1", "v2"], dtype=object),
        variants_pos=np.array([100, 200]),
        variants_filter_pass=np.array(["PASS", "q10;s50"], dtype=object),
        variants_info=np.array(
            ["AC=2;AF=0.5;DB;ANN=missense;CSQ=a,b", "AC=.;AF=.;ANN=synonymous;CSQ=c"],
            dtype=object,
        ),
    )

    BCFWriter(snpobj, output_path, phased=True).write()

    observed = BCFReader(output_path).read(fields=["GT", "GP", "IID", "ALT", "FILTER", "INFO"])
    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)
    np.testing.assert_allclose(observed.calldata_gp, gp, equal_nan=True)
    np.testing.assert_array_equal(observed.variants_alt, snpobj.variants_alt)
    np.testing.assert_array_equal(observed.variants_filter_pass, np.array([True, False]))
    np.testing.assert_array_equal(
        observed.variants_info.astype(str),
        snpobj.variants_info.astype(str),
    )


def test_bcf_writer_outputs_bgzf_with_eof_marker(tmp_path):
    output_path = tmp_path / "bgzf.bcf"
    snpobj = SNPObject(
        genotypes=np.array([[[0, 0]]], dtype=np.int8),
        samples=np.array(["s1"]),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["v1"], dtype=object),
        variants_pos=np.array([10]),
    )

    BCFWriter(snpobj, output_path).write()

    raw = output_path.read_bytes()
    assert raw[:4] == b"\x1f\x8b\x08\x04"
    assert raw[12:16] == b"BC\x02\x00"
    assert raw[-28:] == (
        b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00"
        b"BC\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    assert _read_bgzf_or_gzip(output_path).startswith(b"BCF\x02\x02")
