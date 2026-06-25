import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
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

    # 4. Verify
    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)
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
    observed_1 = BCFReader(str(tmp_path / "partitioned_1.bcf")).read()
    np.testing.assert_array_equal(observed_1.variants_chrom, np.array(["1"], dtype=object))
    np.testing.assert_array_equal(observed_1.genotypes, snpobj.genotypes[[0]])

    # Read partition 2
    observed_2 = BCFReader(str(tmp_path / "partitioned_2.bcf")).read()
    np.testing.assert_array_equal(observed_2.variants_chrom, np.array(["2"], dtype=object))
    np.testing.assert_array_equal(observed_2.genotypes, snpobj.genotypes[[1]])

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

    observed = BCFReader(str(output_path)).read()
    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)

