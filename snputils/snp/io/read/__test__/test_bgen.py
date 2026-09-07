import struct

import numpy as np
import pytest

from snputils import BGENReader, read_bgen, read_snp
from snputils.snp.io.read import bgen as bgen_module
from snputils._utils.genotypes import sum_diploid_genotypes
from snputils.snp.genobj.snpobj import SNPObject


def test_bgen_reader_preserves_probabilities(snpobj_bgen, snpobj_vcf):
    assert snpobj_bgen.genotypes is None
    assert snpobj_bgen.calldata_gp is not None
    assert snpobj_bgen.calldata_gp.shape[:2] == (100, snpobj_vcf.n_samples)
    assert snpobj_bgen.calldata_gp.shape[2] in (3, 4)

    expected_dosage = sum_diploid_genotypes(snpobj_vcf.genotypes[:100], dtype=np.int16)
    np.testing.assert_allclose(snpobj_bgen.dosage(), expected_dosage, atol=1 / 255)


def test_bgen_bulk_probability_read_reports_missing_native_extension(monkeypatch):
    monkeypatch.setattr(bgen_module, "_native_bgen", None)

    with pytest.raises(ImportError, match="compiled snputils.snp.io._bgen extension"):
        BGENReader("unused.bgen").read(fields=["GP"], threads=2)


def test_bgen_missing_native_extension_keeps_serial_fallback(tmp_path, monkeypatch):
    path = tmp_path / "empty.bgen"
    path.write_bytes(struct.pack("<IIII4sI", 20, 20, 0, 0, b"bgen", 2 << 2))
    monkeypatch.setattr(bgen_module, "_native_bgen", None)

    snpobj = BGENReader(path).read(fields=["GP"], threads=1)
    dosage = BGENReader(path).read_dosage(threads=1)

    assert snpobj.calldata_gp.shape == (0, 0, 0)
    assert dosage.shape == (0, 0)


def test_bgen_reader_matches_vcf_metadata(snpobj_bgen, snpobj_vcf):
    np.testing.assert_array_equal(snpobj_bgen.samples, snpobj_vcf.samples)
    np.testing.assert_array_equal(snpobj_bgen.variants_ref, snpobj_vcf.variants_ref[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_alt, snpobj_vcf.variants_alt[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_chrom, snpobj_vcf.variants_chrom[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_id, snpobj_vcf.variants_id[:100])
    np.testing.assert_array_equal(snpobj_bgen.variants_pos, snpobj_vcf.variants_pos[:100])


def test_bgen_auto_reader_and_function(data_path):
    path = data_path + "/bgen/subset.bgen"
    by_auto = read_snp(path, variant_idxs=[0, 1, 2])
    by_function = read_bgen(path, variant_idxs=[0, 1, 2])

    np.testing.assert_allclose(by_auto.calldata_gp, by_function.calldata_gp, equal_nan=True)
    np.testing.assert_array_equal(by_auto.samples, by_function.samples)
    np.testing.assert_array_equal(by_auto.variants_id, by_function.variants_id)


def test_bgen_reader_sample_and_variant_selection(data_path):
    snpobj = BGENReader(data_path + "/bgen/subset.bgen").read(
        sample_ids=["HG00100", "HG00096"],
        variant_idxs=[0, 2],
    )

    np.testing.assert_array_equal(snpobj.samples, np.array(["HG00100", "HG00096"], dtype=object))
    assert snpobj.calldata_gp.shape[:2] == (2, 2)
    assert snpobj.calldata_gp.shape[2] in (3, 4)


def test_bgen_reader_preserves_requested_variant_order(data_path):
    reader = BGENReader(data_path + "/bgen/subset.bgen")
    forward = reader.read(fields=["ID", "POS"], variant_idxs=[0, 2])
    reverse = reader.read(fields=["ID", "POS"], variant_idxs=[2, 0])

    assert reverse.calldata_gp is None
    np.testing.assert_array_equal(reverse.variants_pos, forward.variants_pos[::-1])
    np.testing.assert_array_equal(reverse.variants_id, forward.variants_id[::-1])


def test_bgen_reader_does_not_hard_call(data_path):
    with pytest.raises(NotImplementedError, match="does not hard-call"):
        BGENReader(data_path + "/bgen/subset.bgen").read(fields=["GT"])


def test_bgen_full_read_dosage_matches_indexed_rows(data_path):
    path = data_path + "/bgen/subset.bgen"
    indexes = [0, 2, 159]
    full = BGENReader(path).read(fields=["GP"])
    subset = BGENReader(path).read(fields=["GP"], variant_idxs=indexes)

    np.testing.assert_allclose(full.dosage()[indexes], subset.dosage(), atol=1 / 255, equal_nan=True)


def test_bgen_to_dosage_populates_genotypes(snpobj_bgen):
    dosage_obj = snpobj_bgen.to_dosage()

    assert snpobj_bgen.genotypes is None
    assert dosage_obj.genotypes is not None
    np.testing.assert_allclose(dosage_obj.genotypes, snpobj_bgen.dosage(), equal_nan=True)
    np.testing.assert_allclose(dosage_obj.calldata_gp, snpobj_bgen.calldata_gp, equal_nan=True)


def test_bgen_dosage_handles_variable_ploidy_rows():
    gp = np.array(
        [
            [
                [0.2, 0.8, np.nan, np.nan],
                [0.1, 0.2, 0.3, 0.4],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            [
                [0.2, 0.8, np.nan, np.nan],
                [0.1, 0.9, 0.3, 0.7],
                [np.nan, np.nan, np.nan, np.nan],
            ],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        variants_ref=np.array(["A", "A"], dtype=object),
        variants_alt=np.array(["G", "G"], dtype=object),
    )

    expected = np.array(
        [
            [0.8, 2.0, np.nan],
            [0.8, 1.6, np.nan],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(snpobj.dosage(), expected, equal_nan=True)


def test_bgen_dosage_rejects_multiallelic_probabilities():
    snpobj = SNPObject(
        calldata_gp=np.zeros((1, 1, 6), dtype=np.float32),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["C,T"], dtype=object),
    )

    with pytest.raises(ValueError, match="biallelic"):
        snpobj.dosage()


def test_bgen_sample_compression(data_path, tmp_path):
    import gzip
    import zstandard as zstd
    from snputils import BGENReader
    import shutil
    import pathlib

    bgen_src = pathlib.Path(data_path) / "bgen" / "subset.bgen"
    sample_src = pathlib.Path(data_path) / "bgen" / "subset.sample"

    # Copy files
    shutil.copy(bgen_src, tmp_path / "subset.bgen")
    shutil.copy(sample_src, tmp_path / "subset.sample")

    # Read uncompressed
    reader_uncompressed = BGENReader(tmp_path / "subset.bgen")
    res_uncompressed = reader_uncompressed.read(sample_path=tmp_path / "subset.sample")

    # Read .zst
    # Compress sample to .zst in tmp_path
    cctx = zstd.ZstdCompressor()
    with open(sample_src, "rb") as f_in, open(tmp_path / "subset.sample.zst", "wb") as f_out:
        cctx.copy_stream(f_in, f_out)

    res_zst = reader_uncompressed.read(sample_path=tmp_path / "subset.sample.zst")
    np.testing.assert_array_equal(res_uncompressed.samples, res_zst.samples)

    # Read .gz
    # Compress sample to .gz in tmp_path
    with open(sample_src, "rb") as f_in, gzip.open(tmp_path / "subset.sample.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    res_gz = reader_uncompressed.read(sample_path=tmp_path / "subset.sample.gz")
    np.testing.assert_array_equal(res_uncompressed.samples, res_gz.samples)

    # Test auto-resolution of compressed path when passing base sample path
    # Remove subset.sample to force resolution
    (tmp_path / "subset.sample").unlink()
    res_auto = reader_uncompressed.read(sample_path=tmp_path / "subset.sample")
    np.testing.assert_array_equal(res_uncompressed.samples, res_auto.samples)
