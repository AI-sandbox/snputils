import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.bgen import BGENReader
from snputils.snp.io.write.bgen import BGENWriter


@pytest.mark.parametrize("representation", ["dosage", "alleles", "phased"])
def test_harmonized_hardcalls_preserve_missing_samples_and_variants(tmp_path, representation):
    gt = np.array([[0, np.nan, 2], [np.nan, -1, np.nan]])
    if representation != "dosage":
        gt = np.repeat(gt[:, :, None], 2, axis=2)
        gt[0, 2] = [1, 1]
        gt[0, 1] = [1, np.nan]
    query = SNPObject(genotypes=gt, variants_chrom=np.array(["1", "1"]), variants_pos=np.array([10, 20]),
                      variants_ref=np.array(["A", "A"]), variants_alt=np.array(["G", "G"]))
    reference = SNPObject(variants_chrom=query.variants_chrom, variants_pos=query.variants_pos,
                          variants_ref=query.variants_alt, variants_alt=query.variants_ref)
    aligned = query.correct_flipped_variants(reference)
    path = tmp_path / "aligned.bgen"
    BGENWriter(aligned, path).write(phased=representation == "phased")
    observed = BGENReader(path).read()
    np.testing.assert_allclose(observed.dosage(), [[2, np.nan, 0], [np.nan, np.nan, np.nan]])
    width = 4 if representation == "phased" else 3
    assert observed.calldata_gp.shape == (2, 3, width)


@pytest.mark.parametrize("phased", [False, True])
def test_all_missing_haploid_probabilities_retain_width(tmp_path, phased):
    source = SNPObject(calldata_gp=np.full((1, 2, 2), np.nan), variants_chrom=np.array(["Y"]),
                       variants_pos=np.array([10]), variants_ref=np.array(["A"]), variants_alt=np.array(["G"]))
    path = tmp_path / "haploid.bgen"
    BGENWriter(source, path).write(phased=phased)
    np.testing.assert_array_equal(BGENReader(path).read().calldata_gp, source.calldata_gp)


def test_bgen_writer_roundtrips_probabilities(tmp_path):
    path = tmp_path / "toy.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [np.nan, np.nan, np.nan]],
            [[0.0, 1.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.variants_id, snpobj.variants_id)
    np.testing.assert_array_equal(observed.variants_ref, snpobj.variants_ref)
    np.testing.assert_array_equal(observed.variants_alt, snpobj.variants_alt)
    np.testing.assert_array_equal(observed.variants_chrom, snpobj.variants_chrom)
    np.testing.assert_array_equal(observed.variants_pos, snpobj.variants_pos)
    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)


@pytest.mark.parametrize("inplace", [False, True])
def test_harmonize_bgen_probabilities_roundtrip(tmp_path, inplace):
    nan = np.nan
    gp = np.array([
        [[0.1, 0.2, 0.7, nan], [0.8, 0.2, nan, nan], [nan] * 4],
        [[0.1, 0.9, 0.4, 0.6], [0.3, 0.7, nan, nan], [nan] * 4],
        [[0.2, 0.3, 0.5, nan], [0.6, 0.4, nan, nan], [nan] * 4],
    ], dtype=np.float32)
    source = SNPObject(
        calldata_gp=gp,
        variants_chrom=np.array(["1", "1", "1"]),
        variants_pos=np.array([10, 20, 30]),
        variants_ref=np.array(["A", "A", "A"]),
        variants_alt=np.array(["G", "G", "G"]),
    )
    source.save_bgen(tmp_path / "input.bgen")
    query = BGENReader(tmp_path / "input.bgen").read()
    reference = SNPObject(
        variants_chrom=source.variants_chrom, variants_pos=source.variants_pos,
        variants_ref=np.array(["G", "G", "A"]), variants_alt=np.array(["A", "A", "G"]),
    )
    if inplace:
        query.to_dosage(inplace=True)
        assert query.correct_flipped_variants(reference, inplace=True) is None
        aligned = query
    else:
        original_gp = query.calldata_gp.copy()
        aligned = query.correct_flipped_variants(reference)
        np.testing.assert_array_equal(query.calldata_gp, original_gp)

    expected = gp.copy()
    expected[0, 0] = [0.7, 0.2, 0.1, nan]
    expected[0, 1] = [0.2, 0.8, nan, nan]
    expected[1, 0] = [0.9, 0.1, 0.6, 0.4]
    expected[1, 1] = [0.7, 0.3, nan, nan]
    aligned.save_bgen(tmp_path / "aligned.bgen")
    reopened = BGENReader(tmp_path / "aligned.bgen").read()
    np.testing.assert_array_equal(reopened.variants_ref, reference.variants_ref)
    np.testing.assert_array_equal(reopened.variants_alt, reference.variants_alt)
    np.testing.assert_allclose(reopened.calldata_gp, expected, atol=1 / 255)
    if inplace:
        np.testing.assert_allclose(aligned.genotypes, reopened.dosage(), atol=2 / 255)


@pytest.mark.parametrize("compression", [None, "zlib", "zstd"])
@pytest.mark.parametrize("selection", [{}, {"threads": 2}, {"sample_idxs": [2, 0], "variant_idxs": [1, 0]}])
def test_flip_dosages_read_from_mixed_ploidy_bgen(tmp_path, compression, selection):
    path = tmp_path / "cohort.bgen"
    source = SNPObject(
        samples=np.array(["haploid", "diploid", "missing"]),
        variants_chrom=np.array(["X", "X"]), variants_pos=np.array([10, 20]),
        variants_ref=np.array(["A", "C"]), variants_alt=np.array(["G", "T"]),
        calldata_gp=np.array([
            [[1, 0, np.nan], [0, 1, 0], [np.nan] * 3],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        ], dtype=np.float32),
    )
    BGENWriter(source, path).write(compression=compression)
    reader = BGENReader(path)
    subset = "sample_idxs" in selection
    dosages = reader.read(genotype_mode="dosage", **selection)
    reference = SNPObject(
        variants_chrom=source.variants_chrom, variants_pos=source.variants_pos,
        variants_ref=source.variants_alt, variants_alt=source.variants_ref,
    )
    aligned = dosages.correct_flipped_variants(reference)
    expected = np.array([[1, 1, np.nan], [2, 0, 1]])
    if subset:
        expected = expected[[1, 0]][:, [2, 0]]
    np.testing.assert_allclose(aligned.genotypes, expected)
    assert aligned.calldata_gp is None
    aligned.save_bgen(tmp_path / "aligned.bgen")
    reopened = BGENReader(tmp_path / "aligned.bgen").read()
    np.testing.assert_allclose(reopened.dosage(), expected)
    # The haploid ALT call still has just two genotype probability states.
    haploid = reopened.calldata_gp[1, 1] if subset else reopened.calldata_gp[0, 0]
    np.testing.assert_array_equal(haploid, [0, 1, np.nan])


@pytest.mark.parametrize("compression", ["zlib", "zstd"])
def test_bgen_reader_native_bulk_paths_match_general_reader(tmp_path, compression):
    path = tmp_path / f"toy_bulk_{compression}.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [np.nan, np.nan, np.nan]],
            [[0.0, 1.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(compression=compression, bit_depth=16, phased=False)
    reader = BGENReader(path)
    fast_gp = reader.read(fields=["GP"]).calldata_gp
    parallel_gp = reader.read(fields=["GP"], threads=2).calldata_gp
    general_gp = reader.read(fields=["GP"], sample_idxs=np.arange(gp.shape[1])).calldata_gp
    fast_dosage = reader.read_dosage()
    parallel_dosage = reader.read_dosage(threads=2)
    parallel_object = reader.read(genotype_mode="dosage", threads=2)
    general_dosage = reader.read(fields=["GP"], sample_idxs=np.arange(gp.shape[1])).dosage()

    np.testing.assert_allclose(fast_gp, general_gp, atol=1 / 65535, equal_nan=True)
    np.testing.assert_array_equal(parallel_gp, fast_gp)
    assert np.isnan(fast_dosage[0, 2])
    np.testing.assert_allclose(fast_dosage, general_dosage, atol=1 / 65535, equal_nan=True)
    np.testing.assert_array_equal(parallel_dosage, fast_dosage)
    np.testing.assert_array_equal(parallel_object.genotypes, fast_dosage)

    with pytest.raises(ValueError, match="at least 1"):
        reader.read_dosage(threads=0)
    with pytest.raises(ValueError, match="all samples and variants"):
        reader.read_dosage(threads=2, variant_idxs=[0])
    with pytest.raises(ValueError, match=r"fields=\['GP'\]"):
        reader.read(fields=["GP", "POS"], threads=2)


def test_bgen_reader_falls_back_when_native_zstd_is_unavailable(tmp_path, monkeypatch):
    from snputils.snp.io.read import bgen as bgen_module

    path = tmp_path / "zstd_fallback.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )
    BGENWriter(snpobj, path).write(compression="zstd", bit_depth=16, phased=False)

    def unavailable(*args):
        raise NotImplementedError(
            "Native BGEN zstd decompression requires a shared libzstd library."
        )

    monkeypatch.setattr(bgen_module._native_bgen, "read_file_probabilities", unavailable)
    monkeypatch.setattr(bgen_module._native_bgen, "read_file_dosage", unavailable)

    reader = BGENReader(path)
    with pytest.warns(RuntimeWarning, match="single-threaded Python zstandard"):
        observed_gp = reader.read(fields=["GP"], threads=2).calldata_gp
    with pytest.warns(RuntimeWarning, match="single-threaded Python zstandard"):
        observed_dosage = reader.read_dosage(threads=2)

    np.testing.assert_allclose(observed_gp, gp, atol=1 / 65535)
    np.testing.assert_allclose(
        observed_dosage,
        gp[:, :, 1] + 2 * gp[:, :, 2],
        atol=1 / 65535,
    )


def test_bgen_reader_dosage_mode_returns_metadata_complete_snpobject(tmp_path):
    path = tmp_path / "dosage_mode.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.25, 0.5, 0.25], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(phased=False, bit_depth=16)
    observed = BGENReader(path).read(genotype_mode="dosage")

    expected = gp[:, :, 1] + 2 * gp[:, :, 2]
    np.testing.assert_allclose(observed.genotypes, expected, atol=1 / 65535)
    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.variants_id, snpobj.variants_id)
    np.testing.assert_array_equal(observed.variants_pos, snpobj.variants_pos)
    assert observed.calldata_gp is None
    assert observed.calldata_ploidy is None


def test_bgen_writer_rejects_incompatible_phased_width(tmp_path):
    snpobj = SNPObject(
        calldata_gp=np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32),
        samples=np.array(["s1"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    with pytest.raises(ValueError, match="phased BGEN probabilities require 4 columns"):
        BGENWriter(snpobj, tmp_path / "bad.bgen").write(phased=True)


def test_bgen_writer_encodes_hardcalls_as_probabilities(tmp_path):
    path = tmp_path / "hardcalls.bgen"
    snpobj = SNPObject(
        genotypes=np.array([[0, 1, 2], [2, 0, -1]], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    snpobj.save_bgen(path)
    observed = BGENReader(path).read()

    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(observed.calldata_gp, expected, equal_nan=True)


@pytest.mark.parametrize("compression", ["zlib", "zstd"])
def test_bgen_writer_roundtrips_mixed_probability_widths(tmp_path, compression):
    path = tmp_path / f"mixed_{compression}.bgen"
    gp = np.array(
        [
            [[1.0, 0.0, 0.0, np.nan], [0.0, 1.0, 0.0, np.nan]],
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(compression=compression, bit_depth=16)
    observed = BGENReader(path).read()
    observed_gp_only = BGENReader(path).read(fields=["GP"])
    observed_parallel = BGENReader(path).read(fields=["GP"], threads=2)

    assert observed_parallel.calldata_gp.shape == (2, 2, 4)
    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)
    np.testing.assert_allclose(observed_gp_only.calldata_gp, gp, atol=1 / 65535, equal_nan=True)
    np.testing.assert_allclose(observed_parallel.calldata_gp, gp, atol=1 / 65535, equal_nan=True)


def test_bgen_writer_roundtrips_variable_ploidy_unphased(tmp_path):
    path = tmp_path / "variable_ploidy_unphased.bgen"
    gp = np.array(
        [
            [
                [0.2, 0.8, np.nan],
                [0.2, 0.3, 0.5],
                [np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["haploid", "diploid", "missing"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)


def test_bgen_writer_roundtrips_variable_ploidy_phased(tmp_path):
    path = tmp_path / "variable_ploidy_phased.bgen"
    gp = np.array(
        [
            [
                [0.2, 0.8, np.nan, np.nan],
                [0.2, 0.8, 0.4, 0.6],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["haploid", "diploid", "missing"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    BGENWriter(snpobj, path).write(compression="zlib", bit_depth=16)
    observed = BGENReader(path).read()

    np.testing.assert_allclose(observed.calldata_gp, gp, atol=1 / 65535, equal_nan=True)
    with pytest.raises(ValueError, match="non-diploid call"):
        BGENReader(path).read(genotype_mode="phased")


def test_bgen_reader_hardcalls_phased_probabilities(tmp_path):
    path = tmp_path / "phased_probabilities.bgen"
    gp = np.array(
        [
            [[0.9, 0.1, 0.2, 0.8], [0.1, 0.9, 0.8, 0.2], [np.nan] * 4],
            [[0.4, 0.6, 0.7, 0.3], [0.8, 0.2, 0.1, 0.9], [0.2, 0.8, 0.3, 0.7]],
        ],
        dtype=np.float32,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2", "missing"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BGENWriter(snpobj, path).write(phased=True)
    observed = BGENReader(path).read(genotype_mode="phased")

    expected = np.array(
        [
            [[0, 1], [1, 0], [-1, -1]],
            [[1, 0], [0, 1], [1, 1]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(observed.genotypes, expected)
    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.variants_id, snpobj.variants_id)
    assert observed.calldata_gp is None


def test_bgen_reader_rejects_unphased_probabilities_in_phased_mode(tmp_path):
    path = tmp_path / "unphased_probabilities.bgen"
    snpobj = SNPObject(
        calldata_gp=np.array([[[0.2, 0.3, 0.5]]], dtype=np.float32),
        samples=np.array(["s1"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["G"], dtype=object),
        variants_chrom=np.array(["1"], dtype=object),
        variants_id=np.array(["rs1"], dtype=object),
        variants_pos=np.array([10]),
    )

    BGENWriter(snpobj, path).write(phased=False)

    with pytest.raises(ValueError, match="requires phased probabilities"):
        BGENReader(path).read(genotype_mode="phased")
