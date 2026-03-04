import numpy as np

from snputils.ancestry.io.local.read import MSPReader
from snputils.ancestry.io.local.read.__test__.fixtures import write_msp
from snputils.snp.genobj.snpobj import SNPObject
from snputils.stats import allele_freq_stream


def test_snpobject_allele_freq_default_cohort_from_3d_genotypes():
    gt = np.array(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [1, -1]],
            [[-1, -1], [0, 0]],
        ],
        dtype=float,
    )
    snp = SNPObject(calldata_gt=gt)

    freq = snp.allele_freq()
    np.testing.assert_allclose(freq, np.array([0.75, 1.0 / 3.0, 0.0]))


def test_snpobject_allele_freq_grouped_from_2d_dosages():
    gt = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, -1.0, 1.0],
            [np.nan, 0.0, 0.0],
        ]
    )
    labels = np.array(["A", "A", "B"])
    snp = SNPObject(calldata_gt=gt)

    freq, counts = snp.allele_freq(sample_labels=labels, return_counts=True)

    expected_freq = np.array(
        [
            [0.25, 1.0],
            [1.0, 0.5],
            [0.0, 0.0],
        ]
    )
    expected_counts = np.array(
        [
            [4, 2],
            [2, 2],
            [2, 2],
        ]
    )

    np.testing.assert_allclose(freq, expected_freq)
    np.testing.assert_array_equal(counts, expected_counts)


def test_snpobject_allele_freq_ancestry_masking_uses_lai():
    gt = np.array(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [1, 0]],
        ],
        dtype=float,
    )
    lai = np.array(
        [
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ],
        dtype=int,
    )
    snp = SNPObject(calldata_gt=gt, calldata_lai=lai)

    freq, counts = snp.allele_freq(ancestry=1, return_counts=True)
    np.testing.assert_allclose(freq, np.array([1.0, 0.5]))
    np.testing.assert_array_equal(counts, np.array([2, 2]))


def test_snpobject_allele_freq_handles_missing_values_and_returns_counts():
    gt = np.array(
        [
            [[0, 1], [1, -1]],
            [[np.nan, np.nan], [0, 1]],
        ],
        dtype=float,
    )
    snp = SNPObject(calldata_gt=gt)

    freq, counts = snp.allele_freq(return_counts=True)
    np.testing.assert_allclose(freq, np.array([2.0 / 3.0, 0.5]))
    np.testing.assert_array_equal(counts, np.array([3, 2]))


def test_snpobject_allele_freq_supports_haploid_style_2d_calls():
    gt = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
        ]
    )
    snp = SNPObject(calldata_gt=gt)

    freq, counts = snp.allele_freq(return_counts=True)
    np.testing.assert_allclose(freq, np.array([2.0 / 3.0, 0.5]))
    np.testing.assert_array_equal(counts, np.array([3, 2]))


def test_allele_freq_stream_matches_eager_for_grouped_2d_calls():
    gt = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [2.0, -1.0, 1.0, 0.0],
            [np.nan, 0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    labels = np.array(["A", "A", "B", "B"])
    snp = SNPObject(calldata_gt=gt)

    eager_freq, eager_counts = snp.allele_freq(sample_labels=labels, return_counts=True)
    stream_freq, stream_counts = allele_freq_stream(
        snp,
        sample_labels=labels,
        return_counts=True,
        chunk_size=2,
    )

    np.testing.assert_allclose(stream_freq, eager_freq)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_allele_freq_stream_matches_eager_for_ancestry_specific_calls():
    gt = np.array(
        [
            [[0, 1], [1, 1], [0, 0]],
            [[0, 0], [1, 0], [1, 1]],
            [[1, 0], [1, -1], [0, 1]],
        ],
        dtype=float,
    )
    lai = np.array(
        [
            [[0, 1], [1, 0], [0, 0]],
            [[0, 0], [1, 1], [1, 1]],
            [[1, 0], [1, 1], [0, 1]],
        ],
        dtype=int,
    )
    labels = np.array(["P1", "P1", "P2"])
    snp = SNPObject(calldata_gt=gt, calldata_lai=lai)

    eager_freq, eager_counts = snp.allele_freq(
        sample_labels=labels,
        ancestry=1,
        return_counts=True,
    )
    stream_freq, stream_counts = allele_freq_stream(
        snp,
        sample_labels=labels,
        ancestry=1,
        return_counts=True,
        chunk_size=1,
    )

    np.testing.assert_allclose(stream_freq, eager_freq)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_allele_freq_stream_accepts_msp_reader_for_lai(tmp_path):
    sample_ids = ["S0", "S1"]
    chromosomes = np.array([1, 1, 2, 2], dtype=np.int64)
    starts = np.array([1, 151, 1, 121], dtype=np.int64)
    ends = np.array([150, 300, 120, 220], dtype=np.int64)
    ancestry_map = {0: "AFR", 1: "EUR"}
    lai = np.array(
        [
            [0, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    msp_path = tmp_path / "toy.msp"
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    gt = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 1], [1, 0]],
        ],
        dtype=float,
    )
    variants_chrom = np.array(["1", "1", "2", "2", "2"], dtype=object)
    variants_pos = np.array([100, 200, 50, 160, 400], dtype=np.int64)
    snp = SNPObject(calldata_gt=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)
    snp_eager = SNPObject(calldata_gt=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)

    eager_freq, eager_counts = snp_eager.allele_freq(
        ancestry=1,
        laiobj=MSPReader(msp_path).read(),
        return_counts=True,
    )
    stream_freq, stream_counts = allele_freq_stream(
        snp,
        ancestry=1,
        laiobj=MSPReader(msp_path),
        chunk_size=2,
        lai_chunk_size=2,
        return_counts=True,
    )

    np.testing.assert_allclose(stream_freq, eager_freq)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_allele_freq_stream_accepts_msp_path_for_lai(tmp_path):
    sample_ids = ["S0", "S1"]
    chromosomes = np.array([1, 1], dtype=np.int64)
    starts = np.array([1, 101], dtype=np.int64)
    ends = np.array([100, 200], dtype=np.int64)
    ancestry_map = {0: "AFR", 1: "EUR"}
    lai = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    msp_path = tmp_path / "toy_path.msp"
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    gt = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=float,
    )
    variants_chrom = np.array(["1", "1"], dtype=object)
    variants_pos = np.array([50, 150], dtype=np.int64)
    snp = SNPObject(calldata_gt=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)
    snp_eager = SNPObject(calldata_gt=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)

    eager_freq, eager_counts = snp_eager.allele_freq(
        ancestry=1,
        laiobj=MSPReader(msp_path).read(),
        return_counts=True,
    )
    stream_freq, stream_counts = allele_freq_stream(
        snp,
        ancestry=1,
        laiobj=msp_path,
        chunk_size=1,
        lai_chunk_size=1,
        return_counts=True,
    )

    np.testing.assert_allclose(stream_freq, eager_freq)
    np.testing.assert_array_equal(stream_counts, eager_counts)
