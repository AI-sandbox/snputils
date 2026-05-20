import numpy as np

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.ancestry.io.local.read import MSPReader
from snputils.ancestry.io.local.read.__test__.fixtures import write_msp
from snputils.ancestry.io.local.write import FLAREWriter
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
    snp = SNPObject(genotypes=gt)

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
    snp = SNPObject(genotypes=gt)

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
    snp = SNPObject(genotypes=gt, calldata_lai=lai)

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
    snp = SNPObject(genotypes=gt)

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
    snp = SNPObject(genotypes=gt)

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
    snp = SNPObject(genotypes=gt)

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
    snp = SNPObject(genotypes=gt, calldata_lai=lai)

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
    snp = SNPObject(genotypes=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)
    snp_eager = SNPObject(genotypes=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)

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


def test_allele_freq_stream_accepts_lai_path(tmp_path):
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
    lai_path = tmp_path / "toy_path.msp"
    write_msp(lai_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    gt = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=float,
    )
    variants_chrom = np.array(["1", "1"], dtype=object)
    variants_pos = np.array([50, 150], dtype=np.int64)
    snp = SNPObject(genotypes=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)
    snp_eager = SNPObject(genotypes=gt, variants_chrom=variants_chrom, variants_pos=variants_pos)

    eager_freq, eager_counts = snp_eager.allele_freq(
        ancestry=1,
        laiobj=MSPReader(lai_path).read(),
        return_counts=True,
    )
    stream_freq, stream_counts = allele_freq_stream(
        snp,
        ancestry=1,
        laiobj=lai_path,
        chunk_size=1,
        lai_chunk_size=1,
        return_counts=True,
    )

    np.testing.assert_allclose(stream_freq, eager_freq)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_allele_freq_stream_accepts_flare_path_for_lai(tmp_path):
    sample_ids = ["S0", "S1"]
    chromosomes = np.array(["1", "1"], dtype=object)
    positions = np.array([50, 150], dtype=np.int64)
    ancestry_map = {"0": "AFR", "1": "EUR"}
    lai = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    gt = np.array(
        [
            [[0, 1], [1, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=float,
    )
    snp = SNPObject(genotypes=gt, variants_chrom=chromosomes, variants_pos=positions)
    snp_eager = SNPObject(genotypes=gt, variants_chrom=chromosomes, variants_pos=positions)
    snp_for_flare = SNPObject(
        samples=np.asarray(sample_ids, dtype=object),
        genotypes=gt.astype(np.int8),
        variants_chrom=chromosomes,
        variants_pos=positions,
        variants_id=np.array(["v0", "v1"], dtype=object),
        variants_ref=np.array(["A", "G"], dtype=object),
        variants_alt=np.array(["C", "T"], dtype=object),
    )

    flare_path = tmp_path / "toy_path.anc.vcf.gz"
    laiobj = LocalAncestryObject(
        haplotypes=[f"{sid}.{phase}" for sid in sample_ids for phase in (0, 1)],
        samples=sample_ids,
        ancestry_map=ancestry_map,
        chromosomes=chromosomes,
        physical_pos=np.column_stack([positions, positions]),
        lai=lai,
    )
    FLAREWriter(laiobj, flare_path, snpobj=snp_for_flare).write()

    eager_freq, eager_counts = snp_eager.allele_freq(
        ancestry=1,
        laiobj=laiobj,
        return_counts=True,
    )
    stream_freq, stream_counts = allele_freq_stream(
        snp,
        ancestry=1,
        laiobj=flare_path,
        chunk_size=1,
        lai_chunk_size=1,
        return_counts=True,
    )

    np.testing.assert_allclose(stream_freq, eager_freq)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_allele_freq_pseudohaploid():
    # 2 samples, 4 SNPs
    # Sample 0 is diploid: has hets
    # Sample 1 is pseudohaploid: 0s and 2s only, no 1s
    gt = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [2.0, 2.0],
        ]
    )
    snp = SNPObject(genotypes=gt)

    # 1. Default (pseudohaploid=False)
    freq_false, counts_false = snp.allele_freq(pseudohaploid=False, return_counts=True)
    # S0: 1, 0, 1, 2 (all hap_count=2)
    # S1: 0, 2, 0, 2 (all hap_count=2)
    # Total hap_count = 4 for all SNPs
    # Total alt: 1, 2, 1, 4. Freqs: 0.25, 0.5, 0.25, 1.0
    np.testing.assert_allclose(freq_false, np.array([0.25, 0.5, 0.25, 1.0]))
    np.testing.assert_array_equal(counts_false, np.array([4, 4, 4, 4]))

    # 2. pseudohaploid=True
    # S0 has '1's, treated as diploid (hap_count=2)
    # S1 has no '1's, treated as haploid (hap_count=1). Its alt becomes 0, 1, 0, 1.
    # Total hap_count = 3
    # Total alt: 1+0=1, 0+1=1, 1+0=1, 2+1=3. Freqs: 1/3, 1/3, 1/3, 1.0
    freq_true, counts_true = snp.allele_freq(pseudohaploid=True, return_counts=True)
    np.testing.assert_allclose(freq_true, np.array([1/3, 1/3, 1/3, 1.0]))
    np.testing.assert_array_equal(counts_true, np.array([3, 3, 3, 3]))

    # 3. 3D genotypes
    gt_3d = np.array(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[1, 1], [1, 1]],
        ],
        dtype=float,
    )
    snp_3d = SNPObject(genotypes=gt_3d)
    
    freq_false_3d, counts_false_3d = snp_3d.allele_freq(pseudohaploid=False, return_counts=True)
    np.testing.assert_allclose(freq_false_3d, np.array([0.25, 0.5, 0.25, 1.0]))
    np.testing.assert_array_equal(counts_false_3d, np.array([4, 4, 4, 4]))

    freq_true_3d, counts_true_3d = snp_3d.allele_freq(pseudohaploid=True, return_counts=True)
    np.testing.assert_allclose(freq_true_3d, np.array([1/3, 1/3, 1/3, 1.0]))
    np.testing.assert_array_equal(counts_true_3d, np.array([3, 3, 3, 3]))
