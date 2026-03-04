import numpy as np

from snputils.snp.io.read import BEDReader, PGENReader
from snputils.snp.io.read.vcf import VCFReaderPolars
from snputils.stats import allele_freq_stream

MAX_VARIANTS = 10_000


def _concat_chunks(chunks):
    calldata_gt = np.concatenate([chunk.calldata_gt for chunk in chunks], axis=0)
    variants_id = np.concatenate([chunk.variants_id for chunk in chunks], axis=0)
    variants_pos = np.concatenate([chunk.variants_pos for chunk in chunks], axis=0)
    variants_ref = np.concatenate([chunk.variants_ref for chunk in chunks], axis=0)
    variants_alt = np.concatenate([chunk.variants_alt for chunk in chunks], axis=0)
    variants_chrom = np.concatenate([chunk.variants_chrom for chunk in chunks], axis=0)
    return calldata_gt, variants_id, variants_pos, variants_ref, variants_alt, variants_chrom


def _first_n_variant_idxs(reader, max_variants: int = MAX_VARIANTS) -> np.ndarray:
    n_total = int(reader.read(fields=["ID"]).n_snps)
    return np.arange(min(max_variants, n_total), dtype=np.uint32)


def _write_vcf_head(source_path: str, output_path, n_variants: int = MAX_VARIANTS) -> None:
    written = 0
    with open(source_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
            if line.startswith("#"):
                continue
            written += 1
            if written >= int(n_variants):
                break


def test_bed_iter_read_reconstructs_eager_object(data_path):
    reader = BEDReader(data_path + "/bed/subset")
    subset = _first_n_variant_idxs(reader)

    eager_subset = reader.read(sum_strands=False, variant_idxs=subset)
    chunks = list(reader.iter_read(sum_strands=False, variant_idxs=subset, chunk_size=300))

    assert len(chunks) > 1
    gt, var_id, var_pos, var_ref, var_alt, var_chrom = _concat_chunks(chunks)

    np.testing.assert_array_equal(gt, eager_subset.calldata_gt)
    np.testing.assert_array_equal(var_id, eager_subset.variants_id)
    np.testing.assert_array_equal(var_pos, eager_subset.variants_pos)
    np.testing.assert_array_equal(var_ref, eager_subset.variants_ref)
    np.testing.assert_array_equal(var_alt, eager_subset.variants_alt)
    np.testing.assert_array_equal(var_chrom, eager_subset.variants_chrom)
    np.testing.assert_array_equal(chunks[0].samples, eager_subset.samples)


def test_bed_iter_read_matches_eager_for_unsorted_duplicate_variant_idxs(data_path):
    reader = BEDReader(data_path + "/bed/subset")
    subset = np.array([5, 0, 3, 10, 2, 5, 3], dtype=np.uint32)

    eager_subset = reader.read(sum_strands=False, variant_idxs=subset)
    chunks = list(reader.iter_read(sum_strands=False, variant_idxs=subset, chunk_size=2))

    assert len(chunks) > 1
    gt, var_id, var_pos, var_ref, var_alt, var_chrom = _concat_chunks(chunks)

    np.testing.assert_array_equal(gt, eager_subset.calldata_gt)
    np.testing.assert_array_equal(var_id, eager_subset.variants_id)
    np.testing.assert_array_equal(var_pos, eager_subset.variants_pos)
    np.testing.assert_array_equal(var_ref, eager_subset.variants_ref)
    np.testing.assert_array_equal(var_alt, eager_subset.variants_alt)
    np.testing.assert_array_equal(var_chrom, eager_subset.variants_chrom)


def test_bed_iter_read_matches_eager_for_unsorted_duplicate_variant_ids(data_path):
    reader = BEDReader(data_path + "/bed/subset")
    meta = reader.read(fields=["#CHROM", "POS"])
    subset = np.array([5, 0, 3, 10, 2, 5, 3], dtype=np.uint32)
    variant_ids = np.asarray(
        [f"{meta.variants_chrom[i]}:{meta.variants_pos[i]}" for i in subset],
        dtype=object,
    )

    eager_subset = reader.read(sum_strands=False, variant_ids=variant_ids)
    chunks = list(reader.iter_read(sum_strands=False, variant_ids=variant_ids, chunk_size=2))

    assert len(chunks) > 1
    gt, var_id, var_pos, var_ref, var_alt, var_chrom = _concat_chunks(chunks)

    np.testing.assert_array_equal(gt, eager_subset.calldata_gt)
    np.testing.assert_array_equal(var_id, eager_subset.variants_id)
    np.testing.assert_array_equal(var_pos, eager_subset.variants_pos)
    np.testing.assert_array_equal(var_ref, eager_subset.variants_ref)
    np.testing.assert_array_equal(var_alt, eager_subset.variants_alt)
    np.testing.assert_array_equal(var_chrom, eager_subset.variants_chrom)


def test_pgen_iter_read_reconstructs_subset_eager_object(data_path):
    reader = PGENReader(data_path + "/pgen/subset")
    subset = np.array([0, 1, 2, 5, 9, 20, 21, 45, 80], dtype=np.uint32)

    eager_subset = reader.read(sum_strands=False, variant_idxs=subset)
    chunks = list(reader.iter_read(sum_strands=False, variant_idxs=subset, chunk_size=3))

    assert len(chunks) > 1
    gt, var_id, var_pos, var_ref, var_alt, var_chrom = _concat_chunks(chunks)

    np.testing.assert_array_equal(gt, eager_subset.calldata_gt)
    np.testing.assert_array_equal(var_id, eager_subset.variants_id)
    np.testing.assert_array_equal(var_pos, eager_subset.variants_pos)
    np.testing.assert_array_equal(var_ref, eager_subset.variants_ref)
    np.testing.assert_array_equal(var_alt, eager_subset.variants_alt)
    np.testing.assert_array_equal(var_chrom, eager_subset.variants_chrom)
    np.testing.assert_array_equal(chunks[0].samples, eager_subset.samples)


def test_pgen_iter_read_matches_eager_for_unsorted_duplicate_variant_idxs(data_path):
    reader = PGENReader(data_path + "/pgen/subset")
    subset = np.array([5, 0, 3, 10, 2, 5, 3], dtype=np.uint32)

    eager_subset = reader.read(sum_strands=False, variant_idxs=subset)
    chunks = list(reader.iter_read(sum_strands=False, variant_idxs=subset, chunk_size=2))

    assert len(chunks) > 1
    gt, var_id, var_pos, var_ref, var_alt, var_chrom = _concat_chunks(chunks)

    np.testing.assert_array_equal(gt, eager_subset.calldata_gt)
    np.testing.assert_array_equal(var_id, eager_subset.variants_id)
    np.testing.assert_array_equal(var_pos, eager_subset.variants_pos)
    np.testing.assert_array_equal(var_ref, eager_subset.variants_ref)
    np.testing.assert_array_equal(var_alt, eager_subset.variants_alt)
    np.testing.assert_array_equal(var_chrom, eager_subset.variants_chrom)


def test_allele_freq_stream_from_pgen_reader_matches_eager(data_path):
    reader = PGENReader(data_path + "/pgen/subset")
    subset = _first_n_variant_idxs(reader)

    stream_af, stream_counts = allele_freq_stream(
        reader,
        chunk_size=250,
        variant_idxs=subset,
        sum_strands=False,
        return_counts=True,
    )
    eager_subset = reader.read(sum_strands=False, variant_idxs=subset)
    eager_af, eager_counts = eager_subset.allele_freq(return_counts=True)

    np.testing.assert_allclose(stream_af, eager_af)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_vcf_polars_iter_read_reconstructs_eager_object(data_path, tmp_path):
    mini_vcf = tmp_path / "subset_10k.vcf"
    _write_vcf_head(data_path + "/vcf/subset.vcf", mini_vcf, n_variants=MAX_VARIANTS)
    reader = VCFReaderPolars(str(mini_vcf))
    eager = reader.read(sum_strands=False)
    chunks = list(reader.iter_read(sum_strands=False, chunk_size=300))

    assert len(chunks) > 1
    gt, var_id, var_pos, var_ref, var_alt, var_chrom = _concat_chunks(chunks)

    np.testing.assert_array_equal(gt, eager.calldata_gt)
    np.testing.assert_array_equal(var_id, eager.variants_id)
    np.testing.assert_array_equal(var_pos, eager.variants_pos)
    np.testing.assert_array_equal(var_ref, eager.variants_ref)
    np.testing.assert_array_equal(var_alt, eager.variants_alt)
    np.testing.assert_array_equal(var_chrom, eager.variants_chrom)
    np.testing.assert_array_equal(chunks[0].samples, eager.samples)


def test_allele_freq_stream_from_bed_reader_matches_eager(data_path):
    reader = BEDReader(data_path + "/bed/subset")
    subset = _first_n_variant_idxs(reader)

    stream_af, stream_counts = allele_freq_stream(
        reader,
        chunk_size=250,
        variant_idxs=subset,
        sum_strands=False,
        return_counts=True,
    )
    eager_subset = reader.read(sum_strands=False, variant_idxs=subset)
    eager_af, eager_counts = eager_subset.allele_freq(return_counts=True)

    np.testing.assert_allclose(stream_af, eager_af)
    np.testing.assert_array_equal(stream_counts, eager_counts)


def test_allele_freq_stream_from_vcf_polars_reader_matches_eager(data_path, tmp_path):
    mini_vcf = tmp_path / "subset_10k.vcf"
    _write_vcf_head(data_path + "/vcf/subset.vcf", mini_vcf, n_variants=MAX_VARIANTS)
    reader = VCFReaderPolars(str(mini_vcf))
    eager = reader.read(sum_strands=False)

    stream_af, stream_counts = allele_freq_stream(
        reader,
        chunk_size=250,
        sum_strands=False,
        return_counts=True,
    )
    eager_af, eager_counts = eager.allele_freq(return_counts=True)

    np.testing.assert_allclose(stream_af, eager_af)
    np.testing.assert_array_equal(stream_counts, eager_counts)
