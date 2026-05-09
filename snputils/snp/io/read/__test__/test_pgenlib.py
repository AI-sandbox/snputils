import numpy as np

from snputils.snp.io.read import _pgenlib


class _FakePgenReader:
    def __init__(self, alleles: np.ndarray):
        self._alleles = alleles
        self.range_calls = []
        self.list_calls = []

    def read_alleles_range(self, start: int, stop: int, out: np.ndarray) -> None:
        self.range_calls.append((start, stop))
        out[:] = self._alleles[start:stop]

    def read_alleles_list(self, variant_idxs: np.ndarray, out: np.ndarray) -> None:
        idxs = np.asarray(variant_idxs, dtype=np.uint32)
        self.list_calls.append(idxs.copy())
        out[:] = self._alleles[idxs]


def test_chunked_separate_strands_contiguous_range_and_list_fallback_match(monkeypatch):
    num_samples = 3
    allele_cols = 2 * num_samples
    all_alleles = np.arange(12 * allele_cols, dtype=np.int32).reshape(12, allele_cols)
    variant_idxs = np.arange(2, 10, dtype=np.uint32)
    num_variants = variant_idxs.size

    monkeypatch.setattr(_pgenlib, "PHASED_ALLELE_FULL_READ_BYTES", 1)

    reader_with_range = _FakePgenReader(all_alleles)
    gt_from_range = _pgenlib.read_separate_strands(
        reader_with_range, variant_idxs, num_variants, num_samples
    )

    monkeypatch.setattr(_pgenlib, "_is_contiguous_variant_chunk", lambda _: False)
    reader_with_list = _FakePgenReader(all_alleles)
    gt_from_list = _pgenlib.read_separate_strands(
        reader_with_list, variant_idxs, num_variants, num_samples
    )

    assert reader_with_range.range_calls == [(2, 10)]
    assert reader_with_range.list_calls == []
    assert reader_with_list.range_calls == []
    assert len(reader_with_list.list_calls) == 1
    np.testing.assert_array_equal(reader_with_list.list_calls[0], variant_idxs)

    assert gt_from_range.shape == (num_variants, num_samples, 2)
    assert gt_from_range.dtype == np.int8
    np.testing.assert_array_equal(gt_from_range, gt_from_list)
