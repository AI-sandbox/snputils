import struct

import numpy as np
import pytest

from snputils import BCFReader, read_bcf, read_snp
from snputils._utils.genotypes import sum_diploid_genotypes
from snputils.snp.io.read.bcf import (
    _BCFHeader,
    _batch_decode_gt,
    _build_indiv_offsets,
    _decode_gt_array,
    _read_bgzf_or_gzip,
)
from snputils.snp.io.write.bcf import _encode_typed_int_list, _encode_typed_string


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
        genotype_mode="phased",
        sample_ids=["HG00100", "HG00096"],
        variant_idxs=[0, 2],
    )

    np.testing.assert_array_equal(snpobj.samples, np.array(["HG00100", "HG00096"], dtype=object))
    np.testing.assert_array_equal(snpobj.genotypes, snpobj_vcf.genotypes[[0, 2]][:, [3, 0], :])


def test_bcf_reader_supports_region_filtering(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(
        genotype_mode="phased",
        region="22:10526445-10526445",
        sample_idxs=[0],
    )

    assert snpobj.genotypes.shape == (1, 1, 2)
    np.testing.assert_array_equal(snpobj.variants_pos, np.array([10526445], dtype=np.int64))
    np.testing.assert_array_equal(snpobj.genotypes[0, 0], snpobj_vcf.genotypes[2, 0])


def test_bcf_reader_supports_dosage_genotypes(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(genotype_mode="dosage", variant_idxs=[0, 1, 2])

    np.testing.assert_array_equal(snpobj.genotypes, sum_diploid_genotypes(snpobj_vcf.genotypes[:3]))


def test_bcf_reader_gt_only_sample_selection(data_path, snpobj_vcf):
    reader = BCFReader(data_path + "/bcf/subset.bcf")
    snpobj = reader.read(
        fields=["GT"],
        sample_idxs=[3, 0],
        genotype_mode="dosage",
    )
    parallel = reader.read(
        fields=["GT"],
        sample_idxs=[3, 0],
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
        threads=2,
    )

    expected = sum_diploid_genotypes(snpobj_vcf.genotypes[:, [3, 0], :])
    np.testing.assert_array_equal(snpobj.genotypes, expected)
    np.testing.assert_array_equal(parallel.genotypes, expected)
    assert snpobj.samples is None

    serial_phased = reader.read(
        fields=["GT"],
        sample_idxs=[3, 0],
        genotype_mode="phased",
    )
    parallel_phased = reader.read(
        fields=["GT"],
        sample_idxs=[3, 0],
        genotype_mode="phased",
        threads=2,
    )
    np.testing.assert_array_equal(parallel_phased.genotypes, serial_phased.genotypes)
    np.testing.assert_array_equal(parallel_phased.genotypes, snpobj_vcf.genotypes[:, [3, 0], :])

    with pytest.raises(ValueError, match="at least 1"):
        reader.read(fields=["GT"], genotype_mode="dosage", threads=0)
    parallel_with_metadata = reader.read(
        fields=["GT", "POS", "IID"],
        sample_idxs=[3, 0],
        genotype_mode="dosage",
        chromosome_ploidy="autosomal",
        threads=2,
    )
    np.testing.assert_array_equal(parallel_with_metadata.genotypes, expected)
    np.testing.assert_array_equal(parallel_with_metadata.variants_pos, snpobj_vcf.variants_pos)
    np.testing.assert_array_equal(
        parallel_with_metadata.samples,
        np.array(["HG00100", "HG00096"], dtype=object),
    )


def test_bcf_reader_core_field_subset(data_path, snpobj_vcf):
    snpobj = BCFReader(data_path + "/bcf/subset.bcf").read(
        fields=["GT", "POS", "ID"],
        sample_idxs=[3, 0],
        genotype_mode="dosage",
    )

    expected_gt = sum_diploid_genotypes(snpobj_vcf.genotypes[:, [3, 0], :])
    np.testing.assert_array_equal(snpobj.genotypes, expected_gt)
    np.testing.assert_array_equal(snpobj.variants_pos, snpobj_vcf.variants_pos)
    np.testing.assert_array_equal(snpobj.variants_id, snpobj_vcf.variants_id)
    assert snpobj.samples is None
    assert snpobj.variants_ref is None


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


def test_bgzf_extra_subfield_length_cannot_exceed_xlen(tmp_path):
    path = tmp_path / "malformed.bcf"
    xlen = 4
    header = b"\x1f\x8b\x08\x04" + b"\x00\x00\x00\x00" + b"\x00\xff" + xlen.to_bytes(2, "little")
    path.write_bytes(header + b"ZZ" + (10).to_bytes(2, "little"))

    with pytest.raises(ValueError, match="subfield length extends beyond XLEN"):
        _read_bgzf_or_gzip(path)


def test_build_indiv_offsets_rejects_truncated_record_headers():
    with pytest.raises(ValueError, match="record header is truncated"):
        _build_indiv_offsets(b"\x00" * 7, 0)

    one_empty_record_then_truncated_header = struct.pack("<II", 0, 0) + b"\x00" * 7
    with pytest.raises(ValueError, match="record header is truncated"):
        _build_indiv_offsets(one_empty_record_then_truncated_header, 0)


def test_batch_decode_haploid_dosage_preserves_values():
    data = bytes([2, 4, 0])
    observed = _batch_decode_gt(
        data=data,
        indiv_offsets=np.array([0], dtype=np.int64),
        gt_data_rel_offset=0,
        n_vals=1,
        type_size=1,
        n_samples=3,
        n_records=1,
        sample_index_array=np.array([0, 1, 2], dtype=int),
        return_dosage=True,
    )

    np.testing.assert_array_equal(observed, np.array([[0, 1, -1]], dtype=np.int8))


def test_python_gt_decoders_treat_int8_vector_end_as_missing():
    gt_values = bytes([2, 0x81, 4, 3])
    expected = np.array([[[0, -1], [1, 0]]], dtype=np.int8)

    per_record = _decode_gt_array(
        gt_values,
        0,
        n_samples=2,
        n_vals=2,
        type_size=1,
        require_phase=True,
    )
    batched = _batch_decode_gt(
        data=gt_values,
        indiv_offsets=np.array([0], dtype=np.int64),
        gt_data_rel_offset=0,
        n_vals=2,
        type_size=1,
        n_samples=2,
        n_records=1,
        sample_index_array=np.array([0, 1], dtype=int),
        return_dosage=False,
    )

    np.testing.assert_array_equal(per_record, expected[0])
    np.testing.assert_array_equal(batched, expected)


def test_c_decode_gt_treats_int8_vector_end_as_missing():
    _bcf = pytest.importorskip("snputils.snp.io.read._bcf")
    gt_values = bytes([2, 0x81, 4, 3])
    record = struct.pack("<II", 0, len(gt_values)) + gt_values
    data = record * 2
    expected = np.tile(
        np.array([[[0, -1], [1, 0]]], dtype=np.int8),
        (2, 1, 1),
    )

    for threads in (1, 2):
        gt_buffer, n_records = _bcf.decode_gt(
            data,
            0,
            0,
            2,
            2,
            1,
            len(gt_values),
            None,
            False,
            threads,
        )
        observed = np.frombuffer(gt_buffer, dtype=np.int8).reshape(n_records, 2, 2)
        np.testing.assert_array_equal(observed, expected)


def test_varying_format_layout_warns_and_uses_serial_gt_decoder():
    samples = np.array(["s1", "s2"], dtype=object)
    header = _BCFHeader(
        samples=samples,
        contigs={0: "1"},
        filters={},
        info={},
        formats={1: {"ID": "GT"}, 2: {"ID": "DP"}},
    )

    def format_field(key, values):
        return _encode_typed_int_list([key]) + b"\x21" + bytes(values)

    shared = struct.pack("<iiIfII", 0, 9, 1, 12.5, 2 << 16, (2 << 24) | 2)
    first_indiv = format_field(1, [2, 3, 4, 5]) + format_field(2, [20, 20, 20, 20])
    second_indiv = format_field(2, [20, 20, 20, 20]) + format_field(1, [4, 3, 2, 5])
    data = (
        struct.pack("<II", len(shared), len(first_indiv))
        + shared
        + first_indiv
        + struct.pack("<II", len(shared), len(second_indiv))
        + shared
        + second_indiv
    )

    with pytest.warns(RuntimeWarning, match="FORMAT layouts vary"):
        observed = BCFReader("unused.bcf")._read_all(
            data,
            0,
            header,
            samples,
            np.array([0, 1]),
            ["GT"],
            return_dosage=False,
            detect_non_diploid=False,
            threads=2,
        )
    serial = BCFReader("unused.bcf")._read_all(
        data,
        0,
        header,
        samples,
        np.array([0, 1]),
        ["GT"],
        return_dosage=False,
        detect_non_diploid=False,
        threads=1,
    )

    expected = np.array(
        [
            [[0, 0], [1, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(observed.genotypes, expected)
    np.testing.assert_array_equal(serial.genotypes, expected)


def test_c_decode_gt_haploid_dosage_preserves_values():
    _bcf = pytest.importorskip("snputils.snp.io.read._bcf")
    data = struct.pack("<II", 0, 3) + bytes([2, 4, 0])

    gt_buffer, n_records = _bcf.decode_gt(data, 0, 0, 3, 1, 1, 3, None, True)
    observed = np.frombuffer(gt_buffer, dtype=np.int8).reshape(n_records, 3)
    parallel_buffer, parallel_n_records = _bcf.decode_gt(data, 0, 0, 3, 1, 1, 3, None, True, 2)
    parallel = np.frombuffer(parallel_buffer, dtype=np.int8).reshape(parallel_n_records, 3)

    np.testing.assert_array_equal(observed, np.array([[0, 1, -1]], dtype=np.int8))
    np.testing.assert_array_equal(parallel, observed)


def test_c_decode_core_haploid_dosage_and_missing_pass_fallback():
    _bcf = pytest.importorskip("snputils.snp.io.read._bcf")
    gt_values = bytes([2, 4, 0])
    indiv = _encode_typed_int_list([1]) + b"\x11" + gt_values
    shared = (
        struct.pack("<iiIfII", 0, 9, 1, 12.5, (2 << 16), (1 << 24) | 3)
        + _encode_typed_string("")
        + _encode_typed_string("A")
        + _encode_typed_string("C")
        + _encode_typed_int_list([0])
    )
    data = struct.pack("<II", len(shared), len(indiv)) + shared + indiv
    gt_rel_offset = len(_encode_typed_int_list([1])) + 1

    (
        gt_buffer,
        _chrom_buffer,
        _pos_buffer,
        _qual_buffer,
        filter_buffer,
        _ids,
        _refs,
        _alts,
        n_records,
    ) = _bcf.decode_core(data, 0, gt_rel_offset, 3, 1, 1, len(indiv), None, True, -1)
    observed = np.frombuffer(gt_buffer, dtype=np.int8).reshape(n_records, 3)

    np.testing.assert_array_equal(observed, np.array([[0, 1, -1]], dtype=np.int8))
    np.testing.assert_array_equal(np.frombuffer(filter_buffer, dtype=np.bool_), np.array([False]))
