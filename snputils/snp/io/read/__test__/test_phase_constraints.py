from pathlib import Path

import numpy as np
import pgenlib as pg
import pytest
import snputils.snp.io.read.bed as bed_module
import snputils.snp.io.read.pgen as pgen_module

from snputils import read_pgen
from snputils.snp.io.read import _pgenlib
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read import BCFReader, BEDReader, PGENReader
from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars
from snputils.snp.io.write.bed import BEDWriter
from snputils.snp.io.write.pgen import PGENWriter


def _toy_snpobj(genotypes: np.ndarray) -> SNPObject:
    return SNPObject(
        genotypes=genotypes,
        samples=np.array(["s1", "s2"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([100, 200]),
    )


def test_vcf_unphased_gt_defaults_to_dosage_and_rejects_phased_mode(tmp_path: Path):
    vcf_path = tmp_path / "unphased.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t0/0\t1/1\n"
    )

    # Default auto mode falls back to dosages for unphased GT.
    snpobj = VCFReader(vcf_path).read()
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[1, 1], [0, 2]], dtype=np.int8),
    )

    snpobj_polars = VCFReaderPolars(vcf_path).read()
    np.testing.assert_array_equal(
        snpobj_polars.genotypes,
        np.array([[1, 1], [0, 2]], dtype=np.int8),
    )

    # Explicit genotype_mode="phased" rejects unphased GT.
    with pytest.raises(ValueError, match="unphased VCF genotypes"):
        VCFReader(vcf_path).read(genotype_mode="phased")

    with pytest.raises(ValueError, match="unphased VCF genotypes"):
        VCFReaderPolars(vcf_path).read(genotype_mode="phased")

    # Explicit genotype_mode="dosage" loads dosages
    snpobj = VCFReader(vcf_path).read(genotype_mode="dosage")
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[1, 1], [0, 2]], dtype=np.int8),
    )


def test_vcf_phased_gt_preserves_allele_calls(tmp_path: Path):
    vcf_path = tmp_path / "phased.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
    )

    snpobj = VCFReader(vcf_path).read()
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[[0, 1], [1, 0]]], dtype=np.int8),
    )


@pytest.mark.parametrize("reader_cls", [VCFReader, VCFReaderPolars])
def test_vcf_dosage_rejects_multiallelic_even_without_alt_field(tmp_path: Path, reader_cls):
    vcf_path = tmp_path / "multiallelic.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "1\t100\trs1\tA\tG,T\t.\tPASS\t.\tGT\t1|2\n"
    )

    with pytest.raises(ValueError, match="dosage.*biallelic"):
        reader_cls(vcf_path).read(fields=["GT"], genotype_mode="dosage")

    phased = reader_cls(vcf_path).read(fields=["GT"], genotype_mode="phased")
    np.testing.assert_array_equal(phased.genotypes, np.array([[[1, 2]]], dtype=np.int8))


def test_vcf_dosage_preserves_one_missing_sentinel(tmp_path: Path):
    vcf_path = tmp_path / "missing.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|.\t./.\t1|1\n"
    )

    snpobj = VCFReader(vcf_path).read(genotype_mode="dosage")
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[-1, -1, 2]], dtype=np.int8),
    )


def test_bed_modes(data_path):
    default = BEDReader(data_path + "/bed/subset").read(fields=["GT"])
    automatic = BEDReader(data_path + "/bed/subset").read(fields=["GT"], genotype_mode="auto")
    np.testing.assert_array_equal(default.genotypes, automatic.genotypes)

    parallel = BEDReader(data_path + "/bed/subset").read(
        fields=["GT"],
        genotype_mode="dosage",
        threads=4,
    )
    np.testing.assert_array_equal(default.genotypes, parallel.genotypes)

    with pytest.raises(ValueError, match="BED/BIM/FAM does not store phase"):
        BEDReader("cohort.bed").read(genotype_mode="phased")

    with pytest.raises(ValueError, match="threads must be at least 1"):
        BEDReader("cohort.bed").read(threads=0)

    with pytest.raises(TypeError, match="threads must be an integer"):
        BEDReader("cohort.bed").read(threads=1.5)


def test_bed_parallel_non_diploid_correction_matches_serial(tmp_path: Path, monkeypatch):
    prefix = tmp_path / "mixed"
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0, 2, -1],
                [1, 1, 2],
                [2, 0, -1],
                [0, 1, 2],
            ],
            dtype=np.int8,
        ),
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C", "G", "T"], dtype=object),
        variants_alt=np.array(["G", "T", "A", "C"], dtype=object),
        variants_chrom=np.array(["X", "1", "Y", "MT"], dtype=object),
        variants_id=np.array(["rs1", "rs2", "rs3", "rs4"], dtype=object),
        variants_pos=np.array([100, 200, 300, 400]),
    )
    BEDWriter(snpobj, str(prefix)).write()

    serial = BEDReader(prefix).read(genotype_mode="dosage", threads=1)
    calls = []
    read_parallel = bed_module._read_phased_parallel

    def tracked_read_parallel(*args, **kwargs):
        calls.append((kwargs["num_variants"], kwargs["threads"]))
        return read_parallel(*args, **kwargs)

    monkeypatch.setattr(bed_module, "_read_phased_parallel", tracked_read_parallel)
    parallel = BEDReader(prefix).read(genotype_mode="dosage", threads=4)

    assert calls == [(3, 4)]
    np.testing.assert_array_equal(parallel.genotypes, serial.genotypes)


def test_pgen_unphased_hardcalls_reject_phased_mode(tmp_path: Path):
    prefix = tmp_path / "unphased"
    snpobj = _toy_snpobj(
        np.array(
            [
                [1, 1],
                [0, 2],
            ],
            dtype=np.int8,
        )
    )
    PGENWriter(snpobj, str(prefix)).write()

    observed = PGENReader(prefix).read()
    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)

    with pytest.raises(ValueError, match="hardcall phase information"):
        PGENReader(prefix).read(genotype_mode="phased")


@pytest.fixture
def mixed_phase_prefix(tmp_path):
    prefix = tmp_path / "mixed_phase"
    with pg.PgenWriter(str(prefix.with_suffix(".pgen")).encode(), sample_ct=2, variant_ct=3,
                       hardcall_phase_present=True, allele_ct_limit=3) as writer:
        writer.append_partially_phased(np.array([1, 0, 0, 1], dtype=np.int32),
                                       np.array([True, False]))
        writer.append_alleles(np.array([1, 0, 0, 0], dtype=np.int32), all_phased=True)
        writer.append_alleles(np.array([2, 1, 0, 2], dtype=np.int32), all_phased=True, allele_ct=3)
    prefix.with_suffix(".psam").write_text("#IID\ns1\ns2\n")
    prefix.with_suffix(".pvar").write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n1\t100\tv1\tA\tG\n1\t200\tv2\tC\tT\n1\t300\tv3\tA\tG,T\n"
    )
    return prefix


def test_pgen_assumes_uniform_phase_by_default(mixed_phase_prefix, monkeypatch):
    def unexpected_check(*args):
        pytest.fail("Default PGEN reads should skip per-call phase validation.")

    monkeypatch.setattr(_pgenlib, "_raise_if_unphased_heterozygote", unexpected_check)
    observed = read_pgen(mixed_phase_prefix, fields=["GT"]).genotypes
    assert observed.shape == (3, 2, 2)
    np.testing.assert_array_equal(observed[0, 0], [1, 0])
    # s2 has one copy of each allele, with unknown haplotype order.
    np.testing.assert_array_equal(np.sort(observed[0, 1]), [0, 1])
    reader = PGENReader(mixed_phase_prefix)
    for threads in (1, 2):
        phased = reader.read(fields=["GT"], genotype_mode="phased", threads=threads)
        np.testing.assert_array_equal(phased.genotypes, observed)
    chunks = reader.iter_read(genotype_mode="phased", threads=2, chunk_size=2)
    np.testing.assert_array_equal(np.concatenate([chunk.genotypes for chunk in chunks]), observed)


def test_pgen_checks_phase_in_selected_calls(mixed_phase_prefix):
    reader = PGENReader(mixed_phase_prefix)
    for threads in (1, 2):
        options = dict(fields=["GT"], genotype_mode="phased", threads=threads, assume_uniform_phase=False)
        with pytest.raises(ValueError, match="unphased heterozygous"):
            reader.read(sample_ids=["s2"], **options)
        phased = reader.read(sample_ids=["s1"], **options)
        np.testing.assert_array_equal(phased.genotypes, [[[1, 0]], [[1, 0]], [[2, 1]]])
        with pytest.raises(ValueError, match="unphased heterozygous"):
            next(reader.iter_read(sample_ids=["s2"], chunk_size=2, **options))
    with pytest.raises(ValueError, match="lack complete phase.*multiallelic"):
        reader.read(sample_ids=["s2"], assume_uniform_phase=False)
    dosage = reader.read(sample_ids=["s2"], variant_ids=["v1", "v2"], assume_uniform_phase=False)
    np.testing.assert_array_equal(dosage.genotypes, [[1], [0]])
    checked = reader.read(variant_ids=["v2"], assume_uniform_phase=False)
    np.testing.assert_array_equal(checked.genotypes, [[[1, 0], [0, 0]]])


def test_pgen_dosage_preserves_negative_missing_sentinel(tmp_path: Path):
    prefix = tmp_path / "missing"
    genotypes = np.array(
        [
            [0, -1],
            [2, 1],
        ],
        dtype=np.int8,
    )
    PGENWriter(_toy_snpobj(genotypes), str(prefix)).write()

    observed = PGENReader(prefix).read(genotype_mode="dosage")
    np.testing.assert_array_equal(observed.genotypes[:, 0], genotypes[:, 0])
    assert observed.genotypes[0, 1] < 0
    np.testing.assert_array_equal(observed.genotypes[1, 1], genotypes[1, 1])


def test_save_vcf_preserves_missing_genotypes_from_phased_pgen(tmp_path: Path):
    prefix = tmp_path / "phased"
    source = _toy_snpobj(np.array([[[0, 1], [-1, -1]], [[1, 0], [1, 1]]], dtype=np.int8))
    source.save_pgen(prefix)
    loaded = PGENReader(prefix).read(genotype_mode="phased")
    original = loaded.genotypes.copy()

    path = tmp_path / "converted.vcf"
    loaded.save_vcf(path)

    records = [line.split("\t")[9:] for line in path.read_text().splitlines() if not line.startswith("#")]
    assert records == [["0/1", "./."], ["1/0", "1/1"]]
    observed = VCFReader(path).read(genotype_mode="dosage")
    np.testing.assert_array_equal(observed.genotypes, [[1, -1], [1, 2]])
    np.testing.assert_array_equal(loaded.genotypes, original)


def test_pgen_parallel_dosage_matches_serial(tmp_path: Path):
    prefix = tmp_path / "parallel"
    genotypes = np.array(
        [
            [0, 1],
            [2, -1],
            [1, 2],
            [-1, 0],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(
        genotypes=genotypes,
        samples=np.array(["s1", "s2"], dtype=object),
        variants_ref=np.array(["A", "C", "G", "T"], dtype=object),
        variants_alt=np.array(["G", "T", "A", "C"], dtype=object),
        variants_chrom=np.array(["1", "1", "1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2", "rs3", "rs4"], dtype=object),
        variants_pos=np.array([100, 200, 300, 400]),
    )
    PGENWriter(snpobj, str(prefix)).write()

    serial = PGENReader(prefix).read(
        fields=["GT"],
        genotype_mode="dosage",
        threads=1,
    )
    parallel = PGENReader(prefix).read(
        fields=["GT"],
        genotype_mode="dosage",
        threads=2,
    )
    np.testing.assert_array_equal(parallel.genotypes, serial.genotypes)
    np.testing.assert_array_equal(
        parallel.genotypes,
        np.where(genotypes < 0, -9, genotypes).astype(np.int8),
    )

    filtered_serial = PGENReader(prefix).read(
        fields=["GT"],
        variant_idxs=np.array([3, 0, 2], dtype=np.uint32),
        genotype_mode="dosage",
        threads=1,
    )
    filtered_parallel = PGENReader(prefix).read(
        fields=["GT"],
        variant_idxs=np.array([3, 0, 2], dtype=np.uint32),
        genotype_mode="dosage",
        threads=2,
    )
    np.testing.assert_array_equal(filtered_parallel.genotypes, filtered_serial.genotypes)

    with pytest.raises(ValueError, match="threads must be at least 1"):
        PGENReader(prefix).read(fields=["GT"], genotype_mode="dosage", threads=0)
    with pytest.raises(ValueError, match="explicit genotype_mode"):
        PGENReader(prefix).read(fields=["GT"], genotype_mode="auto", threads=2)


def test_pgen_parallel_non_diploid_correction_matches_serial(tmp_path: Path, monkeypatch):
    prefix = tmp_path / "mixed"
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [1, 1], [0, 1]],
                [[0, 1], [1, 0], [1, 1]],
                [[1, 1], [0, 0], [1, 0]],
                [[0, 0], [0, 1], [1, 1]],
            ],
            dtype=np.int8,
        ),
        samples=np.array(["s1", "s2", "s3"], dtype=object),
        variants_ref=np.array(["A", "C", "G", "T"], dtype=object),
        variants_alt=np.array(["G", "T", "A", "C"], dtype=object),
        variants_chrom=np.array(["X", "1", "Y", "MT"], dtype=object),
        variants_id=np.array(["rs1", "rs2", "rs3", "rs4"], dtype=object),
        variants_pos=np.array([100, 200, 300, 400]),
    )
    PGENWriter(snpobj, str(prefix)).write()

    serial = PGENReader(prefix).read(genotype_mode="dosage", threads=1)
    calls = []
    read_parallel = pgen_module._read_phased_parallel

    def tracked_read_parallel(*args, **kwargs):
        calls.append((kwargs["num_variants"], kwargs["threads"]))
        return read_parallel(*args, **kwargs)

    monkeypatch.setattr(pgen_module, "_read_phased_parallel", tracked_read_parallel)
    parallel = PGENReader(prefix).read(genotype_mode="dosage", threads=4)

    assert calls == [(3, 4)]
    np.testing.assert_array_equal(parallel.genotypes, serial.genotypes)


def test_pgen_phased_hardcalls_preserve_allele_calls(tmp_path: Path):
    prefix = tmp_path / "phased"
    genotypes = np.array(
        [
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.int8,
    )
    snpobj = _toy_snpobj(genotypes)
    PGENWriter(snpobj, str(prefix)).write()

    observed = PGENReader(prefix).read(genotype_mode="phased", threads=1)
    parallel = PGENReader(prefix).read(
        fields=["GT"],
        genotype_mode="phased",
        threads=2,
    )
    np.testing.assert_array_equal(observed.genotypes, genotypes)
    np.testing.assert_array_equal(parallel.genotypes, observed.genotypes)

    filtered_serial = PGENReader(prefix).read(
        fields=["GT"],
        variant_idxs=np.array([1, 0], dtype=np.uint32),
        genotype_mode="phased",
        threads=1,
    )
    filtered_parallel = PGENReader(prefix).read(
        fields=["GT"],
        variant_idxs=np.array([1, 0], dtype=np.uint32),
        genotype_mode="phased",
        threads=2,
    )
    np.testing.assert_array_equal(filtered_parallel.genotypes, filtered_serial.genotypes)


@pytest.mark.parametrize("mode", ["dosage", "phased"])
def test_pgen_assume_biallelic_skips_allele_checks(tmp_path, monkeypatch, mode):
    prefix = tmp_path / "biallelic"
    calls = np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]], dtype=np.int8)
    _toy_snpobj(calls).save_pgen(prefix)
    expected = read_pgen(prefix, fields=["GT"], genotype_mode=mode).genotypes

    def unexpected_check(*args, **kwargs):
        pytest.fail("Trusted genotype reads should skip PVAR allele-count checks.")

    monkeypatch.setattr(pgen_module, "_find_pvar_path", unexpected_check)
    options = dict(fields=["GT"], genotype_mode=mode, assume_biallelic=True)
    np.testing.assert_array_equal(read_pgen(prefix, **options).genotypes, expected)

    monkeypatch.undo()
    string_methods = type(pgen_module.pl.col("ALT").str)
    monkeypatch.setattr(string_methods, "count_matches", unexpected_check)
    chunks = PGENReader(prefix).iter_read(variant_idxs=[1, 0], chunk_size=1, **options)
    np.testing.assert_array_equal(np.concatenate([chunk.genotypes for chunk in chunks]), expected[::-1])


def test_pgen_assume_biallelic_requires_boolean(tmp_path):
    reader = PGENReader(tmp_path / "unused")
    with pytest.raises(TypeError, match="assume_biallelic"):
        reader.read(assume_biallelic="False")
    with pytest.raises(TypeError, match="assume_biallelic"):
        next(reader.iter_read(assume_biallelic="False"))


@pytest.mark.parametrize("threads", [1, 2])
def test_pgen_dosage_checks_only_selected_variants(tmp_path, threads):
    prefix = tmp_path / "mixed_alleles"
    with pg.PgenWriter(
        str(prefix.with_suffix(".pgen")).encode(), sample_ct=2, variant_ct=4,
        hardcall_phase_present=True, allele_ct_limit=3,
    ) as writer:
        for alleles, count in [([0, 1, 1, 1], 2), ([1, 2, 2, 2], 3),
                               ([0, 0, 0, 1], 2), ([0, 1, 0, 0], 2)]:
            writer.append_alleles(np.array(alleles, dtype=np.int32), all_phased=True, allele_ct=count)
    prefix.with_suffix(".pvar").write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n1\t10\tsnp1\tA\tG\n1\t20\tmulti\tA\tG,C\n"
        "1\t30\tsnp2\tA\tG\n1\t40\tsnp3\tA\tG\n"
    )
    prefix.with_suffix(".psam").write_text("#IID\ns1\ns2\n")
    reader = PGENReader(prefix)
    options = dict(fields=["GT"], genotype_mode="dosage", threads=threads)
    result = reader.read(variant_ids=["snp1", "snp2"], **options)
    np.testing.assert_array_equal(result.genotypes, [[1, 2], [0, 1]])
    chunks = reader.iter_read(chunk_size=1, variant_idxs=[2, 0], **options)
    np.testing.assert_array_equal(np.concatenate([c.genotypes for c in chunks]), [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="dosage.*biallelic"):
        reader.read(variant_ids=["multi"], **options)


def test_pgen_dosage_checks_declared_alleles_even_when_alt2_is_unobserved(tmp_path):
    prefix = tmp_path / "unobserved"
    source = _toy_snpobj(np.array([[0, 1], [1, 2]], dtype=np.int8))
    source.variants_alt = np.array(["G", "T,A"])
    source.save_pgen(prefix)
    reader = PGENReader(prefix)
    for selectors in ({}, {"variant_ids": ["rs2"]}):
        with pytest.raises(ValueError, match="dosage.*biallelic"):
            reader.read(fields=["GT"], genotype_mode="dosage", **selectors)
    selected = reader.read(fields=["GT"], variant_ids=["rs1"])
    np.testing.assert_array_equal(selected.genotypes, [[0, 1]])


def test_pgen_multiallelic_modes_and_parallel_allele_offsets(
    tmp_path: Path,
    monkeypatch,
):
    prefix = tmp_path / "multiallelic"
    with pg.PgenWriter(
        filename=str(prefix.with_suffix(".pgen")).encode(),
        sample_ct=2,
        variant_ct=4,
        hardcall_phase_present=True,
        allele_ct_limit=3,
    ) as writer:
        for _ in range(4):
            writer.append_alleles(
                np.array([0, 2, 1, 2], dtype=np.int32),
                all_phased=True,
                allele_ct=3,
            )
    prefix.with_suffix(".pvar").write_text(
        "#CHROM\tPOS\tID\tREF\tALT\n"
        "1\t100\trs1\tA\tG,T\n"
        "1\t200\trs2\tA\tG,T\n"
        "1\t300\trs3\tA\tG,T\n"
        "1\t400\trs4\tA\tG,T\n"
    )
    prefix.with_suffix(".psam").write_text("#IID\ns1\ns2\n")

    with pytest.raises(ValueError, match="dosage.*biallelic"):
        PGENReader(prefix).read(fields=["GT"], genotype_mode="dosage")
    with pytest.raises(ValueError, match="assume_biallelic=False"):
        read_pgen(prefix, fields=["GT"], assume_biallelic=True)

    phased = PGENReader(prefix).read(fields=["GT"], genotype_mode="phased")
    np.testing.assert_array_equal(
        phased.genotypes,
        np.tile(np.array([[[0, 2], [1, 2]]], dtype=np.int8), (4, 1, 1)),
    )
    automatic = PGENReader(prefix).read(fields=["GT"], genotype_mode="auto")
    np.testing.assert_array_equal(automatic.genotypes, phased.genotypes)

    pvar_reader = pg.PvarReader
    pvar_reads = 0

    def tracked_pvar_reader(*args, **kwargs):
        nonlocal pvar_reads
        pvar_reads += 1
        return pvar_reader(*args, **kwargs)

    monkeypatch.setattr(pgen_module.pg, "PvarReader", tracked_pvar_reader)
    parallel = PGENReader(prefix).read(
        fields=["GT"],
        genotype_mode="phased",
        threads=4,
    )

    assert pvar_reads == 1
    np.testing.assert_array_equal(parallel.genotypes, phased.genotypes)


def test_reader_modes_are_validated(tmp_path: Path):
    with pytest.raises(ValueError, match="genotype_mode"):
        VCFReader(tmp_path / "unused.vcf").read(genotype_mode="invalid")


@pytest.mark.parametrize(
    "reader",
    [
        VCFReader("unused.vcf"),
        VCFReaderPolars("unused.vcf"),
        PGENReader("unused.pgen"),
        BEDReader("unused.bed"),
    ],
)
def test_streaming_readers_reject_auto_mode(reader):
    with pytest.raises(ValueError, match="genotype_mode"):
        next(reader.iter_read(genotype_mode="auto"))
