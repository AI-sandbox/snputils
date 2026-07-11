from pathlib import Path

import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read import BCFReader, BEDReader, PGENReader
from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars
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


def test_vcf_dosage_preserves_multiallelic_allele_index_sum(tmp_path: Path):
    vcf_path = tmp_path / "multiallelic.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "1\t100\trs1\tA\tG,T\t.\tPASS\t.\tGT\t1|2\n"
    )

    snpobj = VCFReader(vcf_path).read(fields=["GT"], genotype_mode="dosage")

    np.testing.assert_array_equal(snpobj.genotypes, np.array([[3]], dtype=np.int8))
    assert snpobj.variants_alt.size == 0


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

    with pytest.raises(ValueError, match="BED/BIM/FAM does not store phase"):
        BEDReader("cohort.bed").read(genotype_mode="phased")


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

    observed = PGENReader(prefix).read()
    np.testing.assert_array_equal(observed.genotypes, genotypes)


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
