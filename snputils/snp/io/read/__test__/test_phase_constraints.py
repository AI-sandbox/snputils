from pathlib import Path

import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read import BEDReader, PGENReader
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


def test_vcf_unphased_gt_defaults_to_dosage_and_rejects_separate_strands(tmp_path: Path):
    vcf_path = tmp_path / "unphased.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/0\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t0/0\t1/1\n"
    )

    snpobj = VCFReader(vcf_path).read()
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[1, 1], [0, 2]], dtype=np.int8),
    )

    with pytest.raises(ValueError, match="unphased VCF genotypes"):
        VCFReader(vcf_path).read(sum_strands=False)

    with pytest.raises(ValueError, match="unphased VCF genotypes"):
        VCFReaderPolars(vcf_path).read(sum_strands=False)


def test_vcf_phased_gt_allows_separate_strands(tmp_path: Path):
    vcf_path = tmp_path / "phased.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0\n"
    )

    snpobj = VCFReader(vcf_path).read(sum_strands=False)
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[[0, 1], [1, 0]]], dtype=np.int8),
    )


def test_vcf_summed_gt_preserves_one_missing_sentinel(tmp_path: Path):
    vcf_path = tmp_path / "missing.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|.\t./.\t1|1\n"
    )

    snpobj = VCFReader(vcf_path).read()
    np.testing.assert_array_equal(
        snpobj.genotypes,
        np.array([[-1, -1, 2]], dtype=np.int8),
    )


def test_bed_rejects_separate_strands():
    with pytest.raises(ValueError, match="BED/BIM/FAM does not store phase"):
        BEDReader("cohort.bed").read(sum_strands=False)


def test_pgen_unphased_hardcalls_reject_separate_strands(tmp_path: Path):
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
        PGENReader(prefix).read(sum_strands=False)


def test_pgen_phased_hardcalls_allow_separate_strands(tmp_path: Path):
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

    observed = PGENReader(prefix).read(sum_strands=False)
    np.testing.assert_array_equal(observed.genotypes, genotypes)
