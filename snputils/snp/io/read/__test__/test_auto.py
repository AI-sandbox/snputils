from pathlib import Path

import numpy as np

from snputils.snp.io.read import BCFReader, SNPReader, VCFReader, read_vcf


def test_auto_reader(data_path, snpobj_pgen):
    reader = SNPReader(data_path + "/pgen/subset.pgen")
    snpobj = reader.read(sum_strands=False)

    assert np.array_equal(snpobj.genotypes, snpobj_pgen.genotypes)


def test_auto_reader_uses_default_vcf_backend(data_path):
    reader = SNPReader(data_path + "/vcf/subset.vcf")

    assert isinstance(reader, VCFReader)


def test_auto_reader_detects_bcf(data_path):
    reader = SNPReader(data_path + "/bcf/subset.bcf")

    assert isinstance(reader, BCFReader)


def test_read_vcf_uses_default_backend(tmp_path: Path):
    vcf_path = tmp_path / "tiny.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\trs1\tA\tG\t50\tPASS\t.\tGT\t0|1\t1|0\n"
    )

    snpobj = read_vcf(vcf_path)

    assert snpobj.genotypes.shape == (1, 2)
