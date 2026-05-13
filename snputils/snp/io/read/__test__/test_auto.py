from pathlib import Path

import numpy as np

from snputils.snp.io.read import SNPReader, VCFReader, read_vcf


def test_auto_reader(data_path, snpobj_pgen):
    reader = SNPReader(data_path + "/pgen/subset.pgen")
    snpobj = reader.read(sum_strands=False)

    assert np.array_equal(snpobj.calldata_gt, snpobj_pgen.calldata_gt)


def test_auto_reader_uses_default_vcf_backend(data_path):
    reader = SNPReader(data_path + "/vcf/subset.vcf")

    assert isinstance(reader, VCFReader)


def test_read_vcf_uses_default_backend(tmp_path: Path):
    vcf_path = tmp_path / "tiny.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t100\trs1\tA\tG\t50\tPASS\t.\tGT\t0|1\t1|0\n"
    )

    snpobj = read_vcf(vcf_path)

    assert snpobj.calldata_gt.shape == (1, 2, 2)
