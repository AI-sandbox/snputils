import numpy as np

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.pgen import PGENReader
from snputils.snp.io.write.pgen import PGENWriter


def test_pgen_writer_roundtrips_pvar_info_and_psam_metadata(tmp_path):
    prefix = tmp_path / "toy"
    snpobj = SNPObject(
        genotypes=np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3"]),
        sample_fid=np.array(["P1", "P1", "P2"]),
        sample_sex=np.array(["M", "F", "unknown"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["1:10A,G", "v2"], dtype=object),
        variants_pos=np.array([10, 20]),
        variants_qual=np.array(["50", ""], dtype=object),
        variants_filter_pass=np.array(["PASS", "q10"], dtype=object),
        variants_info=np.array(["AC=1;AF=0.25", ""], dtype=object),
    )

    PGENWriter(snpobj, str(prefix)).write(vzs=False, rename_missing_values=False)

    pvar_lines = prefix.with_suffix(".pvar").read_text().splitlines()
    assert pvar_lines[:3] == [
        "##fileformat=VCFv4.2",
        "##source=snputils",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    assert pvar_lines[3].split("\t") == ["1", "10", "1:10A,G", "A", "G", "50", "PASS", "AC=1;AF=0.25"]
    assert pvar_lines[4].split("\t") == ["1", "20", "v2", "C", "T", ".", "q10", "."]

    psam_lines = prefix.with_suffix(".psam").read_text().splitlines()
    assert psam_lines == [
        "#FID\tIID\tSEX",
        "P1\ts1\t1",
        "P1\ts2\t2",
        "P2\ts3\tNA",
    ]

    observed = PGENReader(prefix).read(sum_strands=True)
    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)
    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.sample_fid, snpobj.sample_fid)
    np.testing.assert_array_equal(observed.sample_sex, np.array(["1", "2", "NA"]))
    np.testing.assert_array_equal(observed.variants_qual, np.array(["50", "."]))
    np.testing.assert_array_equal(observed.variants_filter_pass, snpobj.variants_filter_pass)
    np.testing.assert_array_equal(observed.variants_info, np.array(["AC=1;AF=0.25", "."]))
