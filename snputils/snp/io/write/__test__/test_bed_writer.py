import numpy as np
import pandas as pd
import pytest

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.bed import BEDReader
from snputils.snp.io.write.bed import BEDWriter


@pytest.mark.parametrize("filename", ["toy", "toy.filtered", "toy.filtered.bed"])
def test_bed_writer_roundtrips_summed_genotypes_and_plink_sample_metadata(tmp_path, filename):
    bed_path = tmp_path / ("toy.bed" if filename == "toy" else "toy.filtered.bed")
    snpobj = SNPObject(
        genotypes=np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int8),
        samples=np.array(["s1", "s2", "s3"]),
        sample_fid=np.array(["P1", "P1", "P2"]),
        sample_sex=np.array(["M", "F", "unknown"]),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_cm=np.array([0.125, 1.5]),
        variants_id=np.array(["v1", "v2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )

    BEDWriter(snpobj, tmp_path / filename).write(
        rename_missing_values=False,
        sample_phenotype=["case", "control", "case"],
    )

    observed = BEDReader(bed_path).read(genotype_mode="dosage")
    np.testing.assert_array_equal(observed.genotypes, snpobj.genotypes)
    np.testing.assert_array_equal(observed.samples, snpobj.samples)
    np.testing.assert_array_equal(observed.sample_fid, snpobj.sample_fid)
    np.testing.assert_array_equal(observed.sample_sex, np.array(["1", "2", "0"]))
    np.testing.assert_allclose(observed.variants_cm, snpobj.variants_cm)

    bed_bytes = bed_path.read_bytes()
    assert bed_bytes == bytes([0x6C, 0x1B, 0x01, 0x0B, 0x0E])
    assert len(bed_bytes) == 3 + snpobj.n_snps * ((len(snpobj.samples) + 3) // 4)

    fam = pd.read_csv(
        bed_path.with_suffix(".fam"),
        sep="\t",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "phenotype"],
        dtype=str,
    )
    assert fam["fid"].tolist() == ["P1", "P1", "P2"]
    assert fam["sex"].tolist() == ["1", "2", "0"]
    assert fam["phenotype"].tolist() == ["case", "control", "case"]

    bim = pd.read_csv(
        bed_path.with_suffix(".bim"),
        sep="\t",
        header=None,
        names=["chrom", "id", "cm", "pos", "alt", "ref"],
    )
    assert bim["cm"].tolist() == [0.125, 1.5]
