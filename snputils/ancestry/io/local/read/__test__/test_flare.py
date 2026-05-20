import gzip
from pathlib import Path

import numpy as np
import pytest

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.ancestry.io.local.read import FLAREReader, LAIReader, read_flare, read_lai
from snputils.ancestry.io.local.write import FLAREWriter
from snputils.snp.genobj.snpobj import SNPObject


def _write_flare_vcf(path: Path) -> None:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.1\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write('##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">\n')
        handle.write('##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">\n')
        handle.write('##FORMAT=<ID=ANP1,Number=.,Type=Float,Description="Posterior ancestry probabilities for first haplotype">\n')
        handle.write('##FORMAT=<ID=ANP2,Number=.,Type=Float,Description="Posterior ancestry probabilities for second haplotype">\n')
        handle.write("##ANCESTRY=<AFR=0,EUR=1,AMR=2>\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n")
        handle.write("1\t101\trs1\tA\tC\t.\t.\t.\tGT:AN1:AN2:ANP1:ANP2\t0|1:0:1:0.9,0.1,0:0.2,0.8,0\t1|1:2:2:0,0,1:0,0,1\n")
        handle.write("1\t205\trs2\tG\tT\t.\t.\t.\tGT:AN1:AN2:ANP1:ANP2\t0|0:1:1:0.1,0.9,0:0.1,0.9,0\t0|1:0:2:0.8,0.2,0:0,0,1\n")
        handle.write("2\t300\trs3\tC\tG\t.\t.\t.\tGT:AN1:AN2:ANP1:ANP2\t1|1:2:0:0,0,1:0.7,0.3,0\t0|0:1:0:0.1,0.9,0:0.9,0.1,0\n")


def _write_gt_vcf(path: Path) -> None:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.1\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n")
        handle.write("1\t10\trs10\tA\tC\t50\tPASS\tAC=2\tGT\t0|1\t1|1\n")
        handle.write("1\t20\trs20\tG\tT\t60\tq10\tAC=1\tGT\t0|0\t0|1\n")


def _make_snpobj() -> SNPObject:
    return SNPObject(
        samples=np.array(["S1", "S2"], dtype=object),
        genotypes=np.array(
            [
                [[0, 1], [1, 1]],
                [[0, 0], [0, 1]],
            ],
            dtype=np.int8,
        ),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_pos=np.array([10, 20], dtype=np.int64),
        variants_id=np.array(["rs10", "rs20"], dtype=object),
        variants_ref=np.array(["A", "G"], dtype=object),
        variants_alt=np.array(["C", "T"], dtype=object),
        variants_qual=np.array(["50", "60"], dtype=object),
        variants_filter_pass=np.array(["PASS", "q10"], dtype=object),
        variants_info=np.array(["AC=2", "AC=1"], dtype=object),
    )


def _make_superset_snpobj() -> SNPObject:
    return SNPObject(
        samples=np.array(["S1", "S2"], dtype=object),
        genotypes=np.array(
            [
                [[1, 1], [0, 0]],
                [[0, 1], [1, 1]],
                [[0, 0], [0, 1]],
            ],
            dtype=np.int8,
        ),
        variants_chrom=np.array(["1", "1", "1"], dtype=object),
        variants_pos=np.array([5, 10, 20], dtype=np.int64),
        variants_id=np.array(["extra", "rs10", "rs20"], dtype=object),
        variants_ref=np.array(["T", "A", "G"], dtype=object),
        variants_alt=np.array(["G", "C", "T"], dtype=object),
        variants_qual=np.array(["40", "50", "60"], dtype=object),
        variants_filter_pass=np.array(["PASS", "PASS", "q10"], dtype=object),
        variants_info=np.array(["AC=3", "AC=2", "AC=1"], dtype=object),
    )


def test_flare_reader_reads_gzipped_vcf_and_autodetects(tmp_path: Path):
    flare_path = tmp_path / "toy.anc.vcf.gz"
    _write_flare_vcf(flare_path)

    lai = read_lai(flare_path)

    assert isinstance(LAIReader(flare_path), FLAREReader)
    assert lai.samples == ["S1", "S2"]
    assert lai.haplotypes == ["S1.0", "S1.1", "S2.0", "S2.1"]
    assert lai.ancestry_map == {"0": "AFR", "1": "EUR", "2": "AMR"}
    np.testing.assert_array_equal(lai.chromosomes.astype(str), np.array(["1", "1", "2"]))
    np.testing.assert_array_equal(lai.physical_pos, np.array([[101, 101], [205, 205], [300, 300]]))
    np.testing.assert_array_equal(
        lai.lai,
        np.array(
            [
                [0, 1, 2, 2],
                [1, 1, 0, 2],
                [2, 0, 1, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_flare_iter_windows_matches_full_read_and_subsets_samples(tmp_path: Path):
    flare_path = tmp_path / "toy.anc.vcf"
    _write_flare_vcf(flare_path)
    full = read_flare(flare_path)

    chunks = list(FLAREReader(flare_path).iter_windows(chunk_size=2, sample_indices=np.array([1])))

    np.testing.assert_array_equal(np.concatenate([chunk["chromosomes"] for chunk in chunks]), full.chromosomes)
    np.testing.assert_array_equal(np.concatenate([chunk["physical_pos"] for chunk in chunks]), full.physical_pos)
    np.testing.assert_array_equal(np.concatenate([chunk["lai"] for chunk in chunks]), full.lai[:, 2:4])


def test_flare_writer_roundtrips_local_ancestry_object(tmp_path: Path):
    lai = np.array([[0, 1, 2, 2], [1, 1, 0, 2]], dtype=np.uint8)
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1", "S2.0", "S2.1"],
        samples=["S1", "S2"],
        ancestry_map={"0": "AFR", "1": "EUR", "2": "AMR"},
        chromosomes=np.array(["1", "1"], dtype=object),
        physical_pos=np.array([[10, 10], [20, 20]], dtype=np.int64),
        lai=lai,
    )
    snpobj = _make_snpobj()

    out_path = tmp_path / "roundtrip.anc.vcf.gz"
    FLAREWriter(laiobj, out_path, snpobj=snpobj).write()
    roundtrip = read_lai(out_path)

    assert roundtrip.samples == laiobj.samples
    assert roundtrip.ancestry_map == laiobj.ancestry_map
    np.testing.assert_array_equal(roundtrip.chromosomes.astype(str), laiobj.chromosomes.astype(str))
    np.testing.assert_array_equal(roundtrip.physical_pos, laiobj.physical_pos)
    np.testing.assert_array_equal(roundtrip.lai, laiobj.lai)

    with gzip.open(out_path, "rt", encoding="utf-8") as handle:
        records = [line.rstrip("\n").split("\t") for line in handle if not line.startswith("#")]
    assert records[0][:9] == ["1", "10", "rs10", "A", "C", "50", "PASS", "AC=2", "GT:AN1:AN2"]
    assert records[0][9:] == ["0|1:0:1", "1|1:2:2"]
    assert records[1][:9] == ["1", "20", "rs20", "G", "T", "60", "q10", "AC=1", "GT:AN1:AN2"]
    assert records[1][9:] == ["0|0:1:1", "0|1:0:2"]


def test_flare_writer_accepts_genotype_file(tmp_path: Path):
    genotype_path = tmp_path / "genotypes.vcf.gz"
    _write_gt_vcf(genotype_path)
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1", "S2.0", "S2.1"],
        samples=["S1", "S2"],
        ancestry_map={"0": "AFR", "1": "EUR", "2": "AMR"},
        chromosomes=np.array(["1", "1"], dtype=object),
        physical_pos=np.array([[10, 10], [20, 20]], dtype=np.int64),
        lai=np.array([[0, 1, 2, 2], [1, 1, 0, 2]], dtype=np.uint8),
    )

    out_path = tmp_path / "from_file.anc.vcf.gz"
    FLAREWriter(laiobj, out_path, genotype_file=genotype_path).write()
    loaded = read_flare(out_path)

    np.testing.assert_array_equal(loaded.lai, laiobj.lai)
    np.testing.assert_array_equal(loaded.physical_pos, laiobj.physical_pos)


def test_flare_writer_accepts_genotype_source_with_extra_variants(tmp_path: Path):
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1", "S2.0", "S2.1"],
        samples=["S1", "S2"],
        ancestry_map={"0": "AFR", "1": "EUR", "2": "AMR"},
        chromosomes=np.array(["1", "1"], dtype=object),
        physical_pos=np.array([[10, 10], [20, 20]], dtype=np.int64),
        lai=np.array([[0, 1, 2, 2], [1, 1, 0, 2]], dtype=np.uint8),
    )

    out_path = tmp_path / "superset.anc.vcf.gz"
    FLAREWriter(laiobj, out_path, snpobj=_make_superset_snpobj()).write()

    with gzip.open(out_path, "rt", encoding="utf-8") as handle:
        records = [line.rstrip("\n").split("\t") for line in handle if not line.startswith("#")]
    assert [record[1] for record in records] == ["10", "20"]
    assert records[0][:9] == ["1", "10", "rs10", "A", "C", "50", "PASS", "AC=2", "GT:AN1:AN2"]
    assert records[0][9:] == ["0|1:0:1", "1|1:2:2"]


def test_flare_writer_formats_missing_scalar_fields_as_missing_values(tmp_path: Path):
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1"],
        samples=["S1"],
        ancestry_map={"0": "AFR", "1": "EUR"},
        chromosomes=np.array(["1"], dtype=object),
        physical_pos=np.array([[10, 10]], dtype=np.int64),
        lai=np.array([[0, 1]], dtype=np.uint8),
    )
    snpobj = SNPObject(
        samples=np.array(["S1"], dtype=object),
        genotypes=np.array([[[0, 1]]], dtype=np.int8),
        variants_chrom=np.array(["1"], dtype=object),
        variants_pos=np.array([10], dtype=np.int64),
        variants_id=np.array(["rs10"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["C"], dtype=object),
        variants_qual=np.array([np.nan]),
        variants_filter_pass=np.array(["PASS"], dtype=object),
        variants_info=np.array([np.nan]),
    )

    out_path = tmp_path / "missing_fields.anc.vcf.gz"
    FLAREWriter(laiobj, out_path, snpobj=snpobj).write()

    with gzip.open(out_path, "rt", encoding="utf-8") as handle:
        records = [line.rstrip("\n").split("\t") for line in handle if not line.startswith("#")]
    assert records[0][:9] == ["1", "10", "rs10", "A", "C", ".", "PASS", ".", "GT:AN1:AN2"]


def test_flare_writer_requires_genotype_source(tmp_path: Path):
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1"],
        samples=["S1"],
        ancestry_map={"0": "AFR", "1": "EUR"},
        chromosomes=np.array(["1"], dtype=object),
        physical_pos=np.array([[42, 42]], dtype=np.int64),
        lai=np.array([[0, 1]], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="requires genotype data"):
        FLAREWriter(laiobj, tmp_path / "missing_source.anc.vcf.gz").write()


def test_local_ancestry_object_save_flare_requires_genotype_source(tmp_path: Path):
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1"],
        samples=["S1"],
        ancestry_map={"0": "AFR", "1": "EUR"},
        chromosomes=np.array(["1"], dtype=object),
        physical_pos=np.array([[42, 42]], dtype=np.int64),
        lai=np.array([[0, 1]], dtype=np.uint8),
    )

    snpobj = SNPObject(
        samples=np.array(["S1"], dtype=object),
        genotypes=np.array([[[0, 1]]], dtype=np.int8),
        variants_chrom=np.array(["1"], dtype=object),
        variants_pos=np.array([42], dtype=np.int64),
        variants_id=np.array(["rs42"], dtype=object),
        variants_ref=np.array(["A"], dtype=object),
        variants_alt=np.array(["C"], dtype=object),
    )

    out_path = tmp_path / "saved.anc.vcf.gz"
    with pytest.raises(ValueError, match="save_flare"):
        laiobj.save(out_path)

    laiobj.save_flare(out_path, snpobj=snpobj)
    loaded = read_flare(out_path)

    np.testing.assert_array_equal(loaded.lai, laiobj.lai)
