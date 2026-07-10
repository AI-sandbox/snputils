from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from snputils.snp.genobj import SNPObject
from snputils.snp.io.write import BCFWriter, BEDWriter, PGENWriter
from snputils.snp.io.write.bcf import _native_bcf
from snputils.tools.cli import main
from snputils.tools.sex_check import read_reported_sex, run_sex_check


def _sex_check_snpobject() -> SNPObject:
    x_genotypes = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [2, 1],
            [2, 2],
            [2, 2],
        ],
        dtype=np.int8,
    )
    genotypes = np.vstack([np.array([[1, 1]], dtype=np.int8), x_genotypes])
    return SNPObject(
        genotypes=genotypes,
        samples=np.array(["s0", "s1"]),
        sample_fid=np.array(["f0", "f1"]),
        sample_sex=np.array(["1", "2"]),
        variants_ref=np.array(["A"] * 7),
        variants_alt=np.array(["G"] * 7),
        variants_chrom=np.array(["1"] + ["X"] * 6),
        variants_cm=np.arange(7, dtype=float),
        variants_id=np.array([f"v{i}" for i in range(7)]),
        variants_pos=np.arange(1, 8),
    )


def _write_sex_check_vcf(path: Path) -> None:
    snpobj = _sex_check_snpobject()
    gt = snpobj.genotypes
    gt_strings = {0: "0/0", 1: "0/1", 2: "1/1"}
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1",
    ]
    for index in range(snpobj.n_snps):
        sample_gt = "\t".join(gt_strings[int(value)] for value in gt[index])
        lines.append(
            f"{snpobj.variants_chrom[index]}\t{index + 1}\tv{index}\tA\tG\t.\tPASS\t.\tGT\t{sample_gt}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sex_check_cli_reads_vcf_sex_table_and_writes_one_report(tmp_path: Path) -> None:
    vcf_path = tmp_path / "cohort.vcf"
    sex_path = tmp_path / "reported.tsv"
    output_path = tmp_path / "results" / "sex_check.tsv"
    _write_sex_check_vcf(vcf_path)
    sex_path.write_text("Individual ID\tGender\ns0\t1\ns1\t2\n", encoding="utf-8")

    assert main(
        [
            "sex-check",
            "--snp-path",
            str(vcf_path),
            "--sex-path",
            str(sex_path),
            "--results-path",
            str(output_path),
        ]
    ) == 0

    observed = pd.read_csv(output_path, sep="\t")
    assert observed["sample"].tolist() == ["s0", "s1"]
    assert observed["reported_sex"].tolist() == ["male", "female"]
    assert observed["inferred_sex"].tolist() == ["male", "female"]
    assert observed["status"].tolist() == ["match", "match"]
    assert observed["n_called"].tolist() == [6, 6]
    assert observed["qc_status"].tolist() == ["low_information", "low_information"]

    without_reported_sex = run_sex_check(vcf_path)
    assert without_reported_sex["status"].tolist() == ["not_compared", "not_compared"]


@pytest.mark.parametrize("file_format", ["bed", "pgen"])
def test_run_sex_check_uses_snputils_plink_readers(
    tmp_path: Path, file_format: str
) -> None:
    prefix = tmp_path / f"cohort_{file_format}"
    snpobj = _sex_check_snpobject()
    if file_format == "bed":
        BEDWriter(snpobj, str(prefix)).write(rename_missing_values=False)
        input_path = prefix.with_suffix(".bed")
    else:
        PGENWriter(snpobj, str(prefix)).write(vzs=False, rename_missing_values=False)
        input_path = prefix.with_suffix(".pgen")

    results = run_sex_check(input_path, low_information_threshold=0)
    assert results["sample"].tolist() == ["s0", "s1"]
    assert results["inferred_sex"].tolist() == ["male", "female"]
    assert results["status"].tolist() == ["match", "match"]
    assert results["n_called"].tolist() == [6, 6]
    assert results["qc_status"].tolist() == ["pass", "pass"]


def test_run_sex_check_uses_snputils_bcf_reader(tmp_path: Path) -> None:
    if _native_bcf is None:
        pytest.skip("Native BCF writer extension is unavailable in this development environment.")
    input_path = tmp_path / "cohort.bcf"
    BCFWriter(_sex_check_snpobject(), input_path).write()

    results = run_sex_check(
        input_path,
        reported_sex={"s0": "male", "s1": "female"},
        low_information_threshold=0,
    )
    assert results["inferred_sex"].tolist() == ["male", "female"]
    assert results["status"].tolist() == ["match", "match"]
    assert results["n_called"].tolist() == [6, 6]


def test_read_reported_sex_accepts_common_headers_and_rejects_duplicates(tmp_path: Path) -> None:
    valid = tmp_path / "valid.tsv"
    duplicate = tmp_path / "duplicate.tsv"
    valid.write_text("IID\tSEX\ns0\tM\ns1\tfemale\n", encoding="utf-8")
    duplicate.write_text("sample,Gender\ns0,1\ns0,2\n", encoding="utf-8")

    assert read_reported_sex(valid) == {"s0": "M", "s1": "female"}
    with pytest.raises(ValueError, match="unique"):
        read_reported_sex(duplicate)


def test_run_sex_check_rejects_bgen_probability_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="genotype probabilities"):
        run_sex_check(tmp_path / "cohort.bgen")
