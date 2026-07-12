import gzip
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

from snputils.phenotype import MultiPhenReader, MultiPhenotypeObject, read_pheno


def _write_compressed_text(path: Path, contents: str) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(contents)
    elif path.suffix == ".zst":
        with zstd.open(path, "wt", encoding="utf-8") as handle:
            handle.write(contents)
    else:
        path.write_text(contents, encoding="utf-8")


def test_multi_phenotype_object_validates_and_reorders_sample_column():
    phen = MultiPhenotypeObject(
        pd.DataFrame(
            {
                "height": [170.0, 180.0],
                "IID": ["S0", "S1"],
                "case_control": [0, 1],
            }
        ),
        sample_column="IID",
    )

    assert phen.sample_column == "IID"
    assert phen.samples == ["S0", "S1"]
    assert phen.phenotype_names == ["height", "case_control"]
    assert phen.phen_df.columns.tolist() == ["IID", "height", "case_control"]


def test_multi_phenotype_object_rejects_duplicate_samples():
    with pytest.raises(ValueError, match="must be unique"):
        MultiPhenotypeObject(
            pd.DataFrame(
                {
                    "IID": ["S0", "S0"],
                    "height": [170.0, 180.0],
                }
            )
        )


def test_multi_phen_reader_uses_iid_convention_and_drops_fid(tmp_path: Path):
    path = tmp_path / "phen.tsv"
    pd.DataFrame(
        {
            "FID": ["F0", "F1"],
            "IID": ["S0", "S1"],
            "height": [170.0, 180.0],
            "case_control": [1, 2],
        }
    ).to_csv(path, sep="\t", index=False)

    phen = MultiPhenReader(path).read()

    assert phen.sample_column == "IID"
    assert phen.samples == ["S0", "S1"]
    assert phen.phen_df.columns.tolist() == ["IID", "height", "case_control"]


def test_multi_phen_reader_rejects_arbitrary_samples_idx(tmp_path: Path):
    path = tmp_path / "phen.tsv"
    pd.DataFrame(
        {
            "FID": ["F0", "F1"],
            "IID": ["S0", "S1"],
            "height": [170.0, 180.0],
        }
    ).to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="IID column"):
        MultiPhenReader(path).read(samples_idx=2)


@pytest.mark.parametrize("compression_suffix", [".gz", ".zst"])
def test_read_pheno_accepts_compressed_text(tmp_path: Path, compression_suffix: str):
    path = tmp_path / f"phenotype.pheno{compression_suffix}"
    _write_compressed_text(path, "FID IID height\nF1 S1 170\nF2 S2 180\n")

    phenotype = read_pheno(path)

    assert phenotype.samples == ["S1", "S2"]
    assert phenotype.phenotype_name == "height"
    assert phenotype.values.tolist() == [170, 180]


@pytest.mark.parametrize("requested", ["BMI", "IID"])
def test_read_pheno_rejects_requested_nonphenotype_column(tmp_path: Path, requested: str):
    path = tmp_path / "phenotype.pheno"
    path.write_text("FID IID HEIGHT\nF1 S1 170\nF2 S2 180\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"Phenotype column '{requested}'.*not found"):
        read_pheno(path, col=requested)


@pytest.mark.parametrize("compression_suffix", [".gz", ".zst"])
def test_multi_phen_reader_accepts_compressed_text(tmp_path: Path, compression_suffix: str):
    path = tmp_path / f"phenotypes.tsv{compression_suffix}"
    _write_compressed_text(
        path,
        "FID\tIID\theight\tcase_control\nF1\tS1\t170\t1\nF2\tS2\t180\t2\n",
    )

    phenotype = MultiPhenReader(path).read()

    assert phenotype.samples == ["S1", "S2"]
    assert phenotype.phenotype_names == ["height", "case_control"]
