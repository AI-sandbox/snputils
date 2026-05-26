from pathlib import Path

import pandas as pd
import pytest

from snputils.phenotype import MultiPhenReader, MultiPhenotypeObject


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
