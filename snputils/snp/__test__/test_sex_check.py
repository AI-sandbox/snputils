from importlib import resources

import numpy as np
import pandas as pd
import pytest

import snputils as su
from snputils.snp._sex_check import _predict_zigo
from snputils.snp.genobj import SNPObject


def _mixed_chromosome_snpobject() -> SNPObject:
    x_genotypes = np.array(
        [
            [0, 0, -1],
            [0, 0, -1],
            [0, 1, -1],
            [2, 1, -1],
            [2, 2, -1],
            [2, 2, -1],
        ],
        dtype=np.int8,
    )
    genotypes = np.vstack(
        [
            np.array([[1, 0, 0]], dtype=np.int8),
            x_genotypes,
            np.array([[1, 0, 0]], dtype=np.int8),
        ]
    )
    return SNPObject(
        genotypes=genotypes,
        samples=np.array(["male_sample", "female_sample", "no_x_calls"]),
        sample_fid=np.array(["F1", "F2", "F3"]),
        sample_sex=np.array(["1", "F", "male"]),
        variants_chrom=np.array(["1", "X", "chrX", "23", "X", "chrX", "23", "2"]),
    )


def test_sex_check_selects_x_and_compares_reported_sex() -> None:
    results = su.sex_check(_mixed_chromosome_snpobject(), low_information_threshold=7)

    assert results["family_id"].tolist() == ["F1", "F2", "F3"]
    assert results["sample"].tolist() == ["male_sample", "female_sample", "no_x_calls"]
    assert results["reported_sex"].tolist() == ["male", "female", "male"]
    assert results["inferred_sex"].tolist()[:2] == ["male", "female"]
    assert pd.isna(results.loc[2, "inferred_sex"])
    assert results["status"].tolist() == ["match", "match", "unknown"]
    assert results["qc_status"].tolist() == ["low_information", "low_information", "no_data"]
    assert results["n_called"].tolist() == [6, 6, 0]
    np.testing.assert_allclose(
        results.loc[0, ["genotype_0_frequency", "genotype_1_frequency", "genotype_2_frequency"]].to_numpy(dtype=float),
        [0.5, 0.0, 0.5],
    )
    np.testing.assert_allclose(
        results.loc[1, ["genotype_0_frequency", "genotype_1_frequency", "genotype_2_frequency"]].to_numpy(dtype=float),
        [1 / 3, 1 / 3, 1 / 3],
    )
    assert results.loc[0, "p_male"] > 0.99
    assert results.loc[1, "p_female"] > 0.99
    assert np.isnan(results.loc[2, "p_male"])
    assert np.isnan(results.loc[2, "p_female"])


def test_sex_check_reports_mismatch_and_not_compared_statuses() -> None:
    results = su.sex_check(
        _mixed_chromosome_snpobject(),
        reported_sex={"male_sample": "female"},
    )

    assert results["status"].tolist() == ["mismatch", "not_compared", "unknown"]


def test_sex_check_applies_zigo_allele_orientation_normalization() -> None:
    genotypes = np.array(
        [
            [2, 2, 2, 0],
            [1, 1, 1, 0],
            [1, 2, 2, 2],
        ],
        dtype=np.int8,
    ).T
    snpobj = SNPObject(
        genotypes=genotypes,
        samples=["diploid_swap", "haploid_swap", "single_sample"],
        variants_chrom=["X"] * 4,
    )

    results = su.sex_check(snpobj, low_information_threshold=0)
    observed = results[
        ["genotype_0_frequency", "genotype_1_frequency", "genotype_2_frequency"]
    ].to_numpy()
    np.testing.assert_allclose(
        observed,
        [
            [0.75, 0.0, 0.25],
            [0.75, 0.25, 0.0],
            [0.0, 0.25, 0.75],
        ],
    )
    assert results["qc_status"].tolist() == ["pass", "pass", "pass"]


def test_sex_check_supports_haploid_3d_calls_and_ignores_invalid_alleles() -> None:
    genotypes = np.array(
        [
            [[0, -1], [0, 0], [2, 0]],
            [[1, -1], [0, 1], [-1, -1]],
            [[1, -1], [1, 1], [-1, -1]],
            [[0, -1], [0, 1], [-1, -1]],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(
        genotypes=genotypes,
        samples=["haploid", "diploid", "invalid"],
        variants_chrom=["X"] * 4,
    )

    results = su.sex_check(
        snpobj,
        reported_sex={"haploid": "M", "diploid": 2},
        low_information_threshold=4,
    )

    assert results["n_called"].tolist() == [4, 4, 0]
    np.testing.assert_allclose(
        results.loc[0, ["genotype_0_frequency", "genotype_1_frequency", "genotype_2_frequency"]].to_numpy(dtype=float),
        [0.5, 0.5, 0.0],
    )
    np.testing.assert_allclose(
        results.loc[1, ["genotype_0_frequency", "genotype_1_frequency", "genotype_2_frequency"]].to_numpy(dtype=float),
        [0.25, 0.5, 0.25],
    )
    assert results["inferred_sex"].tolist()[:2] == ["male", "female"]
    assert results["status"].tolist() == ["match", "match", "unknown"]


def test_sex_check_model_matches_zigo_regression_points() -> None:
    features = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.9, 0.1, 0.0],
            [0.34, 0.33, 0.33],
            [0.0, 0.0, 1.0],
        ]
    )
    p_male, p_female = _predict_zigo(features)

    np.testing.assert_allclose(
        p_male,
        [0.9985400181, 0.9999909773, 0.864049109, 0.000151050894, 1.0],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(p_female, 1.0 - p_male)


def test_sex_check_requires_x_metadata_unless_explicitly_overridden() -> None:
    snpobj = SNPObject(genotypes=np.array([[0], [2]], dtype=np.int8), samples=["s1"])

    with pytest.raises(ValueError, match="chromosome metadata"):
        su.sex_check(snpobj)

    results = su.sex_check(snpobj, assume_x=True, low_information_threshold=0)
    assert results.loc[0, "inferred_sex"] == "male"
    assert results.loc[0, "n_called"] == 2


def test_sex_check_rejects_missing_hard_calls_and_invalid_threshold() -> None:
    probabilities = SNPObject(
        genotypes=None,
        calldata_gp=np.ones((2, 1, 3), dtype=np.float32) / 3,
        samples=["s1"],
        variants_chrom=["X", "X"],
    )
    with pytest.raises(ValueError, match="hard-call genotypes"):
        su.sex_check(probabilities)

    snpobj = _mixed_chromosome_snpobject()
    with pytest.raises(ValueError, match="non-negative integer"):
        su.sex_check(snpobj, low_information_threshold=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        su.sex_check(snpobj, low_information_threshold=1.5)


def test_zigo_model_is_available_as_package_data() -> None:
    model = resources.files("snputils.snp").joinpath("models", "zigo.json")
    assert model.is_file()
    assert '"degree": 6' in model.read_text(encoding="utf-8")
