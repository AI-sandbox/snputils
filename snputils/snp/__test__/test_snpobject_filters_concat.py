import numpy as np
import pandas as pd
import pytest

from snputils.ibd.genobj.ibdobj import IBDObject
from snputils.phenotype.genobj.phenobj import PhenotypeObject
from snputils.snp.genobj.snpobj import SNPObject


def _toy_snpobj() -> SNPObject:
    return SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1], [1, 1]],
                [[0, 0], [-1, -1], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 1], [0, 0]],
                [[0, 0], [0, 1], [0, 0]],
            ],
            dtype=np.int8,
        ),
        samples=np.array(["s1", "s2", "s3"]),
        sample_fid=np.array(["A", "A", "B"]),
        variants_ref=np.array(["A", "C", "G", "AT", "T"], dtype=object),
        variants_alt=np.array(["G", "T", "A", "A", "C,G"], dtype=object),
        variants_chrom=np.array(["1", "1", "1", "1", "1"], dtype=object),
        variants_id=np.array(["v1", "v2", "v3", "v4", "v5"], dtype=object),
        variants_pos=np.array([10, 20, 30, 40, 50]),
    )


def test_variant_filters_cover_biallelic_complete_and_variable_genotypes():
    snpobj = _toy_snpobj()

    filtered = (
        snpobj
        .filter_biallelic_variants(snv_only=True)
        .filter_complete_genotypes()
        .filter_variable_genotypes()
    )

    assert filtered.variants_id.tolist() == ["v1"]
    assert filtered.genotypes.shape == (1, 3, 2)


def test_filter_variants_accepts_boolean_mask():
    snpobj = _toy_snpobj()

    filtered = snpobj.filter_variants(mask=[True, False, True, False, False])

    assert filtered.variants_id.tolist() == ["v1", "v3"]


def test_filter_variants_drops_unloaded_placeholder_info_column():
    """VCF reader uses length-0 arrays when INFO was not loaded; mask must still apply."""
    snpobj = SNPObject(
        genotypes=np.zeros((4, 1, 2), dtype=np.int8),
        samples=np.array(["s1"], dtype=object),
        variants_ref=np.array(["A", "C", "G", "T"], dtype=object),
        variants_alt=np.array(["G", "T", "A", "C"], dtype=object),
        variants_chrom=np.array(["1", "1", "1", "1"], dtype=object),
        variants_pos=np.array([1, 2, 3, 4], dtype=np.int32),
        variants_info=np.array([], dtype=object),
    )
    filtered = snpobj.filter_variants(mask=[True, False, True, False])
    assert filtered.n_snps == 2
    assert filtered.variants_info is None


def test_filter_samples_by_index_works_without_sample_ids():
    genotypes = np.array(
        [
            [[0, 0], [0, 1], [1, 1]],
            [[1, 0], [1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(genotypes=genotypes)

    filtered = snpobj.filter_samples(indexes=[0, 2])

    assert filtered.samples is None
    np.testing.assert_array_equal(filtered.genotypes, genotypes[:, [0, 2], :])


def _call_rate_snpobj() -> SNPObject:
    return SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1], [1, 1]],
                [[0, -1], [-1, -1], [0, 0]],
                [[np.nan, 0], [1, 1], [0, 1]],
                [[-1, -1], [-1, -1], [-1, -1]],
            ]
        ),
        samples=np.array(["s1", "s2", "s3"]),
        sample_fid=np.array(["F1", "F2", "F3"]),
        variants_id=np.array(["v1", "v2", "v3", "v4"], dtype=object),
    )


def _manual_grm(dosages: np.ndarray, min_variants: int = 1) -> tuple[np.ndarray, np.ndarray]:
    dosages = np.asarray(dosages, dtype=float)
    called = np.isfinite(dosages)
    called_per_variant = called.sum(axis=1)
    allele_sum = np.where(called, dosages, 0.0).sum(axis=1)
    allele_freq = np.full(dosages.shape[0], np.nan, dtype=float)
    nonempty = called_per_variant > 0
    allele_freq[nonempty] = allele_sum[nonempty] / (2.0 * called_per_variant[nonempty])
    variance = 2.0 * allele_freq * (1.0 - allele_freq)
    informative = np.isfinite(variance) & (variance > 0)

    called = called[informative]
    allele_freq = allele_freq[informative]
    standardized = (
        dosages[informative] - (2.0 * allele_freq[:, None])
    ) / np.sqrt(2.0 * allele_freq[:, None] * (1.0 - allele_freq[:, None]))
    standardized[~called] = 0.0
    counts = called.astype(np.int64).T @ called.astype(np.int64)
    numerator = standardized.T @ standardized
    relatedness = np.full(numerator.shape, np.nan, dtype=float)
    valid = counts >= min_variants
    relatedness[valid] = numerator[valid] / counts[valid]
    return relatedness, counts


def _toy_ibdobj() -> IBDObject:
    return IBDObject(
        sample_id_1=np.array(["A", "A", "B", "A", "A"], dtype=object),
        haplotype_id_1=np.array([1, 1, 1, 1, 1], dtype=int),
        sample_id_2=np.array(["B", "B", "C", "C", "D"], dtype=object),
        haplotype_id_2=np.array([2, 2, 2, 2, 2], dtype=int),
        chrom=np.array(["1", "1", "1", "1", "1"], dtype=object),
        start=np.array([1, 101, 201, 301, 401], dtype=int),
        end=np.array([100, 200, 300, 400, 500], dtype=int),
        length_cm=np.array([20.0, 10.0, 5.0, 2.0, 100.0], dtype=float),
        segment_type=np.array(["IBD1", "IBD2", "IBD1", "IBD1", "IBD1"], dtype=object),
    )


def test_call_rate_from_3d_genotypes_treats_partial_missing_and_nan_as_missing():
    snpobj = _call_rate_snpobj()

    np.testing.assert_allclose(
        snpobj.variant_call_rate(),
        np.array([1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0]),
    )
    np.testing.assert_allclose(
        snpobj.sample_call_rate(),
        np.array([0.25, 0.5, 0.75]),
    )

    call_rate_df = snpobj.variant_call_rate(as_dataframe=True)
    assert call_rate_df.columns.tolist() == ["variant_call_rate"]
    np.testing.assert_allclose(
        call_rate_df["variant_call_rate"],
        np.array([1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0]),
    )


def test_call_rate_from_2d_dosages_treats_negative_and_nan_as_missing():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 1.0, np.nan],
                [2.0, -1.0, 0.0],
                [-1.0, np.nan, -1.0],
            ]
        )
    )

    np.testing.assert_allclose(
        snpobj.variant_call_rate(),
        np.array([2.0 / 3.0, 2.0 / 3.0, 0.0]),
    )
    np.testing.assert_allclose(
        snpobj.sample_call_rate(),
        np.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )


def test_call_rate_filters_keep_entities_at_or_above_threshold():
    snpobj = _call_rate_snpobj()
    snpobj.sample_metadata = pd.DataFrame(
        {"sample": ["s1", "s2", "s3"], "population": ["A", "B", "C"]}
    )

    variants = snpobj.filter_variants_by_call_rate(min_call_rate=2.0 / 3.0)
    samples = snpobj.filter_samples_by_call_rate(min_call_rate=0.5)

    assert variants.variants_id.tolist() == ["v1", "v3"]
    assert samples.samples.tolist() == ["s2", "s3"]
    assert samples.sample_fid.tolist() == ["F2", "F3"]
    assert samples.sample_metadata["sample"].tolist() == ["s2", "s3"]


@pytest.mark.parametrize("method_name", ["filter_variants_by_call_rate", "filter_samples_by_call_rate"])
@pytest.mark.parametrize("min_call_rate", [-0.01, 1.01, "not-a-number"])
def test_call_rate_filters_validate_threshold(method_name, min_call_rate):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="min_call_rate"):
        getattr(snpobj, method_name)(min_call_rate=min_call_rate)


def test_differential_missingness_aligns_phenotype_groups_and_filters_variants():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, -1.0],
                [0.0, 0.0, -1.0, -1.0],
            ]
        ),
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
        variants_id=np.array(["complete", "case_missing", "balanced", "control_missing"], dtype=object),
    )
    phenotype = PhenotypeObject(
        samples=["s3", "s1", "s4", "s2"],
        values=[0, 1, 0, 1],
        phenotype_name="case_control",
        quantitative=False,
    )

    p_values = snpobj.differential_missingness(phenotype)
    report = snpobj.differential_missingness(phenotype, as_dataframe=True)
    filtered = snpobj.filter_differential_missingness(phenotype, min_p=0.5)

    np.testing.assert_allclose(p_values, np.array([1.0, 1.0 / 3.0, 1.0, 1.0 / 3.0]))
    assert report.columns.tolist() == [
        "differential_missingness_pvalue",
        "statistic",
        "test",
        "missing_count",
        "called_count",
        "n_samples",
        "missing_count_1",
        "called_count_1",
        "missing_rate_1",
        "missing_count_0",
        "called_count_0",
        "missing_rate_0",
    ]
    assert report["test"].tolist() == ["chi2", "fisher", "fisher", "fisher"]
    assert report["missing_count_1"].tolist() == [0, 2, 1, 0]
    assert report["missing_count_0"].tolist() == [0, 0, 1, 2]
    assert filtered.variants_id.tolist() == ["complete", "balanced"]


def test_differential_missingness_supports_multigroup_chi_square():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            ]
        ),
        samples=np.array(["s1", "s2", "s3", "s4", "s5", "s6"], dtype=object),
    )
    groups = pd.Series(
        ["batch_a", "batch_a", "batch_b", "batch_b", "batch_c", "batch_c"],
        index=["s1", "s2", "s3", "s4", "s5", "s6"],
    )

    report = snpobj.differential_missingness(groups, test="chi2", as_dataframe=True)

    assert report["test"].tolist() == ["chi2", "chi2"]
    assert report.loc[0, "differential_missingness_pvalue"] < 0.05
    assert report.loc[1, "differential_missingness_pvalue"] == pytest.approx(1.0)
    assert report["missing_count_batch_a"].tolist() == [2, 1]
    assert report["missing_count_batch_b"].tolist() == [0, 1]
    assert report["missing_count_batch_c"].tolist() == [0, 1]


def test_differential_missingness_supports_bgen_probability_missingness():
    gp = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ],
        dtype=float,
    )
    snpobj = SNPObject(
        calldata_gp=gp,
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
    )

    np.testing.assert_allclose(
        snpobj.differential_missingness(["case", "case", "control", "control"]),
        np.array([1.0, 1.0 / 3.0]),
    )


def test_differential_missingness_uses_group_column_from_table():
    snpobj = SNPObject(
        genotypes=np.array([[-1.0, -1.0, 0.0, 0.0]]),
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
    )
    groups = pd.DataFrame(
        {
            "batch": ["case", "case", "control", "control"],
            "ignore": [0, 1, 0, 1],
        },
        index=["s1", "s2", "s3", "s4"],
    )

    np.testing.assert_allclose(
        snpobj.differential_missingness(groups, group_column="batch"),
        np.array([1.0 / 3.0]),
    )


def test_filter_differential_missingness_include_false_keeps_failing_variants():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, 0.0, 0.0],
            ]
        ),
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
        variants_id=np.array(["complete", "differential"], dtype=object),
    )

    filtered = snpobj.filter_differential_missingness(
        ["case", "case", "control", "control"],
        min_p=0.5,
        include=False,
    )

    assert filtered.variants_id.tolist() == ["differential"]


def test_differential_missingness_validates_inputs():
    snpobj = SNPObject(
        genotypes=np.zeros((1, 4), dtype=float),
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
    )

    with pytest.raises(ValueError, match="'test'"):
        snpobj.differential_missingness(["a", "a", "b", "b"], test="invalid")
    with pytest.raises(ValueError, match="Fisher"):
        snpobj.differential_missingness(["a", "b", "c", "c"], test="fisher")
    with pytest.raises(ValueError, match="length"):
        snpobj.differential_missingness(["a", "b", "c"])
    with pytest.raises(ValueError, match="missing labels"):
        snpobj.differential_missingness({"s1": "a", "s2": "a", "s3": "b"})
    with pytest.raises(ValueError, match="missing labels"):
        snpobj.differential_missingness(["a", "a", np.nan, "b"])
    with pytest.raises(ValueError, match="min_p"):
        snpobj.filter_differential_missingness(["a", "a", "b", "b"], min_p=1.1)
    with pytest.raises(ValueError, match="min_expected"):
        snpobj.differential_missingness(["a", "a", "b", "b"], min_expected=-1)


def test_relatedness_grm_from_dosages_matches_manual_standardization():
    dosages = np.array(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, np.nan, 2.0],
        ]
    )
    snpobj = SNPObject(
        genotypes=dosages,
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
    )
    expected, _ = _manual_grm(dosages)

    relatedness = snpobj.relatedness(block_size=2)
    relatedness_df = snpobj.relatedness(block_size=2, as_dataframe=True)

    np.testing.assert_allclose(relatedness, expected, equal_nan=True)
    assert relatedness_df.index.tolist() == ["s1", "s2", "s3", "s4"]
    assert relatedness_df.columns.tolist() == ["s1", "s2", "s3", "s4"]
    np.testing.assert_allclose(relatedness_df.to_numpy(), expected, equal_nan=True)
    np.testing.assert_allclose(relatedness, relatedness.T, equal_nan=True)


def test_relatedness_kinship_scale_is_half_relationship():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 2.0],
                [2.0, 2.0, 0.0],
                [0.0, 1.0, 2.0],
            ]
        )
    )

    relationship = snpobj.relatedness(scale="relationship")
    kinship = snpobj.relatedness(scale="kinship")

    np.testing.assert_allclose(kinship, relationship / 2.0, equal_nan=True)


def test_relatedness_respects_pairwise_missing_counts_and_min_variants():
    dosages = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, np.nan, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    snpobj = SNPObject(
        genotypes=dosages,
        samples=np.array(["dup1", "dup2", "other"], dtype=object),
    )
    expected, counts = _manual_grm(dosages)

    relatedness = snpobj.relatedness()
    min_count_relatedness = snpobj.relatedness(min_variants=3)
    pairs = snpobj.flag_related_pairs(threshold=0.4)

    np.testing.assert_allclose(relatedness, expected, equal_nan=True)
    assert counts[0, 1] == 2
    assert np.isnan(min_count_relatedness[0, 1])
    assert pairs["sample_1"].tolist() == ["dup1"]
    assert pairs["sample_2"].tolist() == ["dup2"]
    assert pairs["n_variants"].tolist() == [2]
    np.testing.assert_allclose(pairs["kinship"], np.array([0.5]))


def test_relatedness_from_bgen_probabilities_matches_dosage_grm():
    dosages = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    one_hot = np.eye(3, dtype=np.float32)
    gp = one_hot[dosages.astype(int)]
    dosage_snpobj = SNPObject(genotypes=dosages)
    gp_snpobj = SNPObject(calldata_gp=gp)

    np.testing.assert_allclose(
        gp_snpobj.relatedness(),
        dosage_snpobj.relatedness(),
        equal_nan=True,
    )


def test_flag_related_pairs_returns_empty_dataframe_with_expected_columns():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
            ]
        )
    )

    pairs = snpobj.flag_related_pairs(threshold=1.0)

    assert pairs.empty
    assert pairs.columns.tolist() == [
        "sample_index_1",
        "sample_index_2",
        "relationship",
        "kinship",
        "n_variants",
    ]


def test_prune_related_samples_removes_lower_call_rate_sample_and_preserves_metadata():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, np.nan, 2.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        samples=np.array(["dup1", "dup2", "other"], dtype=object),
        sample_fid=np.array(["F1", "F2", "F3"], dtype=object),
    )
    snpobj.sample_metadata = pd.DataFrame(
        {"sample": ["dup1", "dup2", "other"], "batch": ["A", "B", "C"]}
    )

    filtered = snpobj.prune_related_samples(threshold=0.4)

    assert filtered.samples.tolist() == ["dup1", "other"]
    assert filtered.sample_fid.tolist() == ["F1", "F3"]
    assert filtered.sample_metadata["sample"].tolist() == ["dup1", "other"]


def test_prune_related_samples_tie_breaks_by_later_sample_index():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 2.0],
                [2.0, 2.0, 0.0],
            ]
        ),
        samples=np.array(["first", "second", "other"], dtype=object),
    )

    filtered = snpobj.prune_related_samples(threshold=0.4)

    assert filtered.samples.tolist() == ["first", "other"]


def test_relatedness_methods_support_sample_subsets():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 0.0, 0.0],
            ]
        ),
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
    )

    relatedness = snpobj.relatedness(samples=["s1", "s2", "s3"], as_dataframe=True)
    pairs = snpobj.flag_related_pairs(threshold=0.4, samples=np.array([True, True, True, False]))

    assert relatedness.index.tolist() == ["s1", "s2", "s3"]
    assert relatedness.columns.tolist() == ["s1", "s2", "s3"]
    assert pairs["sample_1"].tolist() == ["s1"]
    assert pairs["sample_2"].tolist() == ["s2"]


def test_relatedness_ibd_uses_weighted_segment_lengths_and_dataframe_labels():
    snpobj = SNPObject(samples=np.array(["A", "B", "C"], dtype=object))
    ibdobj = _toy_ibdobj()

    relatedness = snpobj.relatedness(
        method="ibd",
        ibdobj=ibdobj,
        genome_length_cm=100.0,
    )
    kinship_df = snpobj.relatedness(
        method="ibd",
        ibdobj=ibdobj,
        genome_length_cm=100.0,
        scale="kinship",
        as_dataframe=True,
    )

    expected = np.array(
        [
            [1.0, 0.2, 0.01],
            [0.2, 1.0, 0.025],
            [0.01, 0.025, 1.0],
        ]
    )
    np.testing.assert_allclose(relatedness, expected)
    assert kinship_df.index.tolist() == ["A", "B", "C"]
    assert kinship_df.columns.tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(kinship_df.to_numpy(), expected / 2.0)


def test_relatedness_ibd_filters_segments_by_length_type_and_samples():
    snpobj = SNPObject(samples=np.array(["A", "B", "C"], dtype=object))
    ibdobj = _toy_ibdobj()

    only_ibd2 = snpobj.relatedness(
        method="ibd",
        ibdobj=ibdobj,
        genome_length_cm=100.0,
        segment_types=["IBD2"],
    )
    long_segments = snpobj.relatedness(
        method="ibd",
        ibdobj=ibdobj,
        genome_length_cm=100.0,
        min_segment_cm=6.0,
        samples=["A", "B"],
    )

    np.testing.assert_allclose(
        only_ibd2,
        np.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        long_segments,
        np.array(
            [
                [1.0, 0.2],
                [0.2, 1.0],
            ]
        ),
    )


def test_flag_related_pairs_ibd_reports_segment_metrics():
    snpobj = SNPObject(samples=np.array(["A", "B", "C"], dtype=object))
    ibdobj = _toy_ibdobj()

    pairs = snpobj.flag_related_pairs(
        method="ibd",
        ibdobj=ibdobj,
        genome_length_cm=100.0,
        threshold=0.09,
    )

    assert pairs.columns.tolist() == [
        "sample_index_1",
        "sample_index_2",
        "sample_1",
        "sample_2",
        "relationship",
        "kinship",
        "n_segments",
        "ibd1_cm",
        "ibd2_cm",
        "weighted_ibd_cm",
    ]
    assert pairs["sample_1"].tolist() == ["A"]
    assert pairs["sample_2"].tolist() == ["B"]
    np.testing.assert_allclose(pairs["relationship"], np.array([0.2]))
    np.testing.assert_allclose(pairs["kinship"], np.array([0.1]))
    assert pairs["n_segments"].tolist() == [2]
    np.testing.assert_allclose(pairs["ibd1_cm"], np.array([20.0]))
    np.testing.assert_allclose(pairs["ibd2_cm"], np.array([10.0]))
    np.testing.assert_allclose(pairs["weighted_ibd_cm"], np.array([40.0]))


def test_prune_related_samples_ibd_does_not_require_genotypes_and_preserves_metadata():
    snpobj = SNPObject(
        samples=np.array(["A", "B", "C"], dtype=object),
        sample_fid=np.array(["F1", "F2", "F3"], dtype=object),
    )
    snpobj.sample_metadata = pd.DataFrame(
        {"sample": ["A", "B", "C"], "batch": ["x", "y", "z"]}
    )
    ibdobj = _toy_ibdobj()

    filtered = snpobj.prune_related_samples(
        method="ibd",
        ibdobj=ibdobj,
        genome_length_cm=100.0,
        threshold=0.09,
    )

    assert filtered.samples.tolist() == ["A", "C"]
    assert filtered.sample_fid.tolist() == ["F1", "F3"]
    assert filtered.sample_metadata["sample"].tolist() == ["A", "C"]


def test_relatedness_validates_parameters_and_sample_selector():
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="method"):
        snpobj.relatedness(method="king")
    with pytest.raises(ValueError, match="scale"):
        snpobj.relatedness(scale="invalid")
    with pytest.raises(ValueError, match="min_variants"):
        snpobj.relatedness(min_variants=0)
    with pytest.raises(ValueError, match="block_size"):
        snpobj.relatedness(block_size=0)
    with pytest.raises(ValueError, match="threshold"):
        snpobj.flag_related_pairs(threshold=-0.01)
    with pytest.raises(ValueError, match="threshold"):
        snpobj.prune_related_samples(threshold=np.nan)
    with pytest.raises(ValueError, match="not found"):
        snpobj.relatedness(samples=["missing"])


def test_relatedness_ibd_validates_required_inputs_and_filters():
    snpobj = SNPObject(samples=np.array(["A", "B"], dtype=object))
    ibdobj = _toy_ibdobj()
    no_segment_type = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1], dtype=int),
        end=np.array([10], dtype=int),
        length_cm=np.array([1.0], dtype=float),
    )
    no_length = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1], dtype=int),
        end=np.array([10], dtype=int),
    )

    with pytest.raises(ValueError, match="ibdobj"):
        snpobj.relatedness(method="ibd")
    with pytest.raises(ValueError, match="genome_length_cm"):
        snpobj.relatedness(method="ibd", ibdobj=ibdobj, genome_length_cm=0)
    with pytest.raises(ValueError, match="min_segment_cm"):
        snpobj.relatedness(method="ibd", ibdobj=ibdobj, min_segment_cm=-1)
    with pytest.raises(ValueError, match="segment_type"):
        snpobj.relatedness(method="ibd", ibdobj=no_segment_type, segment_types=["IBD1"])
    with pytest.raises(ValueError, match="length_cm"):
        snpobj.relatedness(method="ibd", ibdobj=no_length)
    with pytest.raises(ValueError, match="Sample names"):
        SNPObject(genotypes=np.zeros((1, 2))).relatedness(method="ibd", ibdobj=ibdobj)


def test_imputation_r2_extracts_common_info_keys_and_filters_variants():
    snpobj = SNPObject(
        genotypes=np.zeros((6, 1), dtype=float),
        variants_id=np.array(["v1", "v2", "v3", "v4", "v5", "v6"], dtype=object),
        variants_info=np.array(
            [
                "R2=0.95;AF=0.1",
                "DR2=0.71",
                "INFO=0.82",
                b"imp=0.4;AF=0.2",
                {"rsq": "0.77"},
                ".",
            ],
            dtype=object,
        ),
    )

    r2 = snpobj.imputation_r2(source="info")
    r2_df = snpobj.imputation_r2(source="info", as_dataframe=True)
    filtered = snpobj.filter_imputation_quality(min_r2=0.8, source="info")

    np.testing.assert_allclose(r2, np.array([0.95, 0.71, 0.82, 0.4, 0.77, np.nan]), equal_nan=True)
    assert r2_df.columns.tolist() == ["imputation_r2"]
    np.testing.assert_allclose(r2_df["imputation_r2"], r2, equal_nan=True)
    assert filtered.variants_id.tolist() == ["v1", "v3"]


def test_imputation_r2_honors_custom_info_key_order():
    snpobj = SNPObject(
        genotypes=np.zeros((2, 1), dtype=float),
        variants_info=np.array(["R2=0.1;CUSTOM=0.9", "CUSTOM=0.85"], dtype=object),
    )

    np.testing.assert_allclose(
        snpobj.imputation_r2(source="info", info_keys=["CUSTOM", "R2"]),
        np.array([0.9, 0.85]),
    )


def test_imputation_r2_from_unphased_genotype_probabilities():
    snpobj = SNPObject(
        calldata_gp=np.array(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                [
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                ],
            ],
            dtype=np.float32,
        ),
        variants_id=np.array(["high_quality", "low_quality"], dtype=object),
    )

    np.testing.assert_allclose(snpobj.imputation_r2(source="gp"), np.array([1.0, 0.0]))
    filtered = snpobj.filter_imputation_quality(min_r2=0.8, source="gp")
    assert filtered.variants_id.tolist() == ["high_quality"]


def test_imputation_r2_from_phased_genotype_probabilities():
    snpobj = SNPObject(
        calldata_gp=np.array(
            [
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                ],
            ],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(snpobj.imputation_r2(source="gp"), np.array([1.0, 0.0]))


def test_imputation_r2_auto_prefers_info_and_fills_missing_from_gp():
    snpobj = SNPObject(
        calldata_gp=np.array(
            [
                [
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                    [0.25, 0.5, 0.25],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
            ],
            dtype=np.float32,
        ),
        variants_info=np.array(["R2=0.9", "."], dtype=object),
    )

    np.testing.assert_allclose(snpobj.imputation_r2(source="auto"), np.array([0.9, 1.0]))
    np.testing.assert_allclose(snpobj.imputation_r2(source="gp"), np.array([0.0, 1.0]))


def test_imputation_r2_from_dosages_and_auto_fallback():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        variants_id=np.array(["spread", "flat", "monomorphic"], dtype=object),
    )

    expected = np.array([1.0, 0.0, np.nan])
    np.testing.assert_allclose(snpobj.imputation_r2(source="dosage"), expected, equal_nan=True)
    np.testing.assert_allclose(snpobj.imputation_r2(source="auto"), expected, equal_nan=True)
    assert snpobj.filter_imputation_quality(min_r2=0.8, source="dosage").variants_id.tolist() == ["spread"]


@pytest.mark.parametrize("min_r2", [-0.01, 1.01, "not-a-number"])
def test_filter_imputation_quality_validates_threshold(min_r2):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="min_r2"):
        snpobj.filter_imputation_quality(min_r2=min_r2)


def test_imputation_r2_validates_source():
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="source"):
        snpobj.imputation_r2(source="unknown")


def test_hwe_pvalue_from_3d_hard_calls_ignores_missing_calls():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 0], [0, 1], [0, 1], [1, 1], [1, 1]],
                [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1]],
                [[0, -1], [np.nan, 0], [0, 0], [0, 1], [1, 1], [-1, -1]],
                [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            ]
        ),
        variants_id=np.array(["balanced", "excess_hom", "partial_missing", "all_missing"], dtype=object),
    )

    p_values = snpobj.hwe_pvalue()

    np.testing.assert_allclose(
        p_values,
        np.array([0.4805194805194805, 0.021645021645021644, 1.0, np.nan]),
        equal_nan=True,
    )
    hwe_df = snpobj.hwe_pvalue(as_dataframe=True)
    assert hwe_df.columns.tolist() == ["hwe_pvalue"]
    np.testing.assert_allclose(hwe_df["hwe_pvalue"], p_values, equal_nan=True)


def test_hwe_pvalue_and_filter_support_control_sample_subset():
    controls = np.array([f"ctrl{i}" for i in range(10)], dtype=object)
    cases = np.array([f"case{i}" for i in range(10)], dtype=object)
    samples = np.concatenate([controls, cases])
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            ],
            dtype=float,
        ),
        samples=samples,
        variants_id=np.array(["control_pass", "control_fail"], dtype=object),
    )
    control_mask = np.isin(samples, controls)

    p_by_name = snpobj.hwe_pvalue(samples=controls)
    p_by_mask = snpobj.hwe_pvalue(samples=control_mask)
    filtered = snpobj.filter_hwe(min_p=0.05, samples=controls)

    np.testing.assert_allclose(
        p_by_name,
        np.array([1.0, 0.0013639611162830976]),
    )
    np.testing.assert_allclose(p_by_mask, p_by_name)
    assert filtered.variants_id.tolist() == ["control_pass"]


def test_hwe_pvalue_supports_sample_indexes_and_deduplicates_them():
    snpobj = SNPObject(
        genotypes=np.array([[0, 0, 1, 1, 1, 1, 1, 1, 2, 2]], dtype=float),
        samples=np.array([f"s{i}" for i in range(10)], dtype=object),
    )

    p_by_index = snpobj.hwe_pvalue(samples=[0, 1, 2, 3, 4, 5, 6, 7, 8, -1, 0])

    np.testing.assert_allclose(p_by_index, np.array([1.0]))


def test_hwe_pvalue_rejects_fractional_dosages():
    snpobj = SNPObject(genotypes=np.array([[0.0, 0.5, 2.0]]))

    with pytest.raises(ValueError, match="hard-call dosages"):
        snpobj.hwe_pvalue()


def test_hwe_pvalue_rejects_non_biallelic_allele_calls():
    snpobj = SNPObject(genotypes=np.array([[[0, 0], [0, 2], [1, 1]]], dtype=np.int8))

    with pytest.raises(ValueError, match="0/1 alleles"):
        snpobj.hwe_pvalue()


@pytest.mark.parametrize("min_p", [-0.01, 1.01, "not-a-number"])
def test_filter_hwe_validates_threshold(min_p):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="min_p"):
        snpobj.filter_hwe(min_p=min_p)


def test_hwe_pvalue_validates_sample_subset():
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="not found"):
        snpobj.hwe_pvalue(samples=["missing"])
    with pytest.raises(ValueError, match="Boolean 'samples' mask"):
        snpobj.hwe_pvalue(samples=[True, False])


def test_ld_prune_mask_greedily_removes_later_correlated_variants():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                [0.0, 1.0, 0.0, 2.0, 1.0, 2.0],
                [0.0, 1.0, 0.0, 2.0, 1.0, 2.0],
            ]
        ),
        variants_id=np.array(["v1", "v2", "v3", "v4"], dtype=object),
    )

    mask = snpobj.ld_prune_mask(window_size=2, step_size=1, r2_threshold=0.8)
    filtered = snpobj.filter_ld_pruned(window_size=2, step_size=1, r2_threshold=0.8)
    alias_filtered = snpobj.ld_prune(window_size=2, step_size=1, r2_threshold=0.8)

    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))
    assert filtered.variants_id.tolist() == ["v1", "v3"]
    assert alias_filtered.variants_id.tolist() == ["v1", "v3"]


def test_ld_prune_uses_selected_samples_to_estimate_ld():
    controls = np.array([f"ctrl{i}" for i in range(4)], dtype=object)
    cases = np.array([f"case{i}" for i in range(4)], dtype=object)
    samples = np.concatenate([controls, cases])
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        ),
        samples=samples,
        variants_id=np.array(["v1", "v2"], dtype=object),
    )

    all_sample_mask = snpobj.ld_prune_mask(window_size=2, step_size=1, r2_threshold=0.8)
    control_mask = snpobj.ld_prune_mask(
        window_size=2,
        step_size=1,
        r2_threshold=0.8,
        samples=controls,
    )
    control_bool_mask = snpobj.ld_prune_mask(
        window_size=2,
        step_size=1,
        r2_threshold=0.8,
        samples=np.isin(samples, controls),
    )

    np.testing.assert_array_equal(all_sample_mask, np.array([True, True]))
    np.testing.assert_array_equal(control_mask, np.array([True, False]))
    np.testing.assert_array_equal(control_bool_mask, control_mask)


def test_ld_prune_supports_3d_calls_and_ignores_pairwise_missing_samples():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1], [1, 1], [-1, -1]],
                [[0, 0], [0, 1], [1, 1], [0, -1]],
            ],
            dtype=np.int8,
        ),
        variants_id=np.array(["v1", "v2"], dtype=object),
    )

    pruned = snpobj.ld_prune_mask(window_size=2, step_size=1, r2_threshold=0.8, min_samples=3)
    insufficient_overlap = snpobj.ld_prune_mask(
        window_size=2,
        step_size=1,
        r2_threshold=0.8,
        min_samples=4,
    )

    np.testing.assert_array_equal(pruned, np.array([True, False]))
    np.testing.assert_array_equal(insufficient_overlap, np.array([True, True]))


def test_ld_prune_accepts_fractional_dosages():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 0.5, 1.5, 2.0],
                [0.0, 0.5, 1.5, 2.0],
            ]
        )
    )

    mask = snpobj.ld_prune_mask(window_size=2, step_size=1, r2_threshold=0.8)

    np.testing.assert_array_equal(mask, np.array([True, False]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_size": 1}, "window_size"),
        ({"window_size": 2.5}, "window_size"),
        ({"step_size": 0}, "step_size"),
        ({"r2_threshold": -0.01}, "r2_threshold"),
        ({"r2_threshold": 1.01}, "r2_threshold"),
        ({"min_samples": 1}, "min_samples"),
    ],
)
def test_ld_prune_validates_parameters(kwargs, message):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match=message):
        snpobj.ld_prune_mask(**kwargs)


def test_ld_prune_rejects_invalid_genotype_encodings():
    dosage_snpobj = SNPObject(genotypes=np.array([[0.0, 1.0, 2.5], [0.0, 1.0, 2.0]]))
    allele_snpobj = SNPObject(genotypes=np.array([[[0, 0], [0, 2]], [[0, 0], [0, 1]]], dtype=np.int8))

    with pytest.raises(ValueError, match="between 0 and 2"):
        dosage_snpobj.ld_prune_mask(window_size=2, step_size=1)
    with pytest.raises(ValueError, match="0/1"):
        allele_snpobj.ld_prune_mask(window_size=2, step_size=1)


def test_sample_heterozygosity_and_inbreeding_from_2d_hard_calls():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 1.0, 2.0, 1.0],
                [0.0, 0.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [np.nan, 0.0, -1.0, 2.0],
            ]
        ),
        samples=np.array(["s1", "s2", "s3", "s4"], dtype=object),
    )

    heterozygosity = snpobj.sample_heterozygosity()
    inbreeding = snpobj.sample_inbreeding_coefficient()
    heterozygosity_df = snpobj.sample_heterozygosity(samples=["s2", "s4"], as_dataframe=True)

    np.testing.assert_allclose(heterozygosity, np.array([1.0 / 3.0, 0.5, 1.0 / 3.0, 0.75]))
    np.testing.assert_allclose(
        inbreeding,
        np.array([0.3191489361702128, -0.015873015873015817, 0.3191489361702128, -0.5238095238095237]),
    )
    assert heterozygosity_df.columns.tolist() == ["sample_index", "sample", "heterozygosity"]
    assert heterozygosity_df["sample"].tolist() == ["s2", "s4"]
    np.testing.assert_allclose(heterozygosity_df["heterozygosity"], np.array([0.5, 0.75]))


def test_sample_heterozygosity_from_3d_calls_treats_partial_missing_as_missing():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1], [1, 1], [-1, -1]],
                [[0, 1], [0, -1], [1, 1], [np.nan, 0]],
            ]
        )
    )

    heterozygosity = snpobj.sample_heterozygosity()

    np.testing.assert_allclose(heterozygosity, np.array([0.5, 1.0, 0.0, np.nan]), equal_nan=True)


def test_flag_heterozygosity_outliers_returns_sample_qc_report():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        ),
        samples=np.array(["s1", "s2", "s3", "s4", "s5", "s6"], dtype=object),
    )

    report = snpobj.flag_heterozygosity_outliers(n_sd=2)

    assert report.columns.tolist() == [
        "sample_index",
        "sample",
        "called_genotypes",
        "observed_heterozygotes",
        "expected_heterozygotes",
        "heterozygosity",
        "inbreeding_coefficient",
        "heterozygosity_z",
        "heterozygosity_outlier",
    ]
    assert report["sample"].tolist() == ["s1", "s2", "s3", "s4", "s5", "s6"]
    np.testing.assert_allclose(report["heterozygosity"], np.array([0, 0, 0, 0, 0, 1], dtype=float))
    assert report["heterozygosity_outlier"].tolist() == [False, False, False, False, False, True]


def test_sample_inbreeding_is_nan_when_expected_heterozygosity_is_zero():
    snpobj = SNPObject(genotypes=np.array([[0, 0, 0], [2, 2, 2]], dtype=float))

    np.testing.assert_allclose(
        snpobj.sample_inbreeding_coefficient(),
        np.array([np.nan, np.nan, np.nan]),
        equal_nan=True,
    )


def test_sample_heterozygosity_rejects_invalid_hard_calls():
    dosage_snpobj = SNPObject(genotypes=np.array([[0.0, 0.5, 2.0]]))
    allele_snpobj = SNPObject(genotypes=np.array([[[0, 0], [0, 2], [1, 1]]], dtype=np.int8))

    with pytest.raises(ValueError, match="hard-call dosages"):
        dosage_snpobj.sample_heterozygosity()
    with pytest.raises(ValueError, match="0/1 alleles"):
        allele_snpobj.sample_heterozygosity()


@pytest.mark.parametrize("n_sd", [-1, np.nan, "not-a-number"])
def test_flag_heterozygosity_outliers_validates_n_sd(n_sd):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="n_sd"):
        snpobj.flag_heterozygosity_outliers(n_sd=n_sd)


def test_filter_samples_reorders_sample_metadata():
    snpobj = _toy_snpobj()
    snpobj.sample_metadata = pd.DataFrame(
        {"sample": ["s1", "s2", "s3"], "population": ["A", "A", "B"]},
        index=[10, 11, 12],
    )

    filtered = snpobj.filter_samples(samples=["s3", "s1"], reorder=True)

    assert filtered.samples.tolist() == ["s3", "s1"]
    assert filtered.sample_metadata["sample"].tolist() == ["s3", "s1"]
    assert filtered.sample_metadata.index.tolist() == [0, 1]


def test_filter_samples_excludes_sample_metadata_rows():
    snpobj = _toy_snpobj()
    snpobj.sample_metadata = pd.DataFrame(
        {"sample": ["s1", "s2", "s3"], "population": ["A", "A", "B"]}
    )

    filtered = snpobj.filter_samples(samples=["s2"], include=False)

    assert filtered.samples.tolist() == ["s1", "s3"]
    assert filtered.sample_metadata["sample"].tolist() == ["s1", "s3"]


def test_filter_samples_rejects_misaligned_sample_metadata():
    snpobj = _toy_snpobj()
    snpobj.sample_metadata = pd.DataFrame({"sample": ["s1", "s2"]})

    with pytest.raises(ValueError, match="sample_metadata"):
        snpobj.filter_samples(indexes=[0])


def test_duplicate_sample_ids_report_and_filter_preserves_metadata():
    genotypes = np.arange(10, dtype=float).reshape(2, 5)
    snpobj = SNPObject(
        genotypes=genotypes,
        samples=np.array(["s1", "s2", "s1", "s3", "s2"], dtype=object),
        sample_fid=np.array(["F1a", "F2a", "F1b", "F3", "F2b"], dtype=object),
        sample_sex=np.array(["1", "2", "1", "2", "1"], dtype=object),
    )
    snpobj.sample_metadata = pd.DataFrame(
        {"sample": ["s1", "s2", "s1", "s3", "s2"], "row": [0, 1, 2, 3, 4]}
    )

    mask = snpobj.duplicate_sample_ids()
    report = snpobj.duplicate_sample_ids(as_dataframe=True)
    filtered_first = snpobj.filter_duplicate_samples()
    filtered_last = snpobj.filter_duplicate_samples(keep="last")
    filtered_all = snpobj.filter_duplicate_samples(keep=False)

    np.testing.assert_array_equal(mask, np.array([False, False, True, False, True]))
    assert report.columns.tolist() == ["sample_index", "sample", "is_duplicate", "duplicate_group"]
    assert report["duplicate_group"].tolist() == [0, 1, 0, -1, 1]
    assert filtered_first.samples.tolist() == ["s1", "s2", "s3"]
    assert filtered_first.sample_fid.tolist() == ["F1a", "F2a", "F3"]
    assert filtered_first.sample_sex.tolist() == ["1", "2", "2"]
    assert filtered_first.sample_metadata["row"].tolist() == [0, 1, 3]
    np.testing.assert_array_equal(filtered_first.genotypes, genotypes[:, [0, 1, 3]])
    assert filtered_last.samples.tolist() == ["s1", "s3", "s2"]
    np.testing.assert_array_equal(filtered_last.genotypes, genotypes[:, [2, 3, 4]])
    assert filtered_all.samples.tolist() == ["s3"]
    np.testing.assert_array_equal(filtered_all.genotypes, genotypes[:, [3]])


def test_duplicate_variant_ids_ignore_missing_ids_and_filter():
    snpobj = SNPObject(
        genotypes=np.zeros((7, 2), dtype=float),
        variants_id=np.array(["rs1", ".", "rs2", "rs1", "", "rs2", "."], dtype=object),
        variants_pos=np.arange(7),
    )

    mask = snpobj.duplicate_variant_ids()
    report = snpobj.duplicate_variant_ids(as_dataframe=True)
    filtered = snpobj.filter_duplicate_variants(by="id")
    all_missing_included = snpobj.duplicate_variant_ids(ignore_missing=False)

    np.testing.assert_array_equal(mask, np.array([False, False, False, True, False, True, False]))
    assert report["variant_id"].tolist() == ["rs1", ".", "rs2", "rs1", "", "rs2", "."]
    assert report["duplicate_group"].tolist() == [0, -1, 1, 0, -1, 1, -1]
    assert filtered.variants_id.tolist() == ["rs1", ".", "rs2", "", "."]
    np.testing.assert_array_equal(
        all_missing_included,
        np.array([False, False, False, True, False, True, True]),
    )


def test_duplicate_variant_coordinates_support_configurable_fields_and_filter():
    snpobj = SNPObject(
        genotypes=np.zeros((5, 2), dtype=float),
        variants_id=np.array(["v1", "v2", "v3", "v4", "v5"], dtype=object),
        variants_chrom=np.array(["1", "1", "1", "1", "2"], dtype=object),
        variants_pos=np.array([10, 10, 10, 20, 10], dtype=np.int32),
        variants_ref=np.array(["A", "A", "T", "A", "A"], dtype=object),
        variants_alt=np.array(["C", "C", "A", "C", "C"], dtype=object),
    )

    coordinate_mask = snpobj.duplicate_variant_coordinates()
    position_mask = snpobj.duplicate_variant_coordinates(fields=("chrom", "pos"))
    report = snpobj.duplicate_variant_coordinates(fields="chrom,pos", as_dataframe=True)
    filtered = snpobj.filter_duplicate_variants(by="coordinates")

    np.testing.assert_array_equal(coordinate_mask, np.array([False, True, False, False, False]))
    np.testing.assert_array_equal(position_mask, np.array([False, True, True, False, False]))
    assert report.columns.tolist() == ["variant_index", "chrom", "pos", "is_duplicate", "duplicate_group"]
    assert report["duplicate_group"].tolist() == [0, 0, 0, -1, -1]
    assert filtered.variants_id.tolist() == ["v1", "v3", "v4", "v5"]


def test_duplicate_qc_validates_inputs():
    with pytest.raises(ValueError, match="samples"):
        SNPObject(genotypes=np.zeros((1, 2))).duplicate_sample_ids()

    with pytest.raises(ValueError, match="variants_id"):
        SNPObject(genotypes=np.zeros((1, 2))).duplicate_variant_ids()

    snpobj = SNPObject(
        genotypes=np.zeros((2, 2)),
        variants_id=np.array(["v1", "v1"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_pos=np.array([1, 1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="keep"):
        snpobj.duplicate_variant_ids(keep="middle")
    with pytest.raises(ValueError, match="'by'"):
        snpobj.filter_duplicate_variants(by="unknown")
    with pytest.raises(ValueError, match="variants_ref"):
        snpobj.duplicate_variant_coordinates()
    with pytest.raises(ValueError, match="fields"):
        snpobj.duplicate_variant_coordinates(fields=("chrom", "quality"))


def test_filter_maf_from_3d_genotypes_ignores_missing_calls():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1]],
                [[1, 1], [1, 1]],
                [[0, -1], [-1, -1]],
                [[-1, -1], [-1, -1]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.int8,
        ),
        variants_id=np.array(["v1", "v2", "v3", "v4", "v5"], dtype=object),
    )

    filtered = snpobj.filter_maf(maf=0.25)

    assert filtered.variants_id.tolist() == ["v1", "v5"]


def test_allele_counts_maf_and_mac_from_3d_genotypes_ignore_missing_calls():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1]],
                [[1, 1], [1, 1]],
                [[0, -1], [-1, -1]],
                [[-1, -1], [-1, -1]],
                [[0, 1], [1, 1]],
            ],
            dtype=np.int8,
        ),
        variants_id=np.array(["v1", "v2", "v3", "v4", "v5"], dtype=object),
    )

    alt_counts, called = snpobj.allele_counts(return_called=True)

    np.testing.assert_allclose(alt_counts, np.array([1.0, 4.0, 0.0, np.nan, 3.0]), equal_nan=True)
    np.testing.assert_array_equal(called, np.array([4, 4, 1, 0, 4]))
    np.testing.assert_allclose(snpobj.maf(), np.array([0.25, 0.0, 0.0, np.nan, 0.25]), equal_nan=True)
    np.testing.assert_allclose(snpobj.mac(), np.array([1.0, 0.0, 0.0, np.nan, 1.0]), equal_nan=True)


def test_filter_maf_from_2d_dosages_ignores_missing_calls():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
                [2.0, -1.0, 2.0],
                [-1.0, -1.0, -1.0],
            ]
        ),
        variants_id=np.array(["v1", "v2", "v3", "v4"], dtype=object),
    )

    filtered = snpobj.filter_maf(maf=0.2)

    assert filtered.variants_id.tolist() == ["v1"]


def test_allele_counts_maf_and_mac_from_2d_dosages_ignore_missing_calls():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
                [2.0, -1.0, 2.0],
                [-1.0, -1.0, -1.0],
            ]
        ),
        variants_id=np.array(["v1", "v2", "v3", "v4"], dtype=object),
    )

    alt_counts, called = snpobj.allele_counts(return_called=True)

    np.testing.assert_allclose(alt_counts, np.array([3.0, 1.0, 4.0, np.nan]), equal_nan=True)
    np.testing.assert_array_equal(called, np.array([6, 6, 4, 0]))
    np.testing.assert_allclose(snpobj.maf(), np.array([0.5, 1.0 / 6.0, 0.0, np.nan]), equal_nan=True)
    np.testing.assert_allclose(snpobj.mac(), np.array([3.0, 1.0, 0.0, np.nan]), equal_nan=True)


def test_allele_count_stats_support_grouped_dataframe_output():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1], [1, 1]],
                [[0, 1], [0, 0], [0, -1]],
            ],
            dtype=np.int8,
        )
    )
    labels = ["POP2", "POP1", "POP1"]

    counts = snpobj.allele_counts(sample_labels=labels, as_dataframe=True)
    maf = snpobj.maf(sample_labels=labels, as_dataframe=True)
    mac = snpobj.mac(sample_labels=labels, as_dataframe=True)

    assert counts.columns.tolist() == ["POP1", "POP2"]
    np.testing.assert_allclose(counts.to_numpy(), np.array([[3.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(maf.to_numpy(), np.array([[0.25, 0.0], [0.0, 0.5]]))
    np.testing.assert_allclose(mac.to_numpy(), np.array([[1.0, 0.0], [0.0, 1.0]]))


@pytest.mark.parametrize("maf", [-0.01, 0.51, "not-a-number"])
def test_filter_maf_validates_threshold(maf):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="maf"):
        snpobj.filter_maf(maf=maf)


def test_filter_mac_keeps_variants_at_or_above_threshold():
    snpobj = SNPObject(
        genotypes=np.array(
            [
                [[0, 0], [0, 1]],
                [[1, 1], [1, 1]],
                [[0, 1], [1, 1]],
                [[-1, -1], [-1, -1]],
            ],
            dtype=np.int8,
        ),
        variants_id=np.array(["v1", "v2", "v3", "v4"], dtype=object),
    )

    filtered = snpobj.filter_mac(mac=1)

    assert filtered.variants_id.tolist() == ["v1", "v3"]


@pytest.mark.parametrize("mac", [-1, "not-a-number"])
def test_filter_mac_validates_threshold(mac):
    snpobj = _toy_snpobj()

    with pytest.raises(ValueError, match="mac"):
        snpobj.filter_mac(mac=mac)


def test_to_dosage_preserves_one_missing_sentinel():
    genotypes = np.array(
        [
            [[0, 1], [0, -1], [-1, -1]],
            [[1, 1], [0, 0], [1, 0]],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(genotypes=genotypes)

    expected = np.array([[1, -1, -1], [2, 0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(snpobj.to_dosage().genotypes, expected)
    np.testing.assert_array_equal(snpobj.dosage(), expected.astype(np.float32))
    assert snpobj.is_dosage is False
    assert snpobj.to_dosage().is_dosage is True


def test_to_dosage_rejects_multiallelic_allele_indexes():
    snpobj = SNPObject(genotypes=np.array([[[1, 2]]], dtype=np.int8))

    with pytest.raises(ValueError, match="dosage.*biallelic"):
        snpobj.to_dosage()


def test_to_dosage_rejects_multiallelic_metadata_without_observed_alt2():
    snpobj = SNPObject(
        genotypes=np.array([[[0, 1]]], dtype=np.int8),
        variants_alt=np.array(["C,G"], dtype=object),
    )

    with pytest.raises(ValueError, match="dosage.*biallelic"):
        snpobj.to_dosage()


def test_concat_variants_preserves_sample_metadata_and_validates_order():
    left = _toy_snpobj().filter_variants(indexes=[0, 1])
    right = _toy_snpobj().filter_variants(indexes=[2, 3])

    concatenated = SNPObject.concat_variants([left, right])

    assert concatenated.variants_id.tolist() == ["v1", "v2", "v3", "v4"]
    assert concatenated.samples.tolist() == ["s1", "s2", "s3"]
    assert concatenated.sample_fid.tolist() == ["A", "A", "B"]

    wrong_order = right.filter_samples(samples=["s3", "s2", "s1"], reorder=True)
    with pytest.raises(ValueError, match="sample IDs differ"):
        left.concat(wrong_order)


def test_merge_inplace_preserves_sample_sex_when_left_has_no_fid():
    left = SNPObject(
        genotypes=np.zeros((2, 1, 2), dtype=np.int8),
        samples=np.array(["s1"], dtype=object),
        sample_sex=np.array(["1"], dtype=object),
    )
    right = SNPObject(
        genotypes=np.ones((2, 1, 2), dtype=np.int8),
        samples=np.array(["s2"], dtype=object),
        sample_sex=np.array(["2"], dtype=object),
    )

    merged = left.merge(right, inplace=True)

    assert merged is left
    assert left.samples.tolist() == ["s1", "s2"]
    assert left.sample_sex.tolist() == ["1", "2"]
    assert left.genotypes.shape == (2, 2, 2)


def test_correct_flipped_variants_inverts_dosages_and_preserves_missing_values():
    query = SNPObject(
        genotypes=np.array([[0.0, 1.0, 2.0, -1.0, np.nan]]),
        variants_chrom=np.array(["1"]),
        variants_pos=np.array([10]),
        variants_ref=np.array(["A"]),
        variants_alt=np.array(["G"]),
    )
    reference = SNPObject(
        variants_chrom=np.array(["1"]),
        variants_pos=np.array([10]),
        variants_ref=np.array(["G"]),
        variants_alt=np.array(["A"]),
    )

    corrected = query.correct_flipped_variants(
        reference,
        check_complement=False,
        log_stats=False,
    )

    np.testing.assert_allclose(
        corrected.genotypes,
        np.array([[2.0, 1.0, 0.0, -1.0, np.nan]]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        query.genotypes,
        np.array([[0.0, 1.0, 2.0, -1.0, np.nan]]),
        equal_nan=True,
    )


def test_correct_flipped_variants_inverts_called_alleles_inplace():
    query = SNPObject(
        genotypes=np.array([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, np.nan]]]),
        variants_chrom=np.array(["1"]),
        variants_pos=np.array([10]),
        variants_ref=np.array(["A"]),
        variants_alt=np.array(["G"]),
    )
    reference = SNPObject(
        variants_chrom=np.array(["1"]),
        variants_pos=np.array([10]),
        variants_ref=np.array(["G"]),
        variants_alt=np.array(["A"]),
    )

    result = query.correct_flipped_variants(
        reference,
        check_complement=False,
        log_stats=False,
        inplace=True,
    )

    assert result is None
    np.testing.assert_allclose(
        query.genotypes,
        np.array([[[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [-1.0, np.nan]]]),
        equal_nan=True,
    )
