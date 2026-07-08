import numpy as np
import pandas as pd
import pytest

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


def test_variant_filters_cover_biallelic_complete_and_polymorphic():
    snpobj = _toy_snpobj()

    filtered = (
        snpobj
        .filter_biallelic_variants(snv_only=True)
        .filter_complete_genotypes()
        .filter_polymorphic_variants()
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


def test_sum_strands_and_dosage_preserve_one_missing_sentinel():
    genotypes = np.array(
        [
            [[0, 1], [0, -1], [-1, -1]],
            [[1, 1], [0, 0], [1, 0]],
        ],
        dtype=np.int8,
    )
    snpobj = SNPObject(genotypes=genotypes)

    expected = np.array([[1, -1, -1], [2, 0, 1]], dtype=np.int8)
    np.testing.assert_array_equal(snpobj.sum_strands().genotypes, expected)
    np.testing.assert_array_equal(snpobj.dosage(), expected.astype(np.float32))


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
