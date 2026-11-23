import pathlib

import numpy as np
import pandas as pd
import pytest

from snputils.processing.dimred_tabular import (
    build_embedding_dataframe,
    embedding_dataframe_from_model,
    embedding_column_names,
    pca_row_haplotype_ids,
    save_embedding_table,
    save_embedding_table_from_model,
)
from snputils.processing.pca import PCA
from snputils.snp.genobj.snpobj import SNPObject


def test_embedding_column_names():
    assert embedding_column_names(3, "PC") == ["PC1", "PC2", "PC3"]
    assert embedding_column_names(2, "MDS") == ["MDS1", "MDS2"]


def test_build_embedding_dataframe_includes_method_column():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    df = build_embedding_dataframe(
        x,
        ind_ids=["a", "b"],
        method="unit",
        component_style="PC",
    )
    assert list(df.columns[:3]) == ["indID", "method", "PC1"]
    assert df["method"].tolist() == ["unit", "unit"]
    assert df["PC1"].tolist() == [1.0, 3.0]


def test_save_embedding_table_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / "emb.tsv"
    x = np.zeros((1, 2))
    save_embedding_table(
        path,
        x,
        ind_ids=["s1"],
        method="t",
        component_style="MDS",
    )
    df = pd.read_csv(path, sep="\t")
    assert "MDS1" in df.columns and "indID" in df.columns


def test_pca_row_haplotype_ids_3d_two_rows_per_sample():
    # (n_snps, n_samples, 2) as SNPObject stores
    gt = np.zeros((3, 2, 2))
    samples = np.array(["x", "y"], dtype=object)
    snp = SNPObject(genotypes=gt, samples=samples)
    out = pca_row_haplotype_ids(snp, average_haplotypes=False)
    assert out == ["x|0", "x|1", "y|0", "y|1"]


def test_pca_expanded_rows_preserve_haplotypes_across_variants():
    gt = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.int8,
    )
    snp = SNPObject(genotypes=gt, samples=np.array(["x", "y"], dtype=object))

    pca = PCA(average_haplotypes=False)
    observed = pca._get_data_from_snpobj(snp)

    np.testing.assert_array_equal(
        observed,
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=float,
        ),
    )


def test_pca_expanded_sample_subset_keeps_both_haplotypes():
    gt = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.int8,
    )
    snp = SNPObject(genotypes=gt, samples=np.array(["x", "y"], dtype=object))

    pca = PCA(average_haplotypes=False)
    observed = pca._get_data_from_snpobj(snp, samples_subset=[1])
    row_ids = pca_row_haplotype_ids(
        snp,
        average_haplotypes=False,
        samples_subset=[1],
    )

    np.testing.assert_array_equal(observed, np.array([[1, 0], [0, 1]], dtype=float))
    assert row_ids == ["y|0", "y|1"]


@pytest.mark.parametrize(
    "genotypes",
    [
        np.array([[[0.0, -1.0]]]),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_pca_rejects_missing_and_nonfinite_genotypes(genotypes):
    snp = SNPObject(genotypes=genotypes)

    with pytest.raises(ValueError, match="missing or non-finite"):
        PCA()._get_data_from_snpobj(snp)


@pytest.mark.parametrize("average_haplotypes", [True, False])
def test_pca_rejects_multiallelic_allele_indexes(average_haplotypes):
    snp = SNPObject(genotypes=np.array([[[0, 2]], [[1, 1]]], dtype=np.int8))

    with pytest.raises(ValueError, match="only biallelic allele calls"):
        PCA()._get_data_from_snpobj(snp, average_haplotypes=average_haplotypes)


@pytest.mark.parametrize(
    "genotypes",
    [
        np.array([[0, 1]], dtype=np.int8),
        np.array([[[0, 1], [1, 1]]], dtype=np.int8),
    ],
)
def test_pca_rejects_multiallelic_variant_metadata(genotypes):
    snp = SNPObject(
        genotypes=genotypes,
        variants_alt=np.array(["C,G"], dtype=object),
    )

    with pytest.raises(ValueError, match="only biallelic variants"):
        PCA()._get_data_from_snpobj(snp)


def test_pca_allows_biallelic_subset_of_object_with_multiallelic_variant():
    snp = SNPObject(
        genotypes=np.array([[[0, 2]], [[0, 1]]], dtype=np.int8),
        variants_alt=np.array(["C,G", "T"], dtype=object),
    )

    observed = PCA()._get_data_from_snpobj(snp, snps_subset=[1])

    np.testing.assert_array_equal(observed, np.array([[0.5]]))


def test_save_embedding_table_from_model_writes(tmp_path: pathlib.Path):
    Md = type("mdPCA", (), {})
    o = Md()
    o.X_new_ = np.array([[1.0, 2.0], [3.0, 4.0]])
    o.haplotypes_ = ["h1", "h2"]
    o.samples_ = ["s1", "s2"]

    p = tmp_path / "m.tsv"
    save_embedding_table_from_model(o, p)
    df = pd.read_csv(p, sep="\t")
    assert len(df) == 2
    assert "PC1" in df.columns
    assert df["indID"].tolist() == ["s1", "s2"]


def test_embedding_dataframe_from_model_joins_metadata():
    P = type("PCA", (), {})
    o = P()
    o.X_new_ = np.array([[1.0, 2.0], [3.0, 4.0]])
    o.haplotypes_ = ["s1", "s2"]
    o.samples_ = ["s1", "s2"]
    metadata = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "population": ["CEU", "YRI"],
        }
    )

    df = embedding_dataframe_from_model(o, metadata=metadata)

    assert list(df.columns[:4]) == ["indID", "method", "PC1", "PC2"]
    assert df["population"].tolist() == ["CEU", "YRI"]


def test_embedding_dataframe_from_model_can_require_metadata_match():
    P = type("PCA", (), {})
    o = P()
    o.X_new_ = np.array([[1.0, 2.0], [3.0, 4.0]])
    o.haplotypes_ = ["s1", "s2"]
    o.samples_ = ["s1", "s2"]
    metadata = pd.DataFrame({"sample": ["s1"], "population": ["CEU"]})

    try:
        embedding_dataframe_from_model(o, metadata=metadata, require_metadata_match=True)
    except ValueError as exc:
        assert "metadata is missing rows" in str(exc)
    else:
        raise AssertionError("expected missing metadata to raise")


def test_embedding_dataframe_from_pca_after_transform_has_sample_ids():
    gt = np.array(
        [
            [[0, 1], [1, 1], [0, 0]],
            [[1, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 0], [0, 1]],
        ],
        dtype=float,
    )
    snp = SNPObject(genotypes=gt, samples=np.array(["s1", "s2", "s3"], dtype=object))
    pca = PCA(backend="sklearn", n_components=2, fitting="exact")

    pca.fit(snp)
    pca.transform(snp)
    df = embedding_dataframe_from_model(pca)

    assert df["indID"].tolist() == ["s1", "s2", "s3"]
    assert ["PC1", "PC2"] == [c for c in df.columns if c.startswith("PC")]


@pytest.mark.parametrize("order", [
    [2, 0, 1], [1, 0], 2, np.array([2, 0, 1]),
    [True, True, True], np.array([True, False, True]),
])
def test_transform_selects_cached_samples_and_preserves_ids(order):
    snp = SNPObject(genotypes=np.array([[0, 1, 2], [1, 2, 0]]), samples=["A", "B", "C"])
    pca = PCA(snpobj=snp)
    rows = slice(order) if isinstance(order, int) else order
    expected = pca.X_new_[rows].copy()
    expected_ids = np.asarray(snp.samples)[rows].tolist()

    np.testing.assert_allclose(pca.transform(samples_subset=order), expected, atol=1e-12)
    assert pca.samples_ == expected_ids
    np.testing.assert_allclose(pca.transform(), expected, atol=1e-12)
    assert pca.samples_ == expected_ids


def test_cached_projection_uses_last_object_and_preparation_options():
    gt = np.random.default_rng(42).integers(0, 2, size=(6, 4, 2))
    snp = SNPObject(genotypes=gt, samples=["A", "B", "C", "D"])
    pca = PCA().fit(snp, snps_subset=[4, 1, 5])
    reference = PCA().fit(snp, snps_subset=[4, 1, 5])
    expected = reference.transform(snp, average_haplotypes=False, samples_subset=[3, 0], snps_subset=[4, 1, 5])

    np.testing.assert_allclose(pca.transform(average_haplotypes=False, samples_subset=[3, 0]), expected)
    assert pca.haplotypes_ == ["D|0", "D|1", "A|0", "A|1"]
    other = SNPObject(genotypes=gt[:, ::-1], samples=["E", "F", "G", "H"])
    expected = pca.transform(other, snps_subset=[4, 1, 5]).copy()
    np.testing.assert_allclose(pca.transform(samples_subset=[2, 0]), expected[[2, 0]])
    assert pca.samples_ == ["G", "E"]

    # A failed projection must leave the last successful matrix and labels paired.
    with pytest.raises(ValueError):
        pca.transform(snps_subset=[0, 1])
    np.testing.assert_allclose(pca.transform(), expected[[2, 0]])
    assert pca.samples_ == ["G", "E"]


@pytest.mark.parametrize("mode", ["dosage", "averaged", "expanded"])
def test_pca_preserves_pipe_in_sample_ids(mode, tmp_path):
    gt = np.random.default_rng(42).integers(0, 2, size=(4, 3, 2))
    if mode == "dosage":
        gt = gt.sum(axis=2)
    samples = ["family|A", "family|B", "family|0"]
    snp = SNPObject(genotypes=gt, samples=samples)
    pca = PCA(snpobj=snp, average_haplotypes=mode == "averaged")
    assert pca.samples_ == (np.repeat(samples, 2).tolist() if mode == "expanded" else samples)
    pca.transform(samples_subset=[2, 0])
    pca.transform()

    expected = ["family|0", "family|A"]
    if mode == "expanded":
        haplotypes = [f"{sample}|{i}" for sample in expected for i in range(2)]
        expected = [sample for sample in expected for _ in range(2)]
    else:
        haplotypes = expected
    assert pca.samples_ == expected
    assert pca.haplotypes_ == haplotypes

    metadata = pd.DataFrame({"sample": samples, "batch": [1, 2, 3]})
    frame = embedding_dataframe_from_model(pca, metadata=metadata, require_metadata_match=True)
    assert frame["batch"].tolist() == ([3, 3, 1, 1] if mode == "expanded" else [3, 1])
    path = tmp_path / "projection.tsv"
    save_embedding_table_from_model(pca, path)
    exported = pd.read_csv(path, sep="\t")
    assert exported["indID"].tolist() == expected
    if mode == "expanded":
        assert exported["haplotype_id"].tolist() == haplotypes
    else:
        assert "haplotype_id" not in exported


@pytest.mark.parametrize("subset", [[True, False], [[0, 1]], [1.5]])
def test_pca_rejects_invalid_sample_subset(subset):
    snp = SNPObject(genotypes=np.array([[0, 1, 2], [1, 2, 0]]), samples=["A", "B", "C"])
    with pytest.raises(ValueError, match="subset|mask"):
        PCA(snpobj=snp, samples_subset=subset)


def test_pca_applies_snp_mask_to_genotypes_and_variant_metadata():
    snp = SNPObject(
        genotypes=np.array([[0, 1, 2], [1, 2, 1], [2, 1, 0]]),
        samples=["A", "B", "C"],
        variants_alt=["A", "C,G", "T"],
    )
    pca = PCA(snpobj=snp, samples_subset=np.array([2, 0]), snps_subset=np.array([True, False, True]))
    np.testing.assert_array_equal(pca.X_, snp.genotypes[[0, 2]][:, [2, 0]].T)
    assert pca.samples_ == ["C", "A"]


@pytest.mark.parametrize("axis", ["samples", "variants"])
def test_cached_projection_keeps_its_selection_after_default_changes(axis):
    gt = np.array([[0, 1, 2], [1, 2, 0], [0, 0, 2], [2, 0, 1]])
    snp = SNPObject(genotypes=gt, samples=["A", "B", "C"])
    pca = PCA(snpobj=snp)
    original = pca.X_new_.copy()
    if axis == "samples":
        pca.samples_subset = [2, 0, 1]
        scores = pca.transform(snps_subset=[0, 1, 2, 3])
    else:
        pca.snps_subset = [3, 0, 2, 1]
        scores = pca.transform(samples_subset=[0, 1, 2])

    np.testing.assert_allclose(scores, original, atol=1e-12)
    assert embedding_dataframe_from_model(pca)["indID"].tolist() == ["A", "B", "C"]

    # An explicit source uses the current model defaults.
    if axis == "samples":
        pca.transform(snp)
        np.testing.assert_allclose(pca.X_new_, original[[2, 0, 1]], atol=1e-12)
        assert pca.samples_ == ["C", "A", "B"]
    else:
        with pytest.raises(ValueError, match="fitted variant order"):
            pca.transform(snp)


@pytest.mark.parametrize("backend", ["sklearn", "pytorch"])
def test_projection_rejects_wrong_feature_count_without_changing_cache(backend):
    if backend == "pytorch":
        pytest.importorskip("torch")
    snp = SNPObject(genotypes=np.array([[0, 1, 2], [2, 1, 0], [1, 2, 0]]), samples=["A", "B", "C"])
    model = PCA(snpobj=snp, backend=backend)
    expected = np.asarray(model.X_new_).copy()
    with pytest.raises(ValueError, match="1 feature.*expected 3"):
        model.transform(snps_subset=[0])
    np.testing.assert_allclose(model.transform(), expected, atol=1e-12)
    assert model.samples_ == ["A", "B", "C"]
    if backend == "pytorch":
        with pytest.raises(ValueError, match="1 feature.*expected 3"):
            model.pca.transform(model.X_[:, :1])


@pytest.mark.parametrize("fit_method", ["fit", "fit_transform"])
def test_projection_preserves_fitted_variant_selection(fit_method):
    gt = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0], [0, 0, 1]])
    snp = SNPObject(genotypes=gt, samples=["A", "B", "C"])
    pca = PCA()
    getattr(pca, fit_method)(snp, snps_subset=[0, 2, 3])
    expected = pca.transform(snps_subset=[True, False, True, True], samples_subset=[2, 0]).copy()
    for selection in ([3, 2, 0], [0, 1, 3]):
        with pytest.raises(ValueError, match="fitted variant order"):
            pca.transform(snps_subset=selection, samples_subset=[1, 2])
        np.testing.assert_allclose(pca.transform(), expected)
        assert pca.samples_ == ["C", "A"]


def test_projection_checks_variant_ids_across_objects():
    gt = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    snp = SNPObject(genotypes=gt, samples=["A", "B", "C"], variants_id=["a", "b", "c"])
    pca = PCA().fit(snp, snps_subset=[2, 0])
    expected = pca.transform().copy()
    aligned = SNPObject(genotypes=gt[[2, 0]], samples=snp.samples, variants_id=["c", "a"])
    np.testing.assert_allclose(pca.transform(aligned), expected)
    with pytest.raises(ValueError, match="fitted variant order"):
        pca.transform(aligned, snps_subset=[1, 0])
    np.testing.assert_allclose(pca.transform(), expected)


def test_projection_uses_fitted_coordinates_and_alleles_without_variant_ids():
    gt = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
    snp = SNPObject(genotypes=gt, samples=["A", "B", "C"], variants_chrom=[1, 1, 1],
                    variants_pos=[10, 20, 30], variants_ref=["A", "C", "T"], variants_alt=["C", "G", "A"])
    pca = PCA().fit(snp, snps_subset=[2, 0])
    expected = pca.transform().copy()
    aligned = SNPObject(genotypes=gt[[2, 0]], samples=snp.samples, variants_chrom=["1", "1"],
                        variants_pos=[30, 10], variants_ref=["T", "A"], variants_alt=["A", "C"])
    np.testing.assert_allclose(pca.transform(aligned), expected)
    aligned.variants_ref = ["G", "A"]
    with pytest.raises(ValueError, match="fitted variant order"):
        pca.transform(aligned)
    np.testing.assert_allclose(pca.transform(), expected)
