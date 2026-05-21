import pathlib

import numpy as np
import pandas as pd

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
    out = pca_row_haplotype_ids(snp, average_strands=False)
    assert len(out) == 4
    assert out[0].startswith("x|") and out[1].startswith("x|")


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
