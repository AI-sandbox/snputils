from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from snputils.ancestry.genobj.wide import GlobalAncestryObject
from snputils.datasets import build_synthetic_snp_dataset
from snputils.phenotype import (
    CovariateObject,
    build_association_covariates,
    read_covar_file,
)
from snputils.phenotype.covariates import write_covar_file
from snputils.processing import PCA
from snputils.snp.genobj import SNPObject
from snputils.tools.gwas import run_gwas


def _write_covar(path: Path, sample_ids: Sequence[str], names: Sequence[str], covar: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID " + " ".join(names) + "\n")
        for sid, row in zip(sample_ids, covar):
            values = " ".join(f"{float(v):.12g}" for v in row.tolist())
            handle.write(f"{sid} {sid} {values}\n")


def _write_vcf(path: Path, sample_ids: Sequence[str], dosage: np.ndarray) -> None:
    gt_map = {0: "0|0", 1: "0|1", 2: "1|1"}
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(sample_ids))
        handle.write("\n")
        for vidx in range(dosage.shape[0]):
            genotypes = [gt_map[int(g)] for g in dosage[vidx].tolist()]
            fields = ["1", str(1000 + vidx), f"rs{vidx}", "A", "G", ".", "PASS", ".", "GT", *genotypes]
            handle.write("\t".join(fields) + "\n")


def _write_quantitative_phe(path: Path, sample_ids: Sequence[str], y: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID PHENO\n")
        for sid, yi in zip(sample_ids, y):
            handle.write(f"{sid} {sid} {float(yi):.12g}\n")


def test_from_file_encodes_sex_and_selects_columns(tmp_path: Path):
    path = tmp_path / "covar.txt"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID age sex batch\n")
        handle.write("s1 s1 58 M 1\n")
        handle.write("s2 s2 49 F 2\n")

    covar = CovariateObject.from_file(path, col_nums="1,2")
    assert covar.covariate_names == ["age", "sex"]
    assert covar.samples == ["s1", "s2"]
    np.testing.assert_allclose(covar.values[0], [58.0, 1.0])
    np.testing.assert_allclose(covar.values[1], [49.0, 2.0])

    samples, names, matrix = read_covar_file(path, col_nums="1-2")
    assert names == ["age", "sex"]
    assert samples == ["s1", "s2"]
    assert matrix.shape == (2, 2)


def test_from_file_drops_rows_with_missing_values(tmp_path: Path):
    path = tmp_path / "covar.txt"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#IID age\n")
        handle.write("s1 58\n")
        handle.write("s2 .\n")

    covar = CovariateObject.from_file(path)
    assert covar.samples == ["s1"]
    assert covar.n_covariates == 1


def test_from_embedding_uses_sample_level_pca():
    snpobj = build_synthetic_snp_dataset(n_samples=12, n_snps=40, seed=11)
    pca = PCA(n_components=3, average_strands=True)
    pca.fit_transform(snpobj)

    covar = CovariateObject.from_embedding(pca, n_components=2)
    assert covar.covariate_names == ["PC1", "PC2"]
    assert covar.n_samples == snpobj.n_samples
    assert list(covar.samples) == [str(sample) for sample in snpobj.samples]
    assert covar.values.shape == (snpobj.n_samples, 2)
    np.testing.assert_allclose(covar.values, pca.X_new_[:, :2])


def test_from_embedding_rejects_haplotype_expanded_rows():
    rng = np.random.default_rng(7)
    gt = rng.integers(0, 2, size=(8, 4, 2), dtype=np.int8)
    samples = np.array(["a", "b", "c", "d"], dtype=object)
    snpobj = SNPObject(genotypes=gt, samples=samples)
    pca = PCA(n_components=2, average_strands=False)
    pca.fit_transform(snpobj)

    with pytest.raises(ValueError, match="haplotype-expanded"):
        CovariateObject.from_embedding(pca)


def test_from_global_ancestry_drops_last_column_by_default():
    samples = ["s1", "s2", "s3"]
    q = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float64,
    )
    admobj = GlobalAncestryObject(Q=q, samples=samples)

    covar = CovariateObject.from_global_ancestry(admobj)
    assert covar.covariate_names == ["ANC0", "ANC1"]
    np.testing.assert_allclose(covar.values, q[:, :2])


def test_from_global_ancestry_accepts_explicit_columns_and_names():
    samples = ["s1", "s2"]
    q = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
    admobj = GlobalAncestryObject(Q=q, samples=samples)

    covar = CovariateObject.from_global_ancestry(
        admobj,
        columns=[1],
        ancestry_names=["EUR"],
    )
    assert covar.covariate_names == ["EUR"]
    np.testing.assert_allclose(covar.values[:, 0], q[:, 1])


def test_merge_inner_joins_and_concatenates_columns():
    pc = CovariateObject(["s1", "s2", "s3"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], ["PC1", "PC2"])
    clinical = CovariateObject(["s2", "s1"], [[58.0], [49.0]], ["age"])

    merged = CovariateObject.merge(pc, clinical)
    assert merged.samples == ["s1", "s2"]
    assert merged.covariate_names == ["PC1", "PC2", "age"]
    assert merged.values.shape == (2, 3)
    np.testing.assert_allclose(merged.values[0], [1.0, 2.0, 49.0])
    np.testing.assert_allclose(merged.values[1], [3.0, 4.0, 58.0])


def test_merge_rejects_duplicate_column_names():
    left = CovariateObject(["s1"], [[1.0]], ["PC1"])
    right = CovariateObject(["s1"], [[2.0]], ["PC1"])
    with pytest.raises(ValueError, match="Duplicate covariate names"):
        CovariateObject.merge(left, right)


def test_build_association_covariates_composes_blocks(tmp_path: Path):
    snpobj = build_synthetic_snp_dataset(n_samples=8, n_snps=30, seed=3)
    pca = PCA(n_components=2, average_strands=True)
    pca.fit_transform(snpobj)

    samples = [str(sample) for sample in snpobj.samples]
    q = np.full((len(samples), 3), 1.0 / 3.0, dtype=np.float64)
    admobj = GlobalAncestryObject(Q=q, samples=samples)

    covar_path = tmp_path / "clinical.txt"
    clinical_matrix = np.column_stack(
        [np.arange(len(samples), dtype=np.float64), np.ones(len(samples))]
    )
    _write_covar(covar_path, samples, ["age", "sex"], clinical_matrix)

    covar = build_association_covariates(
        embedding=pca,
        n_components=2,
        global_ancestry=admobj,
        file=covar_path,
    )
    assert covar.n_samples == len(samples)
    assert covar.covariate_names == ["PC1", "PC2", "ANC0", "ANC1", "age", "sex"]


def test_write_covar_file_roundtrip(tmp_path: Path):
    covar = CovariateObject(["s1", "s2"], [[1.0, 2.0], [3.0, 4.0]], ["PC1", "PC2"])
    path = tmp_path / "out.covar"
    write_covar_file(covar, path)

    loaded = CovariateObject.from_file(path)
    assert loaded.covariate_names == covar.covariate_names
    assert loaded.samples == covar.samples
    np.testing.assert_allclose(loaded.values, covar.values)


def test_merged_covariates_work_with_run_gwas(tmp_path: Path):
    rng = np.random.default_rng(17)
    n_samples = 40
    n_variants = 12
    sample_ids = [f"s{i:02d}" for i in range(n_samples)]

    dosage = rng.integers(0, 3, size=(n_variants, n_samples), dtype=np.int8)
    snpobj = build_synthetic_snp_dataset(n_samples=n_samples, n_snps=n_variants, seed=17)
    sample_ids = [str(sample) for sample in snpobj.samples]

    pca = PCA(n_components=2, average_strands=True)
    pca.fit_transform(snpobj)
    pc_covar = CovariateObject.from_embedding(pca, n_components=2)

    clinical_matrix = rng.normal(size=(n_samples, 1))
    clinical = CovariateObject(sample_ids, clinical_matrix, covariate_names=["AGE"])
    covar = CovariateObject.merge(pc_covar, clinical)

    y = (
        0.5 * dosage[0].astype(np.float64)
        + 0.4 * pc_covar.values[:, 0]
        + rng.normal(scale=0.5, size=n_samples)
    )

    vcf_path = tmp_path / "toy.vcf"
    phe_path = tmp_path / "toy.phe"
    _write_vcf(vcf_path, sample_ids, dosage)
    _write_quantitative_phe(phe_path, sample_ids, y)

    results = run_gwas(
        phe_path=phe_path,
        snp_path=vcf_path,
        results_path=tmp_path / "out",
        phe_id="PHENO",
        batch_size=6,
        covar=covar,
    )
    assert not results.empty
    assert "P" in results.columns
