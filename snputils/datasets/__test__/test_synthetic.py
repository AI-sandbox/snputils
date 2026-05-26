import numpy as np
import pandas as pd

from snputils.datasets import (
    build_synthetic_chromosome_painting_dataset,
    build_synthetic_grg,
    build_synthetic_maasmds_dataset,
    build_synthetic_mdpca_dataset,
    build_synthetic_phenotype_dataset,
    build_synthetic_snp_dataset,
)
from snputils.phenotype.genobj import CovariateObject, MultiPhenotypeObject, PhenotypeObject
from snputils.processing import mdPCA, maasMDS
from snputils.snp.genobj.grgobj import GRGObject
from snputils.snp.genobj.snpobj import SNPObject


def test_build_synthetic_snp_dataset_returns_metadata_rich_snpobject():
    snp = build_synthetic_snp_dataset(n_samples=6, n_snps=10, seed=1, missing_rate=0.05)

    assert isinstance(snp, SNPObject)
    assert snp.genotypes.shape == (10, 6, 2)
    assert snp.samples.tolist() == ["S01", "S02", "S03", "S04", "S05", "S06"]
    assert snp.sample_fid.shape == (6,)
    assert snp.sample_sex.tolist() == [1, 2, 1, 2, 1, 2]
    assert snp.variants_id[0] == "rs_syn_00001"
    assert snp.variants_pos.tolist() == list(np.arange(1, 11) * 1000)


def test_build_synthetic_snp_dataset_can_return_summed_dosages():
    snp = build_synthetic_snp_dataset(n_samples=4, n_snps=5, seed=2, phased=False)

    assert snp.genotypes.shape == (5, 4)
    assert np.all((snp.genotypes >= 0) & (snp.genotypes <= 2))


def test_build_synthetic_phenotype_dataset_returns_aligned_objects():
    dataset = build_synthetic_phenotype_dataset(n_samples=10, n_snps=30, seed=7)

    assert isinstance(dataset["snpobj"], SNPObject)
    assert isinstance(dataset["multi_phenotype"], MultiPhenotypeObject)
    assert isinstance(dataset["quantitative"], PhenotypeObject)
    assert isinstance(dataset["binary"], PhenotypeObject)
    assert isinstance(dataset["covariates"], CovariateObject)
    assert dataset["phen_df"].columns.tolist() == [
        "IID",
        "trait_quantitative",
        "trait_binary_01",
        "trait_binary_12",
        "age",
        "batch",
        "sex",
    ]
    assert dataset["binary"].is_quantitative is False
    assert set(np.unique(dataset["binary"].values).tolist()) == {0, 1}
    assert dataset["quantitative"].is_quantitative is True
    assert dataset["covariates"].covariate_names == ["age", "batch", "sex"]
    assert dataset["quantitative"].samples == dataset["covariates"].samples
    assert dataset["shuffled_phen_df"]["IID"].tolist() != dataset["phen_df"]["IID"].tolist()
    assert len(dataset["effect_variant_ids"]) == 3


def test_build_synthetic_mdpca_dataset_works_with_in_memory_labels():
    dataset = build_synthetic_mdpca_dataset(n_samples=8, n_snps=20, seed=3)

    assert isinstance(dataset["snpobj"], SNPObject)
    assert dataset["snpobj"].genotypes.shape == (20, 8, 2)
    assert dataset["laiobj"].lai.shape == (20, 16)
    assert isinstance(dataset["labels"], pd.DataFrame)
    assert list(dataset["labels"].columns) == ["indID", "label"]

    model = mdPCA(
        snpobj=dataset["snpobj"],
        laiobj=dataset["laiobj"],
        labels=dataset["labels"],
        ancestry="AFR",
        average_strands=True,
        min_percent_snps=1,
        group_snp_frequencies_only=False,
        n_components=2,
    )
    assert model.X_new_.shape[1] == 2

    mds = maasMDS(
        snpobj=dataset["snpobj"],
        laiobj=dataset["laiobj"],
        labels=dataset["labels"],
        ancestry="AFR",
        average_strands=True,
        min_percent_snps=1,
        group_snp_frequencies_only=False,
        n_components=2,
    )
    assert mds.X_new_.shape[1] == 2


def test_build_synthetic_maasmds_dataset_has_three_overlapping_arrays():
    dataset = build_synthetic_maasmds_dataset(
        n_samples_per_array=8,
        n_snps_per_array=50,
        seed=4,
        triple_shared_fraction=0.2,
        pair_shared_fraction=0.2,
    )

    assert len(dataset["snpobjs"]) == 3
    assert len(dataset["laiobjs"]) == 3
    assert dataset["labels"].shape[0] == 24
    assert dataset["overlap_counts"]["array_1"] == 50
    assert dataset["overlap_counts"]["array_1_2"] > dataset["overlap_counts"]["array_1_2_3"]
    assert dataset["overlap_counts"]["array_1"] > dataset["overlap_counts"]["array_1_2"]

    model = maasMDS(
        snpobj=dataset["snpobjs"],
        laiobj=dataset["laiobjs"],
        labels=dataset["labels"],
        ancestry="AFR",
        average_strands=True,
        min_percent_snps=1,
        group_snp_frequencies_only=False,
        n_components=2,
    )
    assert model.X_new_.shape[1] == 2
    assert set(model.array_labels_.tolist()) == {1, 2, 3}


def test_build_synthetic_chromosome_painting_dataset_covers_autosomes_and_x_by_default():
    dataset = build_synthetic_chromosome_painting_dataset(
        n_samples=3,
        windows_per_chromosome=3,
        seed=5,
    )

    laiobj = dataset["laiobj"]
    assert len(dataset["chromosomes"]) == 23
    assert laiobj.n_windows == 69
    assert laiobj.chromosomes[0] == "1"
    assert laiobj.chromosomes[-1] == "X"
    assert dataset["sample_sex"]["sex"].tolist() == ["female", "female", "female"]


def test_build_synthetic_chromosome_painting_dataset_can_still_restrict_to_autosomes():
    dataset = build_synthetic_chromosome_painting_dataset(
        n_samples=2,
        windows_per_chromosome=4,
        seed=6,
        chromosomes=[str(chrom) for chrom in range(1, 23)],
        male_samples=["sample1"],
    )

    laiobj = dataset["laiobj"]
    assert len(dataset["chromosomes"]) == 22
    assert laiobj.n_windows == 88
    assert "X" not in set(map(str, laiobj.chromosomes.tolist()))
    assert dataset["sample_sex"]["sex"].tolist() == ["female", "male"]


def test_build_synthetic_grg_returns_convertible_grgobject(tmp_path):
    grg = build_synthetic_grg()

    assert isinstance(grg, GRGObject)
    assert grg.n_samples() == 3
    assert grg.n_snps() == 5

    snp = grg.to_snpobject(chrom="22", sum_strands=False)
    assert snp.genotypes.shape == (5, 3, 2)

    path = tmp_path / "toy.grg"
    grg.save(str(path))
    assert path.exists()
