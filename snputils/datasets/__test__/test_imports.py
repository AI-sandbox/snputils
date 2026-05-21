import importlib


def test_dataset_helpers_are_exposed_from_top_level_package():
    import snputils
    from snputils import (
        available_datasets_list,
        build_synthetic_chromosome_painting_dataset,
        build_synthetic_grg,
        build_synthetic_maasmds_dataset,
        build_synthetic_mdpca_dataset,
        build_synthetic_snp_dataset,
        load_dataset,
    )

    assert load_dataset is snputils.load_dataset
    assert available_datasets_list is snputils.available_datasets_list
    assert build_synthetic_chromosome_painting_dataset is snputils.build_synthetic_chromosome_painting_dataset
    assert build_synthetic_grg is snputils.build_synthetic_grg
    assert build_synthetic_maasmds_dataset is snputils.build_synthetic_maasmds_dataset
    assert build_synthetic_mdpca_dataset is snputils.build_synthetic_mdpca_dataset
    assert build_synthetic_snp_dataset is snputils.build_synthetic_snp_dataset
    assert available_datasets_list() == ["1kgp"]


def test_dataset_helpers_are_exposed_from_datasets_package():
    import snputils.datasets as datasets
    from snputils.datasets import (
        available_datasets_list,
        build_synthetic_chromosome_painting_dataset,
        build_synthetic_grg,
        build_synthetic_maasmds_dataset,
        build_synthetic_mdpca_dataset,
        build_synthetic_snp_dataset,
        load_dataset,
    )

    assert load_dataset is datasets.load_dataset
    assert available_datasets_list is datasets.available_datasets_list
    assert build_synthetic_chromosome_painting_dataset is datasets.build_synthetic_chromosome_painting_dataset
    assert build_synthetic_grg is datasets.build_synthetic_grg
    assert build_synthetic_maasmds_dataset is datasets.build_synthetic_maasmds_dataset
    assert build_synthetic_mdpca_dataset is datasets.build_synthetic_mdpca_dataset
    assert build_synthetic_snp_dataset is datasets.build_synthetic_snp_dataset
    assert datasets.load_dataset.available_datasets_list is datasets.available_datasets_list
    assert available_datasets_list() == ["1kgp"]


def test_documented_load_dataset_module_path_remains_discoverable():
    from snputils.datasets.load_dataset import available_datasets_list, load_dataset

    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")

    assert load_dataset is load_dataset_module.load_dataset
    assert available_datasets_list is load_dataset_module.available_datasets_list
    assert callable(load_dataset_module.load_dataset)
    assert load_dataset_module.available_datasets_list() == ["1kgp"]


def test_import_load_dataset_path_can_find_available_datasets_list():
    import snputils.datasets.load_dataset as load_dataset_api

    assert callable(load_dataset_api)
    assert load_dataset_api.available_datasets_list() == ["1kgp"]
