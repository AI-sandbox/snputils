# Datasets

Helpers for discovering bundled example datasets and generating synthetic data for tests, tutorials, and quick experiments. Synthetic builders return fully populated objects matching the shapes described in {doc}`../user_guide/data-model`.

## Example Datasets

The 1000 Genomes Project dataset is registered as `1kgp`. Its `resource` options include
`grch38_biallelic`, `phase3`, and `high_coverage_2022`. The `high_coverage_2022`
resource corresponds to the phased 30x high-coverage SNV/INDEL/SV panel from
Byrska-Bishop et al., *Cell* 2022, "High-coverage whole-genome sequencing of the
expanded 1000 Genomes Project cohort including 602 trios". Because these VCFs are
large, `load_dataset("1kgp", resource="high_coverage_2022")` defaults to chromosome 1;
pass `chromosomes=` explicitly to load additional chromosomes.

```{eval-rst}
.. autofunction:: snputils.available_datasets_list
```

```{eval-rst}
.. autofunction:: snputils.load_dataset
```

## Synthetic Data Builders

```{eval-rst}
.. autofunction:: snputils.build_synthetic_snp_dataset
```

```{eval-rst}
.. autofunction:: snputils.build_synthetic_admixture_dataset
```

```{eval-rst}
.. autofunction:: snputils.build_synthetic_chromosome_painting_dataset
```

```{eval-rst}
.. autofunction:: snputils.build_synthetic_grg
```

```{eval-rst}
.. autofunction:: snputils.build_synthetic_maasmds_dataset
```

```{eval-rst}
.. autofunction:: snputils.build_synthetic_mdpca_dataset
```

```{eval-rst}
.. autofunction:: snputils.build_synthetic_phenotype_dataset
```
