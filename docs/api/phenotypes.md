# Phenotypes

Phenotype data containers and readers for single-trait and multi-trait tables. Phenotype objects align sample IDs with trait values and feed {func}`~snputils.run_gwas`, {func}`~snputils.run_admixture_mapping`, and covariate-aware association workflows.

## Objects

```{eval-rst}
.. autoclass:: snputils.PhenotypeObject
   :members:
```

```{eval-rst}
.. autoclass:: snputils.MultiPhenotypeObject
   :members:
```

```{eval-rst}
.. autoclass:: snputils.CovariateObject
   :members:
```

## Readers

```{eval-rst}
.. autofunction:: snputils.read_pheno
```

```{eval-rst}
.. autoclass:: snputils.PhenotypeReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.MultiPhenReader
   :members:
```
