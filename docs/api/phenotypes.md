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

## Covariate construction

```{eval-rst}
.. autofunction:: snputils.phenotype.read_covar_file
```

```{eval-rst}
.. autofunction:: snputils.phenotype.build_association_covariates
```

**CovariateObject factories** (class methods on {class}`~snputils.CovariateObject`):

- ``from_file(path, col_nums=None)`` — read a PLINK-style covariate table (`IID` plus numeric columns).
- ``from_embedding(model, n_components=None)`` — PCs or MDS coordinates from a fitted PCA, mdPCA, or maasMDS model.
- ``from_global_ancestry(admobj, columns=None, drop_ancestry=-1)`` — ADMIXTURE ``Q`` proportions; drops the last ancestry column by default.
- ``merge(*objs)`` — inner-join sample IDs and concatenate covariate columns.

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
