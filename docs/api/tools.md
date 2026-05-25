# Association and Utilities

High-level association testing helpers and label-table utilities used across plotting and dimensionality-reduction workflows.

## GWAS and admixture mapping

```{eval-rst}
.. autofunction:: snputils.run_gwas
```

```{eval-rst}
.. autofunction:: snputils.run_admixture_mapping
```

Both functions accept file paths or in-memory objects and return a DataFrame with per-variant statistics suitable for {doc}`../user_guide/visualization`.

## Sample labels

```{eval-rst}
.. autofunction:: snputils.read_labels
```

Label tables must include `indID` and `label` columns. They are used by scatter plots, mdPCA, and maasMDS.
