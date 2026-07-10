# Analysis Tools and Utilities

High-level file-backed analysis helpers and label-table utilities.

## Genetic sex checking

```{eval-rst}
.. autofunction:: snputils.run_sex_check
```

For an already-loaded `SNPObject`, use {func}`~snputils.sex_check` directly.
The file-backed helper accepts VCF, BCF, BED, and PGEN hard-call inputs and can
write the resulting TSV through `results_path`.

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
