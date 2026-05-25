# User Guide

The user guide groups the library by workflow rather than by module path. Each page stands alone, but the sections below follow a typical analysis order.

## Recommended reading order

1. **{doc}`data-model`** — Object shapes, attributes, and filtering before you load real data.
2. **{doc}`file-io`** — Format dispatch, reader/writer options, and synthetic datasets for experiments.
3. **{doc}`analysis`** — Dimensionality reduction, allele frequencies, *f*-statistics, GWAS, admixture mapping, simulation, and the CLI.
4. **{doc}`visualization`** — Scatter, local ancestry, admixture, Manhattan, and Q–Q plots.

After the quickstart, open the {doc}`../tutorials/index` notebooks for end-to-end examples with saved outputs.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Data Model
:link: data-model
:link-type: doc
SNPObject, LocalAncestryObject, GlobalAncestryObject, IBDObject, phenotype containers, and GRG.
:::

:::{grid-item-card} File I/O
:link: file-io
:link-type: doc
Readers and writers for genotype, ancestry, phenotype, and IBD formats.
:::

:::{grid-item-card} Analysis
:link: analysis
:link-type: doc
PCA, mdPCA, maasMDS, *f*-statistics, GWAS, admixture mapping, simulation, and CLI workflows.
:::

:::{grid-item-card} Visualization
:link: visualization
:link-type: doc
Embedding scatter plots, chromosome painting, admixture bars, Manhattan, and Q–Q plots.
:::

::::

```{toctree}
:maxdepth: 1

data-model
file-io
analysis
visualization
```
