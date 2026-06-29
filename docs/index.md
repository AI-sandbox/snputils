# snputils

**snputils** is a Python library for processing genetic variation, ancestry, phenotype, and relatedness data. It focuses on fast file I/O, ergonomic data containers, population-genetic statistics, dimensionality reduction, and visualization workflows for genomic analyses.

Developed in collaboration between Stanford University's Department of Biomedical Data Science, UC Santa Cruz Genomics Institute, and collaborators worldwide.

:::{note}
**snputils** is stable and ready for production workflows. The core API is documented, tested, and suitable for day-to-day genomic analysis. The project is actively maintained: we ship regular releases, welcome contributions, and continue to extend format support, analyses, and performance. See {doc}`changelog` for release history and {doc}`contributing` to get involved.
:::

## Why snputils?

- One API across genotype, local ancestry, global ancestry, phenotype, and IBD data
- Fast readers and writers for common population-genetics formats
- In-memory Python workflows and file-backed CLI workflows in the same package
- Ancestry-aware analyses including PCA, mdPCA, maasMDS, admixture mapping, and ancestry-specific allele frequencies
- Built-in plotting for embeddings, local ancestry, admixture, and association results

## Supported formats

High-level dispatchers such as {func}`~snputils.read_snp`, {func}`~snputils.read_lai`, {func}`~snputils.read_admixture`, {func}`~snputils.read_pheno`, and {func}`~snputils.read_ibd` cover VCF/BCF/BGEN/PLINK BED and PGEN, MSP/FLARE/admix-kit LANC local ancestry, ADMIXTURE `.Q`/`.P`, phenotype tables, hap-IBD and ancIBD segments, and GRG graphs. See {doc}`user_guide/file-io` for format tables and reader options.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Install
:link: installation
:link-type: doc
Set up the package, optional extras, and local documentation build.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
Load SNP, ancestry, phenotype, and IBD files with the high-level API.
:::

:::{grid-item-card} User Guide
:link: user_guide/index
:link-type: doc
Workflow-oriented guides for data objects, I/O, analysis, and visualization.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Rendered notebooks for SNP objects, PCA, allele frequency, local ancestry, admixture mapping, and GRG workflows.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc
Browse objects, readers, writers, analysis classes, statistics, datasets, and visualization functions by topic.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc
Development setup, tests, documentation builds, and pull-request guidelines.
:::

::::

## Citation

If you use **snputils** in your research, please cite [our paper](https://www.biorxiv.org/content/10.64898/2026.02.28.708618):

```bibtex
@article{snputils2026,
    author    = {Bonet, David and Comajoan Cara, Marçal and Barrabés, Míriam and Smeriglio, Riccardo and Agrawal, Devang and Aounallah, Khaled and Geleta, Margarita and Dominguez Mantes, Albert and Thomassin, Christophe and Shanks, Cole and Huang, Edward C. and Franquesa Monés, Marc and Luis, Aina and Saurina, Joan and Perera, Maria and López, Cayetana and Sabat, Benet Oriol and Abante, Jordi and Moreno-Grau, Sonia and Mas Montserrat, Daniel and Ioannidis, Alexander G.},
    title     = {{snputils}: A High-Performance {Python} Library for Genetic Variation and Population Structure},
    year      = {2026},
    doi       = {10.64898/2026.02.28.708618},
    url       = {https://www.biorxiv.org/content/10.64898/2026.02.28.708618},
    journal   = {bioRxiv},
    publisher = {Cold Spring Harbor Laboratory},
}
```

```{toctree}
:hidden:
:maxdepth: 2

Installation <installation>
Quickstart <quickstart>
User guide <user_guide/index>
Tutorials <tutorials/index>
API reference <api/index>
Contributing <contributing>
Changelog <changelog>
License <license>
```
