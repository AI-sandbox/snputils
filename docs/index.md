# snputils

**snputils** is a Python library for processing genetic variation, ancestry, phenotype, and relatedness data. It focuses on fast file I/O, ergonomic data containers, population-genetic statistics, dimensionality reduction, and visualization workflows for genomic analyses.

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

::::

## Citation

If you use **snputils** in your research, please cite [our paper](https://www.biorxiv.org/content/10.64898/2026.02.28.708618):

```bibtex
@article{snputils2026,
  author = {Bonet, David and Comajoan Cara, Marçal and Barrabés, Míriam and Smeriglio, Riccardo and Agrawal, Devang and Aounallah, Khaled and Geleta, Margarita and Dominguez Mantes, Albert and Thomassin, Christophe and Shanks, Cole and Huang, Edward C. and Franquesa Monés, Marc and Luis, Aina and Saurina, Joan and Perera, Maria and López, Cayetana and Sabat, Benet Oriol and Abante, Jordi and Moreno-Grau, Sonia and Mas Montserrat, Daniel and Ioannidis, Alexander G.},
  title = {{snputils}: A High-Performance {Python} Library for Genetic Variation and Population Structure},
  year = {2026},
  doi = {10.64898/2026.02.28.708618},
  url = {https://www.biorxiv.org/content/10.64898/2026.02.28.708618},
  journal = {bioRxiv},
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
```
