# Analysis

snputils includes analysis routines for dimensionality reduction, allele frequencies, population-genetic statistics, admixture mapping, and simulation.

## PCA and Ancestry-Aware Embeddings

```python
from snputils.processing import PCA, mdPCA, maasMDS

pca = PCA(n_components=2)
coords = pca.fit_transform(snpobj)
```

`mdPCA` handles missing data, and `maasMDS` uses local ancestry masks for ancestry-specific multidimensional scaling.

## Allele Frequencies and F-Statistics

```python
from snputils.stats import allele_freq_stream, fst, f2, f3, f4, d_stat

freq = allele_freq_stream("cohort.pgen", chunk_size=50_000)
```

The statistics module supports D, f2, f3, f4, f4-ratio, and FST with jackknife standard errors where applicable.

## Command-Line Tools

The package installs a `snputils` command with analysis-oriented subcommands for workflows that start from files and produce files:

```bash
snputils --help
snputils --version
```

CLI-backed workflows include standard PCA, mdPCA, maasMDS, admixture mapping, GWAS, simulation, and Manhattan/QQ plots:

```bash
snputils pca --help
snputils mdpca --help
snputils maasmds --help
snputils admixture-map --help
snputils gwas --help
snputils simulate --help
snputils plot-manhattan --help
snputils plot-qq --help
```

Use the Python API for lower-level data containers, IBD object operations, f-statistics, allele-frequency helpers, and custom visualization workflows that use in-memory composition.
