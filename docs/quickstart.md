# Quickstart

The top-level package exposes the most common readers and data containers.

```python
import snputils as su

snpobj = su.read_snp("your.vcf.gz")
```

`read_snp` dispatches from the file extension and returns a {class}`snputils.SNPObject`.

```python
bed = su.read_bed("cohort.bed")
pgen = su.read_pgen("cohort.pgen")
vcf = su.read_vcf("cohort.vcf.gz")
```

## Local and Global Ancestry

```python
lai = su.read_msp("local_ancestry.msp")
adm = su.read_admixture("global_ancestry")
```

Local ancestry files are represented as {class}`snputils.LocalAncestryObject`; global ancestry and ADMIXTURE-style outputs use {class}`snputils.GlobalAncestryObject`.

## Phenotypes and IBD

```python
phen = su.PhenotypeReader("phenotypes.tsv").read()
ibd = su.read_ibd("hap.ibd")
```

Phenotype data are represented by {class}`snputils.PhenotypeObject` or {class}`snputils.MultiPhenotypeObject`. Identity-by-descent segments are represented by {class}`snputils.IBDObject`.

## Analysis and Visualization

```python
from snputils.processing import PCA
from snputils.stats import allele_freq_stream
from snputils.visualization import scatter

pca = PCA(n_components=2)
coords = pca.fit_transform(snpobj)

freq = allele_freq_stream("cohort.pgen", chunk_size=50_000)
fig = scatter(coords[:, 0], coords[:, 1])
```

For complete workflows, see the tutorials.
