# Quickstart

The top-level package exposes the most common readers, data containers, and analysis helpers.

```python
import snputils as su

snp = su.read_snp("cohort.vcf.gz")          # VCF, BED, or PGEN
snp = snp.filter_biallelic_variants()
snp.save("cohort.pgen")                     # convert VCF -> PLINK2

af = snp.allele_freq()                      # per-SNP allele frequencies
pcs = su.PCA().fit_transform(snp)

lai = su.read_msp("local_ancestry.msp")     # local ancestry
adm = su.read_admixture("admixture")        # global ancestry
labels = su.read_labels("labels.tsv")       # sample metadata
afr_af = snp.allele_freq(ancestry="AFR", laiobj=lai)
su.viz.chromosome_painting(lai, "paintings/")               # one PNG per sample

mdpca = su.mdPCA(snpobj=snp, laiobj=lai, labels_file=labels, ancestry="AFR")
su.viz.scatter(mdpca, labels, save_path="mdpca.pdf", show=False)

phen = su.PhenotypeReader("phenotypes.tsv").read(phenotype_col="trait")
ibd = su.read_ibd("hap.ibd")
gwas = su.run_gwas(phen, snp, "gwas.tsv.gz")
admix = su.run_admixture_mapping(phen, lai, "admixmap.tsv.gz")
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
pca = su.PCA(n_components=2)
coords = pca.fit_transform(snp)

freq = snp.allele_freq()
su.viz.scatter(pca, "labels.tsv", save_path="pca.png", show=False)
```

For complete workflows, see the tutorials.
