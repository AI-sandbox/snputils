# Quickstart

The top-level package exposes the most common readers, data containers, and analysis helpers.

```python
import snputils as su

snp = su.read_snp("cohort.vcf.gz")                   # Read VCF, BGEN, BED, PGEN,...
snp = snp.filter_biallelic_variants()                # filter to biallelic
snp.save("cohort.pgen")                              # convert VCF -> PLINK2

af = snp.allele_freq()                               # per-SNP allele frequencies (AF)
pcs = su.PCA().fit_transform(snp)                    # PCA and project data

lai = su.read_lai("local_ancestry.msp")              # local ancestry  (LAI; MSP or FLARE)
adm = su.read_admixture("admixture")                 # global ancestry (ADMIXTURE)
labels = su.read_labels("labels.tsv")                # sample metadata
afr_af = snp.allele_freq(ancestry="AFR", laiobj=lai) # ancestry-specific AF
su.viz.chromosome_painting(lai, "chr_paintings/")    # chromosome paintings

mdpca = su.mdPCA(snp, lai, labels, ancestry="AFR")   # mdPCA (AFR-specific)
su.viz.scatter(mdpca, labels)                        # plot mdPCA

pheno = su.read_pheno("phenotypes.tsv", col="trait") # read phenotype data
ibd = su.read_ibd("hap.ibd")                         # read IBD data
gwas = su.run_gwas(pheno, snp)                       # GWAS
admix = su.run_admixture_mapping(pheno, lai)         # admixture mapping
su.viz.qq_plot(gwas)                                 # GWAS Q–Q plot
su.viz.qq_plot(admix)                                # admixture mapping Q–Q plot
su.viz.manhattan_plot(gwas)                          # GWAS manhattan plot
su.viz.manhattan_plot(admix)                         # admixture mapping manhattan plot
```

`read_snp` dispatches from the file extension and returns a {class}`snputils.SNPObject`.

```python
bed = su.read_bed("cohort.bed")  
bgen = su.read_bgen("cohort.bgen")                 # probabilities in calldata_gp
pgen = su.read_pgen("cohort.pgen")
vcf = su.read_vcf("cohort.vcf.gz")
```

## Local and Global Ancestry

```python
lai = su.read_lai("local_ancestry.msp")
adm = su.read_admixture("global_ancestry")
```

Local ancestry files are represented as {class}`snputils.LocalAncestryObject`; global ancestry and ADMIXTURE-style outputs use {class}`snputils.GlobalAncestryObject`.

## Phenotypes and IBD

```python
pheno = su.read_pheno("phenotypes.tsv")
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
