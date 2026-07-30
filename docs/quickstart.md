# Quickstart

Start with a fully self-contained synthetic dataset—no external files or downloads required:

```python
import snputils as su

snp = su.build_synthetic_snp_dataset(
    n_samples=30,
    n_snps=100,
    seed=42,
)

af = snp.allele_freq()
pcs = su.PCA(n_components=2).fit_transform(snp)

print(snp.n_samples, snp.n_snps)
print(af[:5])
print(pcs.shape)
```

## Working with your own files

The top-level package exposes the most common readers, data containers, and analysis helpers. A typical file-backed workflow loads genotypes, optional ancestry and phenotypes, runs analyses, and plots results:

```python
import snputils as su

snp = su.read_snp("cohort.vcf.gz")                   # VCF, BCF, BGEN, BED, PGEN, ...
snp = snp.filter_biallelic_variants()
snp.save("cohort.pgen")                              # convert VCF -> PLINK2

lai = su.read_lai("local_ancestry.msp")              # MSP, FLARE, or .lanc local ancestry
adm = su.read_admixture("admixture")                 # global ancestry (ADMIXTURE)
labels = su.read_labels("labels.tsv")                # sample metadata for plots
pheno = su.read_pheno("phenotypes.tsv", col="trait")
ibd = su.read_ibd("hap.ibd")

af = snp.allele_freq()                               # per-SNP allele frequencies
sex_report = su.sex_check(snp)                       # chromosome-X genetic sex QC
pcs = su.PCA(n_components=2).fit_transform(snp)
afr_af = snp.allele_freq(ancestry="AFR", laiobj=lai) # ancestry-specific AF

gwas = su.run_gwas(pheno, snp)
admix = su.run_admixture_mapping(pheno, lai)

su.viz.scatter(pcs, labels, save_path="pca.png", show=False)
su.viz.chromosome_painting(lai, "chr_paintings/")
su.viz.qq_plot(gwas)
su.viz.manhattan_plot(admix)
```

`read_snp` dispatches from the file extension and returns a {class}`snputils.SNPObject`. Explicit readers are available when you need format-specific options:

```python
bed = su.read_bed("cohort.bed")
bcf = su.read_bcf("cohort.bcf")
bgen = su.read_bgen("cohort.bgen")                   # probabilities in calldata_gp
pgen = su.read_pgen("cohort.pgen")
vcf = su.read_vcf("cohort.vcf.gz")
```

## Sample labels

{func}`snputils.read_labels` loads a TSV with `indID` and `label` columns. The table is accepted anywhere a labels path is expected—for example, pass the returned DataFrame directly to {func}`~snputils.visualization.scatter`.

## Next steps

- **Object internals and filtering** — {doc}`user_guide/data-model`
- **Format tables and writer options** — {doc}`user_guide/file-io`
- **Analysis and CLI** — {doc}`user_guide/analysis`
- **End-to-end notebooks** — {doc}`tutorials/index`
