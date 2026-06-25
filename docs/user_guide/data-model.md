# Data Model

snputils centers work around typed objects that hold NumPy arrays together with the metadata needed to interpret them. Every object supports `obj["attr"]` and `obj["attr"] = value` dict-style access alongside the usual property interface.

## SNPObject

Stores genotype calls, optional genotype probabilities, and variant/sample metadata.

```python
import snputils as su

snpobj = su.read_snp("cohort.pgen")

snpobj.n_samples      # number of samples
snpobj.n_snps         # number of variants
snpobj.genotypes      # (n_snps, n_samples) or (n_snps, n_samples, 2)
snpobj.calldata_gp    # (n_snps, n_samples, n_probs) — BGEN genotype probabilities
snpobj.calldata_lai   # (n_snps, n_samples*2) or (n_snps, n_samples, 2) — SNP-level LAI
snpobj.samples        # (n_samples,)
snpobj.variants_chrom # (n_snps,)
snpobj.variants_pos   # (n_snps,)
snpobj.variants_id    # (n_snps,) — rsID or chrom_pos
snpobj.variants_ref   # (n_snps,)
snpobj.variants_alt   # (n_snps,)
snpobj.variants_cm    # (n_snps,) — centimorgan positions
snpobj.sample_fid     # (n_samples,) — PLINK family ID / population label
snpobj.sample_sex     # (n_samples,) — PLINK sex code
snpobj.ancestry_map   # dict mapping ancestry codes to region names
```

**Filtering and manipulation:**

```python
# Filter by sample name or index (include or exclude; inplace or copy)
sub = snpobj.filter_samples(samples=["sample1", "sample2"])
sub = snpobj.filter_samples(indexes=[0, 1, 2], include=False)

# Filter by chromosome, position, index, or boolean mask
chr1 = snpobj.filter_variants(chrom="1")
region = snpobj.filter_variants(chrom="22", pos=range(1_000_000, 2_000_000))
sub = snpobj.filter_variants(mask=maf_mask)

# Collapse phased haplotypes to allele counts
snpobj.sum_strands(inplace=True)

# Deep copy
snpobj2 = snpobj.copy()
```

**Writing:**

```python
su.BEDWriter(snpobj, "out.bed").write()
su.PGENWriter(snpobj, "out.pgen").write(vzs=True)      # .pvar.zst compression
su.VCFWriter(snpobj, "out.vcf", phased=True).write(chrom_partition=True)
su.BGENWriter(snpobj, "out.bgen").write(compression="zstd", bit_depth=16)
```

## LocalAncestryObject

Stores window-level LAI calls. Each row is a genomic window; each column is a haplotype.

```python
laiobj = su.read_lai("local_ancestry.msp")

laiobj.n_samples      # number of diploid samples
laiobj.n_ancestries   # number of unique ancestry codes
laiobj.lai            # (n_windows, n_haplotypes)
laiobj.haplotypes     # list of haplotype identifiers
laiobj.samples        # list of sample identifiers
laiobj.ancestry_map   # dict of int code → region name
laiobj.chromosomes    # (n_windows,)
laiobj.physical_pos   # (n_windows, 2) — start/end bp
laiobj.centimorgan_pos # (n_windows, 2) — start/end cM
laiobj.window_sizes   # (n_windows,) — SNPs per window
```

## GlobalAncestryObject

Stores ADMIXTURE-style ancestry proportions.

```python
admobj = su.read_admixture("run_prefix")

admobj.Q   # (n_samples, n_ancestries) — per-sample proportions
admobj.P   # (n_snps, n_ancestries) — per-ancestry allele frequencies
admobj.n_samples
admobj.n_ancestries
admobj.n_snps
```

## IBDObject

Stores IBD segment tables (hap-IBD or ancIBD).

```python
ibdobj = su.read_ibd("segments.hapibd")

ibdobj.n_segments
ibdobj.sample_id_1    # (n_segments,)
ibdobj.sample_id_2    # (n_segments,)
ibdobj.haplotype_id_1 # (n_segments,) — 1 or 2; -1 if unknown
ibdobj.haplotype_id_2 # (n_segments,)
ibdobj.chrom          # (n_segments,)
ibdobj.start          # (n_segments,) — bp
ibdobj.end            # (n_segments,) — bp
ibdobj.length_cm      # (n_segments,)
ibdobj.segment_type   # (n_segments,) — 'IBD1' / 'IBD2' for ancIBD

# Restrict segments to windows where both individuals carry a target ancestry
ancestry_ibd = ibdobj.restrict_to_ancestry(
    laiobj=laiobj,
    ancestry="AFR",
    method="clip",   # 'clip' trims to tract boundaries; 'strict' drops entire segment
    min_cm=2.0,
)
```

## PhenotypeObject / MultiPhenotypeObject

```python
phen = su.read_pheno("pheno.tsv")           # single-phenotype file
mphen = su.MultiPhenReader("multi.tsv").read()  # multiple phenotypes

phen.phenotype    # (n_samples,)
phen.samples      # (n_samples,)
mphen.phenotypes  # (n_samples, n_phenotypes)
```

## CovariateObject

Numeric covariates aligned to sample IDs for {func}`~snputils.run_gwas` and {func}`~snputils.run_admixture_mapping`. Values must be finite floating-point numbers with explicit column names.

Factory helpers build blocks from common sources: {meth}`~snputils.CovariateObject.from_file` (clinical tables), {meth}`~snputils.CovariateObject.from_embedding` (fitted PCA, mdPCA, or maasMDS), and {meth}`~snputils.CovariateObject.from_global_ancestry` (ADMIXTURE `Q`, dropping the last ancestry by default). Combine blocks with {meth}`~snputils.CovariateObject.merge` or {func}`~snputils.phenotype.build_association_covariates`.

```python
covar = su.CovariateObject(
    samples=["sample1", "sample2"],
    values=[[58.0, 1.0], [49.0, 2.0]],
    covariate_names=["age", "sex"],
)

covar.samples           # sample IDs
covar.values            # (n_samples, n_covariates)
covar.covariate_names   # column names
covar.n_samples
covar.n_covariates
```

## GRGObject

Genotype Representation Graph (requires `pip install "snputils[grg]"`).

```python
grg = su.read_grg("cohort.grg")
snpobj = grg.to_snpobject(sum_strands=False)   # materialise dense genotype matrix

# Build a GRG from VCF
su.vcf_to_grg("cohort.vcf.gz", "cohort.grg")
su.vcf_to_igd("cohort.vcf.gz", "cohort.igd")  # intermediate IGD step
```
