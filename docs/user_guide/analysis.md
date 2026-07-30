---
myst:
  html_meta:
    description: Run PCA, calculate allele frequencies and f-statistics, perform GWAS and admixture mapping, and simulate genomic data in Python with snputils.
---

# Population-genetics analysis in Python

Use snputils for population-genetics workflows including dimensionality reduction, allele frequencies, f-statistics, association analysis, simulation, and command-line execution.

## Dimensionality Reduction

### PCA

Standard PCA backed by scikit-learn or PyTorch. Accepts a `SNPObject` directly.

```python
from snputils.processing import PCA, TorchPCA

pca = PCA(n_components=10)
pca.fit_transform(snpobj)

pca.X_new_    # (n_samples, n_components) embedding
pca.samples_  # sample IDs aligned with embedding rows

# GPU-accelerated via PyTorch (requires torch)
torch_pca = TorchPCA(n_components=10, device="cuda")
torch_pca.fit_transform(snpobj)
```

### mdPCA

PCA with missing-data support; ancestry-specific masking replaces non-target LAI segments with `NaN`.

```python
from snputils.processing import mdPCA

mdpca = mdPCA(
    snpobj=snpobj,
    laiobj=laiobj,
    labels_file="labels.tsv",  # columns: indID, label
    ancestry="AFR",
    is_masked=True,
    average_haplotypes=False,
    n_components=2,
    embedding_table_path="mdpca_coords.tsv",  # optional TSV export
)
# fit_transform runs automatically when all four arguments are passed at construction

mdpca.fit_transform(snpobj, laiobj, "labels.tsv", ancestry="AFR")
mdpca.X_new_   # embedding
```

### maasMDS

Multi-array, ancestry-specific multidimensional scaling. Distances are computed on ancestry-masked genotypes; multiple SNP arrays are harmonized to a shared reference and linearly calibrated so embeddings are comparable.

```python
from snputils.processing import maasMDS

mds = maasMDS(
    snpobj=[snpobj_array1, snpobj_array2],   # single object or list
    laiobj=[laiobj_array1, laiobj_array2],
    labels_file="labels.tsv",
    ancestry=0,
    is_masked=True,
    n_components=2,
)

mds.X_new_         # (n_samples, n_components)
mds.array_labels_  # array index per row
```

### Saving Embeddings

```python
from snputils.processing import save_embedding_table, embedding_dataframe_from_model

df = embedding_dataframe_from_model(pca)         # returns a DataFrame
save_embedding_table(pca, "coords.tsv")          # writes TSV
```

---

## Allele Frequencies

```python
import numpy as np
from snputils.stats import allele_freq_stream

# Streaming computation — does not load all genotypes at once
for chunk in allele_freq_stream("cohort.pgen", chunk_size=50_000):
    freqs = chunk   # (chunk_size, n_pops)

# Ancestry-specific frequencies
all_freqs = []
for chunk in allele_freq_stream(
    snpobj,
    sample_labels=labels,
    ancestry="EUR",
    laiobj=laiobj,
    chunk_size=50_000,
):
    all_freqs.append(chunk)
freqs = np.concatenate(all_freqs, axis=0)
```

---

## F-Statistics

All functions accept either a `SNPObject` or a pre-computed `(afs, counts, pops)` tuple. Block-jackknife standard errors are computed by default.

```python
from snputils.stats import f2, f3, f4, f4_ratio, d_stat, fst

# f2: pairwise branch length between populations
df = f2(snpobj, sample_labels=labels)

# f3: test whether C is admixed between A and B  (f3(A, B; C))
df = f3(snpobj, a=["EUR"], b=["AFR"], c=["AMR"], sample_labels=labels)

# f4: test for gene flow  (f4(A, B; C, D))
df = f4(snpobj, a=["EUR"], b=["AFR"], c=["EAS"], d=["AMR"], sample_labels=labels)

# f4-ratio: admixture proportion estimate
df = f4_ratio(snpobj, sample_labels=labels)

# D-statistic
df = d_stat(snpobj, a=["EUR"], b=["AFR"], c=["EAS"], d=["AMR"], sample_labels=labels)

# FST (Hudson, Weir-Cockerham, or Tsallis)
df = fst(snpobj, method="hudson", sample_labels=labels)
df = fst(snpobj, method="weir_cockerham", sample_labels=labels)
```

All statistics support ancestry-specific computation:

```python
df = f2(snpobj, sample_labels=labels, ancestry="AFR", laiobj=laiobj)
```

---

## GWAS and Admixture Mapping

### Genotype QC and preprocessing

`SNPObject` exposes genotype QC checks used across association testing, admixture
mapping, PCA and ancestry analysis, relatedness estimation, and population-genetic
summaries. The right set depends on the analysis: duplicate checks and call-rate
filters are broadly useful, differential missingness targets group-specific
artifacts, MAF/MAC and HWE filters are common in variant-level association QC,
imputation quality applies to imputed variants, and LD pruning is mainly for
sample-level summaries such as PCA, relatedness, and heterozygosity.

```python
import snputils as su

snpobj = su.read_snp("cohort.pgen")
phen = su.read_pheno("phenotypes.tsv", col="trait")

# Remove duplicate identifiers. Missing variant IDs such as "." are ignored by default.
snpobj = snpobj.filter_duplicate_samples()
snpobj = snpobj.filter_duplicate_variants(by="id")
snpobj = snpobj.filter_duplicate_variants(
    by="coordinates",
    fields=("chrom", "pos", "ref", "alt"),
)

# Missingness and allele-frequency QC. These are common before association
# testing and can also improve many downstream summaries.
snpobj = snpobj.filter_variants_by_call_rate(min_call_rate=0.98)
snpobj = snpobj.filter_samples_by_call_rate(min_call_rate=0.98)
snpobj = snpobj.filter_maf(maf=0.01)
snpobj = snpobj.filter_mac(mac=20)

# Phenotype-, cohort-, ancestry-, or batch-specific missingness. Groups can be
# a PhenotypeObject, sample-aligned labels, a mapping keyed by sample ID, or a
# pandas object.
snpobj = snpobj.filter_differential_missingness(phen, min_p=1e-5)

# HWE is often checked in controls for case-control association studies.
if not phen.is_quantitative:
    snpobj = snpobj.filter_hwe(min_p=1e-6, samples=phen.controls)

# Imputed data can be filtered using INFO/R2 fields, genotype probabilities, or
# dosage-derived estimates.
snpobj = snpobj.filter_imputation_quality(min_r2=0.8, source="auto")
```

For PCA, relatedness pruning, heterozygosity, and inbreeding QC, use autosomal,
reasonably common, preferably LD-pruned variants. This pruned object is used to
make sample-level QC decisions; those decisions can then be applied back to the
full dataset used for association testing, admixture mapping, or other analyses:

```python
qc_snps = (
    snpobj
    .filter_maf(maf=0.05)
    .filter_ld_pruned(window_size=50, step_size=5, r2_threshold=0.2)
)

het_report = qc_snps.flag_heterozygosity_outliers(n_sd=3)
related_pairs = qc_snps.flag_related_pairs(threshold=0.0884)
unrelated_qc_snps = qc_snps.prune_related_samples(threshold=0.0884)
snpobj = snpobj.filter_samples(samples=unrelated_qc_snps.samples)
```

Use {meth}`~snputils.SNPObject.variant_call_rate`, {meth}`~snputils.SNPObject.sample_call_rate`,
{meth}`~snputils.SNPObject.maf`, {meth}`~snputils.SNPObject.mac`,
{meth}`~snputils.SNPObject.hwe_pvalue`, {meth}`~snputils.SNPObject.imputation_r2`,
{meth}`~snputils.SNPObject.relatedness`, and related `as_dataframe=True` reports
when you want to inspect QC metrics before filtering. Some filters, especially
MAF/MAC and HWE, should be chosen with the analysis design in mind rather than
applied as universal defaults.

### Genetic sex checking with Zigo

{func}`~snputils.sex_check` infers genetic sex from normalized chromosome-X
genotype-class frequencies using the distilled Zigo polynomial model. It accepts
an in-memory {class}`~snputils.SNPObject`, automatically recognizes chromosome
labels `X`, `chrX`, and PLINK `23`, and supports either summed 0/1/2 dosages or
separate biallelic allele calls.

```python
import snputils as su

snpobj = su.read_snp("cohort.pgen")
report = su.sex_check(snpobj)

report[[
    "sample", "reported_sex", "inferred_sex", "status",
    "p_male", "p_female", "n_called", "qc_status",
]]
```

When `SNPObject.sample_sex` is populated from a PLINK FAM/PSAM file, it is used
as the reported sex automatically. A sample-aligned sequence or mapping can be
provided explicitly with `reported_sex=...`. Recognized values are `1`, `M`, or
`male`, and `2`, `F`, or `female`.

The result is one row per sample:

| Column | Meaning |
|--------|---------|
| `reported_sex` | Normalized sex supplied by the user or SNP metadata |
| `inferred_sex` | Zigo inference (`male`, `female`, or missing when no X calls are available) |
| `status` | `match`, `mismatch`, `not_compared`, or `unknown` |
| `p_male`, `p_female` | Model probabilities, reported explicitly rather than as a PLINK-style `F` column |
| `n_called` | Number of callable chromosome-X genotypes used |
| `genotype_0_frequency`, `genotype_1_frequency`, `genotype_2_frequency` | Normalized Zigo model features |
| `qc_status` | `pass`, `low_information`, or `no_data` |

By default, non-empty samples with fewer than 500 calls are retained but marked
`low_information`; pass `low_information_threshold=0` to disable that flag.
Samples with zero callable X genotypes are never converted into a prediction.
If chromosome metadata was deliberately omitted from an X-only object, use
`assume_x=True` explicitly.

For file-backed use, the CLI reads chromosome X through snputils' own readers:

```bash
snputils sex-check \
    --snp-path cohort.pgen \
    --sex-path reported_sex.tsv \
    --results-path sex_check.tsv
```

`--sex-path` is optional and accepts a headered table with an `IID`,
`Individual ID`, or `sample` column and a `SEX` or `Gender` column. BED and PGEN
sex metadata are used automatically when no separate table is supplied. VCF,
BCF, BED, and PGEN hard calls are supported; BGEN probabilities are not
hard-called implicitly.

The model is intended for sample QC, not diagnosis of sex-chromosome
aneuploidies. If you use this sex-check functionality in research, please cite:

```bibtex
@article{zigo2026,
    author  = {Molina-Sedano, Oscar and Mas Montserrat, Daniel and Ioannidis, Alexander G.},
    title   = {Sex checking by zygosity distributions},
    year    = {2026},
    doi     = {10.64898/2026.03.15.711924},
    url     = {https://www.biorxiv.org/content/10.64898/2026.03.15.711924},
    journal = {bioRxiv},
}
```

### Association testing

```python
import snputils as su

# GWAS: phenotype + genotype paths or in-memory objects
results = su.run_gwas(phen, snpobj)

# Admixture mapping: phenotype + local ancestry path or object
results = su.run_admixture_mapping(phen, laiobj)
```

Optional covariates go on the `covar` argument as a file path or {class}`~snputils.CovariateObject`. Covariate files are whitespace-separated tables with an `IID` column and numeric columns after it (for example `age`, `sex`). A column named `SEX` accepts `M`/`F` or `MALE`/`FEMALE` and is coded as 1/2.

For typical GWAS and admixture-mapping models, build covariates from PCs, global ancestry proportions, and clinical variables, then merge:

```python
pc_covar = su.CovariateObject.from_embedding(pca, n_components=10)
anc_covar = su.CovariateObject.from_global_ancestry(admobj)
clinical = su.CovariateObject.from_file("covariates.txt")
covar = su.CovariateObject.merge(pc_covar, clinical)

results = su.run_gwas(phen, snpobj, covar=covar)
results = su.run_admixture_mapping(phen, laiobj, covar=covar)
```

`from_global_ancestry` drops the last ancestry column by default. `from_embedding` requires sample-level coordinates (`average_haplotypes=True` on phased PCA). The same blocks can be composed with {func}`~snputils.phenotype.build_association_covariates`.

File-only or manual construction still works:

```python
results = su.run_gwas(phen, snpobj, covar="covariates.txt")
```

Both return a DataFrame with per-variant statistics (p-values, effect sizes) suitable for Manhattan/QQ plotting. See {doc}`../api/tools` for full signatures and {doc}`../api/cli` for file-backed CLI equivalents.

---

## Simulation

`OnlineSimulator` generates admixed haplotypes from real reference panels using crossover models.

```python
from snputils.simulation.simulator import OnlineSimulator
import pandas as pd

meta = pd.read_csv("metadata.tsv", sep="\t")   # columns: Sample, Population, Latitude, Longitude
sim = OnlineSimulator(
    snp_data=snpobj,
    meta=meta,
    genetic_map=gmap_df,   # columns: chm, pos, cM  (optional)
    window_size=1000,
    expand_haplotypes=True,
)

snps, labels_d, labels_c, changepoints = sim.simulate(
    batch_size=256,
    num_generation_max=10,
    device="cuda",         # or 'cpu'
)
# snps:       (batch_size, n_snps)  tensor
# labels_d:   (batch_size, n_windows)  discrete population labels
# labels_c:   (batch_size, n_windows, 3)  continuous lat/lon n-vectors
# changepoints: (batch_size, n_windows)  ancestry change-point mask
```

Via CLI:

```bash
snputils simulate \
    --snp cohort.pgen \
    --metadata metadata.tsv \
    --genetic-map gmap.tsv \
    --output-dir sim_batches/ \
    --batch-size 256 \
    --n-batches 10 \
    --num-generations 10
```

---

## Command-Line Interface

The `snputils` CLI mirrors these analyses for file-backed workflows. Subcommands include `pca`, `mdpca`, `maasmds`, `gwas`, `admixture-map`, `simulate`, `plot-manhattan`, and `plot-qq`.

```bash
snputils gwas \
    --phe-id trait \
    --phe-path phenotypes.tsv \
    --snp-path cohort.pgen \
    --results-path gwas.tsv.gz

snputils mdpca \
    --snp-path cohort.pgen \
    --lai-path local_ancestry.msp \
    --labels-file labels.tsv \
    --ancestry AFR \
    --coords mdpca_coords.tsv \
    --plot mdpca.pdf
```

Full subcommand reference and additional examples: {doc}`../api/cli`.
