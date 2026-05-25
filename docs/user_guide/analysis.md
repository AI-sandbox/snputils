# Analysis

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
    average_strands=False,
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

`from_global_ancestry` drops the last ancestry column by default. `from_embedding` requires sample-level coordinates (`average_strands=True` on phased PCA). The same blocks can be composed with {func}`~snputils.phenotype.build_association_covariates`.

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
    make_haploid=True,
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
