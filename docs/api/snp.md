# SNP Data

Objects, readers, writers, and convenience functions for genotype data.

## Genotype QC and Preprocessing

`SNPObject` includes genotype QC helpers used before association testing, admixture
mapping, PCA and ancestry analysis, relatedness estimation, population-genetic
summaries, and other workflows that depend on clean sample and variant alignment.
Methods that compute metrics return NumPy arrays by default and usually accept
`as_dataframe=True` for labeled reports. Filter methods return a filtered copy
unless `inplace=True`.

### Variant and Sample Missingness

Call-rate filters are useful in most genotype-based workflows. Differential
missingness is useful when missing calls differ by phenotype, cohort, array,
sequencing center, ancestry group, or other user-defined groups.

- {meth}`~snputils.SNPObject.variant_call_rate`
- {meth}`~snputils.SNPObject.sample_call_rate`
- {meth}`~snputils.SNPObject.filter_variants_by_call_rate`
- {meth}`~snputils.SNPObject.filter_samples_by_call_rate`
- {meth}`~snputils.SNPObject.differential_missingness`
- {meth}`~snputils.SNPObject.filter_differential_missingness`

```python
variant_qc = snpobj.variant_call_rate(as_dataframe=True)
sample_qc = snpobj.sample_call_rate(as_dataframe=True)

snpobj = snpobj.filter_variants_by_call_rate(min_call_rate=0.98)
snpobj = snpobj.filter_samples_by_call_rate(min_call_rate=0.98)

# Group-specific missingness. `groups` can be labels aligned to samples, a
# mapping keyed by sample ID, a pandas Series/DataFrame, or a snputils
# phenotype/covariate object.
missingness = snpobj.differential_missingness(groups, as_dataframe=True)
snpobj = snpobj.filter_differential_missingness(groups, min_p=1e-5)
```

### Allele Frequency and Hardy-Weinberg Filters

MAF and MAC filters are common before single-variant association testing and can
also stabilize sample-level summaries such as PCA, admixture analysis or relatedness. HWE filters are
most often used as variant QC in diploid biallelic data; for case-control studies,
controls are usually preferred.

- {meth}`~snputils.SNPObject.allele_counts`
- {meth}`~snputils.SNPObject.maf`
- {meth}`~snputils.SNPObject.mac`
- {meth}`~snputils.SNPObject.filter_maf`
- {meth}`~snputils.SNPObject.filter_mac`
- {meth}`~snputils.SNPObject.hwe_pvalue`
- {meth}`~snputils.SNPObject.filter_hwe`

```python
maf = snpobj.maf(as_dataframe=True)
mac = snpobj.mac(as_dataframe=True)

snpobj = snpobj.filter_maf(maf=0.01)
snpobj = snpobj.filter_mac(mac=20)

# In case-control association studies, HWE QC is usually run in controls.
hwe = snpobj.hwe_pvalue(samples=controls, as_dataframe=True)
snpobj = snpobj.filter_hwe(min_p=1e-6, samples=controls)
```

### Duplicate Sample and Variant Identifiers

Duplicate sample IDs can break phenotype, covariate, ancestry, and IBD alignment.
Duplicate variant IDs or coordinates can make variant-level results ambiguous
across association tests, population-genetic summaries, and file conversion.

- {meth}`~snputils.SNPObject.duplicate_sample_ids`
- {meth}`~snputils.SNPObject.duplicate_variant_ids`
- {meth}`~snputils.SNPObject.duplicate_variant_coordinates`
- {meth}`~snputils.SNPObject.filter_duplicate_samples`
- {meth}`~snputils.SNPObject.filter_duplicate_variants`

```python
duplicate_samples = snpobj.duplicate_sample_ids(as_dataframe=True)
duplicate_ids = snpobj.duplicate_variant_ids(as_dataframe=True)
duplicate_coords = snpobj.duplicate_variant_coordinates(as_dataframe=True)

snpobj = snpobj.filter_duplicate_samples(keep="first")
snpobj = snpobj.filter_duplicate_variants(by="id", keep="first")
snpobj = snpobj.filter_duplicate_variants(
    by="coordinates",
    fields=("chrom", "pos", "ref", "alt"),
)
```

### LD, Heterozygosity, Imputation, and Relatedness

These helpers are mostly for sample-level QC and imputed-data QC. LD pruning is
recommended before PCA, relatedness, and heterozygosity/inbreeding summaries.
Relatedness pruning can be used before association tests, admixture mapping, PCA,
or ancestry analysis when the model assumes unrelated samples. Imputation R2/INFO
filters are specific to imputed variants and dosage/probability data.

- {meth}`~snputils.SNPObject.ld_prune_mask`
- {meth}`~snputils.SNPObject.filter_ld_pruned`
- {meth}`~snputils.SNPObject.ld_prune`
- {meth}`~snputils.SNPObject.sample_heterozygosity`
- {meth}`~snputils.SNPObject.sample_inbreeding_coefficient`
- {meth}`~snputils.SNPObject.flag_heterozygosity_outliers`
- {meth}`~snputils.SNPObject.imputation_r2`
- {meth}`~snputils.SNPObject.filter_imputation_quality`
- {meth}`~snputils.SNPObject.relatedness`
- {meth}`~snputils.SNPObject.flag_related_pairs`
- {meth}`~snputils.SNPObject.prune_related_samples`

```python
# LD pruning is useful before PCA, relatedness, and heterozygosity QC.
ld_mask = snpobj.ld_prune_mask(window_size=50, step_size=5, r2_threshold=0.2)
pruned = snpobj.filter_ld_pruned(window_size=50, step_size=5, r2_threshold=0.2)

het = pruned.sample_heterozygosity(as_dataframe=True)
outliers = pruned.flag_heterozygosity_outliers(n_sd=3)

# Imputed data can be filtered from INFO/R2 fields, genotype probabilities, or dosages.
r2 = snpobj.imputation_r2(source="auto", as_dataframe=True)
snpobj = snpobj.filter_imputation_quality(min_r2=0.8)

# GRM-relatedness is genotype-derived. IBD mode summarizes already-called IBD segments.
grm = pruned.relatedness(method="grm", scale="kinship", as_dataframe=True)
pairs = pruned.flag_related_pairs(threshold=0.0884)
unrelated = pruned.prune_related_samples(threshold=0.0884)
snpobj = snpobj.filter_samples(samples=unrelated.samples)

ibd_pairs = snpobj.flag_related_pairs(method="ibd", ibdobj=ibdobj)
```

## Genetic Sex Checking

```{eval-rst}
.. autofunction:: snputils.sex_check
```

The function uses the internally bundled Zigo distilled polynomial model. See
{doc}`../user_guide/analysis` for output interpretation, limitations, and the
Zigo paper citation.

## Objects

```{eval-rst}
.. autoclass:: snputils.SNPObject
   :members:
```

```{eval-rst}
.. autoclass:: snputils.GRGObject
   :members:
```

## Readers

```{eval-rst}
.. autoclass:: snputils.SNPReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.BEDReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.BCFReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.BGENReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.PGENReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.VCFReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.snp.io.read.vcf.VCFReaderPolars
   :members:
```

```{eval-rst}
.. autoclass:: snputils.GRGReader
   :members:
```

## Read Functions

```{eval-rst}
.. autofunction:: snputils.read_snp
```

```{eval-rst}
.. autofunction:: snputils.read_bed
```

```{eval-rst}
.. autofunction:: snputils.read_bcf
```

```{eval-rst}
.. autofunction:: snputils.read_bgen
```

```{eval-rst}
.. autofunction:: snputils.read_pgen
```

```{eval-rst}
.. autofunction:: snputils.read_vcf
```

```{eval-rst}
.. autofunction:: snputils.read_grg
```

## Writers

```{eval-rst}
.. autoclass:: snputils.BEDWriter
   :members:
```

```{eval-rst}
.. autoclass:: snputils.BGENWriter
   :members:
```

```{eval-rst}
.. autoclass:: snputils.PGENWriter
   :members:
```

```{eval-rst}
.. autoclass:: snputils.VCFWriter
   :members:
```

```{eval-rst}
.. autoclass:: snputils.BCFWriter
   :members:
```

```{eval-rst}
.. autoclass:: snputils.GRGWriter
   :members:
```
