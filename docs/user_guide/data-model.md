# Data Model

snputils centers work around lightweight objects that hold data arrays plus the sample, variant, ancestry, phenotype, or segment metadata needed to interpret them.

## Core Objects

- {class}`snputils.SNPObject` stores genotype calls and variant metadata.
- {class}`snputils.LocalAncestryObject` stores ancestry calls along chromosomes.
- {class}`snputils.GlobalAncestryObject` stores ADMIXTURE-style ancestry proportions.
- {class}`snputils.PhenotypeObject` and {class}`snputils.MultiPhenotypeObject` store trait data.
- {class}`snputils.GRGObject` stores genotype representation graph data.
- {class}`snputils.IBDObject` stores identity-by-descent segment tables.

Most objects provide a `copy()` method, shape/count properties, and filtering helpers that return object-level views or copies instead of loose arrays.

```python
snpobj = su.read_snp("cohort.pgen")
snpobj.n_samples, snpobj.n_snps
```

Use the API reference when you need exact constructor arguments or property names.
